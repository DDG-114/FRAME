import copy
from dataclasses import replace
import tempfile
from pathlib import Path
import unittest
import torch
from model import Config, MODES, Adapter, PaperForecaster, objective, masked_token_mean
from train import evaluation_mode, rng_state, restore_rng

torch.set_num_threads(1)


def batch(length=32, horizon=6, size=2):
    return dict(history=torch.randn(size,length,30),history_mask=torch.ones(size,length,30,dtype=torch.bool),
        future=torch.randn(size,horizon,6),future_mask=torch.ones(size,horizon,6,dtype=torch.bool),
        availability=torch.ones(size,10),task_id=torch.zeros(size,dtype=torch.long),
        cadence_id=torch.zeros(size,dtype=torch.long),semantic=torch.randn(size,4096))


class PaperEquations(unittest.TestCase):
    def test_operator_matches_explicit_A_diag_beta_B(self):
        torch.manual_seed(2027)
        a=Adapter(Config()).eval()
        with torch.no_grad():a.B.normal_(std=.02)
        h,code=torch.randn(2,7,128),torch.randn(2,128)
        out,penalty,gamma,alpha,candidate,applied=a(h,code,'dual')
        matrices=torch.stack([a.A@torch.diag(gamma[i]*alpha[i])@a.B for i in range(2)])
        torch.testing.assert_close(out,h+torch.bmm(h,matrices),rtol=1e-5,atol=2e-7)
        torch.testing.assert_close(penalty,candidate.square().mean((1,2)))

    def test_adapter_parameter_count_matches_manuscript(self):
        m=PaperForecaster()
        self.assertEqual([sum(p.numel() for p in a.parameters()) for a in m.adapters],[3201]*4)
        self.assertEqual(sum(p.numel() for a in m.adapters for p in a.parameters()),12804)

    def test_penalty_is_before_strength_gate(self):
        a=Adapter(Config()).eval()
        with torch.no_grad():a.B.normal_(std=.02)
        h,code=torch.randn(2,5,128),torch.randn(2,128)
        one=a(h,code,'dual')
        with torch.no_grad():a.strength.bias.fill_(2.)
        two=a(h,code,'dual')
        self.assertTrue(torch.equal(one[1],two[1]))
        self.assertGreater(two[-1].norm(),one[-1].norm())

    def test_zero_initialization_preserves_all_four_control_predictions(self):
        b=batch()
        reference=None
        for mode in MODES:
            if mode == 'no_availability':
                continue
            torch.manual_seed(2027)
            m=PaperForecaster(Config(mode=mode)).eval()
            r=m(b)
            self.assertEqual(float(r['residual_penalty_per_sample'].sum()),0.)
            if reference is None:reference=r['prediction'].detach()
            else:self.assertTrue(torch.equal(reference,r['prediction']))

    def test_first_gradient_updates_B_then_A_and_gates(self):
        m=PaperForecaster()
        b=batch();o=torch.optim.AdamW(m.parameters(),lr=3e-4)
        r=m(b);r['prediction'].square().mean().backward()
        self.assertGreater(float(m.adapters[0].B.grad.norm()),0.)
        self.assertEqual(float(m.adapters[0].A.grad.norm()),0.)
        o.step();o.zero_grad(set_to_none=True)
        m(b)['prediction'].square().mean().backward()
        self.assertGreater(float(m.adapters[0].A.grad.norm()),0.)
        self.assertGreater(float(m.adapters[0].rank_gate.weight.grad.norm()),0.)

    def test_frozen_semantic_input_and_trainable_projection(self):
        m=PaperForecaster()
        with torch.no_grad():
            for a in m.adapters:a.B.normal_(std=.01)
        b=batch();b['semantic'].requires_grad_(True)
        m(b)['prediction'].square().mean().backward()
        self.assertIsNone(b['semantic'].grad)
        for name in ['semantic_projection','availability_projection','W1','W2','W3','W4']:
            self.assertGreater(float(getattr(m,name).weight.grad.norm()),0.,name)
        self.assertGreater(float(m.pool_query.grad.norm()),0.)

    def test_four_blocks_each_have_one_following_adapter(self):
        m=PaperForecaster().eval();trace=[]
        handles=[]
        for i in range(4):
            handles.append(m.blocks[i].register_forward_hook(lambda module,args,out,i=i:trace.append('block'+str(i))))
            handles.append(m.adapters[i].register_forward_hook(lambda module,args,out,i=i:trace.append('adapter'+str(i))))
        r=m(batch())
        self.assertEqual(trace,[name+str(i) for i in range(4) for name in ['block','adapter']])
        self.assertEqual(len(r['diagnostics']),4)
        for h in handles:h.remove()

    def test_masked_values_cannot_change_prediction(self):
        m=PaperForecaster().eval();a=batch();a['history_mask'][:,2:8,4]=False;a['future_mask'][:,:,5]=False
        b=copy.deepcopy(a);b['history'][:,2:8,4]=float('nan');b['future'][:,:,5]=float('nan')
        self.assertTrue(torch.equal(m(a)['prediction'],m(b)['prediction']))

    def test_patch_count_and_horizon_shape(self):
        m=PaperForecaster().eval()
        for length,horizon,count in [(168,24,20),(1008,144,125),(336,52,41)]:
            r=m(batch(length,horizon,1))
            self.assertEqual(r['patch_count'],count)
            self.assertEqual(tuple(r['prediction'].shape),(1,horizon,1))

    def test_future_observed_zero_is_distinguishable_from_missing(self):
        m=PaperForecaster().eval();a=batch();a['future'][:,:,5]=0.;b=copy.deepcopy(a)
        b['future_mask'][:,:,5]=False
        self.assertFalse(torch.equal(m(a)['prediction'],m(b)['prediction']))
        self.assertEqual(m.known_projection.in_features,22)

    def test_objective_uses_physical_error_over_TRAIN_kappa(self):
        prediction=torch.tensor([[[2.],[100.]]],requires_grad=True)
        result=dict(prediction=prediction,residual_penalty_per_sample=torch.tensor([3.]))
        loss,pred,energy=objective(result,torch.zeros_like(prediction),torch.tensor([[[True],[False]]]),torch.tensor([[[4.]]]),2.)
        self.assertAlmostEqual(float(pred),3.5)
        self.assertAlmostEqual(float(loss),3.5003,places=6)
        loss.backward();self.assertEqual(float(prediction.grad[0,1,0]),0.)

    def test_token_pool_excludes_padding_without_L2_or_centering(self):
        h=torch.tensor([[[2.,4.],[4.,8.],[100.,200.]]])
        self.assertTrue(torch.equal(masked_token_mean(h,torch.tensor([[1,1,0]])),torch.tensor([[3.,6.]])))

    def test_full_state_roundtrip_predictions_optimizer_and_rng(self):
        torch.manual_seed(2027);m=PaperForecaster();o=torch.optim.AdamW(m.parameters(),lr=3e-4,betas=(.9,.999))
        b=batch()
        for _ in range(3):
            o.zero_grad(set_to_none=True);m(b)['prediction'].square().mean().backward();o.step()
        m.eval();expected=m(b)['prediction'].detach();rng=rng_state()
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'checkpoint.pt';torch.save(dict(model=m.state_dict(),optimizer=o.state_dict(),rng=rng),path)
            payload=torch.load(path,map_location='cpu',weights_only=False);other=PaperForecaster().eval();other.load_state_dict(payload['model'])
            second=torch.optim.AdamW(other.parameters(),lr=3e-4,betas=(.9,.999));second.load_state_dict(payload['optimizer'])
            self.assertTrue(torch.equal(expected,other(b)['prediction']))
            self.assertEqual([int(v['step']) for v in second.state.values()],[int(v['step']) for v in o.state.values()])
            restore_rng(payload['rng']);a=torch.rand(5);restore_rng(rng);self.assertTrue(torch.equal(a,torch.rand(5)))

    def test_evaluation_restores_mode_and_random_state(self):
        m=PaperForecaster().train();before=torch.get_rng_state().clone()
        with evaluation_mode(m):
            self.assertFalse(m.training);m(batch())
        self.assertTrue(m.training);self.assertTrue(torch.equal(before,torch.get_rng_state()))

    def test_forecast_has_no_anchor_or_task_specific_head(self):
        m=PaperForecaster().eval();b=batch();r=m(b)['prediction'];b['seasonal_anchor']=torch.full_like(r,10000.)
        self.assertTrue(torch.equal(r,m(b)['prediction']))
        self.assertFalse(any('lora' in n or 'private' in n or 'anchor' in n for n,_ in m.named_parameters()))

    def test_BF16_math_attention_forward_and_gradient_are_finite(self):
        torch.backends.mha.set_fastpath_enabled(False)
        m=PaperForecaster();b=batch()
        with torch.autocast('cpu',dtype=torch.bfloat16):loss=m(b)['prediction'].square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))


if __name__=='__main__':unittest.main()
