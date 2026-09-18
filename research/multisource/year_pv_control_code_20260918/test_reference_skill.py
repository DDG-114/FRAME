import torch

from train import reference_skill_weights


target = torch.zeros(2, 3, 4, device='cuda')
reference = target + target.new_tensor([1., 2., 3., 0.])
alternatives = torch.stack((target + target.new_tensor([2., 1., 3., 0.]),
                            target + target.new_tensor([3., 3., 6., 0.])))
weights = reference_skill_weights(reference, alternatives, target)
torch.testing.assert_close(weights, target.new_tensor([.75, 0., 0., 1.]))
residual = torch.ones_like(target, requires_grad=True)
(.1 * (residual.square() * weights).mean()).backward()
assert residual.grad[..., 0].abs().sum() > 0
assert residual.grad[..., 1:3].abs().sum() == 0
assert residual.grad[..., 3].abs().sum() > 0
print('REFERENCE_SKILL_GPU_CHECK_PASSED')
