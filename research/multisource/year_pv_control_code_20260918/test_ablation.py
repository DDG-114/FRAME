import torch
from model import InternalAdapter

torch.manual_seed(42)
torch.cuda.set_per_process_memory_fraction(.04)
block = InternalAdapter(16, affine_adapter=True, adapter_after_ff=True).cuda().eval()
x = torch.randn(2, 3, 4, 16, device='cuda')
c = torch.randn_like(x)
block.ablation = 'no_adapter'
torch.testing.assert_close(block(x, c), x + block.ff(x))
torch.testing.assert_close(block(x, c), block(x, c * 2))
block.ablation = 'static_gate'
inputs = []
handle = block.control.register_forward_pre_hook(lambda module, args: inputs.append(args[0].detach()))
block(x, c)
handle.remove()
assert torch.count_nonzero(inputs[-1]) == 0
assert inputs[-1].shape == (2, 3, 4, 32)
print('ABLATION_BLOCK_CHECK_PASSED')
