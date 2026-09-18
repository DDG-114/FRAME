import torch
from model import scale_pv_control

x = torch.randn(2, 5, 4, 96, requires_grad=True)
original = x.detach().clone()
for scale in (0.25, 0.5):
    assert scale_pv_control(x, 2, scale) is x
    for h in (24, 168):
        z = scale_pv_control(x, h, scale)
        torch.testing.assert_close(z[:, :, [0, 1, 3]], x[:, :, [0, 1, 3]])
        torch.testing.assert_close(z[:, :, 2], x[:, :, 2] * scale)
        grad = torch.autograd.grad(z.sum(), x)[0]
        torch.testing.assert_close(grad[:, :, 2], torch.full_like(grad[:, :, 2], scale))
        torch.testing.assert_close(x, original)
assert scale_pv_control(x, 168, 1.) is x
print('PV_CONTROL_CHECK_PASSED')
