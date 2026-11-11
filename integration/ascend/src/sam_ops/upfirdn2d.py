# Pure-PyTorch replacement for SAM/StyleGAN2 CUDA upfirdn2d (rosinality reference upfirdn2d_native).
import torch
import torch.nn.functional as F
def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    b, c, h, w = input.shape
    out = upfirdn2d_native(input.reshape(-1, h, w, 1), kernel, up, up, down, down, pad[0], pad[1], pad[0], pad[1])
    return out.reshape(b, c, out.shape[1], out.shape[2])
def upfirdn2d_native(input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1):
    _, in_h, in_w, minor = input.shape; kernel_h, kernel_w = kernel.shape
    out = input.view(-1, in_h, 1, in_w, 1, minor); out = F.pad(out, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1])
    out = out.view(-1, in_h * up_y, in_w * up_x, minor)
    out = F.pad(out, [0, 0, max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)])
    out = out[:, max(-pad_y0, 0): out.shape[1] - max(-pad_y1, 0), max(-pad_x0, 0): out.shape[2] - max(-pad_x1, 0), :]
    out = out.permute(0, 3, 1, 2).reshape([-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1])
    out = F.conv2d(out, torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w))
    out = out.reshape(-1, minor, in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1, in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1).permute(0, 2, 3, 1)
    return out[:, ::down_y, ::down_x, :]
