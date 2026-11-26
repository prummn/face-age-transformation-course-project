"""Same pooling/resize forwards with deterministic matrix-transpose backwards.

CUDA adaptive-pool/interpolate backward may use atomic additions. W+ Adam can
amplify those rounding differences. These small loss-network resizes use fixed
separable weights for their input gradients; network weights remain unchanged.
"""
import math
import torch
import torch.nn.functional as F

_CACHE={}


def weights(source,target,kind,device,dtype):
    key=(source,target,kind,str(device),dtype)
    if key not in _CACHE or torch.is_inference(_CACHE[key]):
        # Model warmup/age estimation use inference_mode. These constants are
        # later saved by autograd during W+ fitting, so cache normal tensors.
        with torch.inference_mode(False):
            matrix=torch.zeros(target,source,dtype=dtype)
            for j in range(target):
                if kind=='area':
                    low=j*source//target;high=math.ceil((j+1)*source/target)
                    matrix[j,low:high]=1/(high-low)
                else:
                    coordinate=max(0.,(j+.5)*source/target-.5);low=math.floor(coordinate);fraction=coordinate-low
                    matrix[j,min(low,source-1)]+=1-fraction;matrix[j,min(low+1,source-1)]+=fraction
            _CACHE[key]=matrix.to(device)
    return _CACHE[key]


class _Resize(torch.autograd.Function):
    @staticmethod
    def forward(ctx,image,size,kind):
        h,w=image.shape[-2:];oh,ow=size
        ctx.save_for_backward(weights(h,oh,kind,image.device,image.dtype),weights(w,ow,kind,image.device,image.dtype))
        if kind=='area':return F.adaptive_avg_pool2d(image,size)
        return F.interpolate(image,size,mode='bilinear',align_corners=False)

    @staticmethod
    def backward(ctx,gradient):
        wy,wx=ctx.saved_tensors
        return torch.matmul(wy.t(),torch.matmul(gradient,wx)),None,None


def resize(image,size,kind='area'):
    return _Resize.apply(image,tuple(size),kind)


class AreaPool(torch.nn.Module):
    def __init__(self,size):
        super().__init__();self.size=(size,size) if isinstance(size,int) else tuple(size)
    def forward(self,image):return resize(image,self.size)


def replace_pools(module):
    for name,child in list(module.named_children()):
        if isinstance(child,torch.nn.AdaptiveAvgPool2d):setattr(module,name,AreaPool(child.output_size))
        else:replace_pools(child)


def verify():
    # Check forwards and CPU reference input gradients with independent native ops.
    cases=[]
    for kind,shape,size in [('area',(2,3,19,23),(11,13)),('bilinear',(2,3,19,23),(11,13))]:
        x=torch.randn(*shape,requires_grad=True);y=resize(x,size,kind)
        native=F.adaptive_avg_pool2d(x,size) if kind=='area' else F.interpolate(x,size,mode='bilinear',align_corners=False)
        gradient=torch.randn_like(y);a=torch.autograd.grad(y,x,gradient,retain_graph=True)[0];b=torch.autograd.grad(native,x,gradient)[0]
        error=float((a-b).abs().max());assert torch.equal(y,native) and error<4e-6,(kind,error)
        cases.append(dict(kind=kind,forward_exact=True,cpu_gradient_max_error=error))
    for kind,size in [('area',(1,1)),('bilinear',(3,5))]:
        with torch.inference_mode():resize(torch.ones(1,3,7,9),size,kind)
        x=torch.randn(1,3,7,9,requires_grad=True)
        resize(x,size,kind).sum().backward()
        assert torch.isfinite(x.grad).all()
        cases.append(dict(kind=kind,inference_warmup_then_backward=True))
    return cases
