"""Optional slow-quality SAM inference. Optimize per-session W+, never weights."""
import time


def resolve_wplus_precision(torch, device, mode, requested):
    if requested not in ("auto", "fp32", "bf16", "bf16-full"):
        raise ValueError("FACE_AGE_WPLUS_PRECISION must be auto, fp32, bf16, or bf16-full")
    if requested == "bf16-full" and mode != "constrained":
        raise ValueError("Full BF16 requires constrained W+ refinement")
    if requested == "fp32" or (requested == "auto" and mode != "constrained"):
        return "fp32"
    with torch.cuda.device(device):
        supported = torch.cuda.is_bf16_supported(including_emulation=False)
    if supported:
        return "bf16-full" if requested == "bf16-full" else "bf16"
    if requested in ("bf16", "bf16-full"):
        raise ValueError(f"Native BF16 is unavailable on {device}")
    return "fp32"


def _fp32_modulation(torch, forward):
    def project(style):
        with torch.autocast(device_type="cuda", enabled=False):
            return forward(style.float())
    return project


class LatentQuality:
    def __init__(self, engine, mode, *, precision="fp32"):
        import lpips
        from sam_losses import IDLoss
        self.engine=engine;self.mode=mode;self.torch=engine.torch
        self.precision = resolve_wplus_precision(self.torch, engine.dev, mode, precision)
        if self.precision == "bf16":
            from models.stylegan2.model import ModulatedConv2d

            # Leave the SAM checkpoint in FP32. With FP32 style projection and
            # weights, demodulation stays in FP32 before autocast runs the convolution.
            for layer in engine.net.decoder.modules():
                if isinstance(layer, ModulatedConv2d):
                    projection = layer.modulation
                    projection.forward = _fp32_modulation(self.torch, projection.forward)
        self.identity=IDLoss().to(engine.dev).eval().requires_grad_(False)
        self.perceptual=lpips.LPIPS(net='alex',verbose=False).to(engine.dev).eval().requires_grad_(False)
        for model in (engine.net, engine.identity_encoder, engine.age_net):
            model.eval().requires_grad_(False)
        if self.precision == "bf16-full":
            dtype = self.torch.bfloat16
            self.identity.to(dtype=dtype)
            self.perceptual.to(dtype=dtype)

    def _decode(self, code):
        options = dict(input_code=True, input_is_full=True, randomize_noise=False, resize=False)
        if self.precision == "bf16":
            with self.torch.autocast(device_type="cuda", dtype=self.torch.bfloat16):
                image = self.engine.net(code, **options)
            # Keep all downstream loss and optimizer math in FP32.
            return image.float()
        return self.engine.net(code, **options)

    def run(self, item, target_age, assert_active):
        return self.run_many(item, [target_age], assert_active)

    def run_many(self, item, target_ages, assert_active):
        torch=self.torch;engine=self.engine
        import torch.nn.functional as F
        import numpy as np
        import cv2
        with torch.inference_mode(False),torch.enable_grad():
            # Tensors created inside the caller's inference_mode cannot be used
            # for autograd. Create normal tensors inside this explicit scope.
            x=engine.tensor(item.aligned).contiguous()
            def condition(age):return torch.cat([x,torch.full_like(x[:,:1],age/100)],1)
            decode = self._decode
            if item.quality_state is None:
                assert_active();started=time.monotonic()
                reference=item.reference_rgb
                if reference is None:raise RuntimeError('High-quality mode requires the original aligned reference')
                ref=engine.tensor(cv2.resize(np.asarray(reference),(512,512),interpolation=cv2.INTER_AREA)).contiguous()
                with torch.no_grad():
                    base=engine.net.pretrained_encoder(x)+engine.net.latent_avg
                    source_code=base+engine.identity_encoder(condition(item.reference_age))
                    identity_reference=self.identity.feats(x).detach()
                grid=torch.linspace(0,1,512,device=engine.dev,dtype=x.dtype);yy,xx=torch.meshgrid(grid,grid,indexing='ij')
                mask=(.2+.8*((1-((xx-.5)/.31)**2-((yy-.53)/.36)**2)/.25).clamp(0,1))[None,None]
                pivot=source_code.clone().requires_grad_(True)
                optimizer=torch.optim.Adam([pivot],lr=.02)
                for step in range(200):
                    if step%10==0:assert_active()
                    optimizer.param_groups[0]['lr']=.02*(.2+.8*(1-step/200))
                    y=decode(pivot);y256=F.interpolate(y,(256,256),mode='area');y512=F.interpolate(y,(512,512),mode='area')
                    pixel=((y512-ref).abs()*mask).sum()/(mask.sum()*3)
                    ident=(1-(self.identity.feats(y256)*identity_reference).sum(1)).mean()
                    loss=.5*pixel+.8*self.perceptual(y256,x).mean()+.8*ident+.002*(pivot-source_code).square().mean()
                    optimizer.zero_grad(set_to_none=True);loss.backward()
                    if not torch.isfinite(loss) or not torch.isfinite(pivot.grad).all():raise RuntimeError('Invalid quality anchor gradient')
                    optimizer.step()
                assert_active()
                item.quality_state=dict(base=base.detach(),source=source_code.detach(),anchor=pivot.detach(),identity=identity_reference,
                    anchor_seconds=time.monotonic()-started)
            state = item.quality_state
            cross_indices = [i for i, age in enumerate(target_ages) if age != item.reference_age]
            with torch.no_grad():
                inputs = torch.cat([condition(age) for age in target_ages])
                original = state['base'] + engine.identity_encoder(inputs)
                target = state['anchor'] + original - state['source']
                if self.mode in ('texture', 'texture_robust'):
                    direction = original - state['source']
                    target = state['anchor'] + torch.cat([direction[:, :8], .5 * direction[:, 8:]], 1)
                if self.mode == 'anchor' or not cross_indices:
                    return decode(target).detach()

            if self.mode not in ('constrained', 'texture', 'texture_robust'):
                raise RuntimeError('Unknown quality profile')
            refined = self._fit_targets(x, state['identity'], target[cross_indices],
                                        original[cross_indices], assert_active)
            with torch.no_grad():
                target[cross_indices] = refined
                return decode(target).detach()

    def _fit_targets(self, x, identity_reference, target, original, assert_active):
        import torch.nn.functional as F

        torch = self.torch
        engine = self.engine

        def age_logits(image):
            value = F.interpolate((image + 1) / 2, (224, 224), mode='bilinear', align_corners=False)
            logits = engine.age_net.net((value - engine.age_net.m) / engine.age_net.sd)[2]
            return logits if self.precision == "bf16-full" else logits.float()

        def age_views(image):
            return [image, F.avg_pool2d(image, 3, 1, 1), image[:, :, 8:-8, 8:-8]]

        with torch.no_grad():
            prior = F.interpolate(self._decode(target), (256, 256), mode='area')
            teacher = F.interpolate(self._decode(original), (256, 256), mode='area')
            teacher_views = age_views(teacher) if self.mode == 'texture_robust' else [teacher]
            target_probs = [age_logits(view).softmax(1) for view in teacher_views]
            reference_color = F.avg_pool2d(x, 32)

        delta = torch.zeros_like(target[:, :8]).requires_grad_(True)
        tail = torch.zeros_like(target[:, 8:])
        optimizer = torch.optim.Adam([delta], lr=.012)
        for step in range(180):
            if step % 10 == 0:
                assert_active()
            optimizer.param_groups[0]['lr'] = .012 * (.2 + .8 * (1 - step / 180))
            code = target + torch.cat([delta, tail], 1)
            image = F.interpolate(self._decode(code), (256, 256), mode='area')
            identity = 1 - (self.identity.feats(image) * identity_reference).sum(1)
            views = age_views(image) if self.mode == 'texture_robust' else [image]
            age_loss = sum(F.kl_div(age_logits(view).log_softmax(1), probability, reduction='none').sum(1)
                           for view, probability in zip(views, target_probs)) / len(views)
            color = (F.avg_pool2d(image, 32) - reference_color).abs().flatten(1).mean(1)
            perceptual = self.perceptual(image, prior).flatten(1).mean(1)
            regularization = delta.square().flatten(1).mean(1)
            # Sum independent per-age losses to keep each lane's gradient scale.
            loss = (.8 * identity + .5 * age_loss + .6 * perceptual + .05 * regularization + .1 * color).sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if not torch.isfinite(loss) or not torch.isfinite(delta.grad).all():
                raise RuntimeError('Invalid constrained quality gradient')
            optimizer.step()

        assert_active()
        with torch.no_grad():
            return target + torch.cat([delta, tail], 1)
