"""Optional slow-quality SAM inference. Optimize per-session W+, never weights."""
import time


class LatentQuality:
    def __init__(self,engine,mode):
        import lpips
        from sam_losses import IDLoss
        self.engine=engine;self.mode=mode;self.torch=engine.torch
        self.identity=IDLoss().to(engine.dev).eval().requires_grad_(False)
        self.perceptual=lpips.LPIPS(net='alex',verbose=False).to(engine.dev).eval().requires_grad_(False)
        for model in [engine.net,engine.identity_encoder,engine.age_net]:model.eval().requires_grad_(False)

    def run(self,item,target_age,assert_active):
        torch=self.torch;engine=self.engine
        import torch.nn.functional as F
        import numpy as np
        import cv2
        with torch.inference_mode(False),torch.enable_grad():
            # Tensors created inside the caller's inference_mode cannot be used
            # for autograd. Create normal tensors inside this explicit scope.
            x=engine.tensor(item.aligned).contiguous()
            def condition(age):return torch.cat([x,torch.full_like(x[:,:1],age/100)],1)
            def decode(code):return engine.net(code,input_code=True,input_is_full=True,randomize_noise=False,resize=False)
            if item.quality_state is None:
                assert_active();started=time.monotonic()
                reference=item.reference_rgb
                if reference is None:raise RuntimeError('High-quality mode requires the original aligned reference')
                ref=engine.tensor(cv2.resize(np.asarray(reference),(512,512),interpolation=cv2.INTER_AREA)).contiguous()
                with torch.no_grad():
                    base=engine.net.pretrained_encoder(x)+engine.net.latent_avg
                    source_code=base+engine.identity_encoder(condition(item.reference_age))
                    identity_reference=self.identity.feats(x).detach()
                grid=torch.linspace(0,1,512,device=engine.dev);yy,xx=torch.meshgrid(grid,grid,indexing='ij')
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
            state=item.quality_state
            with torch.no_grad():
                original=state['base']+engine.identity_encoder(condition(target_age))
                target=state['anchor']+original-state['source']
            if self.mode=='anchor' or target_age==item.reference_age:
                with torch.no_grad():return decode(target).detach()
            if self.mode!='constrained':raise RuntimeError('Unknown quality profile')
            def age_logits(image):
                value=F.interpolate((image+1)/2,(224,224),mode='bilinear',align_corners=False)
                return engine.age_net.net((value-engine.age_net.m)/engine.age_net.sd)[2].float()
            with torch.no_grad():
                prior=F.interpolate(decode(target),(256,256),mode='area')
                teacher=F.interpolate(decode(original),(256,256),mode='area')
                target_prob=age_logits(teacher).softmax(1)
            delta=torch.zeros_like(target[:,:8]).requires_grad_(True)
            optimizer=torch.optim.Adam([delta],lr=.012)
            for step in range(180):
                if step%10==0:assert_active()
                optimizer.param_groups[0]['lr']=.012*(.2+.8*(1-step/180))
                code=target+torch.cat([delta,torch.zeros_like(target[:,8:])],1)
                y=F.interpolate(decode(code),(256,256),mode='area')
                ident=(1-(self.identity.feats(y)*state['identity']).sum(1)).mean()
                age_loss=F.kl_div(age_logits(y).log_softmax(1),target_prob,reduction='batchmean')
                color=F.l1_loss(F.avg_pool2d(y,32),F.avg_pool2d(x,32))
                loss=.8*ident+.5*age_loss+.6*self.perceptual(y,prior).mean()+.05*delta.square().mean()+.1*color
                optimizer.zero_grad(set_to_none=True);loss.backward()
                if not torch.isfinite(loss) or not torch.isfinite(delta.grad).all():raise RuntimeError('Invalid constrained quality gradient')
                optimizer.step()
            assert_active()
            with torch.no_grad():return decode(target+torch.cat([delta,torch.zeros_like(target[:,8:])],1)).detach()
