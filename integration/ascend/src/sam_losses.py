"""Frozen loss nets (gradients flow to inputs only): IR-SE50 ArcFace ID (official SAM crop), interim age/ethnicity from FairFace classifier."""
import sys, torch, torch.nn as nn, torch.nn.functional as F
B = "/data/home/1120231322/scow/face-age"; sys.path.insert(0, f"{B}/src/SAM"); sys.path.insert(0, f"{B}/src")
from models.encoders.model_irse import Backbone
from faceattr import Net
class IDLoss(nn.Module):
    def __init__(s):
        super().__init__(); s.net = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode="ir_se")
        s.net.load_state_dict(torch.load(f"{B}/models/irse50/model_ir_se50.pth", map_location="cpu")); s.net.eval(); s.pool = nn.AdaptiveAvgPool2d((112, 112))
        for p in s.net.parameters(): p.requires_grad_(False)
    def feats(s, x): return s.net(s.pool(x[:, :, 35:223, 32:220]))
    def cos(s, a, b): return (s.feats(a) * s.feats(b).detach()).sum(1)
class AgeEst(nn.Module):
    def __init__(s):
        super().__init__(); s.net = Net(); s.net.load_state_dict(torch.load(f"{B}/models/faceattr/convnext_t_fairface.pt", map_location="cpu")); s.net.eval()
        for p in s.net.parameters(): p.requires_grad_(False)
        s.register_buffer("mid", torch.tensor([1, 6, 15, 25, 35, 45, 55, 65, 75.]) / 100)
        s.register_buffer("m", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)); s.register_buffer("sd", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    def forward(s, x):
        x = F.interpolate((x + 1) / 2, size=(224, 224), mode="bilinear", align_corners=False); r, g, a = s.net((x - s.m) / s.sd)
        return a.float().softmax(1) @ s.mid, r.float().softmax(1)[:, [0, 6]].sum(1)
