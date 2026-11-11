"""Independent ethnicity judge (eval only): Qwen2.5-VL-7B next-token probs over A-G. p_asian_vlm = P(A)+P(B)."""
import sys, os, glob, json, torch, torch_npu
from torch_npu.contrib import transfer_to_npu
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image
B = "/data/home/1120231322/scow/face-age"; M = f"{B}/models/Qwen2.5-VL-7B-Instruct"; dev = "npu:0"; torch.npu.set_device(dev)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(M, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
proc = AutoProcessor.from_pretrained(M, min_pixels=128 * 28 * 28, max_pixels=256 * 28 * 28)
Q = ("Look only at the face of the person in this photo. Which group does the person most likely appear to belong to? "
     "A) East Asian B) Southeast Asian C) White D) Black E) Indian F) Middle Eastern G) Latino. Answer with one letter.")
letters = list("ABCDEFG"); ids = [proc.tokenizer.encode(l, add_special_tokens=False)[0] for l in letters]
paths = sorted(sum([glob.glob(g) for g in sys.argv[2:]], []))[int(os.environ.get("SHARD", 0))::int(os.environ.get("NSHARD", 1))]; res = {}
with torch.no_grad():
    for p in paths:
        text = proc.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": Q}]}], tokenize=False, add_generation_prompt=True)
        inp = proc(text=[text], images=[Image.open(p).convert("RGB")], return_tensors="pt").to(dev)
        pr = torch.softmax(model(**inp).logits[0, -1].float()[ids], 0).cpu().numpy(); res[p] = dict(zip(letters, [round(float(v), 4) for v in pr]))
json.dump(res, open(sys.argv[1], "w")); print("VLM_DONE", len(res))
