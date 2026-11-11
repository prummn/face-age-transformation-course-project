# Install pure-PyTorch ops into the SAM checkout (originals kept as *.cuda_orig).
D=/data/home/1120231322/scow/face-age/src/SAM/models/stylegan2/op
for f in fused_act upfirdn2d; do [ -f $D/$f.py.cuda_orig ] || cp $D/$f.py $D/$f.py.cuda_orig; cp /data/home/1120231322/scow/face-age/src/sam_ops/$f.py $D/$f.py; done; echo SAM_PATCHED
