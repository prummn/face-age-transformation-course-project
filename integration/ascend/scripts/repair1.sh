cd /data/home/1120231322/scow/face-age
mark(){ echo "[$(date +%T)] $*" | tee -a logs/fetch_all.status; }
# 1) SAM code (retry)
for i in 1 2 3 4 5; do [ -d src/SAM/models ] && break; rm -rf src/SAM; timeout 300 git clone -q https://github.com/yuval-alaluf/SAM.git src/SAM && (cd src/SAM && git checkout -q c1895aef275e702fba7560284dc16df60d65210e) || sleep 10; done
[ -d src/SAM/models ] && mark "DONE SAM code c1895ae (repair)" || mark "FAIL SAM code clone"
# 2) FairFace corrupted shard: serial re-download + sha check
F=data/fairface/1.25/train-00000-of-00004-e715178553977907.parquet; SHA=9c27e70f0cf6f1ba54e5d7fcc7c69bd5b1c8342205796e9a9bb8f507125ef70c
for i in 1 2 3; do curl -sS -L --retry 5 -o $F.re https://hf-mirror.com/datasets/HuggingFaceM4/FairFace/resolve/main/1.25/$(basename $F) && echo "$SHA  $F.re" | sha256sum -c --quiet && { mv $F.re $F; mark "DONE FairFace shard0 re-fetched sha OK"; break; }; done
echo "$SHA  $F" | sha256sum -c --quiet || mark "FAIL FairFace shard0 still mismatched"
# 3) timm small files (plain curl)
for f in config.json; do curl -sS -L -o models/convnext_tiny/$f https://hf-mirror.com/timm/convnext_tiny.fb_in22k_ft_in1k/resolve/main/$f; done
rm -rf models/convnext_tiny/*.parts models/convnext_tiny/*.tmp; ls -la models/convnext_tiny
