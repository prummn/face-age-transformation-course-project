#!/bin/bash
# Fetch all models/data for face-age into persistent storage. sha256 verified where the hub provides LFS hashes.
cd /data/home/1120231322/scow/face-age; P=scripts/pfetch.sh; mkdir -p models data src logs
mark(){ echo "[$(date +%T)] $*" | tee -a logs/fetch_all.status; }
lfs(){ # lfs TYPE REPO PATTERN -> lines "sha256 size path"
  local api=https://hf-mirror.com/api/$1/$2; curl -s -m 30 "$api?blobs=true" | python3 -c "import sys,json,fnmatch
d=json.load(sys.stdin)
for s in d.get(\"siblings\",[]):
    if fnmatch.fnmatch(s[\"rfilename\"], \"$3\"): print((s.get(\"lfs\") or {}).get(\"sha256\",\"-\"), s.get(\"size\",0), s[\"rfilename\"])"; }
get_repo(){ # get_repo TYPE REPO PATTERN DEST [BASEURL]
  local T=$1 R=$2 PAT=$3 D=$4 BU=${5:-}; mkdir -p $D
  lfs $T $R "$PAT" > $D/.manifest
  while read sha size f; do
    if [ -n "$BU" ]; then BASE=$BU $P $R $f $D/$f 8; elif [ $T = datasets ]; then BASE=https://hf-mirror.com/datasets/$R/resolve/main $P $R $f $D/$f 8; else $P $R $f $D/$f 8; fi || { mark "FAIL $R $f"; continue; }
    [ "$sha" != "-" ] && { echo "$sha  $D/$f" | sha256sum -c --quiet || mark "SHA_MISMATCH $D/$f"; }
  done < $D/.manifest
  mark "DONE $R ($(wc -l < $D/.manifest) files)"; }
get_repo models aimi-models/sam-aging "sam_ffhq_aging.pt" models/sam
get_repo models matbee/model_ir_se50_pytorch "model_ir_se50.pth" models/irse50
get_repo models opencv/face_detection_yunet "face_detection_yunet_2023mar.onnx" models/yunet
curl -s -m 20 -o models/yunet/LICENSE https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/face_detection_yunet/LICENSE
get_repo models timm/convnext_tiny.fb_in22k_ft_in1k "*" models/convnext_tiny
get_repo datasets HuggingFaceM4/FairFace "1.25/*" data/fairface
[ -d src/SAM ] || { git clone -q https://github.com/yuval-alaluf/SAM.git src/SAM && (cd src/SAM && git checkout -q c1895aef275e702fba7560284dc16df60d65210e) && mark "DONE SAM code c1895ae"; }
get_repo models Qwen/Qwen2.5-VL-7B-Instruct "*" models/Qwen2.5-VL-7B-Instruct https://www.modelscope.cn/models/Qwen/Qwen2.5-VL-7B-Instruct/resolve/master
get_repo datasets CaptionEmporium/flickr-megalith-10m-internvl2-multi-caption "train/*.parquet" data/megalith_caps
mark ALL_FETCH_DONE
