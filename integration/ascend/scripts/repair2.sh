#!/bin/bash
# Verify every manifest file by sha256 (LFS) or existence (non-LFS); re-fetch bad/missing ones with ONE connection + backoff (avoids HTTP 429).
cd /data/home/1120231322/scow/face-age
mark(){ echo "[$(date +%T)] $*" | tee -a logs/fetch_all.status; }
fix_repo(){ # fix_repo DEST URLBASE PARALLEL
  local D=$1 U=$2 P=$3; local bad=0
  check_one(){ local sha=$1 f=$2 D=$3 U=$4
    if [ -f "$D/$f" ] && { [ "$sha" = "-" ] || echo "$sha  $D/$f" | sha256sum -c --quiet 2>/dev/null; }; then return 0; fi
    mkdir -p $(dirname "$D/$f")
    for i in 1 2 3 4 5 6; do
      curl -sS -L --fail --retry 8 --retry-delay 15 --retry-all-errors -C - -o "$D/$f.part" "$U/$f" 2>>logs/repair2_curl.log
      if [ "$sha" = "-" ] || echo "$sha  $D/$f.part" | sha256sum -c --quiet 2>/dev/null; then mv "$D/$f.part" "$D/$f"; return 0; fi
      rm -f "$D/$f.part"; sleep $((20*i))
    done; return 1; }
  export -f check_one
  while read sha size f; do echo "$sha $f"; done < $D/.manifest | xargs -P $P -n 2 bash -c "check_one \$0 \$1 $D $U || echo BADFILE \$1" > logs/repair2_$(basename $D).out 2>&1
  bad=$(grep -c BADFILE logs/repair2_$(basename $D).out); echo $bad; }
b=$(fix_repo models/Qwen2.5-VL-7B-Instruct https://www.modelscope.cn/models/Qwen/Qwen2.5-VL-7B-Instruct/resolve/master 3)
[ "$b" = 0 ] && mark "VERIFIED Qwen/Qwen2.5-VL-7B-Instruct" || mark "FAIL Qwen verify ($b bad)"
b=$(fix_repo data/megalith_caps https://hf-mirror.com/datasets/CaptionEmporium/flickr-megalith-10m-internvl2-multi-caption/resolve/main 2)
[ "$b" = 0 ] && mark "DONE CaptionEmporium verified (39 files)" || mark "FAIL CaptionEmporium verify ($b bad)"
grep -q "VERIFIED Qwen" logs/fetch_all.status && grep -q "DONE CaptionEmporium verified" logs/fetch_all.status && mark ALL_FETCH_DONE
