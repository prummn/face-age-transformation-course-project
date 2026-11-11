#!/bin/bash
# repair3: same as repair2 (CaptionEmporium part) but source = huggingface.co via local Xray proxy socks5h://127.0.0.1:10899.
# Verifies every manifest file by sha256; re-fetches bad/missing ones (resume with -C -). Writes the same completion marks as repair2.
cd /data/home/1120231322/scow/face-age
export PROXY=socks5h://127.0.0.1:10899
mark(){ echo "[$(date +%T)] $*" | tee -a logs/fetch_all.status; }
check_one(){ local sha=$1 f=$2 D=$3 U=$4
  if [ -f "$D/$f" ] && { [ "$sha" = "-" ] || echo "$sha  $D/$f" | sha256sum -c --quiet 2>/dev/null; }; then echo "OK $f"; return 0; fi
  mkdir -p $(dirname "$D/$f")
  for i in 1 2 3 4 5 6; do
    curl -sS -L --fail -x $PROXY --retry 8 --retry-delay 15 --retry-all-errors -C - -o "$D/$f.part" "$U/$f" 2>>logs/repair3_curl.log
    if [ "$sha" = "-" ] || echo "$sha  $D/$f.part" | sha256sum -c --quiet 2>/dev/null; then mv "$D/$f.part" "$D/$f"; echo "FETCHED $f"; return 0; fi
    rm -f "$D/$f.part"; sleep $((20*i))
  done; return 1; }
export -f check_one
D=data/megalith_caps; U=https://huggingface.co/datasets/CaptionEmporium/flickr-megalith-10m-internvl2-multi-caption/resolve/main
while read sha size f; do echo "$sha $f"; done < $D/.manifest | xargs -P 4 -n 2 bash -c "check_one \$0 \$1 $D $U || echo BADFILE \$1" > logs/repair3_megalith_caps.out 2>&1
b=$(grep -c BADFILE logs/repair3_megalith_caps.out)
[ "$b" = 0 ] && mark "DONE CaptionEmporium verified (39 files) [hf+proxy]" || mark "FAIL CaptionEmporium verify ($b bad) [hf+proxy]"
grep -q "VERIFIED Qwen" logs/fetch_all.status && grep -q "DONE CaptionEmporium verified" logs/fetch_all.status && mark ALL_FETCH_DONE
