#!/bin/bash
# pfetch.sh REPO FILE OUTPATH [N]   (BASE env overrides URL prefix). Parallel range download with retries; size-checked.
set -e; REPO=$1; FILE=$2; OUT=$3; N=${4:-8}
URL=${BASE:-https://hf-mirror.com/$REPO/resolve/main}/$FILE
TOTAL=$(curl -sL -r 0-0 -D - -o /dev/null $URL | grep -i "^content-range" | tail -1 | sed "s#.*/##" | tr -dc 0-9)
[ -z "$TOTAL" ] && { echo "no size for $URL"; exit 1; }
[ -f "$OUT" ] && [ "$(stat -c %s $OUT)" = "$TOTAL" ] && { echo "skip $OUT"; exit 0; }
if [ "$TOTAL" -lt 1048576 ]; then mkdir -p $(dirname $OUT); curl -sS -L --fail --retry 5 -o $OUT.tmp $URL && [ "$(stat -c %s $OUT.tmp)" = "$TOTAL" ] && mv $OUT.tmp $OUT && echo "OK $OUT $TOTAL" && exit 0; exit 1; fi
mkdir -p $(dirname $OUT); T=$OUT.parts; mkdir -p $T; SEG=$(( (TOTAL+N-1)/N ))
for i in $(seq 0 $((N-1))); do
  a=$((i*SEG)); b=$((a+SEG-1)); [ $b -ge $TOTAL ] && b=$((TOTAL-1))
  ( f=$T/p$i; want=$((b-a+1))
    while [ "$(stat -c %s $f 2>/dev/null || echo 0)" -lt $want ]; do
      have=$(stat -c %s $f 2>/dev/null || echo 0)
      curl -sS -L --fail --speed-limit 20000 --speed-time 45 -r $((a+have))-$b $URL >> $f || sleep 3
    done ) &
done; wait
cat $(for i in $(seq 0 $((N-1))); do echo $T/p$i; done) > $OUT.tmp
[ "$(stat -c %s $OUT.tmp)" = "$TOTAL" ] && mv $OUT.tmp $OUT && rm -rf $T && echo "OK $OUT $TOTAL"
