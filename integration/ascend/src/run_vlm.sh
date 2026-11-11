# run_vlm.sh EVALDIR "npu list"
D=$1; NP=($2); N=${#NP[@]}; cd /data/home/1120231322/scow/face-age; source env.sh
for i in $(seq 0 $((N-1))); do ASCEND_RT_VISIBLE_DEVICES=${NP[$i]} SHARD=$i NSHARD=$N python src/vlm_judge.py $D/_vlm_$i.json "$D/*_rec.png" "$D/*_a??.png" "$D/*_input.png" > logs/vlm_$(basename $D)_$i.log 2>&1 & done; wait
[ $(cat logs/vlm_$(basename $D)_*.log | grep -c "^VLM_DONE") -eq $N ] && echo ALL_VLM_DONE || echo VLM_INCOMPLETE
