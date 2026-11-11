#!/bin/bash
# Resumable face-age flow. Each stage writes state/<name>.done; rerunning skips finished stages. All data/results under $B (persistent).
cd /data/home/1120231322/scow/face-age; source env.sh
NPUS="0 1 4 5 6 7"; VIS=0,1,4,5,6,7          # NPU 2,3 are used by other services: never use
mark(){ echo "[$(date "+%F %T")] $*" | tee -a logs/flow.status; }
stage(){ local n=$1; shift; [ -f state/$n.done ] && { mark "skip $n"; return 0; }; mark "start $n"; if "$@" > logs/stage_$n.log 2>&1; then touch state/$n.done; mark "done $n"; else mark "FAIL $n (see logs/stage_$n.log)"; exit 1; fi; }
waitfor(){ while ! grep -q "$2" $1 2>/dev/null; do sleep 30; done; }
mark "flow start"
waitfor logs/setup_env.log SETUP_DONE; source env.sh; which python
waitfor logs/fetch_all.status "DONE yuval\|DONE SAM code"
stage patch_sam bash src/patch_sam.sh
waitfor logs/fetch_all.status "DONE HuggingFaceM4/FairFace"
stage faceattr env ASCEND_RT_VISIBLE_DEVICES=0 python src/faceattr.py &          # NPU 0, in parallel with data download
FA=$!
waitfor logs/fetch_all.status "DONE CaptionEmporium"
stage stage1 python src/data_stage1.py
stage download python src/data_download.py data/stage1_tierA.parquet
stage stage3a python src/data_stage3a.py
wait $FA; [ -f state/faceattr.done ] || { mark "FAIL faceattr"; exit 1; }
stage stage3b env ASCEND_RT_VISIBLE_DEVICES=0 python src/data_stage3b.py
stage align python src/prep_align.py
waitfor logs/fetch_all.status ALL_FETCH_DONE
evaluate(){ # evaluate NAME ENC
  stage gen_$1 env ASCEND_RT_VISIBLE_DEVICES=0 python src/eval_sam.py --enc "$2" --out out/eval_$1
  stage vlm_$1 bash -c "bash src/run_vlm.sh out/eval_$1 \"1 4 5 6 7\" | grep -q ALL_VLM_DONE"; }
evaluate base ""
stage train_l5 env ASCEND_RT_VISIBLE_DEVICES=$VIS torchrun --nproc_per_node 6 --master_port 29801 src/train_sam_ft.py --steps 250 --aging_lambda 5.0 --save_every 250 --out out/ft_l5
stage train_l1 env ASCEND_RT_VISIBLE_DEVICES=$VIS torchrun --nproc_per_node 6 --master_port 29802 src/train_sam_ft.py --steps 1000 --aging_lambda 1.0 --save_every 250 --out out/ft_l1
evaluate l5_250 out/ft_l5/encoder_step250.pt
for s in 250 500 750 1000; do evaluate l1_$s out/ft_l1/encoder_step$s.pt; done
stage summary python src/summarize.py out/summary.json out/eval_base out/eval_l5_250 out/eval_l1_250 out/eval_l1_500 out/eval_l1_750 out/eval_l1_1000
mark "FLOW_DONE"
