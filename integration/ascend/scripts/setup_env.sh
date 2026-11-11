set -ex
cd /data/home/1120231322/scow/face-age; source env.sh
[ -x venv/bin/python ] || { curl -sS -o /tmp/get-pip.py https://bootstrap.pypa.io/get-pip.py && python3 /tmp/get-pip.py -q && python3 -m pip install -q virtualenv && python3 -m virtualenv -q venv; }; source venv/bin/activate
pip install -U pip wheel setuptools
pip install --no-deps torch==2.12.0+cpu torchvision==0.27.0+cpu --index-url https://download.pytorch.org/whl/cpu
pip install -c constraints.txt filelock typing-extensions sympy networkx jinja2 fsspec
pip install -c constraints.txt torch_npu==2.12.0 numpy pyyaml decorator attrs psutil scipy cloudpickle ml-dtypes tornado absl-py
pip install -c constraints.txt transformers accelerate safetensors huggingface_hub sentencepiece protobuf pillow opencv-python-headless pyarrow pandas timm lpips==0.1.4 qwen-vl-utils tqdm
python -c "import torch, torch_npu, torchvision, transformers, timm, lpips, cv2; print(torch.__version__, torch_npu.__version__, torchvision.__version__, transformers.__version__); x=torch.randn(512,512).npu(); print(\"npu ok\", float((x@x).sum()))"
echo SETUP_DONE
