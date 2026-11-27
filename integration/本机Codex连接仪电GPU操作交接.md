# 本机 Codex 通过 SSH 使用仪电 GPU

更新：2026-11-27。本文交接的是**本机运行 Codex、远端执行代码和 GPU 任务**的工作方式。已有仪电工作区可以复用，不需要在服务器重新安装或启动 Codex。

## 直接给本机 Codex 的提示词

将下面整段交给自己电脑上的 Codex；它应自行连接、阅读文件并执行后续任务，不需要你逐条运行本文的 SDK 示例。

```text
你运行在我的本机电脑上。请通过本机已有 SSH Host 别名 yicloud，操作仪电平台开发工作区，并通过平台任务 SDK 使用远端 GPU。保持 Codex 运行在本机，不在远端安装、启动或嵌套调用 Codex。

先运行一个短 SSH 命令确认连接，再读取远端文档：
/root/face-age-transformation-course-project/integration/本机Codex连接仪电GPU操作交接.md

项目：
- 远端目录 /root/face-age-transformation-course-project
- 仓库 https://github.com/prummn/face-age-transformation-course-project
- 当前研究分支 dev/mobei-gan-portable-20261116
- 先检查远端 git branch/status/log，保留未提交文件；不要 reset、clean、强制切分支或覆盖 deployment。不要误用 main 或旧演示分支。
- 先复用远端现有源码、权重、数据、SDK、缓存；本机不需要下载模型或训练数据。

平台机制：
- ssh yicloud 登录的是无 GPU 的开发工作区；GPU 必须用仪电 SDK 创建 Job，不能在该工作区直接 python train.py 就认为用了 GPU。
- SDK /root/face-age-transformation-course-project/deployment/sdk。
- 复用 integration/deploy_yicloud.py 的 client() 和 IMAGE。
- 服务器现有 YICLOUD_PUBLIC_KEY/YICLOUD_SECRET_KEY 仅供工作区 SDK 使用，不能输出、复制到本机、写入 Git 或放进 GPU Job.Env。
- Project=fdj-infra，ZoneId=cn-shanghai-a，QuotaGroup=fdj_gpu_taskpool。
- MountGPFS=gpfs://gpfs2/haobinzhang:/root，任务和工作区在 /root 下共享文件。
- GPU=1、Replicas=1；本项目所有推理和实验合计最多一张卡。提交前查 Pending/Starting/Inqueue/Running/Stopping，已有本项目任务则等待或在明确不再需要时按回执停止，不结束无关任务。
- 工作区下载准备、GPU 容器离线计算；继承平台镜像 Torch，在任务 /tmp 建独立 venv，不升级共享 Torch/CUDA，不照搬旧昇腾安装脚本。
- 每次实验独立目录、固定源码/配置/样本和权重哈希、保存 request.json/receipt.json、worker.log、device.json 和结果；设置明确时长上限；结束核对平台终态并释放卡。

本地与远端：
- 每次本机命令工具调用都明确使用 ssh yicloud 执行远端命令；上一次 ssh cd 不会让下次本机命令自动在服务器执行。
- 可远端编码，也可 scp 少量源码到本机编辑后传到远端暂存目录，检查差异后应用。不要 scp -r 整仓覆盖已有项目，不改运行任务的源码快照。
- 用 Python/Pillow 等在远端生成少量代表图，再 scp 到本机供你读图；图像文件的远端路径不是本机路径。
- SSH 身份验证沿用我的本机配置及 ssh-agent，不索取或展示私钥、密码。若 BatchMode 失败但交互 SSH 可用，只说明具体缺失条件。

当前研究约束：
- 主线是 GAN，同一输入人物生成多个目标年龄；SAM 是基线，不是强制模型。
- 先读 integration/研究进展与同学接手.md、integration/quality/20261127_课程技术路线与模型选型边界.md、integration/quality/20261127_近期GAN年龄编辑方法筛选.md。
- 当前重点是亚洲高龄输出的纹理自然度、身份保持和足够老化；不要把强度当岁数、把估龄器分数当真实年龄、把同龄重建收益当跨龄成功。
- 保留默认 Release 和所有研究资产，不自动替换未通过的候选。

先完成只读接手并简要报告：实际分支提交、脏文件、平台认证是否可用、本项目活动 GPU 任务、现有资产位置。随后执行我在本机交给你的具体工作；不要为了验证连接无故重跑训练或启动演示。
```

## 三个执行位置

| 位置 | 做什么 | 是否需要 GPU / Codex |
| --- | --- | --- |
| 你的本机，当前是 Windows PowerShell | 运行 Codex、SSH、编辑器、浏览器和图片查看 | Codex 在这里运行；不需要本机 GPU |
| `ssh yicloud` 后的开发工作区 | 保存源码/数据/模型、下载准备、CPU 检查、使用 SDK 提交和管理任务、网页网关 | 当前没有 GPU；不需要在这里运行 Codex |
| 平台创建的 GPU 任务容器 | 执行推理、训练、GPU 评测；写回共享目录 | 只申请一张 GPU；不运行 Codex |

文件与控制关系：

```text
本机 Codex ──SSH──> 仪电开发工作区 ──SDK CreateJob──> GPU 任务容器
                        │                              │
                        └──── /root GPFS 共享目录 ───────┘
本机浏览器 ──SSH 端口转发──> 工作区网页网关 ──已有专用隧道──> GPU 后端
```

不需要先 SSH 到一个固定 GPU IP，也不要在工作区安装 NVIDIA 驱动。任务在哪个物理设备执行由平台分配，以 `device.json` 和容器内 `nvidia-smi` 为准；历史实验分到 H200，不承诺每次相同型号。

## 本机连接与命令执行

在**本机 PowerShell**执行，复用你已经能用的 `yicloud` 别名，其账号、端口和跳板不需要重新填写：

```powershell
ssh -o UpdateHostKeys=no yicloud
```

本机 Codex 的短命令通常这样写：

```powershell
ssh -o BatchMode=yes -o ConnectTimeout=15 -o UpdateHostKeys=no yicloud 'hostname; pwd'
ssh -o BatchMode=yes -o UpdateHostKeys=no yicloud 'cd /root/face-age-transformation-course-project && git branch --show-current && git status --short && git log -1 --oneline'
ssh -o UpdateHostKeys=no yicloud 'cat /root/face-age-transformation-course-project/integration/本机Codex连接仪电GPU操作交接.md'
```

`BatchMode=yes` 适合自动执行，但需要现有密钥/agent认证能直接使用；若只有交互登录能成功，先由用户在本机完成正常认证。不要用 `StrictHostKeyChecking=no`、删除 known_hosts 或把私钥传上服务器来绕过问题。`UpdateHostKeys=no` 沿用此前额外 RSA 主机密钥更新签名报错的处理，仍保留正常主机身份校验。

SSH 配置和程序属于实际运行 Codex 的环境：原生 Windows 与 WSL 的 `~/.ssh/config`、agent、路径可能不同。若 Codex 在 WSL 内，先确认该环境也能解析 `yicloud`；不要假定 PowerShell 能连就代表 WSL 已配置。

对较长 Python 脚本，本机 PowerShell 可以通过标准输入发往服务器执行，避免多层转义。例如只检查凭据是否存在，**不输出值**：

```powershell
$remoteCode = @'
import os
from pathlib import Path
root = Path('/root/face-age-transformation-course-project')
print('repo_exists:', root.is_dir())
print('sdk_exists:', (root / 'deployment/sdk/yicloud').is_dir())
for name in ['YICLOUD_PUBLIC_KEY', 'YICLOUD_SECRET_KEY']:
    print(name + '_present:', bool(os.environ.get(name)))
'@
$remoteCode | ssh -T -o BatchMode=yes -o UpdateHostKeys=no yicloud 'python3 -'
```

本轮已确认当前工作区有 SDK 和两项凭据；本机非交互 SSH 是否继承相同环境需要本机 Codex验证。如果缺失，先比较正常交互登录的环境，使用平台已有的合法初始化方式；不要 `env`、`printenv`、`cat /proc/*/environ` 或将完整 shell 配置打印到对话。不要运行 `setup_codex_cli.sh --start`：那是旧的新服务器安装方式，不是本文目标。

## 远端代码与资产位置

以下全部是**服务器路径**：

| 路径 | 用途 |
| --- | --- |
| `/root/face-age-transformation-course-project` | 现有主仓库，优先复用 |
| `integration/deploy_yicloud.py` | SDK 客户端、镜像常量、演示 start/status/stop/gateway |
| `deployment/sdk` | 本平台已取得的 Python SDK，Git 不包含它 |
| `deployment/model-files/face-age` | 默认演示模型与辅助模型 |
| `deployment/SAM` | 固定版本 SAM 上游及运行适配 |
| `deployment/gan-models` | 其他官方模型；如 StyleGANEX |
| `deployment/wheels` | 已缓存的离线 Python 依赖 |
| `deployment/training-runs/<实验名-时间>` | 实验快照、回执、日志、检查点、图片和指标 |
| `deployment/runs/<时间>` | 演示 GPU 任务的日志和回执 |
| `deployment/current.json` | **演示任务**回执；不是所有实验的统一任务清单 |
| `deployment/logs` | 网页网关和专用隧道日志 |
| `integration/models/*.json`、`integration/quality/evidence/` | 模型/资源清单及可共享证据 |

Git 不含所有模型、图片、数据和 SDK。本机仅克隆 Git 不能恢复仪电现有全部资产；先用 SSH 利用服务器已有文件。不要全盘扫描或递归加载整批图片。

新接手先检查当前分支和工作区，再查远端更新；不要无条件 `git pull` 覆盖或合并。本文编写时分支为 `dev/mobei-gan-portable-20261116`，提交基于 `41033dc`；文档推送后 HEAD 会继续前进，以实际查询为准。已有未提交文件 `integration/quality/20261116_重建与潜变量质量改进.md`，不要提交、还原或覆盖这份既有改动。

推荐由本机 Codex直接经 SSH 读取和修改远端源码，保持现有环境路径。若用本机编辑器，可先传单个文件到暂存目录，核对 diff 后再应用，不直接覆盖已存在且有修改的同名文件。每个新任务在提交前复制自己的源码和配置到独立 `run/source`，记录提交和文件哈希；任务排队后也不要再改快照。

## GPU 平台参数与提交方式

当前项目已使用并核实的参数：

```text
API host       https://gate.yicloud.com.cn
SDK region     cn-shanghai
Project        fdj-infra
ZoneId         cn-shanghai-a
QuotaGroup     fdj_gpu_taskpool
MountGPFS      gpfs://gpfs2/haobinzhang:/root
Image          从 integration/deploy_yicloud.py 的 IMAGE 导入
GPU            "1"
Replicas       1
CPU            "4"（按实验实际需要调整）
Memory         "32Gi"（按实验实际需要调整）
Internet       False
RestartPolicy  "Never"
MaxWaitDuration     "10m"
MaxRunningDuration "20m"（先小规模验证，再按实际任务设置）
```

共享挂载参数属于这份工作区，不要原样用到另一账号或全新工作区。`GPU="1"` 配合 `Replicas=1` 才是一张卡；多个 Job 或多个 TaskSpec 也会分别占卡。演示服务占的卡也计入总预算。停止是异步操作，`Stopping` 期间不能马上再开新任务。

**提交前查活动任务**。下面的代码在服务器仓库根目录运行，或像上一节那样通过本机 SSH 标准输入运行；不要在本机直接导入远端 SDK：

```python
import sys
sys.path.insert(0, '/root/face-age-transformation-course-project/integration')
from deploy_yicloud import client

job, ctx = client()
result = job.list_jobs(ctx, job.models.ListJobsReq(
    Project='fdj-infra', ZoneId='cn-shanghai-a',
    JobNames='face-age', Self=True, Limit=100,
    Phases='Pending,Running,Inqueue,Starting,Stopping'))
for item in result.Items or []:
    print(item.JobId, item.Name, item.Phase)
print('active_count:', len(result.Items or []))
```

该筛选只覆盖约定 `face-age` 名称、当前平台身份的项目任务，不能声称查到了全平台占用。新实验必须沿用 `face-age-...` 命名；若更换平台身份、发现其他名称或结果超过一页，要扩大只读核查。多个 Codex 会话不得同时提交；现有“先查再提交”不是跨会话原子锁。

### 如何向 GPU 下发一段新代码

优先看已成功的提交器 [styleganex_study.py](quality/styleganex_study.py) 中 `prepare/submit` 和 [run_styleganex.sh](quality/run_styleganex.sh)。它是具体旧实验，不要为了测试连接重复执行它。新任务由 Codex 按实际需求准备独立快照与 worker。

下面是 **SDK 提交模板，不是要求现在创建任务**。在服务器仓库根目录执行前，必须已经准备好 `run/source/run.sh` 和全部离线资产。`run` 指向本次独立目录，不填旧实验目录：

```python
import json, sys
from pathlib import Path
ROOT = Path('/root/face-age-transformation-course-project')
sys.path.insert(0, str(ROOT / 'integration'))
from deploy_yicloud import client, IMAGE

run = ROOT / 'deployment/training-runs/实际实验名-UTC时间'
runner = run / 'source/run.sh'
assert runner.is_file(), '先准备源码快照与 worker'
assert not (run / 'request.json').exists(), '已有请求记录，先核对是否已提交'
assert not (run / 'receipt.json').exists(), '不要重复提交'
job, ctx = client()
active = job.list_jobs(ctx, job.models.ListJobsReq(
    Project='fdj-infra', ZoneId='cn-shanghai-a', JobNames='face-age',
    Self=True, Limit=100, Phases='Pending,Running,Inqueue,Starting,Stopping'))
if active.Items:
    raise RuntimeError('本项目已有活动任务，保持总计最多一张卡')

name = 'face-age-' + run.name
payload = dict(
    Name=name, Project='fdj-infra', ZoneId='cn-shanghai-a',
    QuotaGroup='fdj_gpu_taskpool', Priority='3', UsePrivateMachine=False,
    AutoDeleteDuration='168h', TaskSpecs={'experiment': dict(
        Command=['bash', str(runner), str(ROOT), str(run)], Env={}, Image=IMAGE,
        Replicas=1, CPU='4', Memory='32Gi', GPU='1', Internet=False,
        MaxRunningDuration='20m', MaxWaitDuration='10m', RestartPolicy='Never',
        MountGPFS='gpfs://gpfs2/haobinzhang:/root')})
(run / 'request.json').write_text(json.dumps(payload, indent=2))
request = dict(payload)
request['TaskSpecs'] = {
    key: job.models.CreateJobReqTaskSpec(**spec)
    for key, spec in payload['TaskSpecs'].items()}
result = job.create_job(ctx, job.models.CreateJobReq(**request))
receipt = dict(job_id=result.JobId, name=name, run_dir=str(run), gpus=1)
(run / 'receipt.json').write_text(json.dumps(receipt, indent=2))
print(json.dumps(receipt))
```

若 `create_job` 超时，不能当成创建失败后直接重试。先按上述**唯一 name**调用 ListJobs 并核对精确名称，找回 JobId、保存回执；确认没有任务后才决定重试。演示 `deployment/current.json` 不得被这个实验模板覆盖。SDK 完整字段在 `deployment/sdk/yicloud/services/job/models.py`。

worker 以 `bash run.sh ROOT RUN_DIR` 启动，通常做这些事：

1. `set -euo pipefail`，日志写入 `RUN_DIR/worker.log`，设置 Python 无缓冲输出。
2. 容器内核对 `torch.cuda.is_available()`、`device_count()==1` 和 `nvidia-smi`；记录真实 Python/Torch/CUDA/GPU 及显存。
3. 在任务本地 `/tmp` 建 `venv --system-site-packages`，继承镜像 PyTorch；只按具体依赖清单安装必要离线 wheel，不改镜像基础环境。现有 worker 的通配 wheel 安装是历史做法，新实验优先精确选包。
4. 读取共享盘已校验的权重/数据和 `run/source` 快照；容器使用分配到的 `cuda:0`，不硬绑物理卡编号。
5. 执行本次推理/训练，检查数值和输出，写回共享盘；成功后写本次完成标志并正常退出。临时环境随容器释放，持久结果必须写到挂载的 `/root`。

`HF_HUB_OFFLINE=1`、`OMP_NUM_THREADS=4` 等按实验需要设置。不要给 worker 传平台 API 凭据或本机中转站 Key。CPU 准备和下载完成后再申请卡，避免 GPU 等下载。任务中不要后台启动一个进程后让主命令马上退出，也不要用无关常驻进程占卡；退出码不能被忽略。

## 状态、日志和停止

**平台任务独立运行，SSH 断开不等于停止任务。** 本机 Codex 可以短命令提交后退出 SSH，之后再连回来查状态。长 CPU 命令如需独立后台运行，要有日志、PID 和明确退出条件。

下面在服务器执行，`run` 由本次 `receipt.json` 的准确位置确定：

```python
import json, sys
from pathlib import Path
sys.path.insert(0, '/root/face-age-transformation-course-project/integration')
from deploy_yicloud import client
run = Path('/root/face-age-transformation-course-project/deployment/training-runs/实际实验目录')
receipt = json.loads((run / 'receipt.json').read_text())
job, ctx = client()
state = job.get_job(ctx, job.models.GetJobReq(
    Project='fdj-infra', JobId=receipt['job_id']))
print(state.Name, state.Phase, state.Message)
```

常用查看日志的本机命令：

```powershell
ssh -o UpdateHostKeys=no yicloud 'tail -n 60 /root/face-age-transformation-course-project/deployment/training-runs/实际实验目录/worker.log'
```

日志尚未生成且平台仍排队，不代表程序崩溃。任务启动失败时，可用 SDK `get_logs` / `list_job_replicas` 或平台任务详情；`GetLogsReq.Start/End` 用纳秒 Unix 时间，先查 SDK 字段，不要凭空编命令。正常轮询间隔约15–30秒；不要反复提交或持续高频轮询。

确实要停止**本次实验**时，在上述已读取回执和 state 的脚本后加：

```python
if state.Name != receipt['name'] or not state.Name.startswith('face-age-'):
    raise RuntimeError('任务身份不匹配，不停止')
if state.Phase not in ['Stopped', 'Succeeded', 'Failed']:
    job.stop_job(ctx, job.models.StopJobReq(
        Project='fdj-infra', JobId=receipt['job_id']))
    print('stop_requested；稍后复查终态')
```

只停止已确认属于本项目、本次回执的任务，不对全平台 stop、不用无范围 `pkill`。任务已由平台自动删除时记录 Deleted/不存在；网络、认证错误不能伪装成“任务已删”。`Succeeded` 之后仍需检查产物、哈希和报告，不能仅用退出成功证明模型效果。

本文编写时的只读实查：本项目活动任务0；旧演示回执 `j-1795747711-339860` 在平台已为 `Failed` 终态。终态原因未在本次接手文档任务中重新诊断，不将它写成当前在线服务。以后始终重新查询。

## 下载、同步和本机读图

下载依赖/模型在开发工作区完成。需要已有出网代理时：

```bash
# 以下是服务器命令；由本机 Codex 经 SSH 调用。
cd /root/face-age-transformation-course-project
bash integration/with_yicloud_network.sh git ls-remote origin refs/heads/dev/mobei-gan-portable-20261116
```

代理包装只作用于当前命令，不修改全局网络；优先让脚本选择可用副本，不固定长期依赖某个旧私网 IP。GitHub API 限流、raw/git 可用、Hugging Face 传输中断是不同问题。大模型按清单校验完整 SHA，不下载无关的 `--all`。SDK client 自己关闭代理继承，避免将平台内网管理请求错误转到外部代理。

本机 PowerShell 的 SCP 示例：

```powershell
New-Item -ItemType Directory -Force .\yicloud-results | Out-Null
scp yicloud:/root/face-age-transformation-course-project/deployment/training-runs/实际实验目录/代表图.png .\yicloud-results\
scp yicloud:/root/face-age-transformation-course-project/deployment/training-runs/实际实验目录/report.json .\yicloud-results\
```

本机 Codex随后用自己的读图工具打开 `yicloud-results` 下的本地图片。需要传源码时先由它在远端创建新的暂存目录，再 scp 少量文件并核对 diff/哈希。中文文件名和含空格路径须正常引用；不要复制整仓、密钥或不必要的个人照片到本机。

## 本机浏览器查看演示

在**本机 PowerShell 的独立终端**保持以下转发运行：

```powershell
ssh -N -o ExitOnForwardFailure=yes -o UpdateHostKeys=no -L 127.0.0.1:16512:127.0.0.1:65512 yicloud
```

- 首页：`http://localhost:16512/`
- 镜子：`http://localhost:16512/人脸年龄变换原型_v5.html?mode=live&experience=mirror`
- StyleGANEX 已生成研究对照：`http://localhost:16512/age-styleganex-review/`

摄像头来自本机浏览器，无需服务器桌面/摄像头。16512 是用户此前已能用的端口；本机65512、18080曾被 Windows 拒绝绑定。静态页可打开不代表即时生成服务在线。

确实需要上传推理时，由本机 Codex执行：

```powershell
ssh -o UpdateHostKeys=no yicloud 'cd /root/face-age-transformation-course-project && python3 integration/deploy_yicloud.py start --quality original_delta --stable-numerics --max-running-duration 1h'
ssh -o UpdateHostKeys=no yicloud 'cd /root/face-age-transformation-course-project && python3 integration/deploy_yicloud.py status'
```

这一命令会申请 GPU，必须遵守前述总计一张卡；不要与训练任务同时启动。`start` 提交成功不代表后端已 ready，继续看 worker/backend 日志和 health。现有质量预览慢于最初默认，不用旧耗时表当作当前性能。

不再需要演示时：

```powershell
ssh -o UpdateHostKeys=no yicloud 'cd /root/face-age-transformation-course-project && python3 integration/deploy_yicloud.py stop'
```

这个 stop 只管理演示回执和对应网关/专用隧道，不负责停止独立训练实验。按 Ctrl+C 关闭本机转发也不会停止 GPU。只看静态研究页时不要为恢复网关启动一张卡：按[部署说明](仪电部署说明.md)检查已有网关和 `gateway` 行为后再恢复。

GPU→工作区的现有专用回传隧道用18822/18765，工作区→本机网页用65512/16512，两者不同。原 worker 含工作区旧私网 IP，工作区重建时需检查并适配它；正常接手不要改平台全局 sshd、复制 `deployment/ssh` 私钥或把推理端口绑定到公网。

## 接手完成的标准

本机 Codex应交付：实际分支/提交和修改清单、本次任务 ID 与单卡配置、数据/权重版本和哈希、日志/结果的服务器位置、实际测试与失败、下一步命令。只有需要看图时才下载少量代表结果。共享文档/源码按开发分支提交，模型/数据/照片沿用清单与 Release 管理，不混进 Git。

后续研究优先读[研究进展](研究进展与同学接手.md)、[课程模型边界](quality/20261127_课程技术路线与模型选型边界.md)、[近期GAN候选筛选](quality/20261127_近期GAN年龄编辑方法筛选.md)。更换模型前核查官方权重、许可、原生分辨率和年龄接口。当前没有一份已经验证替代 SAM 的“最新通用权重”，不要在新会话里误报。

本次交接文档已在服务器实查 SDK ListJobs/GetJob 和资源路径；没有测试你的本机 SSH 环境，也没有创建 GPU 任务。创建任务模板来自现有成功提交脚本并按 SDK 字段核对，需由接手 Codex结合具体实验执行。
