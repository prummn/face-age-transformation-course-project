# SFE 与 AgeTransGAN 真实模型对照

2026-11-27，莫贝研究分支 `dev/mobei-gan-portable-20261116`。本轮已实际运行两个新增 GAN 候选，保留 SAM 和此前 StyleGANEX 为对照。**SFE 的源重建和轻度编辑值得保留，但其公开年龄方向在大跨度编辑时严重纠缠眼镜、头发和身份；AgeTransGAN 的年龄组响应更完整，原生细节仍明显柔化。两者均未成为已验证的线上替代品。**

这次没有追加训练、生成新训练权重或修改时空镜上传推理。SFE 是 CVPR 2024 的细节编辑方法；AgeTransGAN 是 ECCV 2022 的完整年龄条件 GAN 备用基线，不称它为最新 SOTA。选型理由和其他近期候选的权重障碍见[方法筛选](20261127_近期GAN年龄编辑方法筛选.md)。

## 查看结果

电脑上保持以下隧道运行：

```powershell
ssh -N -o ExitOnForwardFailure=yes -o UpdateHostKeys=no -L 127.0.0.1:16512:127.0.0.1:65512 yicloud
```

- SFE／e4e／SAM：<http://localhost:16512/age-sfe-review/>
- AgeTransGAN／SAM：<http://localhost:16512/age-agetrans-review/>
- 既有 StyleGANEX：<http://localhost:16512/age-styleganex-review/>

这些是本轮真实推理生成的静态对照页，不占 GPU；不是新模型的上传推理接口。可选所有照片、所有强度／年龄组、SAM 目标、原始／保色、眼周／脸颊／口周原生裁图，下载完整 1024 PNG。两页均通过 Chromium 的全部选项切换、8 照片切换、下载解码、390 宽布局和 JS 错误检查。首页时空镜仍需原后端，启动方法见[仪电部署说明](../仪电部署说明.md)。

## 固定设计与边界

共用此前已看过的 8 张 FFHQ 开发图：64376、61908、62644、64919、65817、67140、60764、68335；发布性别标注为 3 女／5 男，5 张有发布的 Asian 标注、3 张没有该标注。包含年轻、中年和高龄源图；**不是新盲测，也不是亚洲总体评估**，可能与上游 FFHQ 预训练数据重合。没有使用剩余 41 张储备图。来源、授权和分组保留于每个运行目录的 `selection.json`，历史 `new-hd-holdout` 来源标签在本轮不再代表未见测试。

各方法使用同一张保存的 FFHQ 对齐原照。SAM 沿用面积插值 256 输入；SFE 官方入口接收 1024，再在模型内部双线性缩至 256；AgeTransGAN 直接接收 1024。因此共用对齐参考，但**不是相同输入像素的严格消融**。SFE 与 e4e 内部对照使用相同输入、生成器与年龄方向，可用于观察空间特征编辑的贡献。

SFE 固定扫描 `[-10,-5,0,2.5,5,7.5,10]`，正值变老；控制量是 InterFaceGAN 强度，计算仍为官方 `w + strength/2 * direction`。每档都保存同方向 e4e 对照。AgeTransGAN 使用官方 10 组检查点，扫描全部 0–9 one-hot 条件；组 9 在官方表中为 70+。README 的中间组边界描述有歧义，页面保留组号，不编造每组精确岁数。

SAM 同龄／20／40／60／75 从 `original-delta-20261124-160055` 已验证留档复制，逐张核验原始 PNG 哈希。候选均禁用修脸、超分、原照像素掩膜融合和非官方训练权重。原始模型结果为主要证据，可选保色使用同款后处理。

## 结果

以下都是独立 SFace 身份余弦与脸区 MiVOLO v2 外观估龄，均非真实年龄。**不同控制量及不同实际老化幅度不能直接当作公平胜负。** 没有按人搜索最优强度。

| 相同 8 图的条件 | SFace 均值 | 外观估龄均值 |
|---|---:|---:|
| SAM 同龄 | .8538 | 45.33 |
| SFE 强度 0 | .9835 | 47.84 |
| e4e 强度 0 | .6747 | 43.30 |
| SFE 强度 2.5 | .9246 | 52.40 |
| SFE 强度 5 | .7544 | 58.15 |
| SFE 强度 7.5 | .6134 | 65.82 |
| SFE 强度 10 | .5112 | 71.55 |
| SAM 目标 75 | .8139 | 60.10 |
| AgeTransGAN 组 8 | .7729 | 58.73 |
| AgeTransGAN 组 9（70+ 条件） | .7421 | 69.59 |

源图独立外观估龄均值 49.10。SFE 强度 0 的全幅 RGB 像素 L1 为 6.12，SAM 同龄为 15.62；该误差含背景，只是重建诊断。SFE 强度 2.5 相对其重建的外观年龄只增加 4.57，不能因身份 .9246 就说解决了年轻到高龄的问题。

另对冻结参考估龄≤35的 4 张源图作描述性检查，避免高龄源图抬高总体输出年龄。该分组依赖原参考估龄，不是真实年龄。以下均值接近不等于逐人匹配：

| 4 张较年轻源图 | SFace | 外观估龄 |
|---|---:|---:|
| SAM 目标 75 | .7564 | 51.48 |
| SFE 强度 7.5 | .5455 | 52.79 |
| SFE 强度 10 | .3971 | 62.18 |
| AgeTransGAN 组 8 | .7060 | 52.55 |
| AgeTransGAN 组 9 | .6546 | 63.33 |

大跨度编辑时，SFE 的重建优势没有转化成同等身份优势；AgeTransGAN 在这组开发图上的高龄身份表现优于 SFE 的大强度结果，但没有证明全面优于 SAM。

### 原生图观察

- SFE 的原照重建接近源图；强老化时 64376、61908、62644、64919 等无眼镜源图被加上眼镜，头发／发际线也变化。68335 的帽子被改成白发并出现胡须，60764 的眼镜与齿列也变化。相同问题出现在 e4e 对照，提示公开年龄方向本身的属性纠缠，不只是特征编辑器或输入模糊。
- AgeTransGAN 的年龄组变化清楚，代表图中没有 SFE 那种普遍新增眼镜；但 61908 原生眼周、毛孔、眉毛和牙齿仍明显柔化／重绘。整脸缩小预览较自然，不能据此称原生高清纹理已解决。其他人物也存在形状与身份变化。
- SFace 下降不完全等于人工身份判断，估龄上升也不等于自然度提高。本轮是代表图定性检查和完整自动检查，没有人工盲评或统计显著性结论。

代表拼图：SFE 页的 `representative.jpg`、`native-detail.jpg`；AgeTransGAN 页的 `representative.jpg`、`representative-first.jpg`、`native-detail.jpg`。各页仍提供全部原始 PNG，不只提供代表图。

## 权重、适配与资源

SFE 官方代码：[ControlGenAI/StyleFeatureEditor](https://github.com/ControlGenAI/StyleFeatureEditor)，提交 `00996c309ba5f04eec9ad7456f521e1501f7cc67`。官方 HF 提交 `5a50cb1b78da946be8163b8b255d486996fd2c73`；只恢复 editor light、e4e 和 StyleGAN2 `.pt` 三份必需权重，完整 SHA256 与官方 LFS 摘要一致，清单为 `integration/models/sfe-*.json`，文件在 `deployment/gan-models/stylefeatureeditor/`。

AgeTransGAN 官方代码：[AvLab-CV/AgeTransGAN](https://github.com/AvLab-CV/AgeTransGAN)，提交 `5d1f8354d903f8858ea9da639c0eac80180cfde2`。官方 Drive 10 组文件完整大小 1,287,079,078 字节，本地完整 SHA256 `4da9b623f4879d1d1ed614c1fa834dfc8205b8cbc9ec3ac5bf0730dfa361f9ef`；作者没有提供独立发布摘要，本地记录不能称为与作者哈希交叉核验。来源及校验见 `integration/models/agetrans-10group.json`，文件在 `deployment/gan-models/ffhq_10group_910k.pt`。

适配均在运行快照内完成，上游检出保持干净：

1. SFE 只实例化推理用组件，省略未用判别器、训练日志及其他编辑模型；跳过随后会被完整覆盖的 ArcFace 初始化下载。特征编辑器、反演器、e4e 编码器和生成器均严格加载。官方 forward 与两个编辑函数的 AST 一致性已留档。
2. 两者使用官方 StyleGANEX 的纯 PyTorch 算子，保留其许可；SFE 对上游 native 滤波参考实现的 4 种采样／填充组合误差为 0。没有改训练张量、生成器网络或 SAM 线上语义。
3. AgeTransGAN 初次尝试 `randomize_noise=False` 时，上游沿用的 4×4 固定噪声与其 16×16 起始特征不一致，任务失败。已定位并恢复官方演示的 `randomize_noise=True`，每次前向固定种子 20261127；同一组结果重复像素完全相同，所有年龄组共享相同噪声取样策略。失败日志保留，不重装环境。
4. 使用平台已有 Torch 镜像和独立临时 venv，只补齐 OpenCV；没有升级共享 Torch/CUDA/CANN。代码、权重、数据、日志均在共享工作区持久化。

最初平台 GPU 组配额满，返回 `insufficient group quota: gpu : 2001/2k`，先用 0 GPU、8 CPU／32GiB 完成 SFE 两图冒烟。随后获配**一张 H200**完成 SFE 全量，GPU worker 107.62 秒，冷加载／指纹及重复前向 6.45 秒；56 次 SFE＋e4e 编辑对均值 .120 秒，Torch 峰值 allocated 3.80 GB。此时间不包含网页请求、完整平台排队／容器启动，也不是 SFE 单独的 API 延迟。

AgeTransGAN 完整 8 图在 **0 GPU、8 CPU／32GiB** 上完成，worker 320.74 秒，冷阶段 7.71 秒，80 次前向均值 3.17 秒。与 H200 时间不能直接作模型速度比较。运行环境均为 Python 3.10.12、Torch 2.6.0+cu124；CPU 任务的 CUDA 字符串只是构建版本，实际 `device_count=0`。最多同时占用一张 GPU；只管理本项目任务。

SFE：152 原始＋152 保色；AgeTransGAN：120 原始＋120 保色，含各自复制的 SAM 对照。共 544 张主实验 PNG（含重复对照），**每张均为可解码 1024 PNG，原始输出 SHA256 完整检查，无漏检，首图重复一致，模型前后状态指纹一致**。本轮没有新训练权重。

## 文件与复现

| 用途 | 共享工作区相对路径／任务 |
|---|---|
| SFE 完整结果、日志、源码快照 | `deployment/training-runs/sfe-20261127-110920/`；`j-1795778635-867943`，Succeeded |
| SFE CPU 两图冒烟 | `deployment/training-runs/sfe-cpu-20261127-112816/`；`j-1795778941-890346`，Succeeded |
| AgeTransGAN 完整结果、日志、源码快照 | `deployment/training-runs/agetrans-cpu-20261127-113329/`；`j-1795779259-932262`，Succeeded |
| AgeTransGAN 固定噪声失败记录 | `deployment/training-runs/agetrans-cpu-20261127-113152/`；`j-1795779171-374459`，Failed |
| 精简证据 | [evidence/gan-candidates-20261127.json](evidence/gan-candidates-20261127.json) |

每个主运行目录有 `experiment.json`、`model-spec*.json`、`selection.json`、`request.json`、`receipt.json`、`worker.log`、`device.json`、`checkpoint-info.json`、`repeatability.json`、`frozen-verification.json`、`evaluation/report.json`、`summary.json`、`artifact-validation.json`、`browser-validation.json`。SFE 另有算子／官方函数一致性记录。

这是当前共享服务器的复现方法，前提是保留上述基线目录及获授权的原生照片；全新机器需先按[本机 Codex／服务器交接](../本机Codex连接仪电GPU操作交接.md)恢复相应资产，不能只克隆源码便算数据齐全。上游代码需按所列提交检出到 `deployment/StyleFeatureEditor`、`deployment/AgeTransGAN`、`deployment/StyleGANEX`。不重新执行上游全量环境安装。

```bash
cd /root/face-age-transformation-course-project
# 校验／按需恢复；已有正确文件直接复用。代理只对当前命令生效。
for model_manifest in integration/models/sfe-*.json; do
  bash integration/with_yicloud_network.sh python3 integration/models/fetch_verified_ranges.py "$model_manifest" deployment/gan-models/stylefeatureeditor --workers 4 --hf-mirror
done
bash integration/with_yicloud_network.sh python3 integration/models/fetch_verified_ranges.py integration/models/agetrans-10group.json deployment/gan-models --workers 4

# SFE 完整单卡对照。prepare 输出运行目录；submit 会检查已有项目任务。
python3 integration/quality/sfe_study.py prepare
python3 integration/quality/sfe_study.py submit deployment/training-runs/sfe-输出时间戳
# 配额不足时可先 prepare --cpu-pilot，固定两图／三强度／0 GPU。

# AgeTransGAN 完整 CPU 对照；省略 --cpu 则准备单卡版，须等前项终态后提交。
python3 integration/quality/agetrans_study.py prepare --cpu
python3 integration/quality/agetrans_study.py submit deployment/training-runs/agetrans-cpu-输出时间戳

# 各自 DONE 后，使用既有 CPU 评估环境。
deployment/asia-adaptation/20261123/age-cpu-venv/bin/python integration/quality/finish_styleganex.py 运行目录 --profile sfe
# AgeTransGAN 改为 --profile agetrans；浏览器检查用 verify_gan_pilot_browser.py。
```

任务停止应读取对应 `receipt.json`，按交接文档调用该 JobId 的 `stop_job`，不要批量停止同项目其他任务。上游源码和模型不随本次 Git 提交重发；SFE 源码 MIT，依赖保留原条款；AgeTransGAN 没有单独找到代码／模型许可，不能把 CAF 数据许可套用于模型或承诺可商用。

## 研究决定

保留 SAM 当前应用基线，AgeTransGAN 作为完整年龄条件 GAN 对照，SFE 作为高保真反演／特征编辑研究基座。**后续的明确方向是学习源／目标年龄条件下的空间特征编辑，同时约束眼镜、头发、身份与局部纹理**；公开统一年龄向量尚不适合直接承担陌生人高龄功能。SFE 与 AgeTransGAN 的潜空间／生成器不相同，不能直接混权重或未经消融相加。

先在固定开发数据上验证控制模块和纹理／身份的取舍，再冻结配方使用储备集及授权实拍验证，随后才决定亚洲高龄域适配与是否更新生成器。需要真实高清高龄数据，但本轮结果已经表明：源重建、年龄控制和高频纹理是三个独立问题，单纯扩充 SAM 的短训练没有足够依据解决它们。
