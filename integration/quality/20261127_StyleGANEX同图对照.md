# StyleGANEX与现有SAM的同图诊断

## 与课程技术路线的关系

课程主线为基于对抗生成网络的人脸年龄变换；课堂报告同时列出SAM／LATS作为生成候选，没有规定只能使用SAM。本项采用官方StyleGANEX年龄编辑检查点，仍属StyleGAN生成与人脸反演路线，但会更换反演和编辑机制；它是独立GAN候选，不是SAM换一份权重。当前SAM应用、模型Release、已有结果与训练资产保留。

StyleGANEX通过空间特征及可选编码器跳连补充W+，值得检验它是否更好保留源细节。跳连传递的是学习特征，不将原照像素直接回贴到生成脸。本轮不是已证明更优的替换，也不称最新SOTA或完成亚洲专项适配。

## 官方来源与运行边界

官方仓库<https://github.com/williamyang1991/StyleGANEX>，固定`e593d16006f64b1e105c04eeb4a7cc149c1aca98`，已取得完整`LICENSE.md`（此前只查`LICENSE`得到404）。S-Lab License1.0允许非商业使用／修改，复用时保留许可及上游依赖条款；本项目只做课程非商业对照。

作者官方Hugging Face镜像<https://huggingface.co/PKUWilliamYang/StyleGANEX>固定`afcf234cdbddc9f9205ab1a71c80efe411cd760f`。只下载`styleganex_edit_age.pt`，1,475,996,775字节，LFS SHA256为`9577fcad114b207511eb96b0071caa6735fa07d80aa5fcca73dc45d7b00e20f3`。不用完整模型合集，不下载超分／卡通等无关权重。清单`integration/models/styleganex-20261127.json`；分段传输支持断点，最终按完整文件SHA256验证，分段收到本身不表示已通过内容哈希。

官方`video_editing.py`的年龄检查点以`editing_w × scale_factor`控制强度，没有数值目标年龄接口。因此本项只能称零强度重建／强度扫描，不能将强度1或2标成60／75岁、不能直接替换当前年龄滑块。独立外观估龄也不能提供真实目标年龄校准保证。

## 事前固定设计

八张曾经看过的高清开发图：64376、61908、62644、64919、65817、67140、60764、68335，涵盖源年龄、男女、眼镜和嘴部难例。选择不依据StyleGANEX结果；发布者Asian标签仅按原清单使用，不按外观补写族裔。全部来自现有公开来源清单与署名，不使用用户会话、不消费41张新储备；不能叫独立盲测。

两模型共用同一FFHQ对齐RGB256输入，已逐像素核对它等于留存原生1024参考的INTER_AREA缩小。SAM复用历史原SAM差值＋约束的same／20／40／60／75全部PNG并检查哈希；不重跑或重拟合它。StyleGANEX先按官方年龄编辑前向做零强度重建，然后固定扫描−1、−.5、0、.5、1、1.5、2，**所有强度与失败都保留，不按人挑最好结果**。

StyleGANEX设置`x1=x2=已对齐256`、`resize=False`、`zero_noise=True`、采用检查点原`use_skip`与编辑方向，batch1输出1024。这是对齐输入的模型机制诊断，不是官方宽视场视频流程或其500步潜变量反演复现。跳过dlib二次对齐，使两路线接收相同像素；这可能限制StyleGANEX的宽视场优势，不能把本项结果推广为完整官方流程性能。

官方训练脚本的合成配对采样强度为0／.25／.5／.75／1／1.25，并从保存的方向中采样；官方演示只取第一条方向。本次同样固定第一条方向，负强度是该方向的反向扫描，1.5／2超过训练脚本列出的强度，单列压力观察，不能用它们的失败否定常规强度，也不能用更强外推掩盖普通强度老化不足。

不训练任何网络，不对生成图做超分、修脸或原照像素混合。原始／可选同款保色均保留；先看原始以免后处理影响判断。独立SFace与CPU MiVOLO仅做评估，不用于优化、强度选择或检查点选择。零强度重建应先检查皮肤／眼镜／牙齿，再观察强度提高时的老化、身份与伪影，不因年龄分数上涨即宣称改善。

先核验官方检查点完整SHA256，严格检查所有网络参数加载，首图强度1前向重复两次PNG像素应完全相同，全部输出finite且可解码1024PNG，冻结模型前后指纹不变。预期8×(5＋7)=96张原始与96张保色PNG；所有同图结果、原生局部与署名通过独立静态页查看。未显现可确认的自然度／身份收益则不扩大；有收益也只允许后续固定开发回归，不自动接入线上。

只申请一张GPU、4核／32GiB、20分钟硬上限，下载／准备期间不占卡，提交前核查本项目无其他活动任务。使用隔离venv继承已有镜像Torch，不安装完整上游requirements。模型使用上游自带纯Torch算子；仅在冻结运行副本中删去未用到的matplotlib导入，不改数学运算或原始检出。

```bash
# 下载进度与校验结果位于deployment/gan-models/styleganex；需要当前网络可达。
python3 integration/models/fetch_verified_ranges.py \
  integration/models/styleganex-20261127.json deployment/gan-models/styleganex
python3 integration/quality/styleganex_study.py prepare
python3 integration/quality/styleganex_study.py submit deployment/training-runs/styleganex-时间戳
```

官方源码／许可缓存`deployment/StyleGANEX/`，来源元数据`deployment/gan-baseline-review/20261127/`，不把外部完整源码、模型或人像提交Git。模型级结果产生后再填写下方实际结论，不以准备完成代替推理验证。
