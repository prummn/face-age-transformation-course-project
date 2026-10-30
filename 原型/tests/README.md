# 浏览器验证

需要 Node.js、Playwright 及其 Chromium。可用环境变量 `PLAYWRIGHT_MODULE` 指向已有 Playwright 模块，`CHROME_PATH` 指向浏览器可执行文件，不设置时使用正常模块解析与 Playwright 浏览器。

从仓库根目录运行，另一个终端先运行 `python 原型/serve.py`：

```powershell
node 原型/tests/regression.cjs
node 原型/tests/mobile-flow.cjs
node 原型/tests/edge-cases.cjs
node 原型/tests/immersive-layout.cjs
node 原型/tests/postcapture-flow.cjs
node 原型/tests/navigation-flow.cjs
node 原型/tests/auto-experience.cjs
node 原型/tests/portrait-display.cjs
node 原型/tests/gesture-controls.cjs
```

涉及照片上传和模拟视频的测试使用被 Git 忽略的 `tmp/qa_portrait.jpg` 与 `tmp/qa_two_faces.jpg`。测试肖像来自 MediaPipe 官方测试素材 https://storage.googleapis.com/mediapipe-assets/portrait.jpg ，不作为产品示例分发。双脸样本由该肖像裁取顶部62%后左右并排组成。请先在tmp放好这些测试文件；测试会将截图、导出PNG也写入tmp。

regression检查空年龄、目标数量上限和画布清除；mobile-flow用本地真实模型检查上传、轮廓、结果、回贴和导出；edge-cases检查多人选择、失败重试、超时、晚到结果、模拟视频、手动兜底与清除。它们没有证明物理手机、RK3288或真实GAN完成。

auto-experience 检查检测乱序及短暂漏检时显示编号与选中对象不变、全局跟踪分配、所有人脸拍照门限、缓慢漂移和推理中断、占位年龄自动继续、轮播/暂停/手动停止、估龄失败或超时兜底及清除后晚到结果。它还使用肖像裁片构造双脸 canvas 视频流，运行真实本地 Face Landmarker，检查选项暂停拍摄与稳定后自动进入结果。视频是模拟输入，不是物理相机实测。

portrait-display 检查1080×1920、720×1280、1440×2560视口下镜头整屏显示、字号与人脸标签放大、结果与各次级页面/弹窗关键操作首屏可见、默认页面无溢出、最多九张总览卡片（原照+八目标），以及横竖旋转后恢复对应布局。该项用合成画布验证排版，不作为真实模型或显示器硬件证据。

gesture-controls 检查拍摄前张开手掌停留触发、手势重置、结果页不再响应翻页手势，并初始化仓库内的 Hand Landmarker 模型，同时确认手部轮廓绘制接口存在。状态机测试不替代真实手部视频下的不同距离、光照、遮挡和多人环境验收；手机拍摄后会关闭相机，因此手势只在大屏取景页启用。
