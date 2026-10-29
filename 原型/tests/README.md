# 浏览器验证

需要 Node.js、Playwright 及其 Chromium。可用环境变量 `PLAYWRIGHT_MODULE` 指向已有 Playwright 模块，`CHROME_PATH` 指向浏览器可执行文件，不设置时使用正常模块解析与 Playwright 浏览器。

从仓库根目录运行，另一个终端先运行 `python 原型/serve.py`：

```powershell
node 原型/tests/regression.cjs
node 原型/tests/mobile-flow.cjs
node 原型/tests/edge-cases.cjs
node 原型/tests/immersive-layout.cjs
```

后两项使用被 Git 忽略的 `tmp/qa_portrait.jpg` 与 `tmp/qa_two_faces.jpg`。测试肖像来自 MediaPipe 官方测试素材 https://storage.googleapis.com/mediapipe-assets/portrait.jpg ，不作为产品示例分发。双脸样本由该肖像裁取顶部62%后左右并排组成。请先在tmp放好这些测试文件；测试会将截图、导出PNG也写入tmp。

regression检查空年龄、目标数量上限和画布清除；mobile-flow用本地真实模型检查上传、轮廓、结果、回贴和导出；edge-cases检查多人选择、失败重试、超时、晚到结果、模拟视频、手动兜底与清除。它们没有证明物理手机、RK3288或真实GAN完成。
