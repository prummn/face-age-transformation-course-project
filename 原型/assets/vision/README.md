# 本地检测资源

下载日期：2026-10-29。资源固定到 MediaPipe Tasks Vision 0.10.14 与 Face Landmarker float16 第1版，避免课堂演示必须在线拉取约23 MB资源。通过项目配套HTTP服务器访问时优先使用本目录。校验信息见 `manifest.json`。

| 文件 | 原始来源 |
|---|---|
| vision_bundle.mjs | https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs |
| wasm/vision_wasm_internal.js、.wasm | https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm/ |
| wasm/vision_wasm_nosimd_internal.js、.wasm | 同上，供不支持SIMD的运行环境使用 |
| face_landmarker.task | https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task |
| LICENSE | https://raw.githubusercontent.com/google-ai-edge/mediapipe/master/LICENSE |

MediaPipe运行库由Google及贡献者提供，保留上游Apache-2.0许可证。模型说明、用途和模型相关条款应结合[官方Face Landmarker介绍](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker)及其Model链接查阅；附带运行库LICENSE不作为所有模型权重和外部测试照片的统一许可声明。

HTML内还保留同学原型已有的Base64 BlazeFace检测模型，作为轮廓不可用时的降级。未将其解释为年龄估计、GAN或人像语义分割模型。
