# Mobile Face Mirror Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 v5 网页原型改造成适合手机演示的 A+B 交互：克制的电子镜入口、轻量人脸轮廓反馈、多人脸选择和连续年龄扉页结果，并补齐与需求规格相符的前端页面状态。

**Architecture:** 保留单文件 HTML，继续使用现有 MediaPipe Tasks Vision 和 BlazeFace 检测；增加可选的 Face Landmarker 轮廓层、面部选择状态与触控事件。将结果从九宫格主视图改为横向可滑动的年龄扉页，同时保留一个次级“总览/导出拼图”入口以覆盖 FR10。年龄估计和年龄变换接口仍保持可替换的占位实现，不在本次改造中伪装成真实模型。

**Tech Stack:** Vanilla HTML/CSS/JavaScript, MediaPipe Tasks Vision, Canvas 2D, mobile touch/pointer events, Web Share API with download fallback.

---

## 实施结果（2026-10-29）

五个任务的网页实现已落地；当前行为、验证证据与未完成项分别记录在 `原型/README.md`、`原型/验证记录_2026-10-29.md`、`原型/需求覆盖对照_2026-10-29.md`。以下保留原计划供追溯，未勾选的原步骤不表示物理设备已经验收。

与原计划的具体差异：Face Landmarker 自带检测，因此作为主模型，BlazeFace只在失败时降级；未用随机数伪装估龄，默认由用户手填；目标提供数字输入、年差与快捷组合；扉页采用横向滑动，循环按年龄排序，未用淡入淡出；旧Word说明保持原样，新增Markdown说明。已验证真实本地模型、双脸点选、模拟视频、模型不可用时手动拖框、PNG导出、异步清除和超时。物理手机与镜子触发、系统分享面板、RK3288性能均留待实机验收。

### Task 1: Establish the mobile visual system and state containers

**Files:**
- Modify: `原型/人脸年龄变换原型_v5.html` head, body states, and CSS sections

- [ ] **Step 1: Add mobile viewport and semantic state containers**

  Add `viewport-fit=cover`, a compact status region, a camera overlay canvas, a face-choice layer, an age-page result layer, an overview button, and explicit clear/share controls. Keep existing IDs used by the state machine until the new layers are wired.

- [ ] **Step 2: Replace neon presentation styles**

  Replace purple/blue glow, decorative nebula, and large English labels with warm neutral surfaces, one accent color, thin borders, readable 16px mobile text, `touch-action: pan-x` on result pages, and `env(safe-area-inset-*)` padding.

- [ ] **Step 3: Run a static DOM check**

  Run a PowerShell check that confirms the new IDs occur exactly once and that the document still contains the existing A-E state IDs. Expected output: all required IDs found and no duplicate IDs.

### Task 2: Add lightweight live face contour and multi-face choice

**Files:**
- Modify: `原型/人脸年龄变换原型_v5.html` model constants, state objects, detector initialization, detection loop, and pointer handlers

- [ ] **Step 1: Add landmark model configuration with a fallback**

  Keep BlazeFace as the required detector. Add a lazily loaded MediaPipe Face Landmarker model URL and a `LANDMARK` state object. If the landmark model fails to load, keep the detector workflow usable and show a quiet “轮廓增强不可用” status instead of blocking the phone demo.

- [ ] **Step 2: Store all detected faces and selected face index**

  Change `detectFace` into a result that preserves every bounding box. Add `F.faces`, `F.selectedIndex`, and a `chooseFace(index)` function. Phone mode presents numbered face targets; mirror mode defaults to the largest centered face. The selected face drives quality checks and capture.

- [ ] **Step 3: Draw a smoothed contour overlay**

  Use face landmark oval points when available, smooth each point with an exponential moving average, and draw a thin accent contour plus a small selection marker on the overlay canvas. Clear the overlay when no face is present or the task resets.

- [ ] **Step 4: Validate interaction cases manually**

  Run the page on a phone-sized browser viewport and check: one face, two faces, face switching by tap, no face, model-load failure, and a moving face. Expected behavior: only the selected face receives the active contour and only it enters capture.

### Task 3: Complete mobile capture, quality, and age-input states

**Files:**
- Modify: `原型/人脸年龄变换原型_v5.html` capture flow, upload validation, and state transitions

- [ ] **Step 1: Add explicit capture review and retake controls**

  After capture, show the photo, a concise quality status, “重拍”, and “继续”. Preserve the existing automatic mirror path while making the phone path touch-driven.

- [ ] **Step 2: Add input validation and quality messages**

  Validate JPG/PNG/WEBP, decode success, file size up to 15 MB, and short edge at least 256 px. Add face-size, pose, brightness, and blur checks using the existing frame metrics; report one actionable reason at a time.

- [ ] **Step 3: Add current-age correction and target-age selection**

  Present the current age as an explicitly editable estimate. Add a bounded target-age slider/number input and multi-select age chips. Keep the current placeholder generator behind the same `targets` interface.

- [ ] **Step 4: Verify touch and offline fallback**

  Confirm that no keyboard input is required, that upload works without a camera, and that failed remote resources leave a usable local preview with an “仅本地预览” message.

### Task 4: Replace the nine-grid primary result with continuous age pages

**Files:**
- Modify: `原型/人脸年龄变换原型_v5.html` result markup, result rendering, comparison controls, export/share handlers

- [ ] **Step 1: Build an ordered age-page data model**

  Convert `S.targets` and `S.results` into page records containing target age, relative label, canvas, and status. Keep one source face and generate every target independently.

- [ ] **Step 2: Implement horizontal page navigation**

  Render one large result page at a time with CSS scroll snapping, touch swipes, previous/next buttons, and progress dots. Use short crossfades instead of the current warp text carousel.

- [ ] **Step 3: Preserve comparison and FR10 overview**

  Keep the split comparison slider on a selected page. Add a secondary “总览” action that opens a restrained contact sheet with age labels for export, so FR10 remains available without making the grid the main presentation.

- [ ] **Step 4: Add mobile sharing and cleanup**

  Use `navigator.share` when available and fall back to PNG download. Make “清除本次数据” release canvases, reset selection, stop timers, and return to the mirror state; keep exported files outside runtime memory.

### Task 5: Run verification and document the new prototype boundary

**Files:**
- Modify: `原型/人脸年龄变换原型_v5_使用说明.docx` only if the current behavior has been implemented and rendered for review
- Modify: `需求说明资料/README.md` with the implementation status after verification

- [ ] **Step 1: Run static checks**

  Check balanced script/style tags, required state IDs, no duplicate IDs, and the presence of detector, landmark fallback, face-choice, touch navigation, share, clear, and overview functions.

- [ ] **Step 2: Run browser checks**

  Test camera permission denied, upload fallback, one face, multiple faces, no face, contour fallback, capture/retake, age selection, page swipe, comparison slider, export/share fallback, reset, and narrow viewport overflow.

- [ ] **Step 3: Update the usage note only after behavior is verified**

  Record the new phone-first flow, contour fallback, multi-face selection behavior, page navigation, and remaining placeholder-model limitation. Do not claim real age estimation or GAN generation.

- [ ] **Step 4: Commit the verified implementation**

  ```powershell
  git add -- '原型/人脸年龄变换原型_v5.html' '原型/人脸年龄变换原型_v5_使用说明.docx' '需求说明资料/README.md'
  git commit -m "feat: redesign mobile age mirror prototype"
  git push origin main
  ```

**Known boundary after this plan:** FR06, FR08, FR17, FR19, and FR20 still require real age/GAN models, Asian-face evaluation, Android packaging, and LAN service integration. This frontend plan makes their interfaces and fallback states ready; it does not turn placeholder image processing into a trained model.
