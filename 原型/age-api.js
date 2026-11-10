/* Same-origin adapter for the integrated SAM service. Offline prototype stays explicit. */
(() => {
  'use strict';
  if (new URLSearchParams(location.search).get('mode') !== 'live') return;

  const state = {ready:false, checking:false, error:'', session:null, serial:0, health:null};
  const pending = new Map();
  const $ = selector => document.querySelector(selector);

  function display() {
    document.body.dataset.backend = 'live';
    $('#serviceBadge').hidden = false;
    $('#serviceBadge').dataset.ready = String(state.ready);
    $('#serviceBadge').textContent = state.ready ? '真实模型 · 原生 1024' : state.checking ? '正在连接模型…' : '模型服务未就绪';
    $('#servicePanel').hidden = false;
    $('#serviceState').textContent = state.ready ? '真实估龄与年龄生成已连接。' : state.error || '正在连接自有推理服务器…';
    $('#serviceState').classList.toggle('error',!!state.error);
    $('#bReconnect').disabled = state.checking;
    $('#privacyStorage').textContent = '确认人脸后，照片会发送到自有推理服务器，仅在本次会话内存中处理。清除或返回主页会请求删除；断线时服务端到期自动清理。已导出的文件会保留。';
    $('#privacyModel').textContent = '当前年龄由 FairFace 提供参考估计，可手动修正。变龄由 SAM 模型生成，原生结果为 1024×1024。'+(state.health?.features?.source_color_preservation?'“保留原肤色”会参考原照色度进行后处理，保留模型结果的明暗与纹理，可切换查看未校正的原始模型结果。':'')+'结果仅模拟年龄外观，不是真实年龄判定或未来容貌预测。';
    $('#privacyCamera').textContent = '电脑 localhost 可调用摄像头；手机摄像头需要可信 HTTPS。上传照片时请确认本人或被摄者同意。';
    $('#capturePrivacy').hidden = false;
    $('#generationModeNote').textContent = '真实模型生成 · 原生 1024×1024';
    window.dispatchEvent(new CustomEvent('age-service-state',{detail:{ready:state.ready,error:state.error}}));
  }

  async function request(path, options={}, timeout=60000, kind='request') {
    const controller = new AbortController();
    pending.set(controller,kind);
    let timedOut = false;
    const timer = setTimeout(() => {timedOut=true;controller.abort();},timeout);
    try {
      const response = await fetch(path,{...options,headers:{'Content-Type':'application/json',...options.headers},cache:'no-store',signal:controller.signal});
      const payload = await response.json().catch(() => null);
      if (!response.ok) {
        let detail = payload?.detail;
        if (typeof detail !== 'string') detail = payload?.message || `服务返回错误（${response.status}）`;
        if (response.status >= 500) {state.ready=false;state.error=detail;display();}
        throw new Error(detail);
      }
      if (!payload || typeof payload !== 'object') throw new Error('服务返回内容无效，请重新连接。');
      return payload;
    } catch (error) {
      if (error.name === 'AbortError') throw new Error(timedOut ? '服务处理超时，请重试。' : '本次处理已取消。');
      if (error instanceof TypeError) {
        state.ready=false;state.error='无法连接模型服务，请检查连接后重试。';display();
        throw new Error(state.error);
      }
      throw error;
    } finally {clearTimeout(timer);pending.delete(controller);}
  }

  async function health() {
    if (state.checking) return state.ready;
    state.checking=true;state.error='';display();
    try {
      const info = await request('/api/health',{},10000,'health');
      state.health=info;state.ready=info.ready === true;
      if (!state.ready) state.error='模型正在准备，请稍后点击重新连接。';
    } catch(error) {state.ready=false;state.error=error.message;}
    finally {state.checking=false;display();}
    return state.ready;
  }

  function removeSession(id) {
    if (!id) return Promise.resolve();
    const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),8000);
    return fetch(`/api/sessions/${encodeURIComponent(id)}`,{method:'DELETE',cache:'no-store',keepalive:true,signal:controller.signal}).then(response => {
      if (!response.ok) throw new Error('服务器会话清除未确认，将由到期清理机制释放。');
    }).catch(error => {
      state.error='服务器会话清除未确认，将由到期清理机制释放。';
      display();
    }).finally(()=>clearTimeout(timer));
  }

  function cancel() {
    for (const [controller,kind] of pending) if (kind === 'generate') controller.abort();
  }

  function clear() {
    state.serial++;
    // Keep preparation responses reachable so stale session IDs can be deleted.
    // Aborting that HTTP request would leave a newly created server session unknown.
    for (const [controller,kind] of pending) if (kind === 'generate') controller.abort();
    const old = state.session;
    state.session=null;
    return removeSession(old);
  }

  async function decodeImage(url, size) {
    if (typeof url !== 'string' || !url.startsWith('data:image/png;base64,')) throw new Error('服务未返回有效的 PNG 图像。');
    const image = new Image();
    image.src=url;
    await image.decode();
    if (size && (image.naturalWidth !== size || image.naturalHeight !== size)) throw new Error(`返回图像尺寸异常，应为 ${size}×${size}。`);
    const output=document.createElement('canvas');
    output.width=image.naturalWidth;output.height=image.naturalHeight;
    output.getContext('2d').drawImage(image,0,0);
    image.src='';
    return output;
  }

  async function prepare({image,face}) {
    const cleanup=clear(),serial=state.serial;
    await cleanup;
    if (serial !== state.serial) throw new Error('本次处理已取消。');
    if (!state.ready && !await health()) throw new Error(state.error || '模型服务未连接，请在选项中重新连接。');
    if (serial !== state.serial) throw new Error('本次处理已取消。');
    // Preserve browser-decoded RGB pixels; another JPEG pass would alter colour comparisons.
    const body=JSON.stringify({image:image.toDataURL('image/png'),face:{x:face.x,y:face.y,w:face.w,h:face.h,id:face.id,manual:!!face.manual}});
    if (body.length > 21*1024*1024) throw new Error('照片无损编码后超过处理上限，请裁剪照片或使用尺寸较小的图片。');
    const data=await request('/api/prepare',{method:'POST',body},90000,'prepare');
    if (serial !== state.serial) {await removeSession(data.session_id);throw new Error('本次处理已取消。');}
    if (!data.session_id || data.alignment !== 'sam_ffhq' || !Array.isArray(data.transform?.inverse) || data.transform.inverse.length !== 6 || !data.transform.inverse.every(Number.isFinite)) {
      await removeSession(data.session_id);
      throw new Error('人脸对齐信息无效，请重新选择人脸。');
    }
    let aligned,reference=null;
    try {aligned=await decodeImage(data.aligned_image,256);if(data.reference_image)reference=await decodeImage(data.reference_image,1024);} catch(error) {if(aligned)aligned.width=aligned.height=0;if(reference)reference.width=reference.height=0;await removeSession(data.session_id);throw error;}
    if (serial !== state.serial) {aligned.width=aligned.height=0;if(reference)reference.width=reference.height=0;await removeSession(data.session_id);throw new Error('本次处理已取消。');}
    state.session=data.session_id;
    return {face:aligned,referenceFace:reference,transform:data.transform,ageInfo:data.age,meta:data.meta,alignment:data.alignment};
  }

  async function generate({targetAge,requestId}) {
    if (!state.session) throw new Error('本次人脸尚未准备好，请重新选择照片。');
    const serial=state.serial,session=state.session;
    const data=await request('/api/transform',{method:'POST',body:JSON.stringify({session_id:session,target_age:targetAge,request_id:requestId,enhance:false,preserve_color:true})},60000,'generate');
    if (serial !== state.serial || session !== state.session) throw new Error('本次处理已取消。');
    if (data.meta?.output_kind !== 'native' || data.width !== 1024 || data.height !== 1024) throw new Error('返回结果不是原生 1024 图像，请重新处理。');
    const output=await decodeImage(data.image,1024);
    let raw=null;
    try {if(data.raw_image) raw=await decodeImage(data.raw_image,1024);}catch(error){output.width=output.height=0;throw error;}
    if (serial !== state.serial || session !== state.session) {output.width=output.height=0;if(raw)raw.width=raw.height=0;throw new Error('本次处理已取消。');}
    output.ageMeta=data.meta;
    output.ageRaw=raw;
    state.ready=true;state.error='';display();
    return output;
  }

  // Install immediately in live mode. A disconnected service can never become a demo success.
  window.AgeMirrorAdapters={live:true,prepare,generate,cancel,clear};
  window.AgeMirrorAPI={state,health};
  $('#bReconnect').onclick=health;
  $('#serviceBadge').onclick=() => $('#bInfo').click();
  display();
  health();
})();
