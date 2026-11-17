'use strict';
(()=>{
 const $=s=>document.querySelector(s), asset='hd-demo-assets/';
 const conditions=['same','20','40','60','75'];
 const ages={'7-9':'儿童 · 7–9','15-19':'青少年 · 15–19','20-29':'青年 · 20–29','40-49':'中年 · 40–49','50-69':'50–69','70-120':'老人 · 70+'};
 let data, selected, condition='40', color='raw', timer=null, request=0;
 function safeURL(s){const u=new URL(s,location.href);if(!['https:','http:'].includes(u.protocol))throw Error('Invalid source URL');return u.href;}
 function link(text,url){const a=document.createElement('a');a.textContent=text;a.href=safeURL(url);a.target='_blank';a.rel='noopener';return a;}
 function stop(){clearInterval(timer);timer=null;$('#play').textContent='自动轮播';$('#play').setAttribute('aria-pressed','false');}
 function zoom(url,title){
  stop();$('#zoomImage').src=url;$('#zoomTitle').textContent=title+' · 原生 1024';$('#zoom').showModal();
  requestAnimationFrame(()=>{const box=$('.zoom-scroll');box.scrollLeft=(1024-box.clientWidth)/2;box.scrollTop=(1024-box.clientHeight)/2;});
 }
 async function show(){
  const epoch=++request, row=selected, output=row.outputs.find(o=>o.condition===condition);
  const url=new URL(location.href);url.searchParams.set('person',row.id);url.searchParams.set('age',condition);url.searchParams.set('color',color);history.replaceState(null,'',url);
  $('#personTitle').textContent=`样例 ${row.id} · ${row.gender==='female'?'女':'男'}`;
  $('#personMeta').textContent=`原照外观年龄 ${row.ageGroup==='70-120'?'70+':row.ageGroup.replace('-', '–')}（人工区间） · 模型参考估龄 ${row.referenceAge} 岁`;
  const source=asset+row.reference, result=asset+output[color].path;
  const label=condition==='same'?`同龄重建 · 参考 ${row.referenceAge} 岁`:`目标 ${condition} 岁`;
  $('#resultLabel').textContent=label;
  $('#sourceImage').src=source;$('#resultImage').src=result;
  $('#zoomSource').onclick=()=>zoom(source,'原照');$('#zoomResult').onclick=()=>zoom(result,label+(color==='raw'?' · 原始模型':' · 原照保色'));
  $('#downloadSource').href=source;$('#downloadSource').download=`${row.id}_source_1024.png`;
  $('#downloadResult').href=result;$('#downloadResult').download=`${row.id}_${condition}_${color}_1024.png`;
  $('#timing').textContent=`${condition==='same'?'首次同龄重建（含拟合）':'本次年龄变换'} ${output.modelSeconds.toFixed(2)} 秒${condition==='same'?'':`；首次原照拟合另需 ${row.anchorSeconds.toFixed(2)} 秒`}。为 GPU 容器内模型计时，不含浏览器与网络。`;
  const credit=$('#credit');credit.replaceChildren('原照：',link(row.credit.author||'来源作者',row.credit.photo_url),' · ',link(row.credit.license,row.credit.license_url));
  for(const b of document.querySelectorAll('[data-condition]'))b.setAttribute('aria-pressed',String(b.dataset.condition===condition));
  for(const b of document.querySelectorAll('[data-color]'))b.setAttribute('aria-pressed',String(b.dataset.color===color));
  for(const b of document.querySelectorAll('.person'))b.setAttribute('aria-pressed',String(b.dataset.id===row.id));
  try{await Promise.all([$('#sourceImage').decode(),$('#resultImage').decode()]);if(epoch===request)$('#status').textContent='';}
  catch{if(epoch===request)$('#status').textContent='图片未能完整加载，请刷新重试。';}
 }
 function renderPeople(){
  const rows=data.people.filter(r=>(!$('#ageFilter').value||r.ageGroup===$('#ageFilter').value)&&(!$('#genderFilter').value||r.gender===$('#genderFilter').value));
  $('#count').textContent=`${rows.length} / ${data.people.length}`;$('#people').replaceChildren();
  for(const row of rows){
   const b=document.createElement('button');b.className='person';b.dataset.id=row.id;b.setAttribute('aria-label',`${row.id} ${ages[row.ageGroup]} ${row.gender==='female'?'女':'男'}`);b.setAttribute('aria-pressed',String(row===selected));
   const img=new Image();img.src=asset+row.thumbnail;img.alt=`${ages[row.ageGroup]}高清正脸样例`;img.width=512;img.height=512;img.loading='lazy';
   const label=document.createElement('span');label.className='person-text';label.textContent=`${ages[row.ageGroup]} · ${row.gender==='female'?'女':'男'}`;
   const id=document.createElement('small');id.textContent=row.id;label.append(id);b.append(img,label);
   b.onclick=()=>{stop();selected=row;show();};$('#people').append(b);
  }
  if(!rows.includes(selected)&&rows.length){selected=rows[0];stop();show();}
 }
 function step(delta){condition=conditions[(conditions.indexOf(condition)+delta+conditions.length)%conditions.length];show();}
 for(const c of conditions){const b=document.createElement('button');b.textContent=c==='same'?'同龄':`${c} 岁`;b.dataset.condition=c;b.onclick=()=>{stop();condition=c;show();};$('#ageOptions').append(b);}
 $('#previousAge').onclick=()=>{stop();step(-1);};$('#nextAge').onclick=()=>{stop();step(1);};
 $('#play').onclick=()=>{if(timer){stop();return;}$('#play').textContent='暂停轮播';$('#play').setAttribute('aria-pressed','true');timer=setInterval(()=>step(1),3500);};
 for(const b of document.querySelectorAll('[data-color]'))b.onclick=()=>{color=b.dataset.color;show();};
 $('#ageFilter').onchange=renderPeople;$('#genderFilter').onchange=renderPeople;
 $('#closeZoom').onclick=()=>$('#zoom').close();$('#zoom').onclick=e=>{if(e.target===$('#zoom'))$('#zoom').close();};
 document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});
 fetch(asset+'demo.json').then(r=>{if(!r.ok)throw Error('Missing manifest');return r.json();}).then(d=>{
  data=d;if(!data.people.length)throw Error('Empty set');
  for(const age of Object.keys(ages)){const option=document.createElement('option');option.value=age;option.textContent=ages[age];$('#ageFilter').append(option);}
  $('#intro').textContent=`${data.people.length} 张高清正脸 · 男女各 12 张 · 六组年龄段 · 原生 1024`;
  const query=new URLSearchParams(location.search);
  if(conditions.includes(query.get('age')))condition=query.get('age');
  if(['raw','preserved'].includes(query.get('color')))color=query.get('color');
  selected=data.people.find(r=>r.id===query.get('person'))||data.people.find(r=>r.ageGroup==='20-29')||data.people[0];renderPeople();$('#viewer').hidden=false;show();
 }).catch(()=>{$('#status').textContent='高清样例暂未就绪，请稍后刷新；仍可返回拍摄或上传照片。';});
})();
