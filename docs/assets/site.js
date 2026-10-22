(()=>{'use strict';
const $=q=>document.querySelector(q), $$=q=>Array.from(document.querySelectorAll(q));
const D=window.PHORA_CASES||{}, phases=['DRY','ART','VEN','DEL'];
const phaseLong={DRY:'NONCONTRAST',ART:'ARTERIAL',VEN:'PORTAL VENOUS',DEL:'DELAYED'};
let caseId='CASE00001',phase='ART',slicePos=10,opacity=.58,playTimer=null;
const current=()=>D[caseId];
const asset=(item)=>`assets/cases/${caseId}/${item}`;
function img(id,path,show=true){let e=document.getElementById(id);if(!e)return;e.style.display=show?'block':'none';if(show){e.onerror=()=>{e.style.display='none'};e.src=asset(path);}}
function updateMasks(){let d=current(),i=String(slicePos).padStart(2,'0'); const have=(key)=>d.annotation_keys.includes(key);
  img('ct-lesion',`lesion_${i}.png`,$('#mask-lesion').checked);
  img('ct-aphe',`aphe_${i}.png`,have('aphe')&&$('#mask-aphe').checked);
  img('ct-ven',`venous_washout_${i}.png`,have('venous_washout')&&$('#mask-ven').checked);
  img('ct-del',`delayed_washout_${i}.png`,have('delayed_washout')&&$('#mask-del').checked);
  img('evidence-lesion',`lesion_${i}.png`,$('#mask-lesion').checked);
  img('evidence-heat',`${phase}_heat_${i}.png`,phase!=='DRY');
  ['ct-lesion','ct-aphe','ct-ven','ct-del','evidence-lesion'].forEach(k=>{const e=document.getElementById(k);if(e)e.style.opacity=opacity});
}
function drawCurve(){const d=current(),svg=$('#enhancement-chart');svg.replaceChildren();const NS='http://www.w3.org/2000/svg';const el=(type,attr,text)=>{let e=document.createElementNS(NS,type);Object.entries(attr).forEach(([k,v])=>e.setAttribute(k,v));if(text!==undefined)e.textContent=text;svg.append(e);return e;};
 let vs=phases.map(p=>d.phase_stats[p].lesion_mean_hu),max=Math.ceil((Math.max(...vs)+24)/20)*20,min=Math.floor((Math.min(...vs)-18)/20)*20;
 let x=i=>64+i*(523/3),y=v=>224-(v-min)/(max-min)*179;
 for(let i=0;i<=4;i++){let v=min+i*(max-min)/4,yy=y(v);el('line',{x1:57,x2:608,y1:yy,y2:yy,stroke:'#7a9da4','stroke-opacity':'.24'});el('text',{x:47,y:yy+5,fill:'#829faa','font-size':'11','text-anchor':'end','font-family':'monospace'},String(Math.round(v)))}
 let points=vs.map((v,i)=>[x(i),y(v)]);let poly='M'+points.map(p=>p.map(n=>n.toFixed(2)).join(',')).join(' L');el('path',{d:poly,stroke:'#f2b87d','stroke-width':3.2,fill:'none','stroke-linecap':'round','stroke-linejoin':'round'});
 points.forEach((p,i)=>{const active=phases[i]===phase;el('circle',{cx:p[0],cy:p[1],r:active?9:5,fill:active?'#f2b87d':'#8cdbce',stroke:'#071923','stroke-width':2});el('text',{x:p[0],y:p[1]-17,'font-size':active?13:11,fill:active?'#f7d6af':'#a8c5cb','text-anchor':'middle','font-family':'monospace'},vs[i].toFixed(1));el('text',{x:p[0],y:258,'font-size':11,fill:active?'#f2b87d':'#819faa','text-anchor':'middle','font-family':'monospace'},phases[i]);});
}
function update(){const d=current();if(!d)return;const i=String(slicePos).padStart(2,'0'),zi=d.slices[slicePos];
 $('#case-name').textContent=caseId;$('#main-panel-label').textContent=phaseLong[phase]+' · '+phase;
 $('#slice-indicator').textContent=`${zi+1} / ${d.shape[0]}`;$('#slice-tag').textContent=`AXIAL SLICE ${zi+1}`;
 $('#slice-slider').max=String(d.slices.length-1);$('#slice-slider').value=String(slicePos);
 img('ct-image',`${phase}_${i}.webp`);img('evidence-image',`${phase}_${i}.webp`);
 $('#evidence-label').textContent=phase==='DRY'?'NO CONTRAST BASELINE':phase+' − DRY';
 updateMasks();
 $$('.phase-timeline button').forEach(b=>{const on=b.dataset.phase===phase;b.classList.toggle('active',on);b.setAttribute('aria-pressed',String(on))});
 $$('[data-case]').forEach(b=>{const on=b.dataset.case===caseId;b.classList.toggle('active',on);b.setAttribute('aria-pressed',String(on))});
 for(const [id,key] of [['mask-aphe','aphe'],['mask-ven','venous_washout'],['mask-del','delayed_washout']]){const e=document.getElementById(id);e.disabled=!d.annotation_keys.includes(key)}
 $('#mask-availability').textContent=caseId==='CASE00001'?'Case 01 includes lesion and three clinical-feature annotations.':'Case 02 includes the lesion annotation only; the additional concept masks were not provided.';
 $('#current-hu').textContent=phaseLong[phase]+' · '+d.phase_stats[phase].lesion_mean_hu.toFixed(1)+' HU';
 for(const [id,key] of [['sig-art','delta_art_dry'],['sig-ven','delta_ven_art'],['sig-del','delta_del_art']]){let v=d[key];$('#'+id).innerHTML=`${v>=0?'+':''}${v.toFixed(1)} <small>HU</small>`}
 drawCurve();
}
function changeCase(id){caseId=id;phase='ART';let d=current(), peak=d.primary_slice;let closest=d.slices.reduce((a,b,i)=>Math.abs(b-peak)<Math.abs(d.slices[a]-peak)?i:a,0);slicePos=closest;$('#mask-aphe').checked=false;$('#mask-ven').checked=false;$('#mask-del').checked=false;update();}
$$('[data-case]').forEach(b=>b.addEventListener('click',()=>changeCase(b.dataset.case)));
$$('[data-phase]').forEach(b=>b.addEventListener('click',()=>{phase=b.dataset.phase;update()}));
$('#slice-slider').addEventListener('input',e=>{slicePos=+e.target.value;update()});
['mask-lesion','mask-aphe','mask-ven','mask-del'].forEach(id=>$('#'+id).addEventListener('change',updateMasks));
$('#opacity-slider').addEventListener('input',e=>{opacity=+e.target.value/100;$('#opacity-value').textContent=e.target.value+'%';updateMasks()});
$('#phase-play').addEventListener('click',e=>{let b=e.currentTarget;if(playTimer){clearInterval(playTimer);playTimer=null;b.classList.remove('active');b.innerHTML='▶ <span>AUTO PLAY</span>';return;}b.classList.add('active');b.innerHTML='Ⅱ <span>PAUSE</span>';playTimer=setInterval(()=>{phase=phases[(phases.indexOf(phase)+1)%4];update();},1400)});
changeCase(caseId);
const vals={
 composite:[['Oracle concepts',.935,true],['Extra Trees',.763],['Multi-view stack',.734],['PHORA',.718],['Random forest',.705],['Simple hierarchy',.692],['Elastic-net logistic',.690],['Flat XGBoost',.687],['RBF-SVM',.281],['Hist. gradient boost',.200],['Majority class',.050]],
 qwk:[['Oracle concepts',.947,true],['Extra Trees',.785],['Multi-view stack',.741],['PHORA',.738],['Random forest',.709],['Simple hierarchy',.703],['Elastic-net logistic',.702],['Flat XGBoost',.686],['RBF-SVM',.232],['Hist. gradient boost',.113],['Majority class',0]],
 scr:[['Oracle concepts',.870,true],['Multi-view stack',.694],['Hist. gradient boost',.692],['Flat XGBoost',.692],['Random forest',.683],['Extra Trees',.635],['Simple hierarchy',.629],['Elastic-net logistic',.620],['PHORA',.605],['RBF-SVM',.555],['Majority class',.333]]
};let metric='composite';
function showScores(){let p=$('#score-list');p.replaceChildren();for(const [name,value,oracle] of vals[metric]){const row=document.createElement('div');row.className='score-row'+(name==='PHORA'?' phora':'');row.innerHTML=`<span class="name">${name}${oracle?' †':''}</span><div class="track"><span style="width:0"></span></div><b>${value.toFixed(3)}</b>`;p.append(row);requestAnimationFrame(()=>requestAnimationFrame(()=>row.querySelector('.track span').style.width=(value*100)+'%'));}};
$$('[data-metric]').forEach(b=>b.addEventListener('click',()=>{metric=b.dataset.metric;$$('[data-metric]').forEach(c=>c.classList.toggle('active',c===b));showScores()}));showScores();
const quartiles=[['Q1 · lowest',.800],['Q2',.533],['Q3',.643],['Q4 · highest',.467]];for(const [name,v] of quartiles){let div=document.createElement('div');div.className='unc-bar';div.innerHTML=`<strong>${(v*100).toFixed(1)}%</strong><div><i style="height:${100*v}%"></i></div><span>${name}</span>`;$('#uncertainty-bars').append(div)}
const steps=[['01 — IMAGING REPRESENTATION','From voxels to physiological evidence','Four CT phases and a lesion annotation are distilled into morphology, radiomic texture, local context, and multiphase enhancement trajectories.','~900 candidate imaging features → 140 screened variables',['DRY','ART','VEN','DEL']],['02 — LATENT CLINICAL EVIDENCE','Supervision becomes reusable concepts','Ground-truth APHE, washout, and capsule annotations supervise auxiliary heads during training. At inference, the model consumes predicted concept and burden signals—not annotation ground truth.','6 concept probabilities · 5 spatial burdens · 24 clinical interactions',['APHE','WASHOUT','CAPSULE','BURDEN']],['03 — HIERARCHICAL SPECIAL ROUTING','The exceptional branches are separate','A direct gate and a factorized gate estimate whether an observation belongs to the ordered LI-RADS spectrum, LR-M, or LR-TIV. The router reconciles these estimates.','ORDINAL  /  LR-M  /  LR-TIV',['ORD','LR-M','TIV','GATE']],['04 — ORDINAL DECISION','When appropriate, severity is ordered','A class-balanced ordinal regression expert and an all-threshold expert estimate severity, with ordered cut-points and clinically motivated consistency checks.','LR-1   →   LR-2   →   LR-3   →   LR-4   →   LR-5',['LR-1','LR-2','LR-3','LR-4']]];
function drawStep(i){$$('[data-step]').forEach(b=>b.classList.toggle('active',+b.dataset.step===i));let d=steps[i];$('#stage-caption').textContent=d[0];$('#stage-title').textContent=d[1];$('#stage-description').textContent=d[2];$('#stage-bottom').textContent=d[3];const g=$('#process-nodes');g.replaceChildren();let pos=[[95,160],[260,110],[452,205],[630,155]];d[4].forEach((txt,j)=>{const NS='http://www.w3.org/2000/svg';let group=document.createElementNS(NS,'g');group.setAttribute('transform',`translate(${pos[j][0]} ${pos[j][1]})`);let c=document.createElementNS(NS,'circle');c.setAttribute('r',j===i?34:28);c.setAttribute('fill',j===i?'#e7b88b':'#143948');c.setAttribute('stroke','#83c7bc');c.setAttribute('stroke-width','2');group.append(c);let t=document.createElementNS(NS,'text');t.textContent=txt;t.setAttribute('font-family','monospace');t.setAttribute('font-size','11');t.setAttribute('text-anchor','middle');t.setAttribute('dominant-baseline','central');t.setAttribute('fill','#eaf4ed');group.append(t);g.append(group)})}
$$('[data-step]').forEach(b=>b.addEventListener('click',()=>drawStep(+b.dataset.step)));drawStep(0);
// Saved, out-of-sample PHORA validation outputs: never interpret as PHORA outputs for the two illustrated CTs.
const records={
 CASE00015:{truth:'LR-5',pred:'LR-5',reg:4.6194286,at:4.2395357688,aphe:.628316462,values:[.00281834,.00203555,.08715413,.41047385,.289353,.05404095,.1541242]},
 CASE00025:{truth:'LR-5',pred:'LR-M',reg:3.74991,at:3.266715706,aphe:.104699663,values:[.00169895,.00364816,.07221839,.01523445,.00955574,.80560756,.09203678]},
 CASE00035:{truth:'LR-M',pred:'LR-M',reg:3.6034667,at:2.998772598,aphe:.0228804,values:[.00169347,.00514611,.04138306,.00295282,.00275701,.83144975,.11461781]},
 CASE00084:{truth:'LR-TIV',pred:'LR-TIV',reg:4.2551394,at:3.798325146,aphe:.271672159,values:[.00099260,.00122704,.05373015,.05984658,.02979909,.09722343,.7571811]},
 CASE00099:{truth:'LR-1',pred:'LR-2',reg:2.328488,at:2.00224945,aphe:.467544317,values:[.2639041,.28912863,.076860026,.024731351,.04638608,.24680848,.052181423]}
};
function drawRecord(id){let a=records[id];$('#record-id').textContent=id;$('#record-truth').textContent=a.truth;$('#record-pred').textContent=a.pred;$('#record-reg').textContent=a.reg.toFixed(3);$('#record-at').textContent=a.at.toFixed(3);$('#record-aphe').textContent=a.aphe.toFixed(3);$$('[data-record]').forEach(b=>b.classList.toggle('active',b.dataset.record===id));let pane=$('#record-chart');pane.replaceChildren();['LR-1','LR-2','LR-3','LR-4','LR-5','LR-M','LR-TIV'].forEach((k,i)=>{let v=a.values[i],row=document.createElement('div');row.className='record-bar'+(k===a.pred?' active':'');row.innerHTML=`<span>${k}</span><div class="track"><i style="width:${(v*100).toFixed(2)}%"></i></div><b>${v.toFixed(3)}</b>`;pane.append(row)})}
$$('[data-record]').forEach(b=>b.addEventListener('click',()=>drawRecord(b.dataset.record)));drawRecord('CASE00015');
const scroll=()=>{$('#read-progress').style.width=(100*scrollY/Math.max(1,document.documentElement.scrollHeight-innerHeight))+'%'};window.addEventListener('scroll',scroll,{passive:true});scroll();
})();
