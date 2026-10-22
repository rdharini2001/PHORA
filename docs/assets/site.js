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
  img('ct-aphe',`aphe_${i}.png`,phase==='ART'&&have('aphe')&&$('#mask-aphe').checked);
  img('ct-ven',`venous_washout_${i}.png`,phase==='VEN'&&have('venous_washout')&&$('#mask-ven').checked);
  img('ct-del',`delayed_washout_${i}.png`,phase==='DEL'&&have('delayed_washout')&&$('#mask-del').checked);
  img('evidence-lesion',`lesion_${i}.png`,$('#mask-lesion').checked);
  img('evidence-heat',`${phase}_signed_${i}.png`,phase!=='DRY');
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
 const prev={ART:'DRY',VEN:'ART',DEL:'VEN'};$('#evidence-label').textContent=phase==='DRY'?'NONCONTRAST BASELINE':phase+' − '+prev[phase]+' · SIGNED ΔHU';const comp=$('#evidence-comparison');if(comp)comp.textContent=phase==='DRY'?'NONCONTRAST BASELINE':phase+' − '+prev[phase]+' · ΔHU';
 updateMasks();
 $$('.phase-timeline button').forEach(b=>{const on=b.dataset.phase===phase;b.classList.toggle('active',on);b.setAttribute('aria-pressed',String(on))});
 $$('[data-case]').forEach(b=>{const on=b.dataset.case===caseId;b.classList.toggle('active',on);b.setAttribute('aria-pressed',String(on))});
 for(const [id,key] of [['mask-aphe','aphe'],['mask-ven','venous_washout'],['mask-del','delayed_washout']]){const e=document.getElementById(id);e.disabled=!d.annotation_keys.includes(key)}
 $('#mask-availability').textContent=caseId==='CASE00001'?'APHE: arterial · washout: corresponding venous/delayed phase.':'Lesion annotation available; arterial and washout masks are not available for this case.';
 $('#current-hu').textContent=phaseLong[phase]+' · '+d.phase_stats[phase].lesion_mean_hu.toFixed(1)+' HU';
 const focus={DRY:['Noncontrast baseline','Native lesion attenuation establishes the reference for subsequent contrast uptake.'],ART:['Arterial enhancement','Arterial lesion contrast uptake is compared with its precontrast baseline. In Case 01, the annotated APHE region can be inspected alongside the measured HU change.'],VEN:['Portal venous transition','The contrast distribution continues to evolve. Case 01 includes a venous-washout annotation; relative liver–lesion behavior determines clinical washout.'],DEL:['Delayed-phase transition','The delayed acquisition reveals subsequent attenuation decline or persistence. A decrease in lesion HU alone does not establish LI-RADS washout.']};
 $('#reading-title').textContent=focus[phase][0];$('#reading-body').textContent=focus[phase][1];
 $('#reading-phase-tag').textContent=(phases.indexOf(phase)+1).toString().padStart(2,'0')+' / '+phaseLong[phase];
 const sign=d.signed_phase_stats?.[phase];
 if(sign){
  const change=sign.mean_delta_hu;const fmt=(v)=>Math.round(v*100)+'%';
  $('#reading-delta').textContent=(change>0?'+':'')+change.toFixed(1)+' HU';
  $('#reading-reference').textContent=phase+' − '+sign.reference;
  $('#signed-positive').innerHTML='<b>'+fmt(sign.fraction_increase_10hu)+'</b> +10 HU or more';
  $('#signed-negative').innerHTML='<b>'+fmt(sign.fraction_decrease_10hu)+'</b> −10 HU or more';
  $('#evidence-readout').textContent='Change from '+phaseLong[sign.reference].toLowerCase()+' to '+phaseLong[phase].toLowerCase()+', measured within the lesion. A ±10 HU display threshold reduces near-zero variation.';
 }else{
  $('#reading-delta').textContent=d.phase_stats.DRY.lesion_mean_hu.toFixed(1)+' HU';$('#reading-reference').textContent='NATIVE BASELINE';
  $('#signed-positive').innerHTML='<b>—</b> contrast not yet given';$('#signed-negative').innerHTML='<b>—</b> baseline reference';
  $('#evidence-readout').textContent='Noncontrast CT is the reference against which subsequent phases are compared.';
 }

 for(const [id,key] of [['sig-art','delta_art_dry'],['sig-ven','delta_ven_art'],['sig-del','delta_del_art']]){let v=d[key];$('#'+id).innerHTML=`${v>=0?'+':''}${v.toFixed(1)} <small>HU</small>`}
 drawCurve();
}
function selectPhase(p){phase=p;$('#mask-aphe').checked=p==='ART'&&current().annotation_keys.includes('aphe');$('#mask-ven').checked=p==='VEN'&&current().annotation_keys.includes('venous_washout');$('#mask-del').checked=p==='DEL'&&current().annotation_keys.includes('delayed_washout');update();}
function changeCase(id){caseId=id;phase='ART';let d=current(), peak=d.primary_slice;let closest=d.slices.reduce((a,b,i)=>Math.abs(b-peak)<Math.abs(d.slices[a]-peak)?i:a,0);slicePos=closest;$('#mask-aphe').checked=d.annotation_keys.includes('aphe');$('#mask-ven').checked=false;$('#mask-del').checked=false;update();}
$$('[data-case]').forEach(b=>b.addEventListener('click',()=>changeCase(b.dataset.case)));
$$('[data-phase]').forEach(b=>b.addEventListener('click',()=>{selectPhase(b.dataset.phase)}));
$('#slice-slider').addEventListener('input',e=>{slicePos=+e.target.value;update()});
['mask-lesion','mask-aphe','mask-ven','mask-del'].forEach(id=>$('#'+id).addEventListener('change',updateMasks));
$('#opacity-slider').addEventListener('input',e=>{opacity=+e.target.value/100;$('#opacity-value').textContent=e.target.value+'%';updateMasks()});
$('#phase-play').addEventListener('click',e=>{let b=e.currentTarget;if(playTimer){clearInterval(playTimer);playTimer=null;b.classList.remove('active');b.innerHTML='▶ <span>AUTO PLAY</span>';return;}b.classList.add('active');b.innerHTML='Ⅱ <span>PAUSE</span>';playTimer=setInterval(()=>{selectPhase(phases[(phases.indexOf(phase)+1)%4]);},1700)});
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
// PHORA public validation predictions: full multiphase case-level inference.
// Stage animation selects which named feature is spotlighted, not which inputs were seen.
const records={
 CASE00015:{truth:'LR-5',pred:'LR-5',reg:4.6194286,at:4.2395357688,values:[.00281834,.00203555,.08715413,.41047385,.289353,.05404095,.1541242],concepts:[.62831646,.103861205,.48224497,.59280246,.0811517,.17574401],burdens:[.20942904,.7230371,.551988,.14560412,.11892173],direct:[.8281687,.066918716,.10491253],factor:[.75362134,.080931164,.16544749]},
 CASE00025:{truth:'LR-5',pred:'LR-M',reg:3.74991,at:3.266715706,values:[.00169895,.00364816,.07221839,.01523445,.00955574,.80560756,.09203678],concepts:[.104699664,.44604748,.06555115,.17039186,.022860274,.04219675],burdens:[.23140499,.77194065,.8037998,.17682034,.12802061],direct:[.07647932,.83468175,.088838935],factor:[.10531355,.80667996,.08800652]},
 CASE00035:{truth:'LR-M',pred:'LR-M',reg:3.6034667,at:2.998772598,values:[.00169347,.00514611,.04138306,.00295282,.00275701,.83144975,.11461781],concepts:[.0228804,.5223881,.044499353,.04818158,.03654138,.020305088],burdens:[.360891,.5511387,.5575544,.26250002,.14338239],direct:[.045490067,.8804021,.074107826],factor:[.0580557,.81104106,.13090324]},
 CASE00084:{truth:'LR-TIV',pred:'LR-TIV',reg:4.2551394,at:3.798325146,values:[.0009926,.00122704,.05373015,.05984658,.02979909,.09722343,.7571811],concepts:[.27167216,.20648424,.13051426,.087545484,.05455724,.07474881],burdens:[.2701509,.93971115,.7658838,.13806868,.14497492],direct:[.096697725,.11235064,.7909517],factor:[.1558643,.12957314,.71456254]},
 CASE00099:{truth:'LR-1',pred:'LR-2',reg:2.328488,at:2.00224945,values:[.2639041,.28912863,.076860026,.024731351,.04638608,.24680848,.052181423],concepts:[.46754432,.16512717,.031601697,.47151306,.068331845,.07599401],burdens:[.73570585,.7157387,.7652149,.28080508,.18614972],direct:[.59185743,.34432077,.06382175],factor:[.6564325,.3012928,.042274714]}
};
const classOrder=['LR-1','LR-2','LR-3','LR-4','LR-5','LR-M','LR-TIV'];
let selectedRecord='CASE00015',cinemaStage=0,cinemaTimer=null;
const phaseCards=[
 {eyebrow:'THE REFERENCE ACQUISITION',title:'Begin with the unenhanced appearance',body:'Lesion morphology, size, and precontrast attenuation provide context for enhancement measurements. PHORA subsequently integrates radiomics and relative phase kinetics with the named clinical concepts.',keys:[]},
 {eyebrow:'ARTERIAL / APHE',title:'How confidently does PHORA recover arterial enhancement?',body:'The model estimates non-rim versus rim arterial phase hyperenhancement, with a separate spatial-burden estimate for the proportion of annotated-like territory.',keys:[['Non-rim APHE probability','concepts',0],['Rim APHE probability','concepts',1],['APHE burden estimate','burdens',0]]},
 {eyebrow:'PORTAL VENOUS / WASHOUT',title:'Venous signal begins to separate the routes',body:'Venous washout and capsule estimates provide context for major-feature interpretation. The burden estimate reflects a predicted spatial fraction, not class confidence.',keys:[['Venous washout probability','concepts',2],['Venous capsule probability','concepts',4],['Venous washout burden','burdens',1]]},
 {eyebrow:'DELAYED / PERSISTENCE',title:'Delayed washout and capsule complete the evidence',body:'Delayed phase features help distinguish persistent contrast from subsequent decline in lesion attenuation. The model uses this information jointly with the earlier phases.',keys:[['Delayed washout probability','concepts',3],['Delayed capsule probability','concepts',5],['Delayed washout burden','burdens',2]]},
 {eyebrow:'INTEGRATED / HIERARCHICAL ROUTING',title:'One multiphase examination. One routed category.',body:'Direct and factorized experts assess ordinal versus LR-M/LR-TIV evidence; the ordinal experts estimate LR-1–LR-5 severity when appropriate. The displayed result is the model’s recorded final decision.',keys:[]}
];
function stageRecord(){const r=records[selectedRecord],cfg=phaseCards[cinemaStage];
 $('#cinema-id').textContent=selectedRecord;$('#cinema-pred').textContent=r.pred;$('#cinema-match').textContent='REFERENCE '+r.truth;
 $('#cinema-reg').textContent=r.reg.toFixed(3);$('#cinema-at').textContent=r.at.toFixed(3);
 $('#cinema-stage-count').textContent=String(cinemaStage+1).padStart(2,'0')+' / 05';
 $('#cinema-stage-eyebrow').textContent=cfg.eyebrow;$('#cinema-stage-title').textContent=cfg.title;$('#cinema-stage-description').textContent=cfg.body;
 $$('[data-cinema]').forEach(b=>b.classList.toggle('active',b.dataset.cinema===selectedRecord));
 $$('[data-cinema-stage]').forEach(b=>{let ok=Number(b.dataset.cinemaStage)===cinemaStage;b.classList.toggle('active',ok);b.setAttribute('aria-pressed',String(ok))});
 const evidence=$('#cinema-concepts');evidence.replaceChildren();
 if(!cfg.keys.length){let e=document.createElement('div');e.className='cinema-concept-note';e.textContent=cinemaStage===0?'Baseline geometry and attenuation anchor the complete multiphase feature representation.':'Clinical concept heads, ordinal experts and special-category gates are combined in the final router.';evidence.append(e)}
 for(const [name,k,index] of cfg.keys){const value=r[k][index],item=document.createElement('div');item.className='cinema-concept-row';item.innerHTML=`<span>${name}</span><div class="cinema-concept-track"><i style="width:${(value*100).toFixed(2)}%"></i></div><b>${value.toFixed(3)}</b>`;evidence.append(item)}
 const distr=$('#cinema-distribution');distr.replaceChildren();classOrder.forEach((name,i)=>{const v=r.values[i],e=document.createElement('div');e.className='cinema-prob-row'+(name===r.pred?' selected':'');e.innerHTML=`<label>${name}</label><div class="cinema-prob-track"><i style="width:${Math.min(100,100*v).toFixed(2)}%"></i></div><b>${v.toFixed(3)}</b>`;distr.append(e)});
 const gates=$('#cinema-gates');gates.replaceChildren();for(const [name,arr] of [['Direct gate',r.direct],['Factorized gate',r.factor]]){let e=document.createElement('div');e.className='cinema-gate-row';e.innerHTML=`<small>${name}</small><div class="cinema-gate-stack"><span style="width:${(arr[0]*100).toFixed(2)}%" title="Ordinal ${(100*arr[0]).toFixed(1)}%"></span><span style="width:${(arr[1]*100).toFixed(2)}%" title="LR-M ${(100*arr[1]).toFixed(1)}%"></span><span style="width:${(arr[2]*100).toFixed(2)}%" title="LR-TIV ${(100*arr[2]).toFixed(1)}%"></span></div>`;gates.append(e)}
 const lane=document.querySelector('.cinema-timebar');lane.style.setProperty('--progress', (cinemaStage/4*100)+'%');
}
function stopCinema(){if(cinemaTimer){clearInterval(cinemaTimer);cinemaTimer=null;}$('#cinema-play').textContent='▶ AUTOPLAY'}
$$('[data-cinema]').forEach(b=>b.addEventListener('click',()=>{stopCinema();selectedRecord=b.dataset.cinema;cinemaStage=0;stageRecord()}));
$$('[data-cinema-stage]').forEach(b=>b.addEventListener('click',()=>{stopCinema();cinemaStage=Number(b.dataset.cinemaStage);stageRecord()}));
$('#cinema-play').addEventListener('click',()=>{if(cinemaTimer){stopCinema();return;}$('#cinema-play').textContent='Ⅱ PAUSE';cinemaTimer=setInterval(()=>{cinemaStage=(cinemaStage+1)%5;stageRecord()},2400)});
stageRecord();
const scroll=()=>{$('#read-progress').style.width=(100*scrollY/Math.max(1,document.documentElement.scrollHeight-innerHeight))+'%'};window.addEventListener('scroll',scroll,{passive:true});scroll();
})();
