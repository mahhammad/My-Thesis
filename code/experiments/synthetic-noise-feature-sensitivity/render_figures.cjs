"use strict";
/** E-CH4-14: vector figures from audited post-fit tables, no RNG, no training. Run: node render_figures.cjs */
function generateFigures(data) {
 if(data.schema!=="e-ch4-14-derived-analysis-v1"||data.level.length!==12||data.paired.length!==18||data.learning.length!==372)throw Error("Bad derived input");
 const C={shared:"#2563eb",class:"#ea580c",teal:"#0d9488",violet:"#7c3aed",dark:"#172338",mute:"#59667b",grid:"#e5eaf1"};
 const esc=x=>String(x).replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;").replaceAll('"',"&quot;");
 const L=(k,f)=>data.level.find(x=>x.k===k&&x.family===f);
 const vals=(f,key,scale=1)=>Array.from({length:6},(_,k)=>L(k,f)[key+"_mean"]*scale);
 const spread=(f,key,scale=1)=>Array.from({length:6},(_,k)=>[L(k,f)[key+"_min"]*scale,L(k,f)[key+"_max"]*scale]);
 const line=(label,v,color,extras={})=>({label,v,color,...extras});
 const pairLines=(key,scale=1)=>["shared","class"].map(f=>line(f,vals(f,key,scale),C[f],{ranges:spread(f,key,scale)}));
 const fmt=v=>Math.abs(v)>=100?v.toFixed(0):Math.abs(v)>=10?v.toFixed(1):Math.abs(v)>=1?v.toFixed(2):Math.abs(v)>=.1?v.toFixed(3):Math.abs(v)>=.001?v.toFixed(4):v.toExponential(1);
 const rnd=x=>Number(x.toFixed(4));
 function panelSVG(p,x0,y0,pw,ph){
  const left=x0+72,right=x0+pw-24,top=y0+54,bottom=y0+ph-81,n=p.n||6;
  const all=[];for(const s of p.series){for(const v of s.v)if(Number.isFinite(v))all.push(v);if(s.ranges)for(const rr of s.ranges)for(const v of rr)if(Number.isFinite(v))all.push(v);if(s.points)for(const x of s.points)if(Number.isFinite(x.y))all.push(x.y)}
  if(!all.length)throw Error("Empty panel "+p.title);
  let lo=p.min??Math.min(...all),hi=p.max??Math.max(...all);
  if(p.zero){lo=Math.min(lo,0);hi=Math.max(hi,0)}
  const delta=Math.max(hi-lo,Math.abs(hi)*.1,0.001);
  if(p.min==null)lo-=delta*.1;if(p.max==null)hi+=delta*.1;
  const X=i=>left+(right-left)*i/(n-1),Y=v=>bottom-(bottom-top)*(v-lo)/(hi-lo);
  const out=[`<rect x="${rnd(x0)}" y="${rnd(y0)}" width="${rnd(pw)}" height="${rnd(ph)}" rx="10" fill="#fcfdff" stroke="#e3eaf2"/>`,`<text x="${rnd(x0+20)}" y="${rnd(y0+30)}" font-size="17" font-weight="700" fill="${C.dark}">${esc(p.title)}</text>`];
  for(let j=0;j<=4;j++){const v=hi-(hi-lo)*j/4,yy=top+(bottom-top)*j/4;out.push(`<line x1="${left}" y1="${rnd(yy)}" x2="${right}" y2="${rnd(yy)}" stroke="${C.grid}"/>`,`<text x="${left-12}" y="${rnd(yy+4)}" text-anchor="end" font-size="12" fill="${C.mute}">${fmt(v)}</text>`)}
  if(p.zero&&lo<0&&hi>0)out.push(`<line x1="${left}" y1="${rnd(Y(0))}" x2="${right}" y2="${rnd(Y(0))}" stroke="#8792a3" stroke-dasharray="3 4"/>`);
  for(const i of (p.ticks||Array.from({length:n},(_,i)=>i))){out.push(`<text x="${rnd(X(i))}" y="${rnd(bottom+22)}" text-anchor="middle" font-size="12" fill="${C.mute}">${i}</text>`)}
  out.push(`<text x="${rnd((left+right)/2)}" y="${rnd(bottom+44)}" text-anchor="middle" font-size="12" fill="${C.mute}">${esc(n===31?"Adam update step":"Added noise features (k)")}</text>`);
  p.series.forEach((s,j)=>{
   if(s.ranges)s.ranges.forEach(([a,b],i)=>{if(!Number.isFinite(a)||!Number.isFinite(b)||a===b)return;const xx=X(i)+(j%2?4:-4);
    out.push(`<path d="M${rnd(xx)} ${rnd(Y(a))} V${rnd(Y(b))} M${rnd(xx-4)} ${rnd(Y(a))} H${rnd(xx+4)} M${rnd(xx-4)} ${rnd(Y(b))} H${rnd(xx+4)}" fill="none" stroke="${s.color}" stroke-width="1" opacity=".3"/>`)});
   if(s.points)s.points.forEach((q,i)=>out.push(`<circle cx="${rnd(X(q.k)+((i%4)-1.5)*5)}" cy="${rnd(Y(q.y))}" r="4.4" fill="${s.color}" fill-opacity=".4"/>`));
   const points=s.v.map((v,i)=>Number.isFinite(v)?[X(i),Y(v)]:null);
   const path=points.reduce((x,p)=>x+(p?(x?" L":"M")+rnd(p[0])+" "+rnd(p[1]):""),"");
   out.push(`<path d="${path}" fill="none" stroke="${s.color}" stroke-width="${s.dash?1.85:2.65}" stroke-linecap="round" stroke-linejoin="round" ${s.dash?`stroke-dasharray="${s.dash}"`:""}/>`);
   points.forEach(p=>{if(p)out.push(`<circle cx="${rnd(p[0])}" cy="${rnd(p[1])}" r="${s.dash?2.5:3.6}" fill="#fff" stroke="${s.color}" stroke-width="1.6"/>`)});
  });
  const block=(pw-30)/p.series.length; p.series.forEach((s,j)=>{const lx=x0+16+j*block,yy=y0+ph-24;out.push(`<path d="M${rnd(lx)} ${rnd(yy-3)} h14" stroke="${s.color}" stroke-width="2.5" ${s.dash?`stroke-dasharray="${s.dash}"`:""}/>`,`<text x="${rnd(lx+20)}" y="${rnd(yy+1)}" font-size="${p.series.length>4?10:11}" fill="${C.dark}">${esc(s.label)}</text>`)});
  return out.join("\n");
 }
 function fig(title,subtitle,panels,foot){
  const w=1220,gap=26,px=26,pw=(w-2*px-gap)/2,ph=374,h=110+Math.ceil(panels.length/2)*(ph+20)+62;
  const out=[`<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" role="img" aria-label="${esc(title)}">`,`<title>${esc(title)}</title>`,`<desc>${esc(subtitle+" "+foot)}</desc>`,`<rect width="${w}" height="${h}" fill="#fff"/>`,`<g font-family="Arial,DejaVu Sans,sans-serif">`,`<text x="${px}" y="45" font-size="25" font-weight="700" fill="${C.dark}">${esc(title)}</text>`,`<text x="${px}" y="75" font-size="14" fill="${C.mute}">${esc(subtitle)}</text>`];
  panels.forEach((p,i)=>out.push(panelSVG(p,px+(i%2)*(pw+gap),110+Math.floor(i/2)*(ph+20),pw,ph)));
  out.push(`<text x="${px}" y="${h-27}" font-size="12" fill="${C.mute}">${esc(foot)}</text>`,`</g></svg>`);return out.join("\n")+"\n";
 }
 const o={};
 const fullfield=(name)=>[{label:"shared | Validation",v:vals("shared","validation_"+name,100),color:C.shared,ranges:spread("shared","validation_"+name,100)},{label:"class | Validation",v:vals("class","validation_"+name,100),color:C.class,ranges:spread("class","validation_"+name,100)},{label:"shared | Train",v:vals("shared","train_"+name,100),color:C.shared,dash:"7 4"},{label:"class | Train",v:vals("class","train_"+name,100),color:C.class,dash:"7 4"}];
 o["01_accuracy_vs_noise.svg"]=fig("Classification performance as noise increases","Same frozen Train/Validation splits for all 18 dependent subsets.",[{title:"Balanced accuracy (%)",series:fullfield("BA")},{title:"Ordinary accuracy (%)",series:fullfield("Accuracy")}],"Whiskers = selected-subset min/max, not confidence intervals. Dashed = Train.");
 o["02_geometry_and_margin.svg"]=fig("Geometry and decision margin at step 30","Train-only class centers; Validation used for evaluation, not fitting.",
 [{title:"Validation R_JS (x 1000)",series:pairLines("validation_R_JS",1000)},
  {title:"Train center-separation I (x 1000)",series:pairLines("I",1000)},
  {title:"Validation median decision margin (x 1000)",series:pairLines("validation_margin_median",1000)},
  {title:"Validation hard-margin error (%)",series:pairLines("validation_hard_error",100)}],
 "Lower R_JS does not imply greater I or a higher BA. Whiskers show subset dispersion.");
 const cs=[];for(const k of [0,2,5])for(const family of ["shared","class"])cs.push({k,family,color:k===0?C.shared:k===2?C.teal:C.class,dash:family==="class"?"6 4":null,label:"k"+k+" "+(family==="shared"?"S":"C")});
 const learn=(name,key,scale=1)=>({title:name,n:31,ticks:[0,5,10,15,20,25,30],series:cs.map(s=>({label:s.label,v:data.learning.filter(x=>x.k===s.k&&x.family===s.family).map(x=>x[key+"_mean"]==null?null:x[key+"_mean"]*scale),color:s.color,dash:s.dash}))});
 o["03_learning_diagnostics.svg"]=fig("Fixed-budget learning trajectories","S = shared, C = class; k = 0, 2 (four-subset mean), 5.",
 [learn("Train Softplus-JS loss (x 1000)","loss",1000),learn("Validation balanced accuracy (%)","validation_BA",100),learn("Gradient L2 (x 1000)","grad_l2",1000),learn("Adam update L2","update_l2",1)],
 "All jobs used exactly 30 updates; no best-step model selection.");
 const paired=(field,scale,title)=>({title,zero:true,series:[{label:"class - shared",color:C.violet,v:Array.from({length:6},(_,k)=>{const rows=data.paired.filter(x=>x.k===k);return rows.reduce((s,x)=>s+x[field+"_delta"]*scale,0)/rows.length;}),points:data.paired.map(x=>({k:x.k,y:x[field+"_delta"]*scale}))}]});
 o["04_paired_family_differences.svg"]=fig("Matched-subset comparison: class minus shared","Each point is one matched subset, not an independent seed.",
 [paired("validation_BA",100,"Validation BA difference (percentage points)"),paired("validation_Accuracy",100,"Validation accuracy difference (pp)"),paired("I",1000,"Train center-separation I difference (x 1000)"),paired("validation_margin_median",1000,"Validation median margin difference (x 1000)")],
 "Positive values favor class; solid line is the descriptive within-k mean.");
 o["05_prediction_flips.svg"]=fig("Prediction changes relative to noise-free N0","Matched sample IDs and labels; flips count prediction changes, not necessarily errors.",
 [{title:"Train changes (300 samples)",series:pairLines("train_flips")},{title:"Validation changes (90 samples)",series:pairLines("validation_flips")}],
 "Whiskers are the selected-subset min/max (not statistical uncertainty).");
 o["06_model_cost.svg"]=fig("Policy size and CPU computational cost","End-of-job resources include 30 updates and canonical metric computation.",
 [{title:"Trainable policy parameters",zero:true,series:["shared","class"].map(f=>({label:f,color:C[f],v:Array.from({length:6},(_,k)=>L(k,f).parameters)}))},
  {title:"Elapsed per job (seconds)",zero:true,series:pairLines("job_seconds")}],
 "These elapsed times are specific to the implementation and CPU environment.");
 o["07_validation_class_recall.svg"]=fig("Validation recall by true class","Validation class sizes: c0=13, c1=49, c2=28.",
 ["shared","class"].map(f=>({title:"Policy "+f+" (recall %)",series:[0,1,2].map((cl)=>({label:"Class "+cl,color:[C.shared,C.class,C.violet][cl],v:Array.from({length:6},(_,k)=>data.classes.find(x=>x.k===k&&x.family===f&&x.class_id===cl).recall_mean*100)}))})),
 "Class 1 carries most Validation mistakes; means describe dependent subsets.");
 return o;
}
if (typeof require==="function" && typeof module!=="undefined" && require.main===module) {
 const fs=require("node:fs"),path=require("node:path"),root=__dirname;
 const data=JSON.parse(fs.readFileSync(path.join(root,"tables/analysis_summary.json"),"utf8"));
 const figs=generateFigures(data);
 for (const [name,content] of Object.entries(figs)){fs.writeFileSync(path.join(root,"figures",name),content,"utf8");process.stdout.write("Wrote figures/"+name+"\n");}
}
