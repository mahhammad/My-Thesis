"use strict";
/** E-CH4-14: read-only postfit reproducibility + checkpoint SHA verification.
  * node reproduce_postfit.cjs --check (default) verifies; --write regenerates ONLY tables/*.csv and analysis_summary.json.
  * No graph, RNG, optimizer, Test split, training, or external dependencies.
  */
function reproduce(mode){
const fs=require("node:fs"),path=require("node:path"),crypto=require("node:crypto"),assert=require("node:assert/strict");
const dir=__dirname,read=p=>JSON.parse(fs.readFileSync(path.join(dir,p),"utf8")),sha=p=>crypto.createHash("sha256").update(fs.readFileSync(path.join(dir,p))).digest("hex");
const summary=read("results/metrics/metrics_analysis.json"),full=read("results/metrics/metrics_full.json"),coverage=read("results/metrics/coverage.json"),prov=read("results/metrics/provenance.json"),derived=read("inputs/derived/manifest.json");
const ids=summary.individual.map(x=>x.run_id).sort(),set=new Set(ids),runs=[],q={};
assert.equal(ids.length,36);assert.equal(set.size,36);assert.equal(derived.datasets.length,18);assert.equal(prov.test_accessed,false);
const manifests=[["inputs/noise_raw/manifest.json","inputs/noise_raw/new_noise_seed20261102.npz"],["inputs/noise_masks/manifest.json","inputs/noise_masks/new_noise_masks.npz"]];
for(const [mp,fp] of manifests){const m=read(mp);assert.equal(sha(fp),m.file_sha256);assert.equal(m.test_accessed,false);assert.equal(m.spec_sha256,prov.data_spec_sha256)}
const mother=path.join(dir,"../objective-start-diagnostics/inputs/phase2_train_validation/ds20261102_train_val.npz");
assert.equal(crypto.createHash("sha256").update(fs.readFileSync(mother)).digest("hex"),prov.mother_sha256);
for(const row of derived.datasets){assert.equal(sha("inputs/derived/"+row.subset_id+".npz"),row.file_sha256)}
assert.equal(sha("inputs/derived/N2-01.npz"),prov.mother_sha256);
const ledger=fs.readFileSync(path.join(dir,"results/metrics/e14_spent_budget.jsonl"),"utf8").trim().split("\n").map(JSON.parse);
const ec={};for(const x of ledger)ec[x.event]=(ec[x.event]||0)+1;
assert.deepEqual({BUDGET_RESERVED:ec.BUDGET_RESERVED,START:ec.START,UPDATE:ec.UPDATE,SUCCESS:ec.SUCCESS}, {BUDGET_RESERVED:1,START:36,UPDATE:1080,SUCCESS:36});assert.equal(ec.FAIL_STOP,undefined);
const records=(s)=>{const cm=Array.from({length:3},()=>[0,0,0]);assert.equal(s.per_sample.labels.length,s.rows);assert.equal(s.per_sample.predicted.length,s.rows);s.per_sample.labels.forEach((v,j)=>cm[v][s.per_sample.predicted[j]]++);assert.deepEqual(cm,s.confusion_true_rows_pred_cols);const ba=cm.reduce((a,c,j)=>a+c[j]/c.reduce((z,x)=>z+x,0),0)/3,acc=cm.reduce((a,c,j)=>a+c[j],0)/s.rows;assert.ok(Math.abs(ba-s.balanced_accuracy)<1e-12);assert.ok(Math.abs(acc-s.ordinary_accuracy)<1e-12);return {labels:s.per_sample.labels,pred:s.per_sample.predicted}};
for(const id of ids){
 const rep=read("results/metrics/"+id+".json"),trace=read("results/traces/"+id+"/trace.json").trace;
 assert.deepEqual(rep,full.runs[id]);assert.equal(rep.test_accessed,false);assert.equal(trace.length,31);
 assert.equal(fs.readFileSync(path.join(dir,"results/traces/"+id+"/trace.csv"),"utf8").trim().split("\n").length,32);
 const cp="results/checkpoints/"+id+"/step_30.pt";assert.equal(sha(cp),fs.readFileSync(path.join(dir,cp+".sha256"),"utf8").trim().split(/\s+/)[0]);
 const stages=ledger.filter(x=>x.run_id===id),steps=stages.filter(x=>x.event==="UPDATE").map(x=>x.updates_completed);
 assert.equal(stages.filter(x=>x.event==="START").length,1);assert.equal(stages.filter(x=>x.event==="SUCCESS").length,1);assert.deepEqual(steps,Array.from({length:30},(_,i)=>i+1));
 assert.ok(trace.every((x,i)=>x.step===i&&x.update_completed===i&&x.numeric_status==="finite"));
 const k=+id.slice(1,2),d=3+k,family=rep.family,rt=rep.resource_usage,t=rep.train,v=rep.validation;
 assert.equal(rt.policy_parameter_count,(d+1)*d*(family==="class"?3:1));
 const sample=summary.individual.find(x=>x.run_id===id);
 assert.ok(Math.abs(sample.validation_BA-v.balanced_accuracy)<1e-12);assert.ok(Math.abs(sample.train_BA-t.balanced_accuracy)<1e-12);
 const row={run_id:id,subset_id:rep.subset_id,family,k,d,parameters:rt.policy_parameter_count,job_seconds:rt.job_compute_elapsed_s,trace_seconds:trace[30].elapsed_s,rss_mib:rt.process_rss_mib_at_report,
 train_BA:t.balanced_accuracy,validation_BA:v.balanced_accuracy,train_Accuracy:t.ordinary_accuracy,validation_Accuracy:v.ordinary_accuracy,
 train_R_JS:t.R_JS,validation_R_JS:v.R_JS,train_R_sqrt_JS:t.R_sqrt_JS,validation_R_sqrt_JS:v.R_sqrt_JS,train_R_KL:t.R_KL,validation_R_KL:v.R_KL,
 I:t.I,T:t.T,BT:t.BT,TNL:t.TNL,BTNL:t.BTNL,min_sqrt_JS:t.min_pairwise_sqrt_JS,mean_sqrt_JS:t.mean_pairwise_sqrt_JS,
 train_margin_q10:t.margin_quantiles["0.1"],validation_margin_q10:v.margin_quantiles["0.1"],train_margin_median:t.margin_quantiles["0.5"],validation_margin_median:v.margin_quantiles["0.5"],
 train_hard_error:t.hard_margin_error_conservative,validation_hard_error:v.hard_margin_error_conservative,validation_sigmoid_loss:v.balanced_sigmoid_margin_loss,validation_softplus_margin:v.balanced_softplus_all_rivals,
 train_ties:t.tie_count,validation_ties:v.tie_count,step0_loss:trace[0].train_softplus_loss,step30_loss:trace[30].train_softplus_loss,step0_gradient_l2:trace[0].gradient_l2,step30_gradient_l2:trace[30].gradient_l2,step30_update_l2:trace[30].adam_update_l2,
 val_BA_step0:trace[0].validation_BA,val_BA_peak:Math.max(...trace.map(x=>x.validation_BA)),val_BA_peak_step:trace.findIndex(x=>x.validation_BA===Math.max(...trace.map(y=>y.validation_BA))),decreases_every_update:trace.slice(1).every((x,i)=>x.train_softplus_loss<=trace[i].train_softplus_loss+1e-11)};
 runs.push(row);q[id]={rep,trace,train:records(t),validation:records(v)};
}
const base={};for(const fam of ["shared","class"])base[fam]=q["N0-01__"+fam];
for(const row of runs){const cur=q[row.run_id],b=base[row.family];for(const split of ["train","validation"]){assert.deepEqual(cur[split].labels,b[split].labels);const flips=cur[split].pred.reduce((s,x,i)=>s+(x!==b[split].pred[i]?1:0),0);row[split+"_flips"]=flips;assert.equal(flips,summary.individual.find(x=>x.run_id===row.run_id)[split+"_prediction_flips_vs_N0"]);}}
const avg=a=>a.reduce((s,x)=>s+x,0)/a.length,st=a=>({mean:avg(a),sd:a.length>1?Math.sqrt(a.reduce((s,x)=>s+(x-avg(a))**2,0)/(a.length-1)):null,min:Math.min(...a),max:Math.max(...a)});
const fields=["train_BA","validation_BA","train_Accuracy","validation_Accuracy","train_R_JS","validation_R_JS","train_R_sqrt_JS","validation_R_sqrt_JS","train_R_KL","validation_R_KL","I","T","BT","TNL","BTNL","min_sqrt_JS","mean_sqrt_JS","validation_margin_q10","validation_margin_median","validation_hard_error","validation_sigmoid_loss","validation_softplus_margin","train_flips","validation_flips","job_seconds","trace_seconds","rss_mib","step30_gradient_l2","step30_update_l2","val_BA_step0","val_BA_peak","step0_loss","step30_loss"];
const level=[],paired=[],learning=[],classes=[];
for(let k=0;k<=5;k++){
for(const family of ["shared","class"]){
 const group=runs.filter(x=>x.k===k&&x.family===family),g={k,family,n_subsets:group.length,parameters:group[0].parameters};
 for(const field of fields){const s=st(group.map(x=>x[field]));g[field+"_mean"]=s.mean;g[field+"_sd"]=s.sd;g[field+"_min"]=s.min;g[field+"_max"]=s.max;}level.push(g);
 for(let step=0;step<=30;step++){
  const rows=group.map(x=>q[x.run_id].trace[step]),z={k,family,step,n_subsets:group.length};
  for(const [key,src] of Object.entries({loss:"train_softplus_loss",train_BA:"train_BA",validation_BA:"validation_BA",train_Accuracy:"train_Accuracy",validation_Accuracy:"validation_Accuracy",grad_l2:"gradient_l2",elapsed_s:"elapsed_s"})){const s=st(rows.map(x=>x[src]));z[key+"_mean"]=s.mean;z[key+"_sd"]=s.sd;}
  z.update_l2_mean=step?avg(rows.map(x=>x.adam_update_l2)):null;learning.push(z);
 }
 for(let class_id=0;class_id<3;class_id++){const vals=group.map(x=>q[x.run_id].rep.validation.per_class[class_id].accuracy),s=st(vals);classes.push({k,family,class_id,n_subsets:group.length,validation_count:q[group[0].run_id].rep.validation.per_class[class_id].count,recall_mean:s.mean,recall_sd:s.sd});}
}
 for(const sub of [...new Set(runs.filter(x=>x.k===k).map(x=>x.subset_id))]){const a=runs.find(x=>x.subset_id===sub&&x.family==="shared"),b=runs.find(x=>x.subset_id===sub&&x.family==="class"),z={subset_id:sub,k};for(const field of fields)z[field+"_delta"]=b[field]-a[field];paired.push(z);}
}
const audit=mode==="--check"?read("tables/integrity_audit.json"):{source_commit:"7bc66e4d2535fb15dfa5b569933b2d6c5a0962ac"};
if(mode==="--check"){assert.equal(audit.report_count,36);assert.equal(audit.ledger_events.UPDATE,1080);}
const result={schema:"e-ch4-14-derived-analysis-v1",source_commit:audit.source_commit,design:"One mother and dependent feature subsets; SD reflects subset dispersion, not independent seeds.",runs,level,paired,learning,classes};
if(mode==="--check"){
 const existing=read("tables/analysis_summary.json");
 assert.equal(JSON.stringify(existing),JSON.stringify(result),"Committed derived summaries disagree with raw reports");
}
const csv=(rows,fields)=>fields.join(",")+"\n"+rows.map(x=>fields.map(k=>{let v=x[k];if(v==null)return "";if(typeof v==="number")return Number(v.toPrecision(13));if(typeof v==="boolean")return v?"true":"false";let s=String(v);return /[,"\n]/.test(s)?'"'+s.replaceAll('"','""')+'"':s}).join(",")).join("\n")+"\n";
const products={"per_dataset_metrics.csv":csv(runs,Object.keys(runs[0])),
"summary_by_noise_level.csv":csv(level,["k","family","n_subsets","parameters",...fields.flatMap(f=>[f+"_mean",f+"_sd"])]),
"paired_class_minus_shared.csv":csv(paired,Object.keys(paired[0])),
"learning_by_step.csv":csv(learning,Object.keys(learning[0])),
"validation_class_recall.csv":csv(classes,Object.keys(classes[0]))};
for(const [name,body] of Object.entries(products)){
 const location=path.join(dir,"tables",name);
 if(mode==="--write")fs.writeFileSync(location,body,"utf8");
 else assert.equal(fs.readFileSync(location,"utf8"),body,"Generated CSV differs: "+name);
}
if(mode==="--write")fs.writeFileSync(path.join(dir,"tables/analysis_summary.json"),JSON.stringify(result,null,2)+"\n","utf8");
console.log("E14_POSTFIT_REPRO_PASS",JSON.stringify({report_count:36,training_updates:1080,checkpoints_sha256:36,subsets:18,covered_summary_files:Object.keys(products).length,mode}));
}
if (require.main===module){const mode=process.argv[2]||"--check";if(!["--check","--write"].includes(mode))throw Error("Unknown option");reproduce(mode);}
