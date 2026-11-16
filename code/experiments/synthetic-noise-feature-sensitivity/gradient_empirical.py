"""Empirical peak/trough envelopes and damping estimates for E-CH4-14.

No functional shape is imposed on the upper or lower envelope: they are
piecewise-linear curves through actual extrema, iteratively tightened to cover
every observed sample of the 61-step MEAN Gradient L2 curve for each k/family.
No optimizer updates, no Test, no replacement of recorded trajectories.

Separately, compare exponential and power-law regressions of positive
observed peak-to-adjacent-trough amplitudes. These are hypotheses, not bounds.
"""
from __future__ import annotations
import io
import math
import numpy as np
import matplotlib.pyplot as plt

N=61
T=np.arange(N)
COLORS=plt.get_cmap("tab10").colors

def require(ok,message):
    if not ok: raise AssertionError(message)

def extrema(y):
    """Strict-local extrema; endpoints are not treated as interior peaks/troughs."""
    y=np.asarray(y,dtype=float)
    hi=[i for i in range(1,len(y)-1) if y[i]>=y[i-1] and y[i]>y[i+1]]
    lo=[i for i in range(1,len(y)-1) if y[i]<=y[i-1] and y[i]<y[i+1]]
    return hi,lo

def _cover(y,knots,upper):
    """Smallest added set of observed contact anchors for a linear interpolation.

    Starts from the actual interior extrema. An anchor is added only if the
    interpolated envelope crosses a data point; the worst offender is repaired.
    This constructs an empirical observed-data enclosure, not a parametric fit.
    """
    y=np.asarray(y,dtype=float)
    eps=max(float(np.max(np.abs(y)))*1e-13,1e-17)
    knots=set(int(t) for t in knots)
    if not knots:
        # A completely monotone series has no repeated oscillations to fit.
        index=int(np.argmax(y) if upper else np.argmin(y))
        knots.add(index)
    for _ in range(N+1):
        anchors=np.array(sorted(knots),dtype=int)
        fitted=np.interp(T,anchors,y[anchors])
        residual=(y-fitted) if upper else (fitted-y)
        offender=int(np.argmax(residual))
        if residual[offender]<=eps:
            require(np.all((fitted>=y-eps) if upper else (fitted<=y+eps)),
                    "Enclosure check failed")
            require(np.allclose(fitted[anchors],y[anchors],rtol=0,atol=eps),
                    "Empirical envelope does not contact anchors")
            return fitted,[int(i) for i in anchors]
        require(offender not in knots,"Cannot repair empirical envelope")
        knots.add(offender)
    raise AssertionError("Failed to cover all observed gradient samples")

def empirical_envelope(y):
    y=np.asarray(y,dtype=float)
    require(y.shape==(N,) and np.isfinite(y).all() and np.all(y>=0),
            "Need nonnegative 61-point Gradient L2 curve")
    peaks,valleys=extrema(y)
    upper,up_contacts=_cover(y,peaks,True)
    lower,lo_contacts=_cover(y,valleys,False)
    require(np.all(upper>=y-1e-12) and np.all(lower<=y+1e-12)
            and np.all(upper>=lower-1e-12),
            "Empirical envelopes fail to cover the recorded curve")
    gap=np.maximum(upper-lower,0)
    # Stable interior windows avoid the spurious zero gap at an endpoint
    # when both envelopes happen to touch exactly the first/last sample.
    start=float(np.mean(gap[5:16]))
    end=float(np.mean(gap[45:56]))
    return {"upper":upper,"lower":lower,"gap":gap,"peaks":peaks,"valleys":valleys,
            "upper_contacts":up_contacts,"lower_contacts":lo_contacts,
            "early_gap":start,"late_gap":end,
            "late_over_early":float(end/start) if start>0 else None,
            "late_gap_smaller":bool(end<start-1e-13),
            "n_peaks":len(peaks),"n_valleys":len(valleys)}

def peak_amplitudes(y):
    """Each local maximum minus mean of its immediately surrounding valleys."""
    peaks,valleys=extrema(y)
    records=[]
    for p in peaks:
        left=[v for v in valleys if v<p]
        right=[v for v in valleys if v>p]
        if not left or not right:continue
        baseline=(y[left[-1]]+y[right[0]])/2
        amplitude=float(y[p]-baseline)
        if amplitude>0:
            records.append({"step":int(p),"amplitude":amplitude,
                            "peak":float(y[p]),
                            "left_trough":int(left[-1]),
                            "right_trough":int(right[0])})
    return records

def regression(records,window="all"):
    items=records
    if window=="first":items=[a for a in records if a["step"]<=30]
    if window=="last":items=[a for a in records if a["step"]>30]
    if len(items)<3:
        return {"n":len(items),"exp_rate":None,"exp_half_life":None,
                "exp_r2_log":None,"power_exponent":None,"power_r2_log":None,
                "better_log_fit":None}
    t=np.array([a["step"] for a in items],dtype=float)
    loga=np.log(np.array([a["amplitude"] for a in items],dtype=float))
    def line(x):
        slope,intercept=np.polyfit(x,loga,1)
        fitted=slope*x+intercept
        ss=float(np.sum((loga-fitted)**2))
        total=float(np.sum((loga-loga.mean())**2))
        return float(slope),float(1-ss/total if total>1e-20 else 1),ss
    exp_slope,exp_r2,exp_ss=line(t)
    power_slope,power_r2,power_ss=line(np.log1p(t))
    rate=float(-exp_slope)
    exponent=float(-power_slope)
    return {"n":len(items),"exp_rate":rate,
            "exp_half_life":(float(math.log(2)/rate) if rate>0 else None),
            "exp_r2_log":exp_r2,"power_exponent":exponent,
            "power_r2_log":power_r2,
            "better_log_fit":("exponential" if exp_ss<power_ss else
                              "power_law" if power_ss<exp_ss else "tie")}

def gradient_series(traces,family,k):
    chosen=[]
    for id,trace in sorted(traces.items()):
        if id.endswith("__"+family) and int(id[1])==k:
            require(len(trace)==N and [r["step"] for r in trace]==list(range(N)),
                    "Incomplete full trajectory: "+id)
            chosen.append((id,np.array([r["gradient_l2"] for r in trace],dtype=float)))
    require(len(chosen)==(1 if k in (0,5) else 4),
            "Group subset count mismatch")
    require(all(np.isfinite(a).all() and (a>=0).all() for _,a in chosen),
            "Invalid recorded gradients")
    return chosen

def _save(fig):
    buffer=io.BytesIO()
    fig.savefig(buffer,format="svg",bbox_inches="tight",metadata={"Date":None})
    plt.close(fig)
    return buffer.getvalue()

def derive(traces):
    figures={};groups=[];runs=[];peaks_rows=[]
    group_fits={}
    for family in ("shared","class"):
        fig,axs=plt.subplots(2,3,figsize=(15,8.6),sharex=True)
        for k,ax in enumerate(axs.flat):
            chosen=gradient_series(traces,family,k)
            mean=np.mean([curve for _,curve in chosen],axis=0)
            env=empirical_envelope(mean)
            group_fits[(family,k)]=env
            for id,curve in chosen:
                points=peak_amplitudes(curve)
                all_fit=regression(points)
                first_fit=regression(points,"first")
                last_fit=regression(points,"last")
                runs.append({"run_id":id,"family":family,"k":k,
                    "n_matched_peaks":len(points),
                    "amplitude_first":points[0]["amplitude"] if points else None,
                    "amplitude_last":points[-1]["amplitude"] if points else None,
                    "amplitude_last_over_first":(
                       points[-1]["amplitude"]/points[0]["amplitude"]
                       if len(points)>1 else None),
                    **{("full_"+key):val for key,val in all_fit.items()},
                    **{("first_"+key):val for key,val in first_fit.items()},
                    **{("last_"+key):val for key,val in last_fit.items()}})
                peaks_rows.extend([{"run_id":id,"family":family,"k":k,**p} for p in points])
                ax.plot(T,curve,color=COLORS[k],alpha=.20,lw=.7)
            ax.plot(T,mean,color=COLORS[k],lw=1.75,label="Observed mean")
            ax.plot(T,env["upper"],ls="--",lw=1.7,color="#a16207",
                    label="Upper through peaks")
            ax.plot(T,env["lower"],ls="--",lw=1.7,color="#0e7490",
                    label="Lower through troughs")
            ax.fill_between(T,env["lower"],env["upper"],color="gray",alpha=.09)
            if env["peaks"]:
                ax.scatter(env["peaks"],mean[env["peaks"]],s=12,
                           color="#a16207",zorder=4)
            if env["valleys"]:
                ax.scatter(env["valleys"],mean[env["valleys"]],s=12,
                           color="#0e7490",zorder=4)
            ratio=env["late_over_early"]
            title=f"k={k} (n={len(chosen)}) | late/early={ratio:.3f}" if ratio is not None else f"k={k} (n={len(chosen)}) | no gap"
            ax.set(title=title,xlim=(0,60),xlabel="Adam update 0–60")
            ax.grid(alpha=.15)
            ax.ticklabel_format(axis="y",style="sci",scilimits=(-2,2))
            if k%3==0:ax.set_ylabel("Gradient L2")
            if k==0:ax.legend(fontsize=6.5)
            groups.append({"family":family,"k":k,"n_subsets":len(chosen),
                "peak_count_mean_trace":env["n_peaks"],
                "trough_count_mean_trace":env["n_valleys"],
                "upper_anchor_steps":env["upper_contacts"],
                "lower_anchor_steps":env["lower_contacts"],
                "early_gap_mean_steps_5_15":env["early_gap"],
                "late_gap_mean_steps_45_55":env["late_gap"],
                "late_over_early":ratio,
                "late_gap_smaller":env["late_gap_smaller"]})
        fig.suptitle(f"{family}: empirical peak/trough Gradient L2 envelopes, actual 0–60 traces")
        fig.tight_layout()
        figures[f"34_gradient_L2_peak_trough_{family}.svg"]=_save(fig)
    # Dedicated comparison: rows = six noisy-feature cardinalities;
    # columns = policy families. Each empirical envelope encloses its group
    # MEAN (not all members); individual trajectories remain faintly visible.
    big,panels=plt.subplots(6,2,figsize=(13.5,21),sharex=True,sharey="row")
    for k in range(6):
        for col,family in enumerate(("shared","class")):
            ax=panels[k,col]
            selected=gradient_series(traces,family,k)
            mean=np.mean([values for _,values in selected],axis=0)
            env=group_fits[(family,k)]
            for _,values in selected:
                ax.plot(T,values,color=COLORS[k],alpha=.20,lw=.75)
            ax.plot(T,mean,color=COLORS[k],lw=1.65,label="Mean Gradient L2")
            ax.plot(T,env["upper"],ls="--",lw=1.4,color="#a16207",
                    label="Empirical upper")
            ax.plot(T,env["lower"],ls="--",lw=1.4,color="#0e7490",
                    label="Empirical lower")
            ax.fill_between(T,env["lower"],env["upper"],color="gray",alpha=.09)
            if env["peaks"]:
                ax.scatter(env["peaks"],mean[env["peaks"]],
                           color="#a16207",s=10,zorder=4)
            if env["valleys"]:
                ax.scatter(env["valleys"],mean[env["valleys"]],
                           color="#0e7490",s=10,zorder=4)
            ratio=env["late_over_early"]
            note=(f"late/early={ratio:.3f}" if ratio is not None else
                  "no measurable gap")
            ax.set_title(rf"{family} — $|A_N|={k}$  (n={len(selected)}; {note})",
                         fontsize=9)
            ax.set_xlim(0,60);ax.set_xticks([0,10,20,30,40,50,60])
            ax.ticklabel_format(axis="y",style="sci",scilimits=(-2,2))
            ax.grid(alpha=.14)
            if col==0:ax.set_ylabel("Gradient L2")
            if k==5:ax.set_xlabel("Adam update (0–60)")
            if k==0 and col==0:ax.legend(fontsize=7,loc="best",ncol=2)
    big.suptitle("Gradient L2 — empirical peak/trough envelopes | "
                 "rows $|A_N|=0,1,2,3,4,5$; columns shared and class",
                 fontsize=12,y=.996)
    big.tight_layout(rect=[0,0,.995,.988],h_pad=1.5,w_pad=1.0)
    figures["36_gradient_L2_peak_trough_6x2.svg"]=_save(big)
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    for ax,family in zip(axes,("shared","class")):
        for k in range(6):
            env=group_fits[(family,k)]
            gap=env["gap"]
            ax.plot(T,gap,color=COLORS[k],lw=1.6,label=f"k={k}")
        ax.set(xlabel="Adam update 0–60",ylabel="Peak-to-trough envelope gap",
               title=family,xlim=(0,60))
        ax.grid(alpha=.18);ax.legend(ncol=3,fontsize=7)
    fig.suptitle("Empirical Gradient L2 envelope width, no decay imposed")
    fig.tight_layout()
    figures["35_gradient_L2_envelope_width.svg"]=_save(fig)
    return figures,groups,runs,peaks_rows

def self_test():
    x=np.arange(61,dtype=float)
    oscillator=0.06+np.exp(-.045*x)*(.6+.12*np.cos(2*np.pi*x/8))
    env=empirical_envelope(oscillator)
    assert np.all(env["upper"]>=oscillator-1e-12)
    assert np.all(env["lower"]<=oscillator+1e-12)
    assert env["n_peaks"]>=5 and env["n_valleys"]>=5
    flat=np.full(61,.2)
    p=empirical_envelope(flat)
    assert np.allclose(p["gap"],0)
    records=peak_amplitudes(oscillator)
    fit=regression(records)
    assert fit["n"]>=3 and fit["exp_rate"]>0
    # A growing oscillation must not be automatically declared convergent.
    growing=.4+(1+.018*x)*.06*np.cos(2*np.pi*x/8)
    new=empirical_envelope(growing)
    assert not new["late_gap_smaller"]
    return True

if __name__=="__main__":
    self_test()
    print("E14_EMPIRICAL_GRADIENT_ENVELOPE_SELFTEST_PASS")
