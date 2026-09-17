import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROI_ROOT=Path(__file__).resolve().parent
GT=ROI_ROOT/"rgb_ground_truth.csv"
PROP=ROI_ROOT/"thermal_roi_results"/"roi_proposals.csv"
OUT=ROI_ROOT/"roi_margin_evaluation"; OUT.mkdir(exist_ok=True)
W,H=640,480
MARGINS=[0,20,40,60,80,100,120]

def read(path):
    with open(path,encoding="utf-8",newline="") as f:
        return {str(int(r["session"])):r for r in csv.DictReader(f)}

def area(b):
    return max(0,b[2]-b[0])*max(0,b[3]-b[1])

def inter(a,b):
    return max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))

def expand(b,m):
    return (max(0,b[0]-m),max(0,b[1]-m),min(W,b[2]+m),min(H,b[3]+m))

g,p=read(GT),read(PROP)
sessions=sorted(set(g)&set(p),key=int)
details=[]; summary=[]

for m in MARGINS:
    cs=[]; ars=[]; full=[]
    for s in sessions:
        if not p[s]["rgb_x1"]: continue
        gb=tuple(float(g[s][k]) for k in ("x1","y1","x2","y2"))
        pb=tuple(float(p[s][k]) for k in ("rgb_x1","rgb_y1","rgb_x2","rgb_y2"))
        rb=expand(pb,m)
        c=inter(gb,rb)/area(gb) if area(gb) else 0
        ar=area(rb)/(W*H)
        f=int(c>=.999)
        cs.append(c); ars.append(ar); full.append(f)
        details.append(dict(session=s,margin_px=m,touches_image_border=g[s]["touches_image_border"],
            visible_hand_containment=c,roi_area_ratio=ar,full_visible_gt_contained=f,
            gt_x1=gb[0],gt_y1=gb[1],gt_x2=gb[2],gt_y2=gb[3],
            roi_x1=rb[0],roi_y1=rb[1],roi_x2=rb[2],roi_y2=rb[3]))
    summary.append(dict(margin_px=m,n_sessions=len(cs),
        mean_visible_hand_containment=float(np.mean(cs)),
        median_visible_hand_containment=float(np.median(cs)),
        min_visible_hand_containment=float(np.min(cs)),
        full_visible_gt_rate=float(np.mean(full)),
        mean_roi_area_ratio=float(np.mean(ars)),
        median_roi_area_ratio=float(np.median(ars))))

def write(path,rows):
    with open(path,"w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

write(OUT/"margin_sweep_detail.csv",details)
write(OUT/"margin_sweep_summary.csv",summary)

x=[r["mean_roi_area_ratio"]*100 for r in summary]
y=[r["mean_visible_hand_containment"]*100 for r in summary]
plt.figure(figsize=(8,6)); plt.plot(x,y,marker="o")
for xx,yy,r in zip(x,y,summary): plt.annotate(f'{r["margin_px"]}px',(xx,yy),xytext=(5,5),textcoords="offset points")
plt.xlabel("Mean ROI area / full RGB frame (%)"); plt.ylabel("Mean visible-hand containment (%)")
plt.title("Thermal-guided ROI: containment vs. spatial area"); plt.grid(alpha=.3); plt.tight_layout()
plt.savefig(OUT/"containment_vs_roi_area.png",dpi=180); plt.close()

ms=[r["margin_px"] for r in summary]
plt.figure(figsize=(8,6))
plt.plot(ms,y,marker="o",label="Visible-hand containment")
plt.plot(ms,x,marker="o",label="ROI area ratio")
plt.xlabel("Safety margin (px)"); plt.ylabel("Percent (%)"); plt.title("Effect of ROI safety margin")
plt.legend(); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(OUT/"margin_effect.png",dpi=180); plt.close()

print("Margin | Mean containment | Min containment | Full visible GT | Mean ROI area")
for r in summary:
    print(f'{r["margin_px"]:>3}px  | {r["mean_visible_hand_containment"]*100:>6.1f}% | {r["min_visible_hand_containment"]*100:>6.1f}% | {r["full_visible_gt_rate"]*100:>6.1f}% | {r["mean_roi_area_ratio"]*100:>6.1f}%')
print("\nNOTE: containment is for the annotated VISIBLE hand region, not the complete physical hand.")
