import os, time
from datetime import datetime
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

TITLE = "WasteWatch AI"
REF_DIR = "data/reference_images"
CUR_DIR = "data/incoming_images"
os.makedirs(REF_DIR, exist_ok=True)
os.makedirs(CUR_DIR, exist_ok=True)

st.set_page_config(page_title=TITLE, page_icon="♻️", layout="wide")
st.markdown("""
<style>
.stApp{background:linear-gradient(135deg,#09111e,#0d1928 65%,#111d2b)}
[data-testid="stSidebar"]{background:#0b1422;border-right:1px solid #263449}
.block-container{padding-top:1.3rem;max-width:1600px}
h1,h2,h3{color:#f4f8ff!important}
.eyebrow{color:#78b4ff;text-transform:uppercase;letter-spacing:.15em;font-size:.72rem;font-weight:700}
.sub{color:#9aacc1}
.card{background:linear-gradient(145deg,#152236,#101a29);border:1px solid #293b52;border-radius:16px;padding:18px;min-height:105px}
.label{color:#9aacc1;font-size:.82rem}.value{font-size:1.6rem;font-weight:750;color:#f5f9ff}.note{font-size:.78rem;color:#7f95ad}
div[data-testid="stMetric"]{background:#111c2d;border:1px solid #26384e;border-radius:13px;padding:14px}
.stButton>button,.stDownloadButton>button{border-radius:9px;background:#1c4e89;border:1px solid #347fd4;color:white;font-weight:650}
</style>
""", unsafe_allow_html=True)

def features(im):
    rgb = np.asarray(im.convert("RGB"))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return {
        "brightness_mean": float(gray.mean()/255),
        "brightness_std": float(gray.std()/255),
        "saturation_mean": float(hsv[:,:,1].mean()/255),
        "hue_mean": float(hsv[:,:,0].mean()/179),
        "edge_density": float((cv2.Canny(gray,100,200)>0).mean()),
        "width": float(im.width), "height": float(im.height),
        "aspect_ratio": float(im.width/max(im.height,1))
    }

def psi(ref, cur, bins=10, eps=1e-6):
    ref=np.asarray(ref,dtype=float); cur=np.asarray(cur,dtype=float)
    ref=ref[np.isfinite(ref)]; cur=cur[np.isfinite(cur)]
    if len(ref)<2 or len(cur)<1: return np.nan
    edges=np.unique(np.quantile(ref,np.linspace(0,1,bins+1)))
    if len(edges)<3:
        lo,hi=float(ref.min()),float(ref.max())
        if lo==hi: lo-=1e-6; hi+=1e-6
        edges=np.linspace(lo,hi,bins+1)
    edges[0],edges[-1]=-np.inf,np.inf
    a,_=np.histogram(ref,bins=edges); b,_=np.histogram(cur,bins=edges)
    a=np.clip(a/max(a.sum(),1),eps,None); b=np.clip(b/max(b.sum(),1),eps,None)
    return float(np.sum((b-a)*np.log(b/a)))

def images_in(folder):
    result=[]
    for fn in sorted(os.listdir(folder)):
        if fn.lower().endswith((".jpg",".jpeg",".png",".bmp",".webp")):
            try:
                with Image.open(os.path.join(folder,fn)) as im: result.append((fn,im.convert("RGB").copy()))
            except Exception: pass
    return result

def batch_features(batch):
    rows=[]
    for name,im in batch:
        r=features(im); r["image"]=name; rows.append(r)
    return pd.DataFrame(rows)

def load_model(path):
    if YOLO is None: return None, "Install ultralytics to enable detection."
    if not os.path.isfile(path): return None, f"Model weights not found: {path}"
    try: return YOLO(path), None
    except Exception as e: return None, f"Could not load weights: {e}"

def infer(model, im, conf, iou, max_det):
    rgb=np.asarray(im.convert("RGB")); start=time.perf_counter()
    r=model.predict(source=rgb,conf=conf,iou=iou,max_det=max_det,verbose=False)[0]
    ms=(time.perf_counter()-start)*1000
    plot=cv2.cvtColor(r.plot(labels=True,conf=True,line_width=2),cv2.COLOR_BGR2RGB)
    names=model.names
    det=[]
    if r.boxes is not None:
        for b in r.boxes:
            cid=int(b.cls.item())
            name=names.get(cid,str(cid)) if isinstance(names,dict) else names[cid]
            det.append({"class":name,"confidence":float(b.conf.item())})
    return plot,det,ms

if "history" not in st.session_state: st.session_state.history=[]

with st.sidebar:
    st.markdown('<div class="eyebrow">AI Operations Platform</div>',unsafe_allow_html=True)
    st.markdown("## ♻️ WasteWatch AI")
    st.caption("Computer vision · Drift monitoring · Experiments")
    page=st.radio("NAVIGATION",["Live Monitoring","Analytics","Drift Monitoring","Experiments","Reports & Settings"])
    st.divider()
    weights=st.text_input("YOLO weights path","models/best.pt")
    conf=st.slider("Detection confidence",.1,.95,.5,.05)
    iou=st.slider("NMS IoU threshold",.1,.9,.45,.05)
    max_det=st.slider("Maximum detections",1,50,10)
model,model_error=load_model(weights)

st.markdown('<div class="eyebrow">Waste intelligence / model reliability</div>',unsafe_allow_html=True)
st.title("WasteWatch AI")
st.markdown('<div class="sub">Intelligent waste detection and data drift monitoring</div>',unsafe_allow_html=True)

if page=="Live Monitoring":
    st.subheader("Live production interface")
    left,right=st.columns([1.6,1],gap="large")
    with left:
        up=st.file_uploader("Upload a waste image",type=["png","jpg","jpeg","webp","bmp"],key="live")
        if up:
            im=Image.open(up).convert("RGB")
            st.image(im,caption=f"Input · {im.width} × {im.height}",use_container_width=True)
            if model is None: st.warning(model_error+" Drift analysis and experiments remain available.")
            elif st.button("Run waste detection",type="primary"):
                try:
                    plot,dets,ms=infer(model,im,conf,iou,max_det)
                    st.session_state.last_plot=plot; st.session_state.last_dets=dets
                    st.session_state.last_ms=ms; st.session_state.last_name=up.name
                    st.session_state.history.append({"timestamp":datetime.now().isoformat(timespec="seconds"),"image":up.name,
                        "detections":len(dets),"mean_confidence":float(np.mean([d["confidence"] for d in dets])) if dets else 0,"latency_ms":ms})
                except Exception as e: st.error(f"Inference failed: {e}")
        if "last_plot" in st.session_state: st.image(st.session_state.last_plot,caption="Detection result",use_container_width=True)
    with right:
        st.subheader("Detection overview")
        if "last_dets" in st.session_state:
            dets=st.session_state.last_dets
            st.metric("Detections",len(dets))
            st.metric("Mean confidence",f'{np.mean([d["confidence"] for d in dets]):.1%}' if dets else "—")
            st.metric("Measured inference latency",f'{st.session_state.last_ms:.1f} ms')
            if dets: st.dataframe(pd.Series([d["class"] for d in dets]).value_counts().rename_axis("Class").reset_index(name="Count"),hide_index=True,use_container_width=True)
            else: st.info("No detections met the confidence threshold.")
        else: st.info("Upload an image and run detection to populate live measurements.")
        st.markdown("#### Operational status")
        if model is None: st.warning("Detector unavailable — configure valid weights.")
        elif "last_dets" in st.session_state: st.success("Inference completed. This does not imply drift is absent.")
        else: st.info("Waiting for inference.")

elif page=="Analytics":
    st.subheader("Detection and runtime analytics")
    df=pd.DataFrame(st.session_state.history)
    if df.empty: st.info("Run detection on Live Monitoring first.")
    else:
        a,b,c=st.columns(3); a.metric("Images processed",len(df)); b.metric("Total detections",int(df.detections.sum())); c.metric("Mean latency",f'{df.latency_ms.mean():.1f} ms')
        st.line_chart(df.set_index("timestamp")[["latency_ms"]]); st.bar_chart(df.set_index("image")[["detections"]])
        st.dataframe(df,use_container_width=True,hide_index=True)
        st.download_button("Download session history CSV",df.to_csv(index=False).encode(),"wastewatch_history.csv","text/csv")

elif page=="Drift Monitoring":
    st.subheader("Reference vs incoming image features")
    st.write("PSI measures distribution changes in selected image-level features. It does not by itself prove model accuracy has declined.")
    a,b=st.columns(2)
    with a:
        ref_upload=st.file_uploader("Reference images (baseline)",type=["jpg","jpeg","png","bmp","webp"],accept_multiple_files=True,key="ref")
        if ref_upload and st.button("Save reference images"):
            for f in ref_upload:
                with open(os.path.join(REF_DIR,os.path.basename(f.name)),"wb") as out: out.write(f.getbuffer())
            st.success(f"Saved {len(ref_upload)} reference images.")
    with b:
        cur_upload=st.file_uploader("Incoming images (current batch)",type=["jpg","jpeg","png","bmp","webp"],accept_multiple_files=True,key="cur")
        if cur_upload and st.button("Save incoming images"):
            for f in cur_upload:
                with open(os.path.join(CUR_DIR,os.path.basename(f.name)),"wb") as out: out.write(f.getbuffer())
            st.success(f"Saved {len(cur_upload)} incoming images.")
    refs=images_in(REF_DIR); curs=images_in(CUR_DIR)
    st.caption(f"Saved reference images: {len(refs)} · Incoming images: {len(curs)}")
    if st.button("Calculate feature drift",type="primary"):
        if len(refs)<5 or len(curs)<5: st.error("For this prototype, upload at least 5 reference and 5 incoming images. Larger, representative batches are better.")
        else:
            rd=batch_features(refs); cd=batch_features(curs); rows=[]
            for col in [x for x in rd.columns if x!="image"]:
                val=psi(rd[col],cd[col])
                status="Low measured shift" if val<.1 else ("Moderate measured shift" if val<.25 else "Large measured shift")
                rows.append({"Feature":col,"PSI":val,"Status":status,"Reference mean":rd[col].mean(),"Incoming mean":cd[col].mean()})
            st.session_state.drift_df=pd.DataFrame(rows).sort_values("PSI",ascending=False)
            st.session_state.ref_df=rd; st.session_state.cur_df=cd
    if "drift_df" in st.session_state:
        dd=st.session_state.drift_df
        st.dataframe(dd,use_container_width=True,hide_index=True); st.bar_chart(dd.set_index("Feature")[["PSI"]])
        selected=st.selectbox("Feature distribution",dd.Feature.tolist())
        st.line_chart(pd.DataFrame({"Reference":st.session_state.ref_df[selected].reset_index(drop=True),"Incoming":st.session_state.cur_df[selected].reset_index(drop=True)}))
        st.download_button("Download drift report CSV",dd.to_csv(index=False).encode(),"wastewatch_drift.csv","text/csv")
        st.caption("Thresholds are indicative heuristics, not universal. Interpret results alongside sample sizes and image context.")

elif page=="Experiments":
    st.subheader("Controlled robustness experiments")
    up=st.file_uploader("Choose an experiment image",type=["jpg","jpeg","png","bmp","webp"],key="exp")
    if up:
        original=Image.open(up).convert("RGB")
        kind=st.selectbox("Transformation",["Brightness","Gaussian blur","Noise","Contrast"])
        strength=st.slider("Strength",.1,2.,1.,.1)
        if kind=="Brightness": changed=ImageEnhance.Brightness(original).enhance(strength)
        elif kind=="Contrast": changed=ImageEnhance.Contrast(original).enhance(strength)
        elif kind=="Gaussian blur": changed=original.filter(ImageFilter.GaussianBlur(strength*2))
        else:
            rng=np.random.default_rng(42); arr=np.asarray(original).astype(float)+rng.normal(0,strength*12,np.asarray(original).shape)
            changed=Image.fromarray(np.clip(arr,0,255).astype(np.uint8))
        x,y=st.columns(2); x.image(original,caption="Original",use_container_width=True); y.image(changed,caption=f"{kind} · {strength}",use_container_width=True)
        f1, f2=features(original),features(changed)
        comp=pd.DataFrame({"Feature":list(f1),"Original":list(f1.values()),"Transformed":list(f2.values())})
        comp["Absolute change"]=(comp.Transformed-comp.Original).abs()
        st.dataframe(comp,use_container_width=True,hide_index=True); st.bar_chart(comp.set_index("Feature")[["Absolute change"]])
        if model is not None and st.button("Compare YOLO detections"):
            with st.spinner("Running inference..."):
                try:
                    _,d1,t1=infer(model,original,conf,iou,max_det); _,d2,t2=infer(model,changed,conf,iou,max_det)
                    st.dataframe(pd.DataFrame([{"Image":"Original","Detections":len(d1),"Mean confidence":np.mean([d["confidence"] for d in d1]) if d1 else 0,"Latency ms":t1},{"Image":"Transformed","Detections":len(d2),"Mean confidence":np.mean([d["confidence"] for d in d2]) if d2 else 0,"Latency ms":t2}]),hide_index=True,use_container_width=True)
                    st.caption("This compares predictions, not accuracy. Accuracy requires labelled ground truth.")
                except Exception as e: st.error(str(e))

else:
    st.subheader("Reports & settings")
    if model is None: st.warning(model_error)
    else:
        st.success(f"Loaded model from `{weights}`")
        names=model.names
        st.dataframe(pd.DataFrame([{"Class ID":k,"Class name":v} for k,v in names.items()]) if isinstance(names,dict) else pd.DataFrame({"Class name":names}),hide_index=True,use_container_width=True)
    st.markdown("#### Existing training artifacts")
    for label,path in [("Training results","runs/detect/train-5/results.png"),("Confusion matrix","runs/detect/train-5/confusion_matrix.png"),("Training metrics CSV","runs/detect/train-5/results.csv"),("Training arguments","runs/detect/train-5/args.yaml")]:
        if os.path.exists(path):
            st.write(f"✅ {label}: `{path}`")
            if path.endswith(".png"): st.image(path,caption=label,use_container_width=True)
            elif path.endswith(".csv"):
                try: st.dataframe(pd.read_csv(path),use_container_width=True)
                except Exception: pass
        else: st.write(f"— {label}: not found")
    st.info("Model accuracy and mAP require labelled validation data. Drift, confidence and accuracy are different measurements.")
    if st.button("Clear session history"): st.session_state.history=[]; st.success("Session history cleared; saved image batches were kept.")

st.divider()
st.caption("WasteWatch AI · Academic prototype · Validate measurements before reporting results.")
