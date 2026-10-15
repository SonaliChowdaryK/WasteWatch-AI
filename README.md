# WasteWatch AI

Streamlit prototype for YOLO waste detection, image-feature drift monitoring, and controlled robustness experiments.

## Setup (Windows PowerShell)
Open a terminal in this folder and run:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Add your model
Copy your verified trained weights to `models/best.pt`. Your existing checkpoint may be at `runs/detect/train-5/weights/best.pt`. Or change the weights path in the app sidebar.

## Drift monitoring
Open Drift Monitoring, upload and save at least five reference images and five incoming images, then calculate feature drift. The app calculates PSI on image-level features such as brightness, saturation, edge density, and dimensions. PSI indicates distribution change in these selected features; it does not prove model accuracy declined. Use representative batches and interpret thresholds carefully.

## Experiments
Open Experiments and apply brightness, blur, noise, or contrast changes. Compare feature changes and, when weights are available, YOLO predictions. This is not an accuracy evaluation unless ground-truth labels are used.

## Notes
No accuracy or drift values are fabricated. Accuracy, precision, recall and mAP require labelled validation data. Latency varies with hardware and image dimensions. Validate results before reporting them.
