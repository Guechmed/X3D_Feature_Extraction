# X3D_Feature_Extraction

This repository contains code to extract **video features using the `x3d_m` backbone**, given a folder of input videos.

> ⚠️ This code is intended for research purposes .

It can be used in the context of the following paper:

> [**Weakly-supervised Video Anomaly Detection with Robust Temporal Feature Magnitude Learning**](https://arxiv.org/pdf/2101.10030.pdf)

---

## 📦 Overview

This code processes each video in a folder and saves a corresponding NumPy file containing its extracted features.

- **Input**: A folder of videos or subfolders of videos  
- **Output**: `.npy` feature file for each video  
- **Feature shape**: `(n/16, x, 2048)` where `n` is the number of frames

---

## 🔗 Credits

This project is based on the following repositories:

- [I3D Feature Extraction with ResNet](https://github.com/GowthamGottimukkala/I3D_Feature_Extraction_resnet.git)  
- [I3D Feature Extraction with ResNet-50](https://github.com/Guechmed/I3D_Feature_Extraction_resnet_50)

> ✅ The backbone has been replaced with **`x3d_m`** in this version.

---

## ⚙️ Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

### Parameters
<pre>
--datasetpath:       folder of input videos (contains videos or subdirectories of videos)
--outputpath:        folder of extracted features
--frequency:         how many frames between adjacent snippet
--batch_size:        batch size for snippets
</pre>

### Run
```bash
python main.py --datasetpath=dataset_path/ --outputpath=output
```
put the video in a folder and add the path to that folder in the datasetpath
