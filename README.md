# I3D_Feature_Extraction_x3d_vidoes
This repo contains code to extract I3D features with x3d_m backbone given a folder of videos

This code can be used for the below paper. Use at your own risk since this is still untested.
* [Weakly-supervised Video Anomaly Detection with Robust Temporal Feature Magnitude Learning](https://arxiv.org/pdf/2101.10030.pdf)

---




## Credits
The main resnet code and others is collected from the following repositories. 

* [I3D_Feature_Extraction_resnet] (https://github.com/GowthamGottimukkala/I3D_Feature_Extraction_resnet.git)
* [I3D_Feature_Extraction_resnet_50](https://github.com/Guechmed/I3D_Feature_Extraction_resnet_50)

  I updated the code to use the x3d_m as backbone instead of resnet50

## Overview
This code takes a folder of videos as input and for each video it saves ```I3D``` feature numpy file of dimension ```1*n/16*2048``` where n is the no.of frames in the video

## Usage
### Setup
 install the requirements 
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
