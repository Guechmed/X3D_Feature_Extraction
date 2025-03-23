from pathlib import Path
import shutil
import argparse
import numpy as np
import time
import ffmpeg
import torch
from extract_features import run
from resnet import i3_res50
import os


def generate(datasetpath, outputpath, pretrainedpath, frequency, batch_size, sample_mode):
    Path(outputpath).mkdir(parents=True, exist_ok=True)
    temppath = os.path.join(outputpath, "temp")
    rootdir = Path(datasetpath)
    videos = [str(f) for f in rootdir.glob('**/*.mp4')]

    # Setup the model
    i3d = i3_res50(400, pretrainedpath)
    i3d.cuda()
    i3d.train(False)  # Set model to evaluation mode

    for video in videos:
        videoname = Path(video).stem  # Extract filename without extension
        startime = time.time()
        print(f"Generating for {video}")

        # Ensure temporary path exists
        Path(temppath).mkdir(parents=True, exist_ok=True)

        # Extract frames using ffmpeg
        ffmpeg.input(video).output(f'{temppath}/%d.jpg', start_number=0).global_args('-loglevel', 'quiet').run()
        print("Preprocessing done..")

        # Run feature extraction
        features = run(i3d, frequency, temppath, batch_size, sample_mode)

        # Ensure output directory exists
        videodir = os.path.join(outputpath, os.path.relpath(os.path.dirname(video), datasetpath))
        Path(videodir).mkdir(parents=True, exist_ok=True)

        # Save extracted features
        np.save(os.path.join(videodir, f"{videoname}.npy"), features)
        print("Obtained features of size: ", features.shape)

        shutil.rmtree(temppath)
        print(f"Done in {time.time() - startime:.2f} seconds.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasetpath', type=str, default="samplevideos/")
    parser.add_argument('--outputpath', type=str, default="output")
    parser.add_argument('--pretrainedpath', type=str, default="pretrained/i3d_r50_kinetics.pth")
    parser.add_argument('--frequency', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--sample_mode', type=str, default="oversample")
    args = parser.parse_args()
    generate(args.datasetpath, str(args.outputpath), args.pretrainedpath, args.frequency, args.batch_size, args.sample_mode)
