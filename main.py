from pathlib import Path
import shutil
import argparse
import numpy as np
import time
import ffmpeg
import torch
from extract_features import run_x3d, get_x3d_model
import os

def generate(datasetpath, outputpath, frequency, batch_size, sample_mode):
    Path(outputpath).mkdir(parents=True, exist_ok=True)
    temppath = os.path.join(outputpath, "temp")
    videos = list(Path(datasetpath).glob('**/*.mp4'))
    
    print("Loading X3D model...")
    model = get_x3d_model()

    for video in videos:
        videoname = video.stem
        rel_out_dir = Path(outputpath) / os.path.relpath(video.parent, datasetpath)
        rel_out_dir.mkdir(parents=True, exist_ok=True)
        feature_file = rel_out_dir / f"{videoname}_x3d.npy"
        
        if feature_file.exists():
            print(f"Skipping {videoname} (features already exist)")
            continue

        print(f"\nProcessing {videoname}...")
        start_time = time.time()

        # Recreate temp directory
        if Path(temppath).exists():
            shutil.rmtree(temppath)
        Path(temppath).mkdir(parents=True)

        try:
            print("Extracting frames...")
            (
                ffmpeg.input(str(video))
                .output(f"{temppath}/%04d.jpg", start_number=0)
                .global_args("-loglevel", "error")
                .run()
            )
            
            print("Extracting features...")
            features = run_x3d(model, frequency, temppath, batch_size, sample_mode)
            print(f"Features shape: {features.shape}")

            np.save(feature_file, features)
            
        except Exception as e:
            print(f"Error processing {videoname}: {str(e)}")
        finally:
            shutil.rmtree(temppath, ignore_errors=True)
            print(f"Completed in {time.time() - start_time:.2f} seconds")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasetpath', type=str, required=True, help="Path to UCSDped2 directory.")
    parser.add_argument('--outputpath', type=str, default="output", help="Path to save extracted features.")
    parser.add_argument('--frequency', type=int, default=16, help="Sampling frequency for frames.")
    parser.add_argument('--batch_size', type=int, default=20, help="Batch size for feature extraction.")
    parser.add_argument('--sample_mode', type=str, default="oversample", choices=["oversample", "center_crop"], help="Sampling mode.")
    args = parser.parse_args()

    generate(
        args.datasetpath,
        args.outputpath,
        args.frequency,
        args.batch_size,
        args.sample_mode
    )

    