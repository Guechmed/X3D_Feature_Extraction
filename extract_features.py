import os
import numpy as np
import torch
from natsort import natsorted
from PIL import Image

from pytorchvideo.models.hub import x3d_m


def get_x3d_model():
    model = x3d_m(pretrained=True)
    # Remove the final projection layer to get 2048-D features
    model.blocks[-1].proj = torch.nn.Identity()
    return model.cuda().eval()


def load_frame(frame_file):
    try:
        data = Image.open(frame_file)
        if data.mode != 'RGB':
            data = data.convert('RGB')
        data = data.resize((340, 256), Image.Resampling.LANCZOS)
        data = np.array(data, dtype=np.float32)
        data = (data * 2 / 255) - 1
        assert (data.max() <= 1.0) and (data.min() >= -1.0), "Frame values out of [-1,1] range"
        return np.ascontiguousarray(data)
    except Exception as e:
        print(f"Error loading {frame_file}: {str(e)}")
        return np.zeros((256, 340, 3), dtype=np.float32)

def load_rgb_batch(frames_dir, rgb_files, frame_indices):
    batch_data = np.zeros(frame_indices.shape + (256, 340, 3), dtype=np.float32)
    for i in range(frame_indices.shape[0]):
        for j in range(frame_indices.shape[1]):
            batch_data[i,j] = load_frame(os.path.join(frames_dir, rgb_files[frame_indices[i,j]]))
    return batch_data

def oversample_data(data):
    assert data.shape[2] == 256 and data.shape[3] == 340, f"Expected (...,256,340,...), got {data.shape}"
    crops = [
        data[:, :, :224, :224, :],    # Top-left
        data[:, :, :224, -224:, :],    # Top-right
        data[:, :, 16:240, 58:282, :], # Center
        data[:, :, -224:, :224, :],    # Bottom-left
        data[:, :, -224:, -224:, :],   # Bottom-right
    ]
    flipped_crops = [np.ascontiguousarray(c[:,:,:,::-1,:]) for c in crops]
    return [np.ascontiguousarray(c) for c in crops] + flipped_crops

def get_x3d_model():
    model = x3d_m(pretrained=True)
    model.blocks[-1].proj = torch.nn.Identity()  # Remove final linear layer
    return model.cuda().eval()

def run_x3d(model, frequency, frames_dir, batch_size, sample_mode):
    assert sample_mode in ['oversample', 'center_crop']
    chunk_size = 16  

    def forward_batch(b_data):
        b_data = np.ascontiguousarray(b_data).astype(np.float32)
        b_data = torch.from_numpy(b_data.transpose(0, 4, 1, 2, 3)).float().cuda()
        with torch.no_grad():
            features = model(b_data)
        return features.cpu().numpy()  

    rgb_files = natsorted([f for f in os.listdir(frames_dir) if f.endswith(('.jpg', '.tif'))])
    if len(rgb_files) < chunk_size:
        raise ValueError(f"Need at least {chunk_size} frames, got {len(rgb_files)}")

    frame_indices = [
        [int(i * frequency) + j for j in range(chunk_size)]
        for i in range((len(rgb_files) - chunk_size) // frequency + 1)
    ]
    frame_indices = np.array(frame_indices, dtype=np.int32)

    full_features = [[] for _ in range(10)] if sample_mode == 'oversample' else [[]]

    for batch_idx in range(0, len(frame_indices), batch_size):
        batch_indices = frame_indices[batch_idx:batch_idx+batch_size]
        batch_data = load_rgb_batch(frames_dir, rgb_files, batch_indices)
        
        if sample_mode == 'oversample':
            crops = oversample_data(batch_data)
            for i, crop in enumerate(crops):
                full_features[i].append(forward_batch(crop))
        else:
            center_crop = batch_data[:,:,16:240,58:282,:]
            full_features[0].append(forward_batch(center_crop))

    full_features = [np.concatenate(f) for f in full_features]
    return np.stack(full_features, axis=0).transpose(1, 0, 2)  # Shape: (N, 10, 2048)