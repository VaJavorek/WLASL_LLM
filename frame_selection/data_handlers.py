import decord
import numpy as np
from decord import VideoReader, cpu
import cv2
from PIL import Image
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

# give required_fps xor required_frames in parameters
def get_equally_distributed_frames(frames, avg_fps, required_fps=None, required_frames=None):
    if (required_fps is not None and required_frames is not None) or (required_fps is None and required_frames is None):
        print('Method reqires variable: required_fps xor required_frames')
        return
    # vr = decord.VideoReader(test_videos[vid_idx]['video_path'])
    # gloss = test_videos[vid_idx]['ground_truth_gloss']
    # fps = test_videos[vid_idx]['fps']
    if required_fps is not None:
        if required_fps > avg_fps:
            return frames
        required_frames = int(len(frames)/avg_fps*required_fps)
    if required_frames>len(frames):
        return frames
    reqired_idxs = np.linspace(0, len(frames)-1, required_frames).astype(np.int32)
    return frames[reqired_idxs]

def get_frames_of_local_mins(frames, movement_graph):
    peaks, _ = find_peaks(movement_graph, distance=2, prominence=0.00001)
    return frames[peaks]
    

def energy_fom_flow(vr):
    prev_gray = None
    flow_magnitudes = []
    kinetic_energy = []
    
    for i in range(len(vr)):
        # Convert Decord NDArray -> NumPy uint8
        frame = vr[i].asnumpy().astype(np.uint8)
        frame = cv2.resize(frame, (124, 124), interpolation=cv2.INTER_AREA)
        
        # Convert RGB -> grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.uint8)
    
        if prev_gray is not None:
            # Ensure prev_gray is also uint8 numpy
            prev_gray = prev_gray.astype(np.uint8)
    
            # Farnebäck optical flow
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
    
            # Compute magnitude of flow
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            flow_magnitudes.append(np.mean(mag))
    
            KE = 0.5 * (mag ** 2)
    
            # Total kinetic energy across frame
            total_KE = np.sum(KE)
            kinetic_energy.append(total_KE)
    
        prev_gray = gray
    
    
    # flow_magnitudes_normalized = flow_magnitudes/np.linalg.norm(flow_magnitudes)
    # kinetic_energy_normalized = kinetic_energy/np.linalg.norm(kinetic_energy)
    return np.array(flow_magnitudes), np.array(kinetic_energy)

# looks at movement in the video and cuts beggining and ending where signer has hands down
def get_adaptive_frames(vr, required_fps=None, required_frames=None, max_peak_valley_dist=0.5):
    if (required_fps is not None and required_frames is not None) or (required_fps is None and required_frames is None):
        print('Method reqires variable: required_fps xor required_frames')
        return
    frames = [Image.fromarray(frame.asnumpy()) for frame in vr]  # list of PIL images
    frames = np.stack(frames)
    flow_magnitudes, kinetic_energy = energy_fom_flow(vr)
    smoothed = gaussian_filter1d(flow_magnitudes, sigma=3)
    peaks, properties = find_peaks(smoothed, distance=2, prominence=0.1)
    avg_fps = vr.get_avg_fps()
#     frames = get_frames_of_local_mins(frames, flow_magnitudes)
    if len(peaks)<2:
        if required_fps is not None:
            frames = get_equally_distributed_frames(frames, avg_fps, required_fps=required_fps, required_frames=None)
        if required_frames is not None:
            frames = get_equally_distributed_frames(frames, avg_fps, required_fps=None, required_frames=required_frames)
        return frames
    # elif len(peaks)>2:
    #     top_two_indices = peaks[np.argsort(flow_magnitudes[peaks])[-2:]]
    #     top_two_indices.sort()
    #     peaks = top_two_indices
    if len(peaks)>=2:
        valleys, properties = find_peaks(-smoothed, distance=2, prominence=0.001)
        mask = (peaks[0] < valleys) & (valleys < peaks[1])
        valleys = valleys[mask]
        if len(valleys) >= 2:
            #valley needs to be close to peak
            if min(valleys)-min(peaks) < avg_fps*max_peak_valley_dist:
                low_bound = int((min(valleys)+min(peaks))/2)
            else:
                low_bound = min(peaks)
            if max(peaks)-max(valleys) < avg_fps*max_peak_valley_dist:
                high_bound = int((max(valleys)+max(peaks))/2)
            else:
                high_bound = max(peaks)
        elif len(valleys) == 1:
            if max(peaks)-valleys[0] < avg_fps*max_peak_valley_dist:
                high_bound = int((max(valleys)+max(peaks))/2)
                low_bound = min(peaks)
            elif valleys[0]-min(peaks) < avg_fps*max_peak_valley_dist:
                low_bound = int((min(valleys)+min(peaks))/2)
                high_bound = max(peaks)
            else:
                low_bound = min(peaks)
                high_bound = max(peaks)
        else:
            low_bound = min(peaks)
            high_bound = max(peaks)
        # print("low bound: "+ str(low_bound))
        # print("high bound: "+ str(high_bound))
        if low_bound < len(frames)/2 and high_bound > len(frames)/2:
            frames = frames[low_bound:high_bound]
        if required_fps is not None:
            frames = get_equally_distributed_frames(frames, avg_fps, required_fps=required_fps, required_frames=None)
        if required_frames is not None:
            frames = get_equally_distributed_frames(frames, avg_fps, required_fps=None, required_frames=required_frames)
        
    return frames