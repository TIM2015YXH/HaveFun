import vctoolkit as vc
import os
import predict_images_wo_annot as original_wrapper
import numpy as np

# You need to set the following variables manually.
# input video
video_path = '/Users/chenxingyu/Datasets/hand_test/video/wrist_test/IMG_1308.MOV'
# a folder that stores the video frames
frames_dir = video_path.split('.')[0] # '/Users/chenxingyu/Datasets/hand_test/video/wrist_test/IMG_'
# path to save the bounding box results
bbox_save_path = video_path.replace('video', 'det') + '/bbox.pkl' # '/Users/chenxingyu/Datasets/hand_test/det/wrist_test/IMG_5108.MOV/bbox.pkl'
model_path = '/Users/chenxingyu/Datasets/models/model_best.pth'

if not os.path.exists(frames_dir):
    os.makedirs(frames_dir, exist_ok=True)
    video = vc.VideoReader(video_path)
    print(f'{video.n_frames} frames to process...')
    for frame_idx in vc.progress_bar(range(video.n_frames), 'video frames'):
        frame = video.next_frame()
        if frame is None:
            break
        if frame.shape[0] < frame.shape[1]:
            frame = np.rot90(frame, axes=(1, 0))
        vc.save(os.path.join(frames_dir, f'{frame_idx:06d}.jpg'), frame)

file_list = []
for file_name in os.listdir(frames_dir):
    if os.path.splitext(file_name)[1][1:].lower() in ['jpg', 'png', 'jpeg']:
        file_list.append(os.path.join(frames_dir, file_name))
file_list = sorted(file_list)
# file_list = file_list[:10]
detections = original_wrapper.main_wrapper(file_list, model_path)
os.makedirs(os.path.dirname(bbox_save_path), exist_ok=True)
vc.save(bbox_save_path, detections)

