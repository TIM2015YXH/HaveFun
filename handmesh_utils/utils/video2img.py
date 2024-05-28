import os
import cv2
import numpy as np

video_path = '/Users/chenxingyu/Downloads/video/IMG_2202.mov'
save_path = video_path.replace('video', 'frame')
if not os.path.exists(save_path):
    os.makedirs(save_path)
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise Exception('Not exist video {}'.format(video_path))
frame_amount = cap.get(cv2.CAP_PROP_FRAME_COUNT)

frame_num = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # get updated location of objects in subsequent frames
    frame = np.rot90(frame, axes=(1,0))
    frame = cv2.resize(frame, (frame.shape[1]//2, frame.shape[0]//2))

    print(str(frame_num) + '/' + str(frame_amount))
    cv2.imwrite(os.path.join(save_path, str(frame_num).zfill(5) + '.png'), frame)
    frame_num += 1
