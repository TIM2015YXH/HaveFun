import cv2
import os
import imageio

# path = '/Users/chenxingyu/Documents/hand_mesh/lighthand/out/Kwai2D/cmrpng_reg2d_left/infe/IMG_5108.MOV/plot'
# path = '/mnt2/shared/research/3dgan_data/epigraf/video_frames/video_grid/006'
path = 'defeg3d/out/soft_mask4/infer/network-snapshot-001200/face'
imgs = sorted([f for f in os.listdir(path) if '.jpg' in f or '.png' in f]) # key=lambda x: int(x.split('.')[0]
print(len(imgs))
img = cv2.imread(os.path.join(path, imgs[0]))
video_size = (img.shape[1], img.shape[0])
sec = 10
fps = 10#len(imgs) / sec
# videoWriter = cv2.VideoWriter(path + '.avi',cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), fps, video_size)
                              # cv2.VideoWriter_fourcc('M', 'P', '4', 'V'), 20, video_size)
                              # cv2.VideoWriter_fourcc('X', '2', '6', '4'), 15, video_size)
                              # cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), 10, video_size)
video_out = imageio.get_writer('defeg3d/out/soft_mask4/infer/network-snapshot-001200/face.mp4', mode='I', fps=10, codec='libx264', bitrate='10M')

for i, name in enumerate(imgs):
    img = cv2.imread(os.path.join(path, name))[..., ::-1]
    # videoWriter.write(cv2.resize(img, video_size))
    video_out.append_data(img)

# videoWriter.release()
# cv2.destroyAllWindows()
video_out.close()
