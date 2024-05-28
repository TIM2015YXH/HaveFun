import numpy as np
import cv2

name = 'hand_font_crop'
save_name = 'hand_font_r'

img = cv2.imread(f'data/{name}_rgba.png', cv2.IMREAD_UNCHANGED)
img = img[:,::-1]
# img = np.rot90(img, -1, axes=(1, 0))
cv2.imwrite(f'data/{save_name}_rgba.png', img)

img = cv2.imread(f'data/{name}_depth.png', cv2.IMREAD_UNCHANGED)
if img.shape[-1] == 3:
    img = img[..., 0]
img = img[:,::-1]
# img = np.rot90(img, -1, axes=(1, 0))
cv2.imwrite(f'data/{save_name}_depth.png', img)

img = cv2.imread(f'data/{name}_normal.png', cv2.IMREAD_UNCHANGED)
img = img[:,::-1]
# img = np.rot90(img, -1, axes=(1, 0)) 
cv2.imwrite(f'data/{save_name}_normal.png', img)
