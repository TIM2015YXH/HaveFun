import cv2

data = 'data/hand_font.png'

img = cv2.imread(data, cv2.IMREAD_UNCHANGED)

img = img[400:-400, 400:-400]

img = cv2.resize(img, (800, 800), interpolation=cv2.INTER_AREA)

cv2.imwrite('data/hand_font800.png', img)

pass
