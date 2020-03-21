import cv2
import numpy as np
from matplotlib import pyplot as plt

# RESIZING

img = cv2.imread('resizing.jpg')

r = 250.0 / img.shape[1]
dim = (500, int(img.shape[0] * r))
# perform the actual resizing of the image and show it
resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

cv2.imwrite('resizingSave.jpg', resized)

plt.subplot(121), plt.imshow(img), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(resized), plt.title('Resized')
plt.xticks([]), plt.yticks([])
plt.show()

# ROTATION

img = cv2.imread('egg.jpg')
num_rows, num_cols = img.shape[:2]
rotation_matrix = cv2.getRotationMatrix2D((num_cols / 2, num_rows / 2), 90, 1)
img_rotation = cv2.warpAffine(img, rotation_matrix, (num_cols, num_rows))

cv2.imwrite('eggRotated.jpg', img_rotation)

plt.subplot(121), plt.imshow(img), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(img_rotation), plt.title('Rotated')
plt.xticks([]), plt.yticks([])
plt.show()

# CROPPING

img = cv2.imread('cropping.png')
cropped = img[65:335, 75:375]

cv2.imwrite('croppingSave.png', cropped)

plt.subplot(121), plt.imshow(img), plt.title('Original')
plt.subplot(122), plt.imshow(cropped), plt.title('Cropped')
plt.show()

# FLIPPING

img = cv2.imread('flipping.jpg')
flipping_img_along_X = cv2.flip(img, 0)
flipping_img_along_Y = cv2.flip(img, 1)
flipping_img_along_XY = cv2.flip(img, -1)

cv2.imwrite('flippingAlongX.jpg', flipping_img_along_X)
cv2.imwrite('flippingAlongY.jpg', flipping_img_along_Y)
cv2.imwrite('flippingAlongXY.jpg', flipping_img_along_XY)

plt.subplot(2, 2, 1), plt.imshow(img, cmap='gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 2), plt.imshow(flipping_img_along_X, cmap='gray')
plt.title('X-Flipped'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 3), plt.imshow(flipping_img_along_Y, cmap='gray')
plt.title('Y-Flipped'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 4), plt.imshow(flipping_img_along_XY, cmap='gray')
plt.title('XY-Flipped'), plt.xticks([]), plt.yticks([])
plt.show()

# PERSPECTIVE TRANSFORMATION

img = cv2.imread('desen.jpg')
rows, cols, ch = img.shape
pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
M = cv2.getPerspectiveTransform(pts1, pts2)
dst = cv2.warpPerspective(img, M, (300, 300))

cv2.imwrite('desenPerspective.jpg', dst)

plt.subplot(121), plt.imshow(img), plt.title('Original')
plt.subplot(122), plt.imshow(dst), plt.title('Perspective Transformed')
plt.show()

# AFFINE TRANSFORMATION

img = cv2.imread('desen.jpg')
rows, cols, ch = img.shape
pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
M = cv2.getAffineTransform(pts1, pts2)
dst = cv2.warpAffine(img, M, (cols, rows))

cv2.imwrite('desenAffine.jpg', dst)

plt.subplot(121), plt.imshow(img), plt.title('Original')
plt.subplot(122), plt.imshow(dst), plt.title('Affine Transformed')
plt.show()
