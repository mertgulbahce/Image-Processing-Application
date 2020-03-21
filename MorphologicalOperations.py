import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# 1) Erosion
img = cv.imread('morphological.png')
kernel = np.ones((5,5),np.uint8)
erosion = cv.erode(img,kernel,iterations = 1)

cv.imwrite('morphologicalErosion.png', erosion)

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(erosion),plt.title('Erosion')
plt.xticks([]), plt.yticks([])
plt.show()

# 2) Dilation
img = cv.imread('morphological.png')
kernel = np.ones((5,5),np.uint8)
dilation = cv.dilate(img,kernel,iterations = 1)

cv.imwrite('morphologicalDilation.png', dilation)

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(dilation),plt.title('Dilation')
plt.xticks([]), plt.yticks([])
plt.show()

# 3) Opening
img = cv.imread('opening.png')
kernel = np.ones((5,5),np.uint8)
opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)

cv.imwrite('openingSave.png', opening)

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(opening),plt.title('Opening')
plt.xticks([]), plt.yticks([])
plt.show()

# 4) Closing
img = cv.imread('closing.png')
kernel = np.ones((5,5),np.uint8)
closing = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)

cv.imwrite('closingSave.png', closing)

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(closing),plt.title('Closing')
plt.xticks([]), plt.yticks([])
plt.show()

# 5) Gradient
img = cv.imread('morphological.png')
kernel = np.ones((5,5),np.uint8)
gradient = cv.morphologyEx(img, cv.MORPH_GRADIENT, kernel)

cv.imwrite('morphologicalGradient.png', gradient)

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(gradient),plt.title('Gradient')
plt.xticks([]), plt.yticks([])
plt.show()

# 6) Top Hat
img = cv.imread('morphological.png')
kernel = np.ones((9,9),np.uint8)
tophat = cv.morphologyEx(img, cv.MORPH_TOPHAT, kernel)

cv.imwrite('morphologicalTopHat.png', tophat)

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(tophat),plt.title('Top Hat')
plt.xticks([]), plt.yticks([])
plt.show()

# 7) Black Hat
img = cv.imread('morphological.png')
kernel = np.ones((9,9),np.uint8)
blackhat = cv.morphologyEx(img, cv.MORPH_BLACKHAT, kernel)

cv.imwrite('morphologicalBlackHat.png', blackhat)

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blackhat),plt.title('Black Hat')
plt.xticks([]), plt.yticks([])
plt.show()

# 8) Ellipse
img = cv.imread('morph_input.jpg')
kernel = np.ones((5,5),np.uint8)
ellipse = cv.morphologyEx(img, cv.MORPH_ELLIPSE, kernel)

cv.imwrite('morph_inputEllipse.jpg', ellipse)

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(ellipse),plt.title('Ellipse')
plt.xticks([]), plt.yticks([])
plt.show()

# 9) Cross
img = cv.imread('morph_input.jpg')
kernel = np.ones((5,5),np.uint8)
cross = cv.morphologyEx(img, cv.MORPH_CROSS, kernel)

cv.imwrite('morph_inputCross.jpg', cross)

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(cross),plt.title('Cross')
plt.xticks([]), plt.yticks([])
plt.show()

# 10) Rect
img = cv.imread('morph_input.jpg')
kernel = np.ones((5,5),np.uint8)
rect = cv.morphologyEx(img, cv.MORPH_RECT, kernel)

cv.imwrite('morph_inputRect.jpg', rect)

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(rect),plt.title('Rect')
plt.xticks([]), plt.yticks([])
plt.show()


