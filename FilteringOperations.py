import cv2
import numpy as np
from matplotlib import pyplot as plt

# Image Enhancement and Filtering

size = 5
filter_ = np.ones([size,size], dtype=np.float32) / (size*size) # Filter 1 (Averaging Filter)

img = cv2.imread("horoz.jpg", 1)[::,::,::-1]

convolved = cv2.filter2D(img, -1, filter_)

cv2.imwrite('horozAveraging.jpg', convolved)

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(convolved),plt.title('Averaging Filter')
plt.xticks([]), plt.yticks([])
plt.show()

#**************************************************************#

img = cv2.imread('sudoku.jpg',0)

laplacian = cv2.Laplacian(img,cv2.CV_64F) # Filter 2 (Laplacian Filter)
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5) # Filter 3 (Sobel X Filter)
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5) # Filter 4 (Sobel Y Filter)

cv2.imwrite('sudokuLaplacian.jpg', laplacian)
cv2.imwrite('sudokuSobelX.jpg', sobelx)
cv2.imwrite('sudokuSobelY.jpg', sobely)

plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
plt.show()

#**************************************************************#

img = cv2.imread('cat.jpg',0)

# Output dtype = cv2.CV_8U
sobelx8u = cv2.Sobel(img,cv2.CV_8U,1,0,ksize=5) # Filter 5

# Output dtype = cv2.CV_64F. Then take its absolute and convert to cv2.CV_8U
sobelx64f = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
abs_sobel64f = np.absolute(sobelx64f)
sobel_8u = np.uint8(abs_sobel64f) # Filter 6

cv2.imwrite('catSobel8U.jpg', sobelx8u)
cv2.imwrite('catSobel64F.jpg', sobelx64f)

plt.subplot(1,3,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,2),plt.imshow(sobelx8u,cmap = 'gray')
plt.title('Sobel CV_8U'), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,3),plt.imshow(sobelx64f,cmap = 'gray')
plt.title('Sobel abs(CV_64F)'), plt.xticks([]), plt.yticks([])

plt.show()

#**************************************************************#

image = cv2.imread('mertgulbahce.jpeg') # reads the image
image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) # convert to HSV
figure_size = 9 # the dimension of the x and y axis of the kernal.
new_image = cv2.blur(image,(figure_size, figure_size)) # Filter 7 (Gaussian Filter)

cv2.imwrite('mertgulbahceGaussian.jpg', new_image)

plt.figure(figsize=(11,6))
plt.subplot(121), plt.imshow(cv2.cvtColor(image, cv2.COLOR_HSV2RGB)),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_HSV2RGB)),plt.title('Gaussian Filter')
plt.xticks([]), plt.yticks([])
plt.show()

#**************************************************************#

img = cv2.imread('medianfilter.jpg')
median = cv2.medianBlur(img,5) # Filter 8 (Median Filter)

cv2.imwrite('medianFilterSaving.jpg', median)

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(median),plt.title('Median')
plt.xticks([]), plt.yticks([])
plt.show()

#**************************************************************#

img = cv2.imread('bilateral.png')
blur = cv2.bilateralFilter(img,9,75,75) # Filter 9 (Bilateral Filtering)

cv2.imwrite('bilateralSaving.png', blur)

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blur),plt.title('Bilateral Filtering')
plt.xticks([]), plt.yticks([])
plt.show()

#**************************************************************#

image = cv2.imread('atam.jpg')

kernel_7x7 = np.ones((7, 7), np.float32) / 49 # Filter 9 (Kernel 7x7 Blurred)
blurred = cv2.filter2D(image, -1, kernel_7x7)
kernel_12x12 = np.ones((12, 12), np.float32) / 144 # Filter 10 (Kernel 12x12 Blurred)
blurred2 = cv2.filter2D(image, -1, kernel_12x12)

cv2.imwrite('kernel7x7.jpg', kernel_7x7)
cv2.imwrite('kernel12x12.jpg', kernel_12x12)

plt.figure(figsize=(11, 6))
plt.subplot(131), plt.imshow(image), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(blurred), plt.title('Kernel 7x7 Blurred')
plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(blurred2), plt.title('Kernel 12x12 Blurred')
plt.xticks([]), plt.yticks([])
plt.show()