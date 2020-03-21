import cv2
import numpy as np
from matplotlib import pyplot as plt

# Log Transformation(1.Process)

img = cv2.imread('camera.jpg')

c = 255 / (np.log(1 + np.max(img)))
log_transformed = c * np.log(1 + img)

# Specify the data type.
log_transformed = np.array(log_transformed, dtype=np.uint8)

# Save the output.
cv2.imwrite('cameraLogTransformed.jpg', log_transformed)

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(log_transformed),plt.title('Log Transformed')
plt.xticks([]), plt.yticks([])
plt.show()


#********************************************************#

# Power-Law (Gamma) Transformation(2.Process)

img = cv2.imread('sample.jpg')

gamma = float(input("Gamma Transformation için Lütfen Değer Giriniz (0.3 ,1.4 vb.) :"))
gamma_corrected = np.array(255 * (img / 255) ** gamma, dtype='uint8')

cv2.imwrite('sampleGammaTransformed.jpg', gamma_corrected)

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(gamma_corrected),plt.title('Gamma Transformed')
plt.xticks([]), plt.yticks([])
plt.show()

#********************************************************#

# Piecewise-Linear Transformation(3.Process)

def pixelVal(pix, r1, s1, r2, s2):
    if (0 <= pix and pix <= r1):
        return (s1 / r1) * pix
    elif (r1 < pix and pix <= r2):
        return ((s2 - s1) / (r2 - r1)) * (pix - r1) + s1
    else:
        return ((255 - s2) / (255 - r2)) * (pix - r2) + s2

# Open the image.
img = cv2.imread('sample.jpg')

# Define parameters.
r1 = int(input("Linear Transformation için 'r1' Değerini Giriniz:"))
s1 = int(input("Linear Transformation için 's1' Değerini Giriniz:"))
r2 = int(input("Linear Transformation için 'r2' Değerini Giriniz:"))
s2 = int(input("Linear Transformation için 's2' Değerini Giriniz:"))

pixelVal_vec = np.vectorize(pixelVal)
contrast_stretched = pixelVal_vec(img, r1, s1, r2, s2)

cv2.imwrite('sampleLinearTransformed.jpg', contrast_stretched)

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(contrast_stretched),plt.title('Piecewise-Linear Transformed')
plt.xticks([]), plt.yticks([])
plt.show()

#********************************************************#

# Negative Transformation(4.Process)

img = cv2.imread('mammogram.jpg')
negative_Rate = int(input("Negative Transformed için Değer Giriniz (200,255 vb.) :"))
img2 = negative_Rate - img

cv2.imwrite('mammogramNegatived.jpg', img2)

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img2),plt.title('Negative Transformed')
plt.xticks([]), plt.yticks([])
plt.show()



