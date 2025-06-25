import cv2
import numpy as np
import matplotlib.pyplot as plt

# Creating a synthetic image
my_img = np.zeros((100, 100), dtype=np.uint8)

my_img[20:50, 20:50] = 100
my_img[60:90, 60:90] = 200

# Adding gaussian noise
mean = 0
standard_dev = 20
gaussian_noise = np.random.normal(mean, standard_dev, my_img.shape).astype(np.uint8)

noisy_image = my_img.astype(np.int16) + gaussian_noise.astype(np.int16)
noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

# Otsu's Thresholding
_, otsus_threshold = cv2.threshold(noisy_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(my_img, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Noisy Image')
plt.imshow(noisy_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Otsu's Thresholded")
plt.imshow(otsus_threshold, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
