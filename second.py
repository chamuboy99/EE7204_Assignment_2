import cv2
import numpy as np
import matplotlib.pyplot as plt

def seed_growing(img, seeds, threshold):
    height, width = img.shape
    segmented = np.zeros_like(img)
    visited = np.zeros_like(img, dtype=bool)

    for seed in seeds:
        if visited[seed]:
            continue

        seed_val = int(img[seed])
        region_mean = seed_val
        region_size = 1

        stack = [seed]
        segmented[seed] = 255
        visited[seed] = True

        while stack:
            y, x = stack.pop()
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width and not visited[ny, nx]:
                        intensity = int(img[ny, nx])
                        if abs(intensity - region_mean) < threshold:
                            segmented[ny, nx] = 255
                            visited[ny, nx] = True
                            stack.append((ny, nx))

                            region_size += 1
                            region_mean = (region_mean * (region_size - 1) + intensity) / region_size

    return segmented

image_path = 'subject.jpg'
img_color = cv2.imread(image_path)
gray_img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

# Multiple seed points
seed_points = [(1450, 1400), (1600, 1500), (1800, 1350)]

# Threshold for similarity
threshold = 30

segmented = seed_growing(gray_img, seed_points, threshold)

plt.figure(figsize=(15, 6))

plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Grayscale Image with Seeds")
plt.imshow(gray_img, cmap='gray')
for sp in seed_points:
    plt.plot(sp[1], sp[0], 'ro')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Region Grown Segment")
plt.imshow(segmented, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
