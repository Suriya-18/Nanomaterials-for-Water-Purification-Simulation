import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the polluted water image
image = cv2.imread("polluted_water.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

# Convert to grayscale for processing
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# Simulate detection of contaminants using thresholding
_, contaminants = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

# Apply a Gaussian blur to simulate nanoparticle purification effect
cleaned_image = cv2.GaussianBlur(image, (15, 15), 0)

# Simulate nanoparticle action by blending the cleaned image with original
alpha = 0.7  # Purification intensity
final_image = cv2.addWeighted(image, 1 - alpha, cleaned_image, alpha, 0)

# Plot the results
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(image)
ax[0].set_title("Original Polluted Water")
ax[0].axis("off")

ax[1].imshow(contaminants, cmap="gray")
ax[1].set_title("Detected Contaminants")
ax[1].axis("off")

ax[2].imshow(final_image)
ax[2].set_title("Simulated Purified Water")
ax[2].axis("off")

plt.show()
