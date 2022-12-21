# This is a direct implementation of A. Chambolle's algorithm for image denoising
# based on total variation minimization [1].
#
# Author: Pan Weiheng
# 
# Tested with:
#   Python      v3.6.5
#   SciPy       v1.1.0
#   NumPy       v1.15.1
#   Matplotlib  v2.2.3
#
# License: MIT license.
# 
# Reference:
#   [1] Antonin Chambolle, An Algorithm for Total Variation Minimization 
#       and Applications 20 (January 2004), no. 1/2, 89–97.

import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
from tv_denoise_algo import *
from ops import *

# <------ Example Usage ------>

img = scipy.misc.face() # Lena is missing from scipy now :(
#img = scipy.misc.face()[:,:,0] # For grayscale (red channel only)

# Crop image to square
(h,w) = img.shape[:2]
if img.ndim == 2:
    img = img[:,(w-h)//2:(w+h)//2] 
else:
    img = img[:,(w-h)//2:(w+h)//2,:]

# Plot image
plt.figure(figsize=(10,10))
plt.imshow(img, cmap='gray')
plt.show()

# Generate and plot noisy image
noisy_img = img.astype(np.float)
sigma = 30 # Standard deviation of noise
noisy_img += sigma * np.random.randn(*img.shape) # components of randn() are unit Gaussians
plt.figure(figsize=(10,10))
plt.imshow(noisy_img.astype(int), cmap='gray')
plt.show()

# Generate and plot denoised image
denoised_img = tv_improved_denoise_rgb(noisy_img, sigma=30, max_iter=50, epsilon=1e-5, log=True)
# denoised_img = tv_improved_denoise(noisy_img, sigma=30, max_iter=50, epsilon=1e-5, log=True) # For grayscale
plt.figure(figsize = (10,10))
plt.imshow(denoised_img.astype(int), cmap='gray')
plt.show()

# Plot difference image
diff_img = noisy_img - denoised_img
plt.figure(figsize = (10,10))
plt.imshow(diff_img.astype(int), cmap='gray')
plt.show()