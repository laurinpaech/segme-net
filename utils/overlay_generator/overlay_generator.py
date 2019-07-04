import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os as os
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap

"""
Overlay generator:

- generates an overlay of the predictions on the training image with a custom color map.
- Make sure that the original images are inside normal_img/ directory
- Make sure that predictions are inside submit_img/ directory
- Set the settable parameters image_nr and img_nr_overlay to the name of the original image and predictions used  and run

As an example we use 123.png as the original image in normal_img
In submit_img we have 2 different overlays, 123.png and 123_1_0.png, with and without overlay respectively.
"""

base_dir_path = ""
normal_img_path = os.path.join(base_dir_path, "normal_img/")
submission_img_path = os.path.join(base_dir_path, "submit_img/")

#########################
# Settable Parameter
#########################
# Add images and change these paths and run this skript
image_nr = '123'  # original image
image_nr_overlay = '123_1_0'  # Example without cutoff
alpha = 0.55

## Images path
img_path = os.path.join(normal_img_path, image_nr + ".png")
overlay_img_path = os.path.join(submission_img_path, image_nr_overlay + ".png")

## Read images
normal_img = mpimg.imread(img_path)
overlay_img = mpimg.imread(overlay_img_path)

## Resize pre_ensembled
overlay_img = Image.fromarray(np.float64(overlay_img))
overlay_img = overlay_img.resize((608, 608), Image.ANTIALIAS)

## If other computation necessary transform into np array
normal_img = np.array(normal_img)
overlay_img = np.array(overlay_img)

## Remove background
# overlay_img2 = ma.masked_array(overlay_img, overlay_img <= 0.1)
# overlay_img = 1 - overlay_img
# overlay_img2 = ma.masked_array(overlay_img, overlay_img >= 0.8)

## Custom colormap
cmap = LinearSegmentedColormap.from_list('mycmap', ['blue', 'green', 'yellow', 'red'])

cdict1 = {'red': ((0.0, 0.0, 0.0),
                  (0.1, 0.0, 0.0),
                  (0.5, 0.0, 0.0),
                  (0.75, 1.0, 1.0),
                  (1.0, 1.0, 1.0)),

          'green': ((0.0, 0.0, 0.0),
                    (0.1, 0.0, 0),
                    (0.5, 1.0, 1.0),
                    (0.75, 1.0, 1.0),
                    (1.0, 0.0, 1.0)),

          'blue': ((0.0, 1.0, 0.2),
                   (0.1, 1.0, 1.0),
                   (0.5, 1.0, 0.0),
                   (0.75, 0.0, 0.0),
                   (1.0, 0.0, 0.0))
          }

custom_cmp = LinearSegmentedColormap('BlueRed1', cdict1)

## Plot
# fig = plt.figure()
fig, ax = plt.subplots(figsize=(6, 6))
plt.axis('off')
plt.grid(None)
plt.imshow(normal_img)
plt.imshow(overlay_img, cmap=custom_cmp, alpha=alpha)
# colorbar.ColorbarBase(ax = ax, cmap=custom_cmp, orientation='horizontal')
# plt.imshow(overlay_img2, alpha=alpha, cmap=cmap)
plt.show()
