import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.colorbar as colorbar
import numpy as np
import os as os
import numpy.ma as ma
from PIL import Image
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

"""
Overlay generator:

- Overlay predicted images on the actual satellite image with a custom color map.
- Make sure that you satellite images are inside normal_img/ directory
- Make sure that you predicted images are inside submit_img/ directory
- Set the settable parameters image_nr and img_nr_overlay to the name of the images and run
"""

base_dir_path = ""
normal_img_path = os.path.join(base_dir_path, "normal_img/")
submission_img_path = os.path.join(base_dir_path, "submit_img/")

#########################
# Settable Parameter
#########################
image_nr = '123'
image_nr_overlay = '123_1_0'
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
#overlay_img2 = ma.masked_array(overlay_img, overlay_img <= 0.1)
#overlay_img = 1 - overlay_img
#overlay_img2 = ma.masked_array(overlay_img, overlay_img >= 0.8)

## Custom colormap
cmap = LinearSegmentedColormap.from_list('mycmap', ['blue', 'green', 'yellow', 'red'])

cdict1 = {'red':   ((0.0, 0.0, 0.0),
                    (0.1, 0.0, 0.0),
                    (0.5, 0.0, 0.0),
                    (0.75, 1.0, 1.0),
                    (1.0, 1.0, 1.0)),

         'green': ((0.0, 0.0, 0.0),
                   (0.1, 0.0, 0),
                   (0.5, 1.0, 1.0),
                   (0.75, 1.0, 1.0),
                   (1.0, 0.0, 1.0)),

         'blue':  ((0.0, 1.0, 0.2),
                   (0.1, 1.0, 1.0),
                    (0.5, 1.0, 0.0),
                    (0.75, 0.0, 0.0),
                    (1.0, 0.0, 0.0))
         }

custom_cmp = LinearSegmentedColormap('BlueRed1', cdict1)


## Plot
#fig = plt.figure()
fig, ax = plt.subplots(figsize=(6, 6))
plt.axis('off')
plt.grid(None)
plt.imshow(normal_img)
plt.imshow(overlay_img, cmap=custom_cmp,  alpha = alpha)
#colorbar.ColorbarBase(ax = ax, cmap=custom_cmp, orientation='horizontal')
#plt.imshow(overlay_img2, alpha=alpha, cmap=cmap)
plt.show()
