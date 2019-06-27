import numpy as np
import os
from PIL import Image

# Image path
IMAGE_DIR = './chicago/'

# Save path for validation images
VALID_DIR = './valid/'

# create valid path
os.makedirs(VALID_DIR, exist_ok=True)
print("validation path is created")

num_images = 100

# get all image names in the image path
filenames = os.listdir(IMAGE_DIR)
# sort them to alternate between gt and image
filenames.sort()

for i, filename in enumerate(filenames):
    # stop when we created num_images of images and labels
    if i == (2*num_images):
        break

    print("generating " + filename)

    if i % 2 == 0:
        # Load image and convert to np array
        im = Image.open(IMAGE_DIR + filename)
        imarray = np.array(im)

        # Get patch of image
        img = imarray[:1333, -1333:, :]
        img = Image.fromarray(img)
        img.thumbnail((400, 400), Image.ANTIALIAS)

        # Save patch
        quality_val = 100
        img.save(os.path.join(VALID_DIR, filename), 'PNG', quality=quality_val)
    else:
        # Load labels and convert to np array
        gt = Image.open(IMAGE_DIR + filename)
        gtarray = np.array(gt)

        # preprocess
        mask = np.all(gtarray == [255, 255, 255], axis=2)
        gtarray[mask] = [0, 0, 0]
        gtarray[:, :, 0] = 0
        mask = np.all(gtarray == [0, 0, 255], axis=2)
        gtarray[mask] = [255, 255, 255]

        # Get patch of image
        img = gtarray[:1333, -1333:, :]
        img = Image.fromarray(img)
        img.thumbnail((400, 400), Image.ANTIALIAS)

        # Save patch
        quality_val = 100
        img.save(os.path.join(VALID_DIR, filename), 'PNG', quality=quality_val)
