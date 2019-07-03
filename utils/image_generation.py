import numpy as np
import os
from PIL import Image

# Image sizes
# 3328 × 2560

# Image path
IMAGE_DIR = './chicago/'

# Save path for validation images
VALID_DIR = './valid/'

# set paths
CENTER_DIR = VALID_DIR + 'center/'
CENTER_GT_DIR = CENTER_DIR + 'groundtruth/'
CENTER_IMG_DIR = CENTER_DIR + 'images/'

UR_DIR = VALID_DIR + 'up_right/'
UR_GT_DIR = UR_DIR + 'groundtruth/'
UR_IMG_DIR = UR_DIR + 'images/'

UL_DIR = VALID_DIR + 'up_left/'
UL_GT_DIR = UL_DIR + 'groundtruth/'
UL_IMG_DIR = UL_DIR + 'images/'

BR_DIR = VALID_DIR + 'bottom_right/'
BR_GT_DIR = BR_DIR + 'groundtruth/'
BR_IMG_DIR = BR_DIR + 'images/'

BL_DIR = VALID_DIR + 'bottom_left/'
BL_GT_DIR = BL_DIR + 'groundtruth/'
BL_IMG_DIR = BL_DIR + 'images/'

# create valid path
os.makedirs(VALID_DIR, exist_ok=True)

os.makedirs(CENTER_GT_DIR, exist_ok=True)
os.makedirs(CENTER_IMG_DIR, exist_ok=True)

os.makedirs(UR_GT_DIR, exist_ok=True)
os.makedirs(UR_IMG_DIR, exist_ok=True)

os.makedirs(UL_GT_DIR, exist_ok=True)
os.makedirs(UL_IMG_DIR, exist_ok=True)

os.makedirs(BR_GT_DIR, exist_ok=True)
os.makedirs(BR_IMG_DIR, exist_ok=True)

os.makedirs(BL_GT_DIR, exist_ok=True)
os.makedirs(BL_IMG_DIR, exist_ok=True)

print("validation paths created")

# set params
num_images = 5
img_patch = 1333
img_patch_half = 666
img_size = 400

# Set patch location flags
center = True
up_right = True
up_left = True
bottom_right = True
bottom_left = True

# get all image names in the image path
filenames = os.listdir(IMAGE_DIR)

# sort them to alternate between gt and image
filenames.sort()

for i, filename in enumerate(filenames):
    # stop when we created num_images of images and labels
    # if i == (2*num_images):
    #     break

    print("generating image " + str(i) + " - " + filename)

    if i % 2 == 0:
        # Load image and convert to np array
        im = Image.open(IMAGE_DIR + filename)
        imarray = np.array(im)

        if center:
            # Get patch of image
            mid_pt = [int(imarray.shape[0] / 2) - 1, int(imarray.shape[1] / 2) - 1]
            img = imarray[mid_pt[0] - img_patch_half:mid_pt[0] + img_patch_half, mid_pt[1] - img_patch_half:mid_pt[1] + img_patch_half, :]

            img = Image.fromarray(img)
            img.thumbnail((img_size, img_size), Image.ANTIALIAS)

            # Save patch
            quality_val = 100

            # Fix naming so that groundtruth and image have same name, i.e. are pulled together by generator
            filename_center = filename.replace('_image', '_center')
            img.save(os.path.join(CENTER_IMG_DIR, filename_center), 'PNG', quality=quality_val)

        if up_right:
            # Get patch of image
            img = imarray[:1333, -1333:, :]
            img = Image.fromarray(img)
            img.thumbnail((400, 400), Image.ANTIALIAS)

            # Save patch
            quality_val = 100
            filename_ur = filename.replace('_image', '_ur')
            img.save(os.path.join(UR_IMG_DIR, filename_ur), 'PNG', quality=quality_val)

        if up_left:
            # Get patch of image
            img = imarray[:1333, :1333, :]
            img = Image.fromarray(img)
            img.thumbnail((400, 400), Image.ANTIALIAS)

            # Save patch
            quality_val = 100
            filename_ul = filename.replace('_image', '_ul')
            img.save(os.path.join(UL_IMG_DIR, filename_ul), 'PNG', quality=quality_val)

        if bottom_right:
            # Get patch of image
            img = imarray[-1333:, -1333:, :]
            img = Image.fromarray(img)
            img.thumbnail((400, 400), Image.ANTIALIAS)

            # Save patch
            quality_val = 100
            filename_br = filename.replace('_image', '_br')
            img.save(os.path.join(BR_IMG_DIR, filename_br), 'PNG', quality=quality_val)

        if bottom_left:
            # Get patch of image
            img = imarray[-1333:, :1333, :]
            img = Image.fromarray(img)
            img.thumbnail((400, 400), Image.ANTIALIAS)

            # Save patch
            quality_val = 100
            filename_bl = filename.replace('_image', '_bl')
            img.save(os.path.join(BL_IMG_DIR, filename_bl), 'PNG', quality=quality_val)

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

        if center:
            # Get patch of image
            mid_pt = [int(gtarray.shape[0] / 2)-1, int(gtarray.shape[1] / 2)-1]
            img = gtarray[mid_pt[0] - img_patch_half:mid_pt[0] + img_patch_half, mid_pt[1] - img_patch_half:mid_pt[1] + img_patch_half, :]
            img = Image.fromarray(img)
            img.thumbnail((img_size, img_size), Image.ANTIALIAS)

            # Save patch
            quality_val = 100
            filename_center = filename.replace('_labels', '_center')
            img.save(os.path.join(CENTER_GT_DIR, filename_center), 'PNG', quality=quality_val)

        if up_right:
            # Get patch of image
            img = gtarray[:1333, -1333:, :]
            img = Image.fromarray(img)
            img.thumbnail((400, 400), Image.ANTIALIAS)

            # Save patch
            quality_val = 100
            filename_ur = filename.replace('_labels', '_ur')
            img.save(os.path.join(UR_GT_DIR, filename_ur), 'PNG', quality=quality_val)

        if up_left:
            # Get patch of image
            img = gtarray[:1333, :1333, :]
            img = Image.fromarray(img)
            img.thumbnail((400, 400), Image.ANTIALIAS)

            # Save patch
            quality_val = 100
            filename_ul = filename.replace('_labels', '_ul')
            img.save(os.path.join(UL_GT_DIR, filename_ul), 'PNG', quality=quality_val)

        if bottom_right:
            # Get patch of image
            img = gtarray[-1333:, -1333:, :]
            img = Image.fromarray(img)
            img.thumbnail((400, 400), Image.ANTIALIAS)

            # Save patch
            quality_val = 100
            filename_br = filename.replace('_labels', '_br')
            img.save(os.path.join(BR_GT_DIR, filename_br), 'PNG', quality=quality_val)

        if bottom_left:
            # Get patch of image
            img = gtarray[-1333:, :1333, :]
            img = Image.fromarray(img)
            img.thumbnail((400, 400), Image.ANTIALIAS)

            # Save patch
            quality_val = 100
            filename_bl = filename.replace('_labels', '_bl')
            img.save(os.path.join(BL_GT_DIR, filename_bl), 'PNG', quality=quality_val)
