import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import skimage.io as io


# If you want to run it on Colab first run this:
"""
import os
import shutil

filepaths = "/content/"

# Folder where you need to put test set
test_set_path = os.path.join(filepaths, 'test_set')
os.makedirs(test_set_path, exist_ok = True)

# Folder where you need to put submission images (output)
submissions_path = os.path.join(filepaths, 'submissions_path')
os.makedirs(submissions_path, exist_ok = True)

# Folder where augmented images will be saved (if flag is set)
save_path = os.path.join(filepaths, 'saved_images')
os.makedirs(save_path, exist_ok = True)

# Remove uneccessary stuff
# shutil.rmtree(os.path.join(filepaths, 'sample_data'))
"""


# Load your submission images inside the folder /content/
# Which is the main folder when clicking Files.
# Run the script and see the comparison.

#############################################
#            SETTABLE PARAMETERS
#############################################
# TODO: find good parameters
minLineLength = 5  # Minimum length of line. Line segments shorter than this are rejected.
maxLineGap = 500  # Maximum allowed gap between line segments to treat them as single line.
threshold = 30  # Minimum vote it should get for it to be considered as a line

test_mode = True  # In test mode not all images are computed, but only nr_images_to_compute
nr_images_to_compute = 40
save = True  # when happy save

#############################################
#               HOUGH TRANSFORM
#############################################
nr_imgs = 3
if (save):
    nr_imgs = 4

filepaths = "/content/"
test_set_path = os.path.join(filepaths, 'test_set')
submissions_path = os.path.join(filepaths, 'submissions_path')
save_path = os.path.join(filepaths, 'saved_images')

count = 0
for image in os.listdir(test_set_path):
    if (image != ".config" and image != ".ipynb_checkpoints"):
        if (test_mode):
            count += 1
            print("TEST MODE: Image {0} out of {1}".format(count, nr_images_to_compute))
            if (count == nr_images_to_compute):
                break

        nr = os.path.splitext(image)[0]

        print("Computing on image: {0}... ".format(nr))

        img_intact = cv.imread(os.path.join(submissions_path, image))
        img = cv.imread(os.path.join(submissions_path, image))
        if (save):
            img_save = cv.imread(os.path.join(submissions_path, image))

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        lines = cv.HoughLinesP(gray, 1, np.pi / 180, threshold, minLineLength, maxLineGap)
        if (lines is not None):
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                if (save):
                    cv.line(img_save, (x1, y1), (x2, y2), (255, 255, 255), 2)

        fig = plt.figure(figsize=(15, 7))
        fig.suptitle(image, fontsize=10, y=0.72)
        plt.subplot(1, nr_imgs, 1)
        plt.grid(False);
        plt.axis('off');
        plt.imshow(img_intact)

        plt.subplot(1, nr_imgs, 2)
        plt.grid(False);
        plt.axis('off');
        plt.imshow(img)

        img_test = cv.imread(os.path.join(test_set_path, nr + ".png"))
        last = 3
        if (save):
            last = 4
        plt.subplot(1, nr_imgs, last)
        plt.grid(False);
        plt.axis('off');
        plt.imshow(img_test)

        if (save):
            if (not test_mode):
                io.imsave(os.path.join(save_path, nr + ".png"), img_save)
            plt.subplot(1, nr_imgs, 3)
            plt.grid(False);
            plt.axis('off');
            plt.imshow(img_save)





