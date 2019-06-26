from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
import glob
import skimage.io as io
import skimage.transform as trans


Road = [255,255,255]
Unlabelled = [0,0,0]

COLOR_DICT = np.array([Road, Unlabelled])


def adjustData(img,mask,num_class):
    img = img / 255
    mask = mask /255
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0
    return (img,mask)



def trainGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict,image_color_mode = "rgb",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    num_class = 2,save_to_dir = None,target_size = (400,400),seed = 1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        img,mask = adjustData(img,mask,num_class)
        yield (img,mask)


def testGenerator(test_path,target_size = (400,400)):
    '''
    create generator for test data.
    Using resizing from 608x608 to 400x400, since network is trained on 400x400, and test is 608x608x
    '''
    for filename in os.listdir(test_path):
        img = io.imread(os.path.join(test_path,filename))
        img = trans.resize(img,target_size)
        img = np.reshape(img,(1,)+img.shape)
        yield img

def prepare_ensamble(operation_path, output_path):
    '''
    Augment data from operation_path and add some changes to them, save new data into output_path.
    :param operation_path: data to be augmented.
    :param output_path: augmented saved data.
    :return:
    '''
    print("===== Prepare Ensamble Data - Saved Inside {0} =====".format(output_path))
    for image in os.listdir(operation_path):
        img = io.imread(os.path.join(operation_path, image))
        nr = os.path.splitext(image)[0]

        # split images into 4 identicall images
        img1 = img
        img2 = img
        img3 = img
        img4 = img

        img5 = img[:, ::-1]
        img6 = img[:, ::-1]
        img7 = img[:, ::-1]
        img8 = img[:, ::-1]

        # augment images
        img2 = trans.rotate(img2, 90)
        img3 = trans.rotate(img3, 180)
        img4 = trans.rotate(img4, 270)

        img6 = trans.rotate(img6, 90)
        img7 = trans.rotate(img7, 180)
        img8 = trans.rotate(img8, 270)

        # save images into output_path
        io.imsave(os.path.join(output_path, nr + "_1.png"), img1)
        io.imsave(os.path.join(output_path, nr + "_2.png"), img2)
        io.imsave(os.path.join(output_path, nr + "_3.png"), img3)
        io.imsave(os.path.join(output_path, nr + "_4.png"), img4)
        io.imsave(os.path.join(output_path, nr + "_5.png"), img5)
        io.imsave(os.path.join(output_path, nr + "_6.png"), img6)
        io.imsave(os.path.join(output_path, nr + "_7.png"), img7)
        io.imsave(os.path.join(output_path, nr + "_8.png"), img8)


def saveSubmitResizeEnsamble(temp_path, output_path_ensambled, output_path):
    print("Saving submission results inside: ", output_path)
    print("Using temp path to resize the images before saving: " + temp_path)
    os.makedirs(temp_path, exist_ok=True)
    for image in os.listdir(output_path_ensambled):
        img = io.imread(os.path.join(output_path_ensambled, image))
        io.imsave(os.path.join(temp_path, image), img)

        img2 = io.imread(os.path.join(temp_path, image))
        img2 = trans.resize(img2, [608, 608])
        io.imsave(os.path.join(output_path, image), img2)

def ensamble_predictions(predict_path, output_path_pre_ensambled, output_path_ensambled, alpha = 0.5):
    '''
    Ensamble using output_path_pre_ensambled and names from predict_path and save to output_path.
    Strategy is avg.
    :param predict_path:
    :param output_path_pre_ensambled:
    :param output_path_ensambled:
    :return:
    '''
    print("Ensamble predictions, and save them inside: ", output_path_ensambled)
    norm = np.ones([400, 400]) * 8

    for image in os.listdir(predict_path):

        nr = os.path.splitext(image)[0]
        recover1 = io.imread(os.path.join(output_path_pre_ensambled, nr + "_1.png"))
        recover2 = io.imread(os.path.join(output_path_pre_ensambled, nr + "_2.png"))
        recover3 = io.imread(os.path.join(output_path_pre_ensambled, nr + "_3.png"))
        recover4 = io.imread(os.path.join(output_path_pre_ensambled, nr + "_4.png"))
        recover5 = io.imread(os.path.join(output_path_pre_ensambled, nr + "_5.png"))
        recover6 = io.imread(os.path.join(output_path_pre_ensambled, nr + "_6.png"))
        recover7 = io.imread(os.path.join(output_path_pre_ensambled, nr + "_7.png"))
        recover8 = io.imread(os.path.join(output_path_pre_ensambled, nr + "_8.png"))

        print("MAX of recover1", np.max(recover1))
        # if value is 0 try casting recover1 as float and normalize
        # 255 * 255
        recover1 = recover1 / 65025
        recover2 = recover2 / 65025
        recover3 = recover3 / 65025
        recover4 = recover4 / 65025
        recover5 = recover5 / 65025
        recover6 = recover6 / 65025
        recover7 = recover7 / 65025
        recover8 = recover8 / 65025

        print("AFTER NORM: MAX of recover1", np.max(recover1))

        # restore to original form
        recover2 = trans.rotate(recover2, -90)
        recover3 = trans.rotate(recover3, -180)
        recover4 = trans.rotate(recover4, -270)

        recover6 = trans.rotate(recover6, -90)
        recover7 = trans.rotate(recover7, -180)
        recover8 = trans.rotate(recover8, -270)

        recover5 = recover5[:, ::-1]
        recover6 = recover6[:, ::-1]
        recover7 = recover7[:, ::-1]
        recover8 = recover8[:, ::-1]

        # combine using avg
        avg = recover1 + recover2 + recover3 + recover4 + recover5 + recover6 + recover7 + recover8
        avg = np.divide(avg, norm)

        val = np.where(avg > alpha, 1, 0)
        val = val * 255
        io.imsave(os.path.join(output_path_ensambled, nr + ".png"), val)



def prepare_4to1data(predict_path, predict_4to1_path):
    '''
    create generator for test data.
    Since network is trained on 400x400, and test is 608x608x. We split the test images into 4 images of size 400x400.
    And after prediction will recombine them.
    '''
    print("===== Prepare 4 to 1 Data =====")
    os.makedirs(predict_4to1_path, exist_ok=True)
    for image in os.listdir(predict_path):
        img = io.imread(os.path.join(predict_path, image))
        nr = os.path.splitext(image)[0]
        # split image into 4
        part1 = img[0:400, 0:400, :]
        part2 = img[0:400, -400:, :]
        part3 = img[-400:, 0:400, :]
        part4 = img[-400:, -400:, :]
        io.imsave(os.path.join(predict_4to1_path, nr + "_1.png"), part1)
        io.imsave(os.path.join(predict_4to1_path, nr + "_2.png"), part2)
        io.imsave(os.path.join(predict_4to1_path, nr + "_3.png"), part3)
        io.imsave(os.path.join(predict_4to1_path, nr + "_4.png"), part4)

def postprocess_4to1data_max(predict_path, output_path_4to1, output_path, alpha=0.5):
    '''
    Recombining 4 test images into 1, for the overlapping part we just use the maximum.
    '''
    for image in os.listdir(predict_path):
        init = np.zeros([608, 608])
        nr = os.path.splitext(image)[0]
        # Images have been saved in folder using "imagenr_1.png, imagenr_2.png, .."
        recover1 = io.imread(os.path.join(output_path_4to1, nr + "_1.png"))
        recover2 = io.imread(os.path.join(output_path_4to1, nr + "_2.png"))
        recover3 = io.imread(os.path.join(output_path_4to1, nr + "_3.png"))
        recover4 = io.imread(os.path.join(output_path_4to1, nr + "_4.png"))

        # calculating max result (from 1 outputs)
        init[0:400, 0:400] = np.maximum(init[0:400, 0:400], recover1)
        init[0:400, -400:] = np.maximum(init[0:400, -400:], recover2)
        init[-400:, 0:400] = np.maximum(init[-400:, 0:400], recover3)
        init[-400:, -400:] = np.maximum(init[-400:, -400:], recover4)

        init = np.where(init > alpha, 1, 0)

        io.imsave(os.path.join(output_path, nr + ".png"), init)

def postprocess_4to1data_avg(predict_path, output_path_4to1_pre, output_path, alpha=0.5, norm_val=1):
    '''
    Recombining 4 test images into 1, for the overlapping part we use the average.
    Note: we use the images before having a probability cutoff (of alpha) to sum up.
        Then we normalize element wise using the norm-matrix, and after do the
        probability cutoff.
    '''
    norm=np.zeros([608,608])
    norm[0:400,0:400]+=np.ones([400,400]) * norm_val
    norm[0:400,-400:]+=np.ones([400,400]) * norm_val
    norm[-400:,0:400]+=np.ones([400,400]) * norm_val
    norm[-400:,-400:]+=np.ones([400,400]) * norm_val

    for image in os.listdir(predict_path):
        init = np.zeros([608, 608])
        nr = os.path.splitext(image)[0]
        recover1 = io.imread(os.path.join(output_path_4to1_pre, nr + "_1.png"))
        recover2 = io.imread(os.path.join(output_path_4to1_pre, nr + "_2.png"))
        recover3 = io.imread(os.path.join(output_path_4to1_pre, nr + "_3.png"))
        recover4 = io.imread(os.path.join(output_path_4to1_pre, nr + "_4.png"))

        # calculating max result (from 1 outputs)
        init[0:400, 0:400] += recover1
        init[0:400, -400:] += recover2
        init[-400:, 0:400] += recover3
        init[-400:, -400:] += recover4

        init = np.divide(init, norm)
        post_init = np.where(init > alpha, 1, 0)
        post_init *= norm_val
        io.imsave(os.path.join(output_path, nr + ".png"), post_init)

def geneTrainNpy(image_path,mask_path,num_class = 2,image_prefix = "image",mask_prefix = "mask",image_as_gray = True,mask_as_gray = True):
    """
    probably not used, legacy
    """

    image_name_arr = glob.glob(os.path.join(image_path,"%s*.png"%image_prefix))
    image_arr = []
    mask_arr = []
    for index,item in enumerate(image_name_arr):
        img = io.imread(item,as_gray = image_as_gray)
        img = np.reshape(img,img.shape + (1,)) if image_as_gray else img
        mask = io.imread(item.replace(image_path,mask_path).replace(image_prefix,mask_prefix),as_gray = mask_as_gray)
        mask = np.reshape(mask,mask.shape + (1,)) if mask_as_gray else mask
        img,mask = adjustData(img,mask,num_class)
        image_arr.append(img)
        mask_arr.append(mask)
    image_arr = np.array(image_arr)
    mask_arr = np.array(mask_arr)
    return image_arr,mask_arr


def labelVisualize(num_class,color_dict,img):
    """ probably not used, legacy"""
    img = img[:,:,0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i,:] = color_dict[i]
    return img_out / 255

def saveResultunprocessed(save_path,npyfile, filenames):
    """ saver for predictions without probablity cutoff"""
    print("Saving unprocessed results inside: ", save_path)
    for i,item in enumerate(npyfile):
        img = item[:,:,0]
        io.imsave(os.path.join(save_path, filenames[i]), img)

def saveResult(save_path,npyfile, filenames):
    """ saver for predictions with probablity cutoff used"""
    print("Saving results inside: ", save_path)
    for i,item in enumerate(npyfile):
        img = item[:,:,0]*255
        io.imsave(os.path.join(save_path, filenames[i]), img)

def savesubmitResult(temp_path,save_path, npyfile, filenames):
    """ Output of test data should be 608x608. Network output is 400x400.
        We resize output from 400x400 to 608x608
    """
    print("Saving submission results inside: ", save_path)
    print("Using temp path to resize the images before saving: " + temp_path)
    os.makedirs(temp_path, exist_ok=True)
    for i,item in enumerate(npyfile):
        img = item[:,:,0]
        img = img*255
        io.imsave(os.path.join(temp_path, filenames[i]), img)
        img2 = io.imread(os.path.join(temp_path, filenames[i]))
        img2 = trans.resize(img2, [608, 608])
        io.imsave(os.path.join(save_path,filenames[i]),img2)

def savesubmitResult_4to1version(save_path, npyfile, filenames):
    """"
        In main-cil 4 test output predictions have already been combined, thus here just saving of the result.
    """
    print("Saving submission results 4to1 inside: ", save_path)
    for i,item in enumerate(npyfile):
        img = item[:,:,0]
        img = img*255
        io.imsave(os.path.join(save_path, filenames[i]), img)
        #img2 = io.imread(os.path.join(temp_path, filenames[i]))
        #img2 = trans.resize(img2, [608, 608])
        #io.imsave(os.path.join(save_path,filenames[i]),img2)