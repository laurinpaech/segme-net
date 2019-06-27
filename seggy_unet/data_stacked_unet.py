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
                    num_class = 2,save_to_dir = None,target_size = (400,400),seed = 1, nr_of_stacks = 1):
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
        yield (img,np.repeat(mask,nr_of_stacks,axis=3))


def testGenerator(test_path,target_size = (400,400)):
    for filename in os.listdir(test_path):
        img = io.imread(os.path.join(test_path,filename))
        img = trans.resize(img,target_size)
        img = np.reshape(img,(1,)+img.shape)
        yield img

def prepare_4to1data(predict_path, predict_4to1_path):
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
    for image in os.listdir(predict_path):
        init = np.zeros([608, 608])
        nr = os.path.splitext(image)[0]
        recover1 = io.imread(os.path.join(output_path_4to1, nr + "_1.png"))
        recover2 = io.imread(os.path.join(output_path_4to1, nr + "_2.png"))
        recover3 = io.imread(os.path.join(output_path_4to1, nr + "_3.png"))
        recover4 = io.imread(os.path.join(output_path_4to1, nr + "_4.png"))

        # adjust data
        recover1=recover1/255
        recover2=recover2/255
        recover3=recover3/255
        recover4=recover4/255

        # calculating max result (from 1 outputs)
        init[0:400, 0:400] = np.maximum(init[0:400, 0:400], recover1)
        init[0:400, -400:] = np.maximum(init[0:400, -400:], recover2)
        init[-400:, 0:400] = np.maximum(init[-400:, 0:400], recover3)
        init[-400:, -400:] = np.maximum(init[-400:, -400:], recover4)

        io.imsave(os.path.join(output_path, nr + ".png"), init)

def postprocess_4to1data_avg(predict_path, output_path_4to1_pre, output_path, alpha=0.5):

    norm=np.zeros([608,608])
    norm[0:400,0:400]+=np.ones([400,400])
    norm[0:400,-400:]+=np.ones([400,400])
    norm[-400:,0:400]+=np.ones([400,400])
    norm[-400:,-400:]+=np.ones([400,400])

    for image in os.listdir(predict_path):
        init = np.zeros([608, 608])
        nr = os.path.splitext(image)[0]
        recover1 = io.imread(os.path.join(output_path_4to1_pre, nr + "_1_2.png"))
        recover1 = recover1.astype(float)
        recover2 = io.imread(os.path.join(output_path_4to1_pre, nr + "_2_2.png"))
        recover2 = recover2.astype(float)
        recover3 = io.imread(os.path.join(output_path_4to1_pre, nr + "_3_2.png"))
        recover3 = recover3.astype(float)
        recover4 = io.imread(os.path.join(output_path_4to1_pre, nr + "_4_2.png"))
        recover4 = recover4.astype(float)

        #normalize data (apparently here max is 256*256)
        recover1 = recover1 / 65025
        recover2 = recover2 / 65025
        recover3 = recover3 / 65025
        recover4 = recover4 / 65025

        # calculating max result (from 1 outputs)
        init[0:400, 0:400] += recover1
        init[0:400, -400:] += recover2
        init[-400:, 0:400] += recover3
        init[-400:, -400:] += recover4

        init = np.divide(init, norm)
        post_init = np.where(init > alpha, 1, 0)

        io.imsave(os.path.join(output_path, nr + ".png"), post_init*255)

def geneTrainNpy(image_path,mask_path,num_class = 2,image_prefix = "image",mask_prefix = "mask",image_as_gray = True,mask_as_gray = True):
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
    img = img[:,:,0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i,:] = color_dict[i]
    return img_out / 255

def saveResultunprocessed(save_path,npyfile, filenames, nr_of_stacks):
    for i,item in enumerate(npyfile):
        for n in range(nr_of_stacks):
            img = item[:,:,n]
            nr = os.path.splitext(filenames[i])[0]
            io.imsave(os.path.join(save_path, nr+"_"+str(n)+".png"), img)

def saveResult(save_path,npyfile, filenames, nr_of_stacks):
    for i,item in enumerate(npyfile):
        img = item[:,:,nr_of_stacks-1]*255
        io.imsave(os.path.join(save_path, filenames[i]), img)

def savesubmitResult(temp_path,save_path, npyfile, filenames, nr_of_stacks):
    for i,item in enumerate(npyfile):
        img = item[:,:,nr_of_stacks-1]
        img = img*255
        io.imsave(os.path.join(temp_path, filenames[i]), img)
        img2 = io.imread(os.path.join(temp_path, filenames[i]))
        img2 = trans.resize(img2, [608, 608])
        io.imsave(os.path.join(save_path,filenames[i]),img2)

def savesubmitResult_4to1version(save_path, npyfile, filenames, nr_of_stacks):
    for i,item in enumerate(npyfile):
        img = item[:,:,nr_of_stacks-1]
        img = img*255
        io.imsave(os.path.join(save_path, filenames[i]), img)
        #img2 = io.imread(os.path.join(temp_path, filenames[i]))
        #img2 = trans.resize(img2, [608, 608])
        #io.imsave(os.path.join(save_path,filenames[i]),img2)