from model_cil import *
from data_cil import *
import os
from keras.callbacks import TensorBoard
import argparse


# this part is used for argument handling
parser = argparse.ArgumentParser()
parser.add_argument('--desc', type=str,
                    help='How to name this run, defines folder in logs dir, only use "a-z,A-Z,1-9,_" pls')
parser.add_argument('--epochs', type=int,
                    help='Number of epochs to run')
parser.add_argument('--submission', type=bool,
                    help='to create submission or not')
parser.add_argument('--rotation', type=int,
                    help='rotation perturbation in degrees')
parser.add_argument('--width_shift_range', type=float,
                    help='width_shift_range')
parser.add_argument('--height_shift_range', type=float,
                    help='height_shift_range')
parser.add_argument('--shear_range', type=float,
                    help='shear_range')
parser.add_argument('--zoom_range', type=float,
                    help='zoom_range')
parser.add_argument('--horizontal_flip', type=bool,
                    help='horizontal_flip: True or False')
parser.add_argument('--fill_mode', type=str,
                    help='points outside the boundaries of the input are filled according to the given mode, '
                         'standard is nearest')
parser.add_argument('--resize', type=bool,
                    help='either resizes submission images or uses splits image into 4 subimages to make predictions' )
parser.add_argument('--combine_max', type=bool,
                    help='if split image into 4 subimage, can combine using max (True), or using average (False, default)' )
parser.add_argument('--ensamble', type=bool, help='creates 4 subimages with different properties and predicts on them before merging')



args = parser.parse_args()
# end of argument handling


# log folder is created with description as name
log_path=os.path.join("./logs/",args.desc)
tensorboard = TensorBoard(log_dir=log_path, histogram_freq=0,
                          write_graph=True, write_images=False)

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

submission_flag = args.submission
nr_of_epochs = args.epochs
ensamble = args.ensamble

# if submission is true, will generate test output, and train with all data
# if submission is false, will train with 90 samples, and use 10 for validation
if not submission_flag:
    train_path = "data/roadseg/train"
    valid_path = "data/roadseg/valid"
    predict_path = "data/roadseg/valid/image"
    predict_ensamble_path = ""
    predict_4to1_path = ""
    output_path = "data/roadseg/valid/output"
    temp_path = "data/roadseg/temp"
    count=10
else:
    train_path = "data/roadseg/submit_train"
    valid_path = "data/roadseg/valid"
    predict_path = "data/roadseg/submit_test"
    predict_4to1_path = "data/roadseg/temp_4to1_folder"
    predict_ensamble_path = "data/roadseg/test_ensamble"
    output_path = "data/roadseg/submit_output"
    temp_path = "data/roadseg/temp"
    count=94 # total number of test images


data_gen_args = dict(rotation_range=args.rotation,
                    width_shift_range=args.width_shift_range,
                    height_shift_range=args.height_shift_range,
                    shear_range=args.shear_range,
                    zoom_range=args.zoom_range,
                    horizontal_flip=args.horizontal_flip,
                    fill_mode=args.fill_mode)

# training generator, note at each epoch the samples are perturbed using the data_gen_args
# save_to_dir allows to save those perturbed samples if you want to see how it looks
trainGen = trainGenerator(2,train_path,'image', 'label',data_gen_args,save_to_dir = None)

# we only validate, if it's not a submission run
if(not submission_flag):
    validGen = trainGenerator(2,valid_path,'image', 'label',data_gen_args,save_to_dir = None)

# model is saved in log_folder
# only the best model (defined by monitor-metric) is saved
model = unet()
model_checkpoint_train = ModelCheckpoint(os.path.join(log_path,'unet_roadseg.hdf5'), monitor='val_acc',verbose=1, save_best_only=True)
model_checkpoint_submit = ModelCheckpoint(os.path.join(log_path,'unet_roadseg.hdf5'), monitor='acc',verbose=1, save_best_only=True)


if(not submission_flag):
    model.fit_generator(trainGen, steps_per_epoch=90, epochs=nr_of_epochs, callbacks=[model_checkpoint_train, tensorboard],
                        validation_data=validGen, validation_steps=10)
else:
    model.fit_generator(trainGen, steps_per_epoch=100, epochs=nr_of_epochs, callbacks=[model_checkpoint_submit, tensorboard])

# choose predictions
# - for valid resize is just normal predictions
# - for predictions we must deal with 608x608x dimensions (instead of 400x400x during training)
# if resize is true -> just resize images
# if resize is false -> we split 608x608 into 4 (partially overlapping) images, and then after recombine them
if(args.resize==True or not submission_flag):
    if (ensamble):
        count = count * 8
        prepare_ensamble(predict_path, predict_ensamble_path)
        filenames = os.listdir(predict_ensamble_path)
        testGene = testGenerator(predict_ensamble_path)
    else:
        filenames = os.listdir(predict_path)
        testGene = testGenerator(predict_path)

else:
    prepare_4to1data(predict_path, predict_4to1_path)
    count = count * 4

    # take 4 patches and apply ensamble
    if (ensamble):
        count = count * 8
        prepare_ensamble(predict_4to1_path, predict_ensamble_path)
        filenames = os.listdir(predict_ensamble_path)
        testGene = testGenerator(predict_ensamble_path)

    else:
        filenames = os.listdir(predict_4to1_path)
        testGene = testGenerator(predict_4to1_path)

# load best model from training and predict results
model.load_weights(os.path.join(log_path,"unet_roadseg.hdf5"))
results = model.predict_generator(testGene,count,verbose=1)


# output is in range 0 to 1, we want binary output for final predictions
# this 0.5 value can be chosen differently
post_results = np.where(results > 0.5, 1, 0)

# create all required output folders
output_path=os.path.join(output_path,args.desc)
os.mkdir(output_path)
output_path_4to1 = os.path.join(output_path, "split_results")
os.mkdir(output_path_4to1)
output_path_4to1_pre = os.path.join(output_path_4to1, "split_results_pre")
os.mkdir(output_path_4to1_pre)
output_path_pre=os.path.join(output_path,"pre_results")
os.mkdir(output_path_pre)

output_path_pre_ensambled=os.path.join(output_path,"pre_ensambled")
os.mkdir(output_path_pre_ensambled)
output_path_ensambled=os.path.join(output_path,"ensambled")
os.mkdir(output_path_ensambled)

if(not submission_flag):
    if (ensamble):
        saveResult(output_path_pre_ensambled, results, filenames) # saves the results inside the pre ensambled output path
        ensamble(predict_path, output_path_pre_ensambled, output_path) # ensamble using output_path_pre_ensambled and names from predict_path and save to output_path
    else:
        saveResult(output_path, post_results, filenames)
        saveResultunprocessed(output_path_pre, results, filenames)
else:
    if(args.resize==True):
        if (ensamble):
            saveResult(output_path_pre_ensambled, results, filenames)
            ensamble(predict_path, output_path_pre_ensambled, output_path_ensambled) # save into output_path_ensambled because need resize
            saveSubmitResizeEnsamble(temp_path, output_path_ensambled, output_path) # resize imgs in output_path_ensambled to 608x608 and save
        else:
            savesubmitResult(temp_path, output_path, post_results, filenames)
            saveResultunprocessed(output_path_pre, results, filenames)
    else:
        if (ensamble):
            saveResult(output_path_pre_ensambled, results, filenames)
            ensamble(predict_4to1_path, output_path_pre_ensambled, output_path_4to1_pre)
            print("combining image using average")
            postprocess_4to1data_avg(predict_path, output_path_4to1_pre, output_path)

        else:
            savesubmitResult_4to1version(output_path_4to1, post_results, filenames)
            saveResultunprocessed(output_path_4to1_pre, results, filenames)
            if(args.combine_max==True):
                print("combining image using max")
                postprocess_4to1data_max(predict_path, output_path_4to1, output_path)
            else:
                print("combining image using average")
                postprocess_4to1data_avg(predict_path, output_path_4to1_pre, output_path)

