
import os
from keras.callbacks import TensorBoard
import argparse


# this part is used for argument handling
parser = argparse.ArgumentParser()
parser.add_argument('--desc', type=str,
                    help='How to name this run, defines folder in logs dir, only use "a-z,A-Z,1-9,_" pls')
parser.add_argument('--stacked', type=bool,
                    help='either resizes submission images or uses splits image into 4 subimages to make predictions' )
parser.add_argument('--resize', type=bool,
                    help='either resizes submission images or uses splits image into 4 subimages to make predictions' )
parser.add_argument('--combine_max', type=bool,
                    help='if split image into 4 subimage, can combine using max (True), or using average (False, default)' )

args = parser.parse_args()
# end of argument handling

if args.stacked:
    from model_cil_stacked import *
    from data_cil_stacked import *
else:
    from model_cil import *
    from data_cil import *

# log folder is created with description as name
log_path=os.path.join("./logs/",args.desc)
tensorboard = TensorBoard(log_dir=log_path, histogram_freq=0,
                          write_graph=True, write_images=False)

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


train_path = "data/roadseg/submit_train"
valid_path = "data/roadseg/valid"
predict_path = "data/roadseg/submit_test"
predict_4to1_path = "data/roadseg/temp_4to1_folder"
output_path = "data/roadseg/submit_output"
temp_path = "data/roadseg/temp"
count=94


# model is saved in log_folder
# only the best model (defined by monitor-metric) is saved
model = unet()
#model_checkpoint_submit = ModelCheckpoint(os.path.join(log_path,'unet_roadseg.hdf5'), monitor='acc',verbose=1, save_best_only=True)


# choose predictions
# - for valid resize is just normal predictions
# - for predictions we must deal with 608x608x dimensions (instead of 400x400x during training)
# if resize is true -> just resize images
# if resize is false -> we split 608x608 into 4 (partially overlapping) images, and then after recombine them
if(args.resize==True):
    filenames = os.listdir(predict_path)
    testGene = testGenerator(predict_path)
else:
    prepare_4to1data(predict_path, predict_4to1_path)
    filenames = os.listdir(predict_4to1_path)
    testGene = testGenerator(predict_4to1_path)
    count = count*4

# load best model from training and predict results
model.load_weights(os.path.join(log_path,"unet_roadseg.hdf5"))
results = model.predict_generator(testGene,count,verbose=1)
# output is in range 0 to 1, we want binary output for final predictions
# this 0.5 value can be chosen differently
post_results = np.where(results > 0.5, 1, 0)

# create all required output folders
output_path=os.path.join(output_path,args.desc)
if(not os.path.exists(output_path)):
    os.mkdir(output_path)
output_path_4to1 = os.path.join(output_path, "split_results")
if(not os.path.exists(output_path_4to1)):
    os.mkdir(output_path_4to1)
output_path_4to1_pre = os.path.join(output_path_4to1, "split_results_pre")
if(not os.path.exists(output_path_4to1_pre)):
    os.mkdir(output_path_4to1_pre)
output_path_pre=os.path.join(output_path,"pre_results")
if(not os.path.exists(output_path_pre)):
    os.mkdir(output_path_pre)

if(args.resize==True):
    savesubmitResult(temp_path, output_path, post_results, filenames)
    saveResultunprocessed(output_path_pre, results, filenames)
else:
    savesubmitResult_4to1version(output_path_4to1, post_results, filenames)
    saveResultunprocessed(output_path_4to1_pre, results, filenames)
    if(args.combine_max==True):
        print("combining image using max")
        postprocess_4to1data_max(predict_path, output_path_4to1, output_path)
    else:
        print("combining image using average")
        postprocess_4to1data_avg(predict_path, output_path_4to1_pre, output_path)