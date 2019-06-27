from model_cil_stacked_flexible import *
from data_cil_stacked_flexible import *
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
parser.add_argument('--nr_of_stacks', type=int,
                    help='points outside the boundaries of the input are filled according to the given mode, '
                         'standard is nearest')



args = parser.parse_args()
# end of argument handling


log_path=os.path.join("./logs/",args.desc)
tensorboard = TensorBoard(log_dir=log_path, histogram_freq=0,
                          write_graph=True, write_images=False)

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"



submission_flag = args.submission
nr_of_epochs = args.epochs

train_path = "data/roadseg/submit_train"
test_predict_path = "data/roadseg/submit_test"
test_output_path = "data/roadseg/submit_output"
valid_path = "data/roadseg/valid_gen"
valid_predict_path = "data/roadseg/valid_gen/image"
valid_output_path="data/roadseg/valid_gen/output"
temp_4to1_path = "data/roadseg/temp_4to1_folder"
temp_path = "data/roadseg/temp"
valid_count=50
test_count=94

data_gen_args = dict(rotation_range=args.rotation,
                    width_shift_range=args.width_shift_range,
                    height_shift_range=args.height_shift_range,
                    shear_range=args.shear_range,
                    zoom_range=args.zoom_range,
                    horizontal_flip=args.horizontal_flip,
                    fill_mode=args.fill_mode)

trainGen = trainGenerator(2,train_path,'image', 'label',data_gen_args,save_to_dir = None, nr_of_stacks = args.nr_of_stacks)
validGen = trainGenerator(2,valid_path,'image', 'label',data_gen_args,save_to_dir = None, nr_of_stacks = args.nr_of_stacks)

model = unet(nr_of_stacks=args.nr_of_stacks)
model_checkpoint_train = ModelCheckpoint(os.path.join(log_path,'unet_roadseg.hdf5'), monitor='val_acc',verbose=1, save_best_only=True)
model_checkpoint_submit = ModelCheckpoint(os.path.join(log_path,'unet_roadseg.hdf5'), monitor='acc',verbose=1, save_best_only=True)


model.fit_generator(trainGen, steps_per_epoch=100, epochs=nr_of_epochs, callbacks=[model_checkpoint_train, tensorboard],
                        validation_data=validGen, validation_steps=valid_count)


# REFACTORED TILL HERE

# choose prediction type (either resize or 4to1)
if(args.resize==True):
    filenames = os.listdir(test_predict_path)
    testGene = testGenerator(test_predict_path)
else:
    prepare_4to1data(test_predict_path, temp_4to1_path)
    filenames = os.listdir(temp_4to1_path)
    testGene = testGenerator(temp_4to1_path)
    count = count*4

# load best model from training and predict results
model.load_weights(os.path.join(log_path,"unet_roadseg.hdf5"))
results = model.predict_generator(testGene,count,verbose=1)
post_results = np.where(results > 0.5, 1, 0)

output_path=os.path.join(test_output_path,args.desc)
os.mkdir(output_path)


output_path_4to1 = os.path.join(output_path, "split_results")
os.mkdir(output_path_4to1)
output_path_4to1_pre = os.path.join(output_path_4to1, "split_results_pre")
os.mkdir(output_path_4to1_pre)

output_path_pre=os.path.join(output_path,"pre_results")
os.mkdir(output_path_pre)


if(args.resize==True):
    savesubmitResult(temp_path, output_path, post_results, filenames, nr_of_stacks = args.nr_of_stacks)
    saveResultunprocessed(output_path_pre, results, filenames, nr_of_stacks = args.nr_of_stacks)
else:
    savesubmitResult_4to1version(output_path_4to1, post_results, filenames, nr_of_stacks = args.nr_of_stacks)
    saveResultunprocessed(output_path_4to1_pre, results, filenames, nr_of_stacks = args.nr_of_stacks)
    if(args.combine_max==True):
        print("combining image using max")
        postprocess_4to1data_max(predict_path, output_path_4to1, output_path)
    else:
        print("combining image using average")
        postprocess_4to1data_avg(predict_path, output_path_4to1_pre, output_path)