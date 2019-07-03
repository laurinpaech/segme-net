from data_loader.data import *
from keras.callbacks import TensorBoard, ModelCheckpoint
import argparse

# this part is used for argument handling
parser = argparse.ArgumentParser()
parser.add_argument('--desc', type=str, default='stacked_unet_default_desc',
                    help='How to name this run, defines folder in logs dir, only use "a-z,A-Z,1-9,_" pls')
parser.add_argument('--epochs', type=int, default=300,
                    help='Number of epochs to run')
parser.add_argument('--rotation', type=int, default=360,
                    help='rotation perturbation in degrees')
parser.add_argument('--width_shift_range', type=float, default=50,
                    help='width_shift_range between 0 and 255')
parser.add_argument('--height_shift_range', type=float, default=50,
                    help='height_shift_range between 0 and 255')
parser.add_argument('--shear_range', type=float, default=0,
                    help='shear_range between 0 and 255')
parser.add_argument('--zoom_range', type=float, default=0,
                    help='zoom_range BETWEEN 0 AND 1')
parser.add_argument('--horizontal_flip', default=False, action='store_true',
                    help='horizontal_flip: True or False')
parser.add_argument('--vertical_flip', default=False, action='store_true',
                    help='vertical_flip: True or False')
parser.add_argument('--fill_mode', type=str, default='reflect',
                    help='points outside the boundaries of the input are filled according to the given mode, '
                         'standard is nearest')
parser.add_argument('--resize', default=False, action='store_true',
                    help='either resizes submission images or uses splits image into 4 subimages to make predictions')
parser.add_argument('--combine_max', default=False, action='store_true',
                    help='if split image into 4 subimage, combine using max (True), or using average (False, default)')
parser.add_argument('--nr_of_stacks', type=int, default=2,
                    help='points outside the boundaries of the input are filled according to the given mode, '
                         'standard is nearest')
parser.add_argument('--ensemble', default=False, action='store_true',
                    help='predict 8 versions of image with rotations and flipping, and recombine them later')
parser.add_argument('--channel_shift_range', type=float, default=0,
                    help='random channel_shift_range in [-input,input]')
parser.add_argument('--batch_size', type=int, default=2,
                    help='Batch size for training (default: 2) ')
parser.add_argument('--leakyRelu', default=False, action='store_true',
                    help='choose if unet should use leaky Relu')
parser.add_argument('--wavelet', default=False, action='store_true',
                    help='choose if unet should use wavelet inputs')
parser.add_argument('--gpu', type=int, default=-1,
                    help='choose which gpu to use')

args = parser.parse_args()
print(args)

# Set Parser Arguments
if args.gpu != -1:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

nr_of_epochs = args.epochs
ensemble = args.ensemble


# load correct unet model
if args.leakyRelu:
    from model.stacked_unet_leaky import *
else:
    from model.stacked_unet import *


# set logging
log_path = os.path.join("./logs/", args.desc)
tensorboard = TensorBoard(log_dir=log_path, histogram_freq=0,
                          write_graph=True, write_images=False)


# set paths
train_path = "data/train"
test_predict_path = "data/test"
test_output_path = "data/submit_output"
valid_path = "data/valid"
valid_predict_path = "data/valid/image"
valid_output_path = "data/valid/output"
temp_4to1_path = "data/temp_4to1_folder"
temp_path = "data/temp"
temp_ensemble_path = "data/temp_ensemble"
valid_count = 100
test_count = 94


data_gen_args = dict(rotation_range=args.rotation,
                     width_shift_range=args.width_shift_range,
                     height_shift_range=args.height_shift_range,
                     shear_range=args.shear_range,
                     zoom_range=args.zoom_range,
                     horizontal_flip=args.horizontal_flip,
                     vertical_flip=args.vertical_flip,
                     fill_mode=args.fill_mode,
                     channel_shift_range=args.channel_shift_range)

trainGen = trainGenerator(args.batch_size, train_path, 'image', 'label', data_gen_args, save_to_dir=None,
                          nr_of_stacks=args.nr_of_stacks, use_wavelet=args.wavelets)
validGen = trainGenerator(1, valid_path, 'image', 'label', data_gen_args, save_to_dir=None,
                          nr_of_stacks=args.nr_of_stacks, use_wavelet=args.wavelets)

# Initialize model
model = unet(nr_of_stacks=args.nr_of_stacks)


model_checkpoint_train = ModelCheckpoint(os.path.join(log_path, 'unet_roadseg.hdf5'), monitor='real_kaggle_metric_035',
                                         verbose=1, save_best_only=True)
# model_checkpoint_train = ModelCheckpoint(os.path.join(log_path, 'unet_roadseg_{epoch:04d}.hdf5'), monitor='val_acc',
#                                          verbose=1, period=50)

model.fit_generator(trainGen, steps_per_epoch=50, epochs=nr_of_epochs, callbacks=[model_checkpoint_train, tensorboard],
                    validation_data=validGen, validation_steps=valid_count)

# if necessary, create tempfolders
os.makedirs(temp_ensemble_path, exist_ok=True)
os.makedirs(temp_4to1_path, exist_ok=True)
os.makedirs(temp_path, exist_ok=True)

# choose prediction type (either resize or 4to1)
if args.resize:
    if ensemble:
        test_count = test_count * 8
        prepare_ensemble(test_predict_path, temp_ensemble_path)
        filenames = os.listdir(temp_ensemble_path)
        testGene = testGenerator(temp_ensemble_path, use_wavelet=args.wavelets)
    else:
        filenames = os.listdir(test_predict_path)
        testGene = testGenerator(test_predict_path, use_wavelet=args.wavelets)
else:
    if ensemble:
        raise Exception("ENSEMBLE WITH 4TO1 CURRENTLY NOT SUPPORTED")

    prepare_4to1data(test_predict_path, temp_4to1_path)
    filenames = os.listdir(temp_4to1_path)
    testGene = testGenerator(temp_4to1_path)
    test_count = test_count * 4

# load best model from training and predict results
model.load_weights(os.path.join(log_path, "unet_roadseg.hdf5"))
results = model.predict_generator(testGene, test_count, verbose=1)
post_results = np.where(results > 0.5, 1, 0)

# create all required output folders
output_path = os.path.join(test_output_path, args.desc)
os.makedirs(output_path, exist_ok=True)
output_path_4to1 = os.path.join(output_path, "split_results")
os.makedirs(output_path_4to1, exist_ok=True)
output_path_4to1_pre = os.path.join(output_path_4to1, "split_results_pre")
os.makedirs(output_path_4to1_pre, exist_ok=True)
output_path_pre = os.path.join(output_path, "pre_results")
os.makedirs(output_path_pre, exist_ok=True)
output_path_pre_ensembled = os.path.join(output_path, "pre_ensembled")
os.makedirs(output_path_pre_ensembled, exist_ok=True)
output_path_ensembled = os.path.join(output_path, "ensembled")
os.makedirs(output_path_ensembled, exist_ok=True)

if args.resize:
    if ensemble:
        saveResultunprocessed(output_path_pre_ensembled, results, filenames, nr_of_stacks=args.nr_of_stacks)
        ensemble_predictions(test_predict_path, output_path_pre_ensembled,
                             output_path_ensembled,
                             nr_of_stacks=args.nr_of_stacks)  # save into output_path_ensembled because need resize
        saveSubmitResizeEnsemble(temp_path, output_path_ensembled,
                                 output_path)  # resize imgs in output_path_ensembled to 608x608 and save
    else:
        savesubmitResult(temp_path, output_path, post_results, filenames, nr_of_stacks=args.nr_of_stacks)
        saveResultunprocessed(output_path_pre, results, filenames, nr_of_stacks=args.nr_of_stacks)

else:
    savesubmitResult_4to1version(output_path_4to1, post_results, filenames, nr_of_stacks=args.nr_of_stacks)
    saveResultunprocessed(output_path_4to1_pre, results, filenames, nr_of_stacks=args.nr_of_stacks)

    if args.combine_max:
        print("combining image using max")
        postprocess_4to1data_max(test_predict_path, output_path_4to1, output_path)
    else:
        print("combining image using average")
        postprocess_4to1data_avg(test_predict_path, output_path_4to1_pre, output_path)
