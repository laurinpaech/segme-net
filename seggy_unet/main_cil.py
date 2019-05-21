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

args = parser.parse_args()
# end of argument handling


log_path=os.path.join("./logs/",args.desc)
tensorboard = TensorBoard(log_dir=log_path, histogram_freq=0,
                          write_graph=True, write_images=False)

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"






submission_flag = args.submission
nr_of_epochs = args.epochs

if not submission_flag:
    train_path = "data/roadseg/train"
    valid_path = "data/roadseg/valid"
    predict_path = "data/roadseg/valid/image"
    output_path = "data/roadseg/valid/output"
    temp_path = "data/roadseg/temp"
    count=10
else:
    train_path = "data/roadseg/submit_train"
    valid_path = "data/roadseg/valid"
    predict_path = "data/roadseg/submit_test"
    output_path = "data/roadseg/submit_output"
    temp_path = "data/roadseg/temp"
    count=94


data_gen_args = dict(rotation_range=args.rotation,
                    width_shift_range=args.width_shift_range,
                    height_shift_range=args.height_shift_range,
                    shear_range=args.shear_range,
                    zoom_range=args.zoom_range,
                    horizontal_flip=args.horizontal_flip,
                    fill_mode=args.fill_mode)

trainGen = trainGenerator(2,train_path,'image', 'label',data_gen_args,save_to_dir = None)

if(not submission_flag):
    validGen = trainGenerator(2,valid_path,'image', 'label',data_gen_args,save_to_dir = None)

model = unet()
model_checkpoint_train = ModelCheckpoint('unet_roadseg.hdf5', monitor='val_acc',verbose=1, save_best_only=True)
model_checkpoint_submit = ModelCheckpoint('unet_roadseg.hdf5', monitor='acc',verbose=1, save_best_only=True)


if(not submission_flag):
    model.fit_generator(trainGen, steps_per_epoch=90, epochs=nr_of_epochs, callbacks=[model_checkpoint_train, tensorboard],
                        validation_data=validGen, validation_steps=10)
else:
    model.fit_generator(trainGen, steps_per_epoch=100, epochs=nr_of_epochs, callbacks=[model_checkpoint_submit, tensorboard])

filenames = os.listdir(predict_path)
testGene = testGenerator(predict_path)
# load best model from training and predict results
model.load_weights("./unet_roadseg.hdf5")
results = model.predict_generator(testGene,count,verbose=1)
post_results = np.where(results > 0.5, 1, 0)

output_path=os.path.join(output_path,args.desc)
os.mkdir(output_path)

output_path_pre=os.path.join(output_path,"pre_results")
os.mkdir(output_path_pre)

if(not submission_flag):
    saveResult(output_path, post_results, filenames)
    saveResultunprocessed(output_path_pre, results, filenames)
else:
    savesubmitResult(temp_path, output_path, post_results, filenames)
    saveResultunprocessed(output_path_pre, results, filenames)
