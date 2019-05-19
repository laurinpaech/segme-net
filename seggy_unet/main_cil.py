from model_cil import *
from data_cil import *
import os
from keras.callbacks import TensorBoard

tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0,
                          write_graph=True, write_images=False)

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

submission_flag = False
nr_of_epochs = 10

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


data_gen_args = dict(rotation_range=45,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

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

if(not submission_flag):
    saveResult(output_path, post_results, filenames)
    saveResult(temp_path, results, filenames)
else:
    savesubmitResult(temp_path, output_path, post_results, filenames)
    saveResult(temp_path, results, filenames)
