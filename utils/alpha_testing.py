from data_loader.data_stacked_unet import *
from keras.callbacks import TensorBoard
import argparse
import skimage.io as io

# this part is used for argument handling
parser = argparse.ArgumentParser()
parser.add_argument('--desc', type=str, default = 'stacked_unet_default_desc',
                    help='How to name this run, defines folder in logs dir, only use "a-z,A-Z,1-9,_" pls')
parser.add_argument('--nr_of_stacks', type=int, default = 2,
                    help='points outside the boundaries of the input are filled according to the given mode, '
                         'standard is nearest')
parser.add_argument('--ensemble', default=False, action='store_true',
                    help='predict 8 versions of image with rotations and flipping, and recombine them later')
parser.add_argument('--leakyRelu', default=False, action='store_true',
                    help='choose if unet should use leaky Relu')


args = parser.parse_args()
print(args)
# end of argument handling



# load correct unet model
if(args.leakyRelu):
    from model.model_stacked_unet_leaky import *
else:
    from model.model_stacked_unet import *


log_path=os.path.join("./logs/",args.desc)
tensorboard = TensorBoard(log_dir=log_path, histogram_freq=0,
                          write_graph=True, write_images=False)

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

### Initial/Default parameter if none are passed

ensemble = args.ensemble

train_path = "data/roadseg/train"
test_predict_path = "data/roadseg/test"
test_output_path = "data/roadseg/submit_output"
valid_path = "data/roadseg/valid"
valid_predict_path = "data/roadseg/valid/image"
valid_predict_label = "data/roadseg/valid/label"
valid_output_path="data/roadseg/valid/output"
temp_4to1_path = "data/roadseg/temp_4to1_folder"
temp_path = "data/roadseg/temp"
temp_ensemble_path = "data/roadseg/temp_ensemble"
valid_count=100

model = unet(nr_of_stacks=args.nr_of_stacks)

######
# if necessary, create tempfolders
os.makedirs(temp_ensemble_path, exist_ok=True)
os.makedirs(temp_4to1_path, exist_ok=True)
os.makedirs(temp_path, exist_ok=True)

# first we predict and save
if (ensemble):
    valid_count = valid_count * 8
    prepare_ensemble(valid_predict_path, temp_ensemble_path)
    filenames = os.listdir(temp_ensemble_path)
    testGene = testGenerator(temp_ensemble_path)
else:
    filenames = os.listdir(valid_predict_path)
    testGene = testGenerator(valid_predict_path)

# load best model from training and predict results
model.load_weights(os.path.join(log_path,"unet_roadseg.hdf5"))
results = model.predict_generator(testGene,valid_count,verbose=1)


# now we test different alpha parameters
with tf.Session() as sess:
    best=0.
    best_alpha=0

    # test 50 values between 0 and 1
    for current_alpha in np.linspace(0,1,50):
        print("current alpha "+str(current_alpha))
        post_results = np.where(results > current_alpha, 1, 0)

        # load ground truth of validation data
        gt_arr = []
        for index,item in enumerate(filenames):
            mask = io.imread(os.path.join(valid_predict_label, item), as_gray = True)
            gt_arr.append(mask)

        result=real_kaggle_metric(tf.cast(np.expand_dims(gt_arr,axis=3), tf.float32), tf.cast(np.expand_dims(post_results[:, :, :, args.nr_of_stacks - 1],axis=3), tf.float32), current_alpha)
        res_out=sess.run(1-result)
        print(res_out)
        if(best<res_out):
            best=res_out
            best_alpha=current_alpha

    print("final verdict: best out: " + str(best) + " with alpha: " + str(best_alpha))