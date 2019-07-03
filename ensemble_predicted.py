from data_loader.data import *
import argparse
import os

"""
Use this to combine pre_ensemble images to fully ensembled output submission images
Images are in data/submit_output/XXX/pre_ensembled/
"""

parser = argparse.ArgumentParser()
parser.add_argument('--desc', type=str, default='stacked_unet_default',
                    help='How to name this run, defines folder in logs dir, only use "a-z,A-Z,1-9,_" pls')
parser.add_argument('--cutoff', type=float, default=0.5, help='Cutoff value for ensembling data.')
parser.add_argument('--nr', type=float, default=0.5, help='Cutoff value for ensembling data.')
parser.add_argument('--nr_of_stacks', type=int, default=2,
                    help='number of the stack to ensemble and get final results on')
args = parser.parse_args()

# Set parameters
cutoff = args.cutoff
nr = args.nr

# Create paths
test_output_path = "data/submit_output"
output_path = os.path.join(test_output_path, args.desc)
test_predict_path = "data/test"
temp_path = "data/temp"
temp_path_2 = "data/temp_2"

# Create dirs
os.makedirs(temp_path, exist_ok=True)
os.makedirs(temp_path_2, exist_ok=True)

# Should be full with images
output_path_pre_ensembled = os.path.join(output_path, "pre_ensembled")

output_path_ensembled = os.path.join(output_path, "ensembled_{0}".format(nr))
os.makedirs(output_path_ensembled, exist_ok=True)

# create ensemble predictions
ensemble_predictions(test_predict_path, output_path_pre_ensembled,
                     temp_path_2, nr_of_stacks=args.nr_of_stacks,
                     alpha=cutoff)  # save into output_path_ensembled because need resize

# resize ensembled predictions and save them
saveSubmitResizeEnsemble(temp_path, temp_path_2, output_path_ensembled)
