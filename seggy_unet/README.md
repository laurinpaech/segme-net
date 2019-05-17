# Implementation of deep learning framework -- Unet, using Keras

The architecture was inspired by [U-Net: Convolutional Networks for Biomedical Image Segmentation](http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/).

---

## Intro

- does perturbation of images during runtime, perturbations are currently defined in mail_cil in data_gen_args. 
- It's fast, maybe 1-2min per epoch
- i manually split the training data into train and valid (90 images/ 10 images). In submission mode, uses all 100 images for training

### stuff to test/explore

- how they calculate loss on kaggle, and maybe write a loss which is similar
- weighted loss, i.e. roads are valued higher in loss
- leaky relu (might help)
- changes on a model level, haven't tested anything
- impact of perturbations, what helps, what doesn't
## How to use

### How to train a model


1. first load env on leonhard


    module load gcc/4.8.5 python_gpu/3.6.4 hdf5 eth_proxy
    module load cudnn/7.0
2.. then run (currently really fast, so 4 hours is easily enough)


    bsub -n 4 -W 4:00 -R "rusage[mem=2048, ngpus_excl_p=1]" python main_cil.py
check progress with (note, after each epoch, also calculates valid-loss)

    bpeek -f
predictions for validation set in data/valid/output

log folder contains tensorboard files, download to own machine and look at with 
    
    tensorboard --logdir ./logs

### How to create submission
1. set submission_flag to True in main_cil.py
2. do step 1 and 2 from above
3. predictions in data/roadseg/submit_output/
4. copy mask_to_submission.py to output folder, switch to folder and run
5. next_submission.csv in folder can now be uploaded to kaggle

## Overview Files

### main_cil

run everything from here, including switch between training and submission
must parameters can be changed here using basic keras features.

### data_cil

a couple helper functions, in general don't have to be touched

### model_cil

i haven't touched this, literally a copy from the git-code i copied.

### Results
