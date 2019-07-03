# segme-net
Road Segmentation Project for Computational Intelligence Lab 2019

## Folder structure

data
- test
- 100 training
- validation_gen?
- 600 training
- 10k training



```
├──  data
│   ├── roadseg
|   |       ├── submit_test
|   |       ├── submit_train
|   |       ├── train
|   |       └── valid
|   |   
│   |── test_images                 - all test images given for the project
|   |
|   └── training                    - all training images (groundtruth labels and images) given for the project
│
│
├── model                           - this folder contains any model of our project.
│   └── unet.py                     - keras implementation of unet
│
│
├──  main.py                        - main that is responsible for the whole pipeline. (formerly main_cil.py)
│ 
│  
├──  data _loader  
│    └── data_generator.py          - data_generator that is responsible for all data handling (formerly data_cil.py)
│ 
└── utils
     ├── mask_to_submission.py
     ├── submission_to_mask.py
     └── sample_submission.csv
```

## Implementation of U-Net, using Keras

The architecture was inspired by [U-Net: Convolutional Networks for Biomedical Image Segmentation](http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/).


### Intro

- does perturbation of images during runtime, perturbations are currently defined in main in data_gen_args. 
- It's fast, maybe 1-2min per epoch
- i manually split the training data into train and valid (90 images/ 10 images). In submission mode, uses all 100 images for training

#### stuff to test/explore

- how they calculate loss on kaggle, and maybe write a loss which is similar
- weighted loss, i.e. roads are valued higher in loss
- leaky relu (might help)
- changes on a model level, haven't tested anything
- impact of perturbations, what helps, what doesn't

### How to use

#### How to train a model
Notes:
-   sometimes gets stuck from beginning on a loss of ~0.6, then you got to restart

1. first load env on leonhard
    
  
    module load gcc/4.8.5 python_gpu/3.6.4 hdf5 eth_proxy
    
    module load cudnn/7.0

2.. then run (currently really fast, ca. 10s/epoch, so 4 hours is easily enough)

    bsub -n 4 -W 4:00 -R "rusage[mem=2048, ngpus_excl_p=1]" python main_cil.py --desc "my_test_model" \
                     --epochs 300 --rotation 360 --width_shift_range 0.1 --height_shift_range 0.1 \
                     --shear_range 0.1 --zoom_range 0.1 --horizontal_flip=True --fill_mode "reflect" \
                     --resize=True --submission=False

check progress with (note, after each epoch, also calculates valid-loss)

    bpeek -f
predictions for validation set in data/valid/output

log folder contains tensorboard files, download to own machine and look at with 
    
    tensorboard --logdir ./logs

#### How to create submission
1. set submission_flag to True in main_cil.py
2. do step 1 and 2 from above, but add flag "--submission_flag True"
3. predictions in data/roadseg/submit_output/
4. copy mask_to_submission.py to output folder, switch to folder and run
5. next_submission.csv in folder can now be uploaded to kaggle

## Overview Files

#### main_cil

run everything from here, including switch between training and submission
must parameters can be changed here using basic keras features.

#### data_cil

a couple helper functions, in general don't have to be touched

#### model_cil

i haven't touched this, literally a copy from the git-code i copied.

## Results

