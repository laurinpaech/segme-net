# segme-net
Road Segmentation Project for Computational Intelligence Lab 2019


## Folder structure

```
├──  data
|   |
|   |── train                       - all 100 training images given for the project
|   |
|   |── valid                       - validation images generated from chicago dataset
|   |   
│   |── test                        - all test images given for the project
|   |
|   |── train_600                   - training images + additional 600 generated images
|   |
|   └── train_6000                       - training images + additional 6000 generated images
│
│
├── model                               - this folder contains any model of our project.
|   |
|   |── encoder_decoder.py                  - keras implementation of simple encoder decoder
|   |
|   |── segnet.py                           - keras implementation of SegNet
|   |
|   |── stacked_unet.py                     - keras implementation of stacked unet
|   |
|   |── stacked_unet_leaky.py               - keras implementation of stacked unet + leaky relu
|   |
|   |── stacked_unet_leaky_wavelet.py       - keras implementation of stacked unet + leaky relu + wavelets
|   |
│   └── stacked_unet_leaky_wavelet_2.py     - keras implementation of stacked unet + leaky relu + wavelets
│
│
├── main.py                        - main that is responsible for the whole pipeline
│ 
│  
├── data _loader
|   | 
│   └── data.py                 - data generator that is responsible for all data handling
│ 
└── utils
     |
     ├── overlay_generator
     |   |
     |   ├── normal_img             - contains original images for that we want to generate overlays of predictions
     |   |
     |   ├── submit_img             - contains images that are used as overlay for normal_img
     |   |
     |   └── overlay_generator.py   - generates 
     |
     ├── alpha_testing.py           - used for testing best cut-off value for our predictions
     |
     ├── custom_losses.py           - custom loss functions
     |
     ├── custom_layers.py           - custom layers for segnet
     |
     ├── hough_transform.py         - Probabilistic Hough Transform to fill gaps in results
     |
     ├── mask_to_submission.py
     |
     ├── submission_to_mask.py
     |
     └── image_generation.py        - used for generation additional images as data
```

## Getting started

Use python 3.6 and run the following command:
```
pip install -r requirements.txt
```

## Stacked U-Net, using Keras

The architecture was inspired by [U-Net: Convolutional Networks for Biomedical Image Segmentation](http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/).

### How to use

#### Running a model
Notes:
-   sometimes gets stuck from beginning on a loss of ~0.6, then you got to restart

1. first load env on leonhard
    
  
    module load gcc/4.8.5 python_gpu/3.6.4 hdf5 eth_proxy
    
    module load cudnn/7.0

2.. then run (currently really fast, ca. 10s/epoch, so 4 hours is easily enough)

    bsub -n 4 -W 4:00 -R "rusage[mem=2048, ngpus_excl_p=1]" python main_cil.py --desc "my_test_model" \
                     --epochs 300 --rotation 360 --width_shift_range 0.1 --height_shift_range 0.1 \
                     --shear_range 0 --zoom_range 0 --horizontal_flip --fill_mode "reflect" \
                     --resize

check progress with (note, after each epoch, also calculates valid-loss)

    bpeek -f
predictions for validation set in data/valid/output

log folder contains tensorboard files, download to own machine and look at with 
    
    tensorboard --logdir ./logs

#### How to create submission
1. run the model
2. predictions are in data/submit_output/
3. run mask_to_submission.py on output folder
4. next_submission.csv in folder can now be uploaded to kaggle

#### Reproducing Kaggle results

run the following command:

```
python main.py --desc "stacked_unet_2stack" --epochs 1000 --rotation 360 --width_shift_range 0 --height_shift_range 0 --shear_range 0 \
--zoom_range 0 --horizontal_flip --fill_mode "reflect" --nr_of_stacks 2 --resize --ensemble
```

This runs the Stacked U-Net on the 100 training images and saves the result in `data/submit_output/stacked_unet_2stack/`.
