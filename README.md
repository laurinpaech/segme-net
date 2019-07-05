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
     ├── hough_transform.py         - TODO
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

1. first load env on leonhard
    
    module load gcc/4.8.5 python_gpu/3.6.4 hdf5 eth_proxy
    module load cudnn/7.0

2.. then run (estimated runtime 8 hours on a "GeForce GTX 1080 Ti")

    python main.py --desc "reproducing_best_result" --epochs 1000 --rotation 360 --width_shift_range 50 --height_shift_range 50 --shear_range 10 --zoom_range 0.1 --horizontal_flip --fill_mode "reflect" --nr_of_stacks 2 --resize --ensemble

3. result will be located in data/roadseg/submit_output/reproducing_best_result
