#pytorch-image-quality-param-ctrl

PyTorch implementation of
* DeepBIQ(https://arxiv.org/pdf/1602.05531.pdf)
* use DeepBIQ and ppo(https://github.com/ikostrikov/pytorch-a2c-ppo-acktr) to search usb camera image quality param(brightness, contrast, saturation, sharpness)

## Contributions
Contributions are very welcome. If you know how to make this code better, don't hesitate to send a pull request.

## Requirements
* Ubuntu 14.04
* usb camera (must support 640 * 480 resolution)
* python3.6
* [PyTorch](http://pytorch.org/)
* GPU
* LIVE In the Wild Image Quality Challenge Database (ChallengeDB_release, it must locate in: pytorch-image-quality-param-ctrl/deepbiq/ChallengeDB_release)
* if you want ppo to search good image quality param, you should add more sample to ChallengeDB_release, 
    for example: very dark, very bright, very saturated images and so on 
    to train DeepBIQ model

##warning
you should save your usb camera default quality param
```commandline
v4l2-ctl --get-ctrl brightness
v4l2-ctl --get-ctrl contrast
v4l2-ctl --get-ctrl saturation
v4l2-ctl --get-ctrl sharpness
```
maybe you have to install v4l2-ctl
```commandline
sudo apt-get install v4l2-ctl
```

## Training DeepBIQ
```commandline
cd deepbiq/
sh train.sh
```

## use ppo to Search Image Quality Param
```commandline
cd ppo/
sh search.sh
```
if occurred error like：set brightness=106 failed:Input/output error ;
you have to power up again usb camera