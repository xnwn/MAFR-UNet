## MAFR-UNet

Official PyTorch implementation of the paper "MAFR-UNet: Multi-scale Adaptive Feature Reassembly network for aortic CTA segmentation".

### Environment

Please prepare an environment with python=3.8.20, and then use the command `pip install -r requirements.txt` for the dependencies.

### Prepare data

Please download the datasets from the following links:

AVT - https://doi.org/10.6084/m9.figshare.14806362.

Synapse - https://doi.org/10.6084/m9.figshare.31324045. 

ACDC - https://doi.org/10.6084/m9.figshare.31324297.

Among them, the AVT dataset requires preprocessing using the preprocessing scripts provided in the `datasets` folder before training and evaluation.

### Download pre-trained weights

Please click the following link to obtain the pre-trained weights and store them in the `pretrained_ckpt` folder.

https://drive.google.com/drive/folders/18NNUdd2g6SzMKd_KZqZIXepZzzYR5Usb?usp=drive_link
### Train/Test

Please adjust the data storage addresses in run_train.sh and run_test.sh, and then execute `sh run_train.sh` and `sh run_test.sh` in sequence to conduct model training and testing.