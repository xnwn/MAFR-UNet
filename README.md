# MAFR-UNet

### Environment

>python=3.8.20  pytorch=1.3.1 CUDA=11.7 RandomSeed=42
>

Please prepare an environment with python=3.8.20, and then use the command `pip install -r requirements.txt` for the dependencies.

### Prepare data

Please click the following link to obtain the experimental dataset. Then, use the data_preprocess.py file in the utils folder to perform data preprocessing.

AVT - https://doi.org/10.6084/m9.figshare.14806362.

Synapse - https://drive.google.com/drive/folders/1ACJEoTp-uqfFJ73qS3eUObQh52nGuzCd. 

ACDC - https://drive.google.com/drive/folders/1KQcrci7aKsYZi1hQoZ3T3QUtcy7b--n4.

### Train/Test

Please adjust the data storage addresses in run_train.sh and run_test.sh, and then execute `sh run_train.sh` and `sh run_test.sh` in sequence to conduct model training and testing.