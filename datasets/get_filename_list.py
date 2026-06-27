import os

train_file_path = "Rider/train_npz"
train_filename_txt_path = "./train.txt"
test_file_path = "Rider/test_vol_h5"
test_filename_txt_path = "./test.txt"

train_list = sorted(os.listdir(train_file_path))
test_list = sorted(os.listdir(test_file_path))

with open(train_filename_txt_path, 'w') as f:
    for train_name in train_list:
        f.write(train_name.split(".")[0] + '\n')

with open(test_filename_txt_path, 'w') as f:
    for test_name in test_list:
        f.write(test_name.split(".")[0] + '\n')
