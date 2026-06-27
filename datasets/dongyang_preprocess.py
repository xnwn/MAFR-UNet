import h5py
import nibabel as nib
import numpy as np
import os
import torch
import torch.nn.functional as F
import vtk


def nrrd2nii(filename):
    # Create a nrrd reader
    reader = vtk.vtkNrrdReader()
    # Specify the file to be read
    reader.SetFileName(filename)
    # Update the reader information
    reader.Update()
    # Obtain the output content
    image = reader.GetOutput()
    # Obtain information related to the output content
    info = reader.GetInformation()

    # Create a nii writer
    writer = vtk.vtkNIFTIImageWriter()
    # Specify the input object
    writer.SetInputData(image)
    # Specify the output file path
    writer.SetFileName(filename.replace(".nrrd", ".nii"))
    # Specify the output file information
    writer.SetInformation(info)
    # Start writing
    writer.Write()


def convert_files(root_dir, prefix, folder_num):
    for i in range(folder_num):
        nrrd = f"{root_dir}/{prefix}{i + 1}/{prefix}{i + 1}.nrrd"
        nrrd2nii(nrrd)
        # seg.nrrd stores the value of the split category, and the following warning will pop up when reading
        # Unknown field: 'Segmentation_ConversionParameters:=Collapse labelmaps ...
        # This will render the transform performed on seg.nrrd invalid.
        # Therefore, any operation should be avoided before the transformation
        nrrd2nii(nrrd.replace(".nrrd", ".seg.nrrd"))

        print(f"{prefix}{i + 1} convert finish")
    print()


def get_hu_range(image, percent=0.0001):
    # Flatten the image
    image_reshape = np.reshape(image, (1, image.shape[0] * image.shape[1] * image.shape[2]))
    # Convert to a list for easier sorting
    image_hu_list = image_reshape.tolist()[0]
    # Sort by the pixel values of the image
    image_hu_list.sort()
    # Exclude some values proportionally
    min_num = round(len(image_hu_list) * percent)
    max_num = round(len(image_hu_list) * 1 - percent)

    return image_hu_list[min_num], image_hu_list[max_num - 1]


def resample(data, pixdim, new_pixdim=None):
    if new_pixdim is None:
        new_pixdim = [1, 1, 1]

    # Calculate the size of the output data
    size_x = int(data.shape[0] * pixdim[0] / new_pixdim[0])
    size_y = int(data.shape[1] * pixdim[1] / new_pixdim[1])
    size_z = int(data.shape[2] * pixdim[2] / new_pixdim[2])

    # In order to be able to interpolate, convert the data into a tensor and add two dimensions
    data = torch.tensor(data).unsqueeze(0).unsqueeze(0)
    # The pixel value type of.seg.nii is uint8. We will unify it to float32 and restore it later
    data = data.to(torch.float32)
    # Sampling is carried out using trilinear interpolation
    data = F.interpolate(data, size=[size_x, size_y, size_z], mode='trilinear', align_corners=False)

    return data[0, 0, :, :, :]


def file_process(root_dir, prefix, folder_num, all_hu=True, hu_idx=None, slope=1, intercept=-1024):
    for i in range(folder_num):
        # Read the nii file
        image = nib.load(f"{root_dir}/{prefix}{i + 1}/{prefix}{i + 1}.nii")
        label = nib.load(f"{root_dir}/{prefix}{i + 1}/{prefix}{i + 1}.seg.nii")
        # Extract the data as a numpy array
        image_data = np.asarray(image.dataobj)
        label_data = np.asarray(label.dataobj)
        # Read the basic information of the nii file
        image_header = image.header
        label_header = label.header

        # Convert the gray value to the HU value
        if not all_hu and i not in hu_idx:
            image_data = image_data * slope + intercept

        # Extract the maximum and minimum values of pixels for normalization
        # Some samples have a small number of outliers, and we excluded them proportionally
        hu_min, hu_max = get_hu_range(image_data, 0.0001)

        # Extract pixdim for resampling
        pixdim = [image_header['pixdim'][1], image_header['pixdim'][2], image_header['pixdim'][3]]
        # Resampling to 1mm * 1mm * 1mm
        new_pixdim = [1, 1, 1]
        image_data = resample(image_data, pixdim, new_pixdim)
        label_data = resample(label_data, pixdim, new_pixdim)
        label_data = label_data.to(torch.uint8)
        # Record the file information after sampling for subsequent preservation
        scaling_affine = np.diag([new_pixdim[0], new_pixdim[1], new_pixdim[2], 1])

        # Set the range of HU values for adjusting the image
        # Note: In fact, after the previous work, the pixel values of most samples have already fallen within this range
        hu_range = [-1024, 3071]
        # Limit the HU value of the image
        clamp_min = hu_range[0] if hu_min <= hu_range[0] else hu_min
        clamp_max = hu_range[1] if hu_max >= hu_range[1] else hu_max
        image_data = torch.clamp(image_data, min=clamp_min, max=clamp_max)
        image_data = (image_data - clamp_min) / (clamp_max - clamp_min)
        # Save the image
        image = nib.Nifti1Image(image_data, scaling_affine, image_header)
        label = nib.Nifti1Image(label_data, scaling_affine, label_header)
        nib.save(image, f"{root_dir}/{prefix}{i + 1}/{prefix}{i + 1}_new.nii")
        nib.save(label, f"{root_dir}/{prefix}{i + 1}/{prefix}{i + 1}_new.seg.nii")

        print(f"{prefix}{i + 1} process finish")
    print()


def nii2npy_npz(root_dir, out_dir, prefix, folder_num, test_idx):
    # Create a save directory
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    for i in range(folder_num):
        # Read the nii file
        image = nib.load(f"{root_dir}/{prefix}{i + 1}/{prefix}{i + 1}_new.nii")
        image_data = np.asarray(image.dataobj)

        label = nib.load(f"{root_dir}/{prefix}{i + 1}/{prefix}{i + 1}_new.seg.nii")
        label_data = np.asarray(label.dataobj)
        # If the shapes of image and label are inconsistent, it indicates that there was an error in the previous work
        assert image_data.shape == label_data.shape, "image shape is not equal to label shape"

        # Test dataset
        if i + 1 in test_idx:
            # Create a directory for storing test sets
            h5_dir = f"{out_dir}/test_vol_h5"
            if not os.path.exists(h5_dir):
                os.makedirs(h5_dir, exist_ok=True)
            image_slices = []
            label_slices = []
            for slice_idx in range(image_data.shape[-1]):
                # Remove slices without labels
                if np.count_nonzero(label_data[:, :, slice_idx]) == 0:
                    continue
                image_slices.append(np.float32(image_data[:, :, slice_idx]))
                label_slices.append(label_data[:, :, slice_idx])
            with h5py.File(f"{h5_dir}/{prefix}{i + 1}.npy.h5", "w") as hf:
                hf.create_dataset("image", data=image_slices)
                hf.create_dataset("label", data=label_slices)
                hf.close()
        # Train dataset
        else:
            # Create a directory for storing train sets
            npz_dir = f"{out_dir}/train_npz"
            if not os.path.exists(npz_dir):
                os.makedirs(npz_dir, exist_ok=True)
            # Save each training sample in sequence
            for slice_idx in range(image_data.shape[-1]):
                # Remove slices without labels
                if np.count_nonzero(label_data[:, :, slice_idx]) == 0:
                    continue
                image_slice = np.float32(image_data[:, :, slice_idx])
                label_slice = label_data[:, :, slice_idx]
                name_format = "{:04d}".format(slice_idx + 1)
                np.savez(f"{npz_dir}/{prefix}{i + 1}_slice{name_format}.npz", image=image_slice, label=label_slice)
        print(f"{prefix}{i + 1} to npz/h5 finish")
    print()


if __name__ == "__main__":
    '''Convert all nrrd files to nii files
       
        KiTS K1 ~ K20 || Dongyang D1 ~ D18 || Rider R1 ~ R18

    '''
    # Dongyang_origin is the folder where Dongyang.zip is stored after decompression
    # convert_files("Dongyang_origin", "D", 18)

    '''Dongyang preprocess
        Slice thickness(mm): x * x * 2 ~ y * y * 3
        To standardize the data, we resampled it to 1 * 1 * 1
    
        Pixel value: D3 ranges from -1024 to 6484, D17 ranges from -1024 to 14542,
                     the rest range from -1024 to 2576 ~ 3071
        The HU value range of common CT images: -1024 ~ 3071
        The pixel value range of common CT images: 0 ~ 4095
        The samples in this dataset basically conform to the HU value range, excluding only outliers
        In fact, the HU value of the target area of most samples is slightly higher, 
        but in order to simplify the operation, we will not process it.    
        
        Abdominal CT window width(HU): 300 ~ 500
        Abdominal CT window level(HU): 30 ~ 50
        In order to be compatible with our private dataset, we have decided not to extract image regions
        based on the above range, although it may improve the segmentation performance of the model
    
        Most of the work normalizes the image range. We follow this strategy and normalize the pixel values to 0 ~ 1.
    
        After resampling, the size of the image changes and can be uniformly restored to the initial size of 512 * 666
        However, since the input image size of the model is 224 * 224, we will not make adjustments here
        Instead, it is accomplished by the model data preprocessing code,
        and only the size needs to be guaranteed to be no less than 224 (All samples meet the requirements)
        
        We convert the processed nii file into a 2D slice for model training
        The training set is saved in npz format and the test set is saved in npy.h5
        Divide the training set and the test set in a 7:3 ratio
        Generate the sample numbers of the test set using random numbers
        The code we use is as follows:
            import random
            
            idx_list = random.sample(range(1, 18), 6)
            print(idx_list)  # [15, 4, 1, 2, 13, 10]
    '''
    # file_process("Dongyang_origin", "D", 18, True, [], 1, 0)
    nii2npy_npz("Dongyang_origin", "./Dongyang", "D", 18, [15, 4, 1, 2, 13, 10])
