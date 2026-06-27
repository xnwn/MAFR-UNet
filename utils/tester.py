import logging
import numpy as np
import sys
import torch

from medpy import metric
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm
import SimpleITK as sitk
import cv2
from scipy.ndimage import distance_transform_edt

def add_gaussian_noise(img, sigma=0.03):
    noise = np.random.normal(0, sigma, img.shape)
    noise_img = img + noise
    return np.clip(noise_img, 0, 1)

def reduce_contrast(img, alpha=0.5):
    mean = np.mean(img)
    out = alpha * (img - mean) + mean
    return np.clip(out, 0, 1)

def motion_blur(img, kernel_size):
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[kernel_size // 2, :] = np.ones(kernel_size)
    kernel = kernel / kernel_size
    return cv2.filter2D(img, -1, kernel)

def mask_to_boundary(mask, dilation_ratio=0.02):
    """
    Convert binary mask to boundary mask.
    :param mask (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary mask (numpy array)
    """
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1
    # Pad image so mask truncated by the image border is also considered as boundary.
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1: h + 1, 1: w + 1]
    # G_d intersects G in the paper.
    return mask - mask_erode


def test(args, model, dataset, dataloader):
    model.eval()

    all_metric_list = 0.0
    img_size = args.img_size

    logging.basicConfig(filename=f"{args.log_dir}/log.txt", level=logging.INFO,
                        format="[%(asctime)s.%(msecs)03d] %(message)s", datefmt="%H:%M:%S")
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    for _, sampled_batch in enumerate(dataloader):
        image, label, filename = sampled_batch["image"], sampled_batch["label"], sampled_batch["filename"][0]

        # B C H W -> C H W
        image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()

        prediction = np.zeros_like(label)

        for slice_index in range(image.shape[0]):
            img_slice = image[slice_index, :, :]
            height, width = img_slice.shape[0], img_slice.shape[1]
            if height != img_size or width != img_size:
                img_slice = zoom(img_slice, (img_size / height, img_size / width), order=3)
            # H W -> 1 1 H W
            model_input = img_slice
            # model_input = add_gaussian_noise(img_slice, 0.02)
            # model_input = reduce_contrast(img_slice, 0.9)
            # model_input = motion_blur(img_slice, 9)
            model_input = torch.from_numpy(model_input).unsqueeze(0).unsqueeze(0).float().cuda()

            if model_input.size()[1] == 1:
                model_input = model_input.repeat(1, 3, 1, 1)

            with torch.no_grad():
                # 1 3 H W -> 1 num_classes H W
                outputs = model(model_input)
                # After normalizing the elements at the same position in each channel, select the maximum value
                # 1 num_classes H W -> H W
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if height != img_size or width != img_size:
                    predict = zoom(out, (height / img_size, width / img_size), order=0)
                else:
                    predict = out
                prediction[slice_index] = predict

        # image_sitk = sitk.GetImageFromArray(image)
        # label_sitk = sitk.GetImageFromArray(label)
        # pred_sitk = sitk.GetImageFromArray(prediction)
        # sitk.WriteImage(image_sitk, "./image.nii.gz")
        # sitk.WriteImage(label_sitk, "./label.nii.gz")
        # sitk.WriteImage(pred_sitk, "./teacher.nii.gz")
        # print("save finish")
        metric_list = []
        for i in range(1, args.num_classes):
            # True if it belongs to the current class, False otherwise
            predict = prediction == i
            ground_truth = label == i
            # Exclude background
            predict[predict > 0] = 1
            ground_truth[ground_truth > 0] = 1

            smooth = 1e-5
            boundary_IOU = 0
            BF1 = 0

            for i in range(predict.squeeze().shape[0]):
                pred_boundary = mask_to_boundary(np.uint8(predict[i].squeeze()))
                gt_boundary = mask_to_boundary(np.uint8(ground_truth[i].squeeze()))

                boundary_inter = np.sum(pred_boundary * gt_boundary)
                boundary_union = np.sum(pred_boundary + gt_boundary) - boundary_inter
                boundary_IOU += (boundary_inter + smooth) / (boundary_union + smooth) / predict.squeeze().shape[0]

                n_pred = pred_boundary.sum()
                n_gt = gt_boundary.sum()
                if n_pred == 0 and n_gt == 0:
                    BF1 = 1.0
                if n_pred == 0 or n_gt == 0:
                    BF1 = 0.0
                pred_dist = distance_transform_edt(1 - pred_boundary)
                gt_dist = distance_transform_edt(1 - gt_boundary)
                # tolerance = 2
                pred_match = gt_dist[pred_boundary > 0] <= 2

                gt_match = pred_dist[gt_boundary > 0] <= 2
                precision = pred_match.sum() / (n_pred + smooth)
                recall = gt_match.sum() / (n_gt + smooth)
                BF1 += (2 * precision * recall) / (precision + recall + smooth) / predict.squeeze().shape[0]

            # If there is a predicted value, calculate the indicator
            if predict.sum() > 0 and ground_truth.sum() > 0:
                metrics = (metric.binary.dc(predict, ground_truth),
                           metric.binary.hd95(predict, ground_truth),
                           boundary_IOU, BF1)
            elif predict.sum() > 0 and ground_truth.sum() == 0:
                # Predicted value exists, but not actually exists
                metrics = (1, 0, boundary_IOU, BF1)
            else:
                # Prediction is background, actual is background
                metrics = (0, 0, boundary_IOU, BF1)
            metric_list.append(metrics)
        # Add the corresponding elements
        all_metric_list += np.array(metric_list)
        # Calculate mean by column
        logging.info(f"{filename} || Mean DSC: %f || Mean HD: %f || Mean BIoU: %f || Mean BF1: %f" % (
            np.mean(metric_list, axis=0)[0], np.mean(metric_list, axis=0)[1],
            np.mean(metric_list, axis=0)[2], np.mean(metric_list, axis=0)[3]))
    metric_list = all_metric_list / len(dataset)
    print()

    for i in range(1, args.num_classes):
        logging.info(f"Class {i} || Mean DSC: %f || Mean HD: %f || Mean BIoU: %f || Mean BF1: %f" %
                     (metric_list[i - 1][0], metric_list[i - 1][1],
                      metric_list[i - 1][2], metric_list[i - 1][3]))
    print()

    logging.info(
        "Total || Mean DSC: %f || Mean HD: %f || Mean BIoU: %f || Mean BF1: %f" %
        (np.mean(metric_list, axis=0)[0], np.mean(metric_list, axis=0)[1],
         np.mean(metric_list, axis=0)[2], np.mean(metric_list, axis=0)[3]))
    print()
