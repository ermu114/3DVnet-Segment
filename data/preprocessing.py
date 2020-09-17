import utility.data_processor as processor
import utility.parameters as param
import utility.iodicom as dcmio
import os
import cv2 as cv
import datetime

def get_cur_time():
    dt = datetime.datetime.now()
    timestr = dt.strftime('%H:%M:%S.%f')
    return timestr

def save_images(volume, path):
    if not os.path.exists(path):
        os.makedirs(path)
    slices = volume.shape[0]
    for i in range(slices):
        cv.imwrite(os.path.join(path, '%03d.jpg' % (i)), volume[i])
    pass

def set_ww_wl(volume_data, roi_name):
    ww = param.dict_supper_parameters[roi_name].ww
    wl = param.dict_supper_parameters[roi_name].wl
    result = dcmio.set_ww_wl(volume_data, ww, wl)
    return result

def image_resample(first_dataset_item, volume_data, new_thickness):
    slice_thickness = first_dataset_item.SliceThickness
    if slice_thickness != new_thickness:
        sitk_image = processor.create_itk_image(first_dataset_item, volume_data)
        volume_data_resampled = processor.image_resample(sitk_image, new_thickness)
    else:
        volume_data_resampled = volume_data
    return volume_data_resampled

def label_dilate(label_volume_data, se_size):
    if se_size != 0:
        result = processor.image_dilate(label_volume_data, se_size)
    else:
        result = label_volume_data
    return result

def get_one_image_volume_data(one_case_path, modality='CT'):
    _, list_dataset_image = dcmio.read_one_series(one_case_path, modality)
    image_volume_data = dcmio.get_image_volume_data_ctvalue(list_dataset_image)
    return image_volume_data, list_dataset_image

def get_one_label_volume_data(one_case_path, roi_name, list_dataset_image):
    _, dataset_rtstruct = dcmio.read_one_series(one_case_path, 'RTSTRUCT')
    label_volume_data = dcmio.get_label_volume_data(roi_name, list_dataset_image, dataset_rtstruct)
    return label_volume_data

def preprocessing_4_train_loc(image_volume_data, list_dicom_dataset, label_volume_data, roi_name):
    ## 1-image
    # 1.1-对image进行重采样，层厚默认为3mm
    first_dataset_item = list_dicom_dataset[0]
    image_volume = image_resample(first_dataset_item, image_volume_data, param.g_slice_thickness)

    # 1.2-对重采样后的数据由ct值，设置窗宽床位后，变换为0~055 8bit灰度图
    image_volume = set_ww_wl(image_volume, roi_name)

    # 2-label
    # 2.1-对label进行重采样，层厚默认为3mm
    label_volume = image_resample(first_dataset_item, label_volume_data, param.g_slice_thickness)

    # 2.2-对label进行膨胀操作，防止器官太小，缩小图像后label区域消失
    label_volume = label_dilate(label_volume, param.dict_supper_parameters[roi_name].se_size)

    return image_volume, label_volume


