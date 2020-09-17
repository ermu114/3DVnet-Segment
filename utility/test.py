import nibabel as nib
import numpy as np
import utility.iodicom as dcmio
import time
import predict.predictor as prdct

#import vtk

def test_tmp():
    t3 = time.clock()
    img = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 2, 3, 0, 0, 0],
                    [0, 3, 3, 3, 3, 0, 0, 0],
                    [0, 3, 0, 3, 3, 3, 0, 0],
                    [3, 3, 3, 0, 0, 3, 0, 0],
                    [0, 0, 4, 4, 4, 4, 0, 0]])
    img2 = np.reshape(img, (6, 8, 1))
    print(img2.shape)
    img3 = np.reshape(img2, (6, 8))
    print(img3.shape)
    print(img3)
    new_img = np.pad(img, ((1,2), (3, 4)), 'constant', constant_values=8)
    print(new_img)
    x, y = img.nonzero()
    min_x = min(x)
    max_x = max(x)
    min_y = min(y)
    max_y = max(y)
    bbox_height2 = max_x - min_x
    bbox_width2 = max_y - min_y
    print(img[min_x:max_x+1, min_y:max_y+1])
    t4 = time.clock()
    print(bbox_height2, bbox_width2)
    print('调用函数方法耗时：', t4 - t3)
    pass

def test_nii_2_dicomrt():
    src_cases_root_path = 'D:\\datasets\\nii\\HaN_OAR'
    dst_cases_root_path = 'D:\\datasets\\nii\\nii_2_dicomrt\\HaN_OAR'
    src_cases_root_path = 'D:\\datasets\\cstro\\nii\\thoracic-oar'
    dst_cases_root_path = 'D:\\datasets\\cstro\\dicomrt\\thoracic-oar'
    src_cases_root_path = 'D:\\datasets\\cstro\\nii\\lung-gtv'
    dst_cases_root_path = 'D:\\datasets\\cstro\\dicomrt\\lung-gtv'
    src_cases_root_path = 'D:\\datasets\\cstro\\org\\HaN_OAR'
    dst_cases_root_path = 'D:\\datasets\\cstro\\dicomrt\\HaN_OAR'
    src_cases_root_path = 'D:\\datasets\\cstro\\nii\\HaN_OAR'
    dst_cases_root_path = 'D:\\datasets\\cstro\\dicomrt\\HaN_OAR'
    dcmio.nii_2_dicomrt(src_cases_root_path, dst_cases_root_path)
    pass

def test_nrrd_2_dicomrt():
    src_cases_root_path = 'D:\\datasets\\nrrd\\cases'
    dst_cases_root_path = 'D:\\datasets\\nrrd\\generated_dicom'
    dcmio.nrrd_2_dicomrt(src_cases_root_path, dst_cases_root_path)
    pass

def test_dicomrt_2_nii():
    src_cases_root_path = 'D:\\datasets\\nii\\nii_2_dicomrt\\HaN_OAR'
    dst_cases_root_path = 'D:\\datasets\\nii\\dicomrt_2_nii\\HaN_OAR'
    dcmio.dicomrt_2_nii(src_cases_root_path, dst_cases_root_path)
    pass

def test_dicomrt_2_jpg():
    src_cases_root_path = 'D:\\datasets\\nii\\nii_2_dicomrt\\HaN_OAR'
    dst_cases_root_path = 'D:\\datasets\\jpg\\headneck\\train'
    ww = 350
    wl = 40
    dcmio.dicomrt_2_jpg(src_cases_root_path, dst_cases_root_path, ww, wl)
    pass

import cv2 as cv
import os

def __get_connected_component_state(slice1, slice2, slice3):
    assert slice1.shape == slice2.shape and slice2.shape == slice3.shape

    flag1 = False
    flag2 = False

    result1 = np.zeros([slice1.shape[0], slice1.shape[1]], dtype=slice1.dtype)
    cv.bitwise_and(slice1, slice2, result1)
    if np.all(result1 == 0):
        flag1 = False
    else:
        flag1 = True

    result2 = np.zeros([slice2.shape[0], slice2.shape[1]], dtype=slice2.dtype)
    cv.bitwise_and(slice2, slice3, result2)
    if np.all(result2 == 0):
        flag2 = False
    else:
        flag2 = True

    if flag1 is False and flag2 is False:
        return 0
    elif flag1 is False and flag2 is True:
        return 1
    elif flag1 is True and flag2 is False:
        return 2
    else:
        return 1

def get_volume_bbox2(label_volume_data_prediction, depth):
    cv.threshold(label_volume_data_prediction, 127, 255, cv.THRESH_BINARY, label_volume_data_prediction)
    connected_component_list = []
    index_list = []
    for i in range(label_volume_data_prediction.shape[0] - 2):
        state = __get_connected_component_state(
            label_volume_data_prediction[i], label_volume_data_prediction[i + 1], label_volume_data_prediction[i + 2])
        if state == 0:
            continue
        elif state == 1:
            index_list.append(i + 1)
        elif state == 2:
            index_list.append(i + 1)
            connected_component_list.append(index_list.copy())
            index_list.clear()
    max_size = 0
    result_connected_component_index = 0
    for i in range(len(connected_component_list)):
        size = len(connected_component_list[i])
        if size > max_size:
            max_size = size
            result_connected_component_index = i
    # label_volume_data = np.zeros(label_volume_data_prediction.shape, np.uint8)
    # label_volume_data[connected_component_list[result_connected_component_index][:]] = \
    #     label_volume_data_prediction[connected_component_list[result_connected_component_index][:]]
    pass
    index_size = len(connected_component_list[result_connected_component_index])
    assert index_size < depth + 1
    z_min = max(0, connected_component_list[result_connected_component_index][0])
    z_max = min(label_volume_data_prediction.shape[0] - 1, connected_component_list[result_connected_component_index][0] + depth - 1)
    z, y, x = label_volume_data_prediction[z_min:z_max].nonzero()
    y_min = min(y)
    y_max = max(y)
    x_min = min(x)
    x_max = max(x)
    width = x_max - x_min
    height = y_max - y_min
    length = max(width, height)
    padding = length // 2

    start_z = z_min
    end_z = z_max
    start_y = max(y_min - padding, 0)
    end_y = min(start_y + length * 2, label_volume_data_prediction.shape[1] - 1)
    start_x = max(x_min - padding, 0)
    end_x = min(start_x + length * 2, label_volume_data_prediction.shape[2] - 1)
    bbox_size = [start_z, end_z, start_y, end_y, start_x, end_x]

    # volume_data_bbox = volume_data_org[start_z : end_z, start_y : end_y, start_x : end_x]
    volume_data_bbox = label_volume_data_prediction[start_z : end_z, start_y : end_y, start_x : end_x]
    return volume_data_bbox, bbox_size

def test_get_volume_box():
    path = 'C:/Users/SEELE/Desktop/11'
    file_list = os.listdir(path)
    file_list.sort()
    label_volume = np.zeros([len(file_list), 512, 512])
    for i in range(len(file_list)):
        one_image_full_path = os.path.join(path, file_list[i])
        label_volume[i] = cv.imread(one_image_full_path, cv.IMREAD_GRAYSCALE)
    volume_data_bbox, bbox_size = get_volume_bbox2(label_volume, 32)
    print(bbox_size)
    for i in range(volume_data_bbox.shape[0]):
        cv.imshow('%d' % (i), volume_data_bbox[i])
        cv.waitKey()
    pass

import itk
def test_connected_component():
    path = 'C:/Users/SEELE/Desktop/11/a'
    file_list = os.listdir(path)
    file_list.sort()
    label_volume = np.zeros([len(file_list), 512, 512])
    for i in range(len(file_list)):
        one_image_full_path = os.path.join(path, file_list[i])
        one_image_data = cv.imread(one_image_full_path, cv.IMREAD_GRAYSCALE)
        cv.threshold(one_image_data, 127, 255, cv.THRESH_BINARY, one_image_data)
        label_volume[i] = one_image_data
    # cv.imshow('a', label_volume[87])
    # cv.waitKey()
    PixelType = itk.ctype("unsigned char")
    ImageType = itk.Image[PixelType, 3]
    itkimage = itk.GetImageFromArray(label_volume.astype(np.uint8))
    filter = itk.BinaryShapeKeepNObjectsImageFilter[ImageType].New()
    filter.SetAttribute('NumberOfPixels')
    filter.SetForegroundValue(255)
    filter.SetBackgroundValue(0)
    filter.SetNumberOfObjects(1)
    filter.SetInput(itkimage)
    filter.Update()
    output = filter.GetOutput()
    result_label_volume = itk.GetArrayFromImage(output)
    for i in range(result_label_volume.shape[0]):
        one_image_full_path = os.path.join(path, 'a.%d.jpg' % (i))
        cv.imwrite(one_image_full_path, result_label_volume[i])

    pass

from skimage import measure

# a1 = np.array([
#     [255, 0, 0, 0, 0],
#     [0, 0, 255, 255, 0],
#     [0, 0, 255, 255, 0],
#     [255, 0, 0, 0, 0]
# ])
# print(a1)
# b1 = measure.label(a1, connectivity=2)
# print(b1)
# props = measure.regionprops(b1)
# print(props[1].bbox)

# a2 = np.array([
# [
#     [255, 0, 0, 0, 0],
#     [0, 0, 255, 255, 0],
#     [0, 0, 255, 255, 0],
#     [255, 0, 0, 0, 0]
# ],
# [
#     [255, 0, 0, 0, 0],
#     [0, 0, 255, 255, 0],
#     [0, 0, 255, 255, 0],
#     [255, 0, 0, 0, 0]
# ],
# ])
# print(a2)
# b2 = measure.label(a2, connectivity=2)
# print(b2)
# props = measure.regionprops(b2)
# max_area = 0
# max_area_index = 0
# for i in range(len(props)):
#     area = props[i].area
#     if area > max_area:
#         max_area_index = i
#         max_area = area
# print('max area index: %d' % (max_area_index))
# print('max area: %d' % (max_area))
# box = props[max_area_index].bbox
# print('bbox:', box)
# c = np.zeros(a2.shape, np.uint8)
# c[box[0]:box[3], box[1]:box[4], box[2]:box[5]] = a2[box[0]:box[3], box[1]:box[4], box[2]:box[5]]
# print(c)

import data.dataset as ds
def test_ionii():
    one_case_file_path = os.path.join('D:\\datasets\\nii\\HaN_OAR\\1\\data.nii.gz')
    dataset = ds.Dataset4TrainLocation('hippo-L', 'D:\\datasets\\nii\\HaN_OAR', 'MR')
    dataset.next_batch()
    pass

if __name__ == '__main__':
    # test_nrrd_2_dicomrt()
    # test_nii_2_dicomrt()
    #test_dicomrt_2_nii()
    # test_dicomrt_2_jpg()
    test_tmp()
    #test_get_volume_box()
    # test_connected_component()
    #test_ionii()
    pass