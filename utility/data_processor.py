import os
import numpy as np
import cv2 as cv
import SimpleITK as sitk
import itk

def image_resize_3d(image_3d, height, width):
    slices = image_3d.shape[0]
    result_image = np.zeros([slices, height, width])
    for i in range(slices):
        result_image[i] = cv.resize(image_3d[i], (width, height), cv.INTER_CUBIC)
    return result_image

def image_resize_2d(image_2d, height, width):
    result_image = cv.resize(image_2d, (width, height))
    return result_image

def image_threshold_binaray_3d(image_3d, threshold_val):
    slices = image_3d.shape[0]
    for i in range(slices):
        one_slice = image_3d[i]
        cv.threshold(one_slice, threshold_val, 255, cv.THRESH_BINARY, one_slice)
        image_3d[i] = one_slice
    return image_3d

def image_threshold_binaray_2d(image_2d, threshold_val):
    cv.threshold(image_2d, threshold_val, 255, cv.THRESH_BINARY, image_2d)
    return image_2d

def get_bounding_box(image_3d_ref, depth, xy_padding=15, z_padding=8):
    image_3d_ref = image_threshold_binaray_3d(image_3d_ref, 127)

    PixelType = itk.ctype("unsigned char")
    ImageType = itk.Image[PixelType, 3]
    itkimage = itk.GetImageFromArray(image_3d_ref.astype(np.uint8))
    filter = itk.BinaryShapeKeepNObjectsImageFilter[ImageType].New()
    filter.SetAttribute('NumberOfPixels')
    filter.SetForegroundValue(255)
    filter.SetBackgroundValue(0)
    filter.SetNumberOfObjects(1)
    filter.SetInput(itkimage)
    filter.Update()
    output = filter.GetOutput()
    result_label_volume = itk.GetArrayFromImage(output)
    z, y, x = result_label_volume.nonzero()
    z_min = max(0, min(z) - z_padding)
    z_max = min(image_3d_ref.shape[0] - 1, z_min + depth)
    y_min = min(y)
    y_max = max(y)
    x_min = min(x)
    x_max = max(x)
    width = x_max - x_min
    height = y_max - y_min
    length = max(width, height)

    start_z = z_min
    end_z = z_max
    start_y = max(y_min - xy_padding, 0)
    end_y = min(start_y + length + 2 * xy_padding, image_3d_ref.shape[1] - 1)
    start_x = max(x_min - xy_padding, 0)
    end_x = min(start_x + length + 2 * xy_padding, image_3d_ref.shape[2] - 1)
    bbox_size = [start_z, end_z, start_y, end_y, start_x, end_x]

    return bbox_size

def image_keep_max_region(label_volume_data):
    label_volume = image_threshold_binaray_3d(label_volume_data, 127)
    # labels = measure.label(image_3d_ref, connectivity=2)
    # props = measure.regionprops(labels)
    # max_area = 0
    # max_area_index = 0
    # for i in range(len(props)):
    #     area = props[i].area
    #     if area > max_area:
    #         max_area_index = i
    #         max_area = area
    # box = props[max_area_index].bbox
    # label_volume_with_max_region = np.zeros(image_3d_ref.shape, np.uint8)
    # label_volume_with_max_region[box[0]:box[3], box[1]:box[4], box[2]:box[5]] = \
    #     image_3d_ref[box[0]:box[3], box[1]:box[4], box[2]:box[5]]
    # assert box[3] - box[0] < depth + 1
    # for i in range(label_volume_with_max_region.shape[0]):
    #     cv.imwrite('D:\\tmp\\MIS\\other\\%d.jpg' % (i), label_volume_with_max_region[i])

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
    return result_label_volume

def normalize_image_data(image_data):
    image_data = image_data.astype(np.float)
    max_val = np.max(image_data)
    min_val = np.min(image_data)
    if max_val - min_val != 0:
        image_data = image_data / (max_val - min_val)
    return image_data

def image_dilate(image_data, se_size):
    itk_image = sitk.GetImageFromArray(image_data)
    filter = sitk.GrayscaleDilateImageFilter()
    filter.SetKernelType(sitk.sitkBall)
    filter.SetKernelRadius(se_size)
    output = filter.Execute(itk_image)
    result_image_data = sitk.GetArrayFromImage(output)
    return result_image_data

def image_erode(image_data, se_size):
    itk_image = sitk.GetImageFromArray(image_data)
    filter = sitk.GrayscaleErodeImageFilter()
    filter.SetKernelType(sitk.sitkBall)
    filter.SetKernelRadius(se_size)
    output = filter.Execute(itk_image)
    result_image_data = sitk.GetArrayFromImage(output)
    return result_image_data

def create_itk_image(first_dataset_item, image_volume_data):
    pixel_spacing = first_dataset_item.PixelSpacing
    slice_thikness = first_dataset_item.SliceThickness
    direction = first_dataset_item.ImageOrientationPatient
    origin = first_dataset_item.ImagePositionPatient

    d_0_0 = float(direction[0])
    d_1_0 = float(direction[1])
    d_2_0 = float(direction[2])
    d_0_1 = float(direction[3])
    d_1_1 = float(direction[4])
    d_2_1 = float(direction[5])
    d_0_2 = d_1_0 * d_2_1- d_2_0 * d_1_1
    d_1_2 = d_2_0 * d_0_1- d_0_0 * d_2_1
    d_2_2 = d_0_0 * d_1_1- d_1_0 * d_0_1

    sitk_image = sitk.GetImageFromArray(image_volume_data)
    sitk_image.SetOrigin((float(origin[0]), float(origin[1]), float(origin[2])))
    sitk_image.SetSpacing((float(pixel_spacing[0]), float(pixel_spacing[1]), slice_thikness))
    itk_dirction = (d_0_0, d_1_0, d_2_0, d_0_1, d_1_1, d_2_1, d_0_2, d_1_2, d_2_2)
    sitk_image.SetDirection(itk_dirction)

    return sitk_image

def image_resample(sitk_image, new_thickness):
    origin = sitk_image.GetOrigin()
    direction = sitk_image.GetDirection()
    spacing = sitk_image.GetSpacing()
    size = sitk_image.GetSize()
    slices = size[2]
    slice_thickness = spacing[2]

    new_origin = origin
    new_direction = direction
    new_spacing = (float(spacing[0]), float(spacing[1]), new_thickness)
    new_slices = int(round(slices * slice_thickness / new_spacing[2]))
    new_size = (size[0], size[1], new_slices)

    resample = sitk.ResampleImageFilter()
    resample.SetOutputOrigin(new_origin)
    resample.SetOutputDirection(new_direction)
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(new_size)
    resample.SetInterpolator(sitk.sitkLinear)
    resample.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    result = resample.Execute(sitk_image)
    result_volume_data = sitk.GetArrayFromImage(result)
    return result_volume_data

def image_rescale_intensity(image_volume_data, output_max, output_min):
    # input_max = np.max(image_volume_data)
    # input_min = np.min(image_volume_data)
    # input = np.clip(image_volume_data, 0.2*(input_max-input_min), 0.8*(input_max-input_min))
    sitk_image= sitk.GetImageFromArray(image_volume_data)
    rescale_filter = sitk.RescaleIntensityImageFilter()
    rescale_filter.SetOutputMaximum(output_max)
    rescale_filter.SetOutputMinimum(output_min)
    output = rescale_filter.Execute(sitk_image)
    result = sitk.GetArrayFromImage(output)
    return result

# def set_ww_wl(image_volume_data, ww, wl):
#     result_image_data = (255 / ww) * (image_volume_data - wl + ww / 2)
#     result_image_data = np.clip(result_image_data, 0, 255)
#     result_image_data = result_image_data.astype(np.uint8)
#     return result_image_data

def image_padding(volume_data, height, width):
    if height > width:
        padding = (height - width) // 2
        if (height - width) % 2 != 0:
            result = np.pad(volume_data, ((0, 0), (padding, padding+1), (0, 0)), 'constant', constant_values=(0,0))
        else:
            result = np.pad(volume_data, ((0, 0), (padding, padding), (0, 0)), 'constant', constant_values=(0,0))
    elif height < width:
        padding = (width - height) // 2
        if (width - height) % 2 != 0:
            result = np.pad(volume_data, ((0, 0), (padding, padding+1), (0, 0)), 'constant', constant_values=(0,0))
        else:
            result = np.pad(volume_data, ((0, 0), (padding, padding), (0, 0)), 'constant', constant_values=(0,0))
    else:
        result = volume_data
    return result

def image_ipadding(volume_data, height, width):
    if height > width:
        if (height - width) // 2 != 0:
            label_volume_data_ipadded = volume_data[:, :, (height - width) // 2:(height - ((height - width) // 2)) - 1]
        else:
            label_volume_data_ipadded = volume_data[:, :, (height - width) // 2:(height - ((height - width) // 2))]
    elif height < width:
        if (width - height) % 2 != 0:
            label_volume_data_ipadded = volume_data[:, (width - height) // 2:(width - ((width - height) // 2)) - 1, :]
        else:
            label_volume_data_ipadded = volume_data[:, (width - height) // 2:(width - ((width - height) // 2)), :]
    else:
        label_volume_data_ipadded = volume_data

    return label_volume_data_ipadded

def image_crop(volume_data):
    z, y, x = volume_data.nonzero()
    z_min = min(z)
    z_max = min(max(z) + 1, volume_data.shape[0])
    y_min = min(y)
    y_max = max(y)
    x_min = min(x)
    x_max = max(x)
    box = [z_min, z_max, y_min, y_max, x_min, x_max]

    result = volume_data[z_min:z_max, y_min:y_max, x_min:x_max]
    return box, result
