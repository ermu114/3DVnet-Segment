import pydicom
import numpy as np
import os
from pydicom.dataset import Dataset, FileDataset
import nibabel as nib
import SimpleITK as sitk
import utility.iodicom as dcmio
import datetime
import nrrd
import cv2 as cv

#------------------------------------------------------------------------------
dict_threshold_and_roiname = {
    1: 'Lung_L',
    2: 'Lung_R',
    3: 'Trachea',
}
dict_roiname_and_threshold = {
    'Lung_L':1,
    'Lung_R':2,
    'Trachea':3,
}
dict_threshold_and_roiname_rt = {
    1: 'Brainstem',
    2: 'Eye_L',
    3: 'Eye_R',
    4: 'Len_L',
    5: 'Len_R',
    6: 'Optic_Nerve_L',
    7: 'Optic_Nerve_R',
    8: 'Optic_Chiasm',
    9: 'Temporal_Lobe_L',
    10: 'Temporal_Lobe_R',
    11: 'Pituitary',
    12: 'Parotid_Gland_L',
    13: 'Parotid_Gland_R',
    14: 'Inner_Ear_L',
    15: 'Inner_Ear_R',
    16: 'Middle_Ear_L',
    17: 'Middle_Ear_R',
    18: 'Mandible_Joint_L',
    19: 'Mandible_Joint_R',
    20: 'Spinal_Cord',
    21: 'Mandible_L',
    22: 'Mandible_R'
}

dict_roiname_and_threshold_rt = {
    'Brainstem':1,
    'Eye_L':2,
    'Eye_R':3,
    'Len_L':4,
    'Len_R':5,
    'Optic_Nerve_L':6,
    'Optic_Nerve_R':7,
    'Optic_Chiasm':8,
    'Temporal_Lobe_L':9,
    'Temporal_Lobe_R':10,
    'Pituitary':11,
    'Parotid_Gland_L':12,
    'Parotid_Gland_R':13,
    'Inner_Ear_L':14,
    'Inner_Ear_R':15,
    'Middle_Ear_L': 16,
    'Middle_Ear_R': 17,
    'Mandible_Joint_L':18,
    'Mandible_Joint_R':19,
    'Spinal_Cord':20,
    'Mandible_L':21,
    'Mandible_R':22
}
def __generate_sop_instance_info_nii(series_dataset):
    sop_instance_info_list = []

    header = series_dataset.header
    slice_thikness = header['pixdim'][3]
    x = header['qoffset_x']
    y = header['qoffset_y']
    z = header['qoffset_z']
    volume_size = series_dataset.shape
    sop_instance_uid_base = pydicom.uid.generate_uid()
    for i in range(volume_size[2]):
        last_3_num = np.uint16(sop_instance_uid_base[-4:])
        last_3_num += i
        sop_instance_uid = sop_instance_uid_base[:-4] + str(last_3_num)
        instance_number = i + 1
        slice_location = z + slice_thikness * i
        image_position = '%f\\%f\\%f' % (x, y, slice_location)
        sop_instance_info = [instance_number, sop_instance_uid, slice_location, image_position]
        sop_instance_info_list.append(sop_instance_info)
    return sop_instance_info_list

def __create_pixel_data_array_nii(series_dataset, slice_index):
    src_one_slice_data = series_dataset.dataobj[:, :, slice_index]
    dst_one_slice_data = np.transpose(src_one_slice_data) + 1024
    bytes_array = dst_one_slice_data.tobytes()
    return bytes_array

def __write_dicom_series_file_nii(series_dataset, dst_one_case_file_path):
    print('start: write dicom series')
    if not os.path.exists(dst_one_case_file_path):
        os.makedirs(dst_one_case_file_path)
    dataset_list = []

    hdr = series_dataset.header
    print(hdr)
    sop_instance_info_list = __generate_sop_instance_info_nii(series_dataset)
    study_instance_uid = pydicom.uid.generate_uid()
    series_instance_uid = pydicom.uid.generate_uid()
    frame_of_reference_uid = pydicom.uid.generate_uid()
    volume_size = series_dataset.shape
    for j in range(volume_size[2]):
        sop_instance_uid = sop_instance_info_list[j][1]
        filename = os.path.join(dst_one_case_file_path, 'CT.' + sop_instance_uid + '.dcm')

        file_meta = Dataset()
        file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
        file_meta.MediaStorageSOPInstanceUID = sop_instance_uid
        file_meta.ImplementationClassUID = '1.2.246.352.70.2.1.7'

        ds = FileDataset(filename, {}, file_meta=file_meta, preamble=b"\0" * 128)

        ds.InstanceNumber = sop_instance_info_list[j][0]
        ds.SliceLocation = sop_instance_info_list[j][2]
        ds.ImagePositionPatient = sop_instance_info_list[j][3]

        ds.fix_meta_info()
        ds.is_little_endian = True
        ds.is_implicit_VR = True
        ds.StudyInstanceUID = study_instance_uid
        ds.SeriesInstanceUID = series_instance_uid
        ds.FrameOfReferenceUID = frame_of_reference_uid
        ds.SpecificCharacterSet = 'ISO_IR 100'
        ds.ImageType = 'ORIGINAL\\PRIMARY\\AXIAL\\HELIX'
        dt = datetime.datetime.now()
        datestr = dt.strftime('%Y%m%d')
        timestr1 = dt.strftime('%H%M%S.%f')
        timestr2 = dt.strftime('%H%M%S')

        ds.InstanceCreationDate = datestr
        ds.InstanceCreationTime = timestr1
        ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
        ds.SOPInstanceUID = sop_instance_uid
        ds.StudyDate = datestr
        ds.SeriesDate = datestr
        ds.ContentDate = datestr
        ds.StudyTime = timestr2
        ds.SeriesTime = timestr2
        ds.ContentTime = timestr2
        ds.AccessionNumber = '116939'
        ds.Modality = 'CT'
        ds.Manufacturer ='HYHY' #'Philips'
        ds.InstitutionName = 'Cancer Hosipital'
        ds.ReferringPhysicianName = '79309'
        ds.StationName = '-'
        ds.StudyDescription = '-'
        ds.PatientName = 'patientxxx'
        ds.PatientID = 'patientxxx'
        ds.PatientBirthDate = '19000101'
        ds.PatientBirthTime = '000000'
        ds.PatientAge = '119'
        ds.PatientSex = '1'#'-'
        ds.PatientSize = 120
        ds.PatientWeight = 100
        ds.PatientAddress = 'some where'
        ds.PatientComments = 'some'
        ds.SliceThickness = series_dataset.header['pixdim'][3]
        ds.KVP = 1111111
        ds.PatientPosition = 'HFS'
        ds.StudyID = '20200423'
        ds.SeriesNumber = 1
        ds.ImageOrientationPatient = '1\\0\\0\\0\\1\\0'
        ds.SamplesPerPixel = 1
        ds.Rows = volume_size[1]
        ds.Columns = volume_size[0]
        spacing_x = series_dataset.header['pixdim'][1]
        spacing_y = series_dataset.header['pixdim'][2]
        ds.PixelSpacing = '%f\\%f' % (spacing_x, spacing_y)

        niiPixelDataType = series_dataset.header['datatype'].dtype
        if(niiPixelDataType == 'uint16' or niiPixelDataType == 'uint8' ):   #判断有符号数 无符号数 if(niiPixelDataType.find('uint') == 1 )
            ds.PixelRepresentation = 0
        elif (niiPixelDataType == 'int16' or niiPixelDataType == 'int8' ) :
            ds.PixelRepresentation = 1


        niiBitPixel = series_dataset.header['bitpix']
        ds.BitsAllocated = niiBitPixel#16
        ds.BitsStored = niiBitPixel#12
        ds.HighBit = niiBitPixel-1#11

        #ds.WindowCenter = 756#60
        #ds.WindowWidth = 1500#350
        rescale_intercept = series_dataset.header['scl_inter']
        if np.isnan(rescale_intercept):
            ds.RescaleIntercept = 0 #-1024
        else:
            ds.RescaleIntercept = rescale_intercept
        rescale_slope = series_dataset.header['scl_slope']
        if np.isnan(rescale_slope):
            ds.RescaleSlope = 1
        else:
            ds.RescaleSlope = rescale_slope

        ds.PixelData = __create_pixel_data_array_nii(series_dataset, j)
        ds.save_as(filename, write_like_original=False)
        dataset_list.append(ds)
    print('end: write dicom series')
    return dataset_list
#按roi序号 阈值处理为 255
def __create_label_volume_data_list_nii(one_label_file_path):
    result_label_volume_data_list = []
    label_dataset = nib.load(one_label_file_path)
    one_case_label_volume_data = np.transpose(label_dataset.dataobj)
    slice_nums = one_case_label_volume_data.shape[0]
    hdr = label_dataset.header
    print(hdr)
    for i in range(slice_nums):
        cv.imwrite('D:/tmp/01/%03d.jpg' % (i), one_case_label_volume_data[i]*20)#肉眼看不出1、2、3的灰阶变化
    roi_nums = len(dict_threshold_and_roiname)
    for i in range(roi_nums):
        one_case_itk_image = sitk.GetImageFromArray(one_case_label_volume_data)
        threshold_filter = sitk.BinaryThresholdImageFilter()
        threshold_filter.SetOutsideValue(0)
        threshold_filter.SetInsideValue(255)
        threshold_filter.SetLowerThreshold(i + 1)
        threshold_filter.SetUpperThreshold(i + 1)
        output = threshold_filter.Execute(one_case_itk_image)
        result_label_volume_data = sitk.GetArrayFromImage(output)
        roi_name = dict_threshold_and_roiname[i + 1]
        if roi_name == 'Eye_L':
            result_label_volume_data_eye_l = result_label_volume_data
            continue
        elif roi_name == 'Eye_R':
            result_label_volume_data_eye_r = result_label_volume_data
            continue
        elif roi_name == 'Len_L':
            result_label_volume_data_len_l = result_label_volume_data
        elif roi_name == 'Len_R':
            result_label_volume_data_len_r = result_label_volume_data
        result_label_volume_data_list.append([roi_name, result_label_volume_data])
    # if (roi_nums == 22): #头部勾画再放开
    #    result_label_volume_data_eye_l = result_label_volume_data_eye_l + result_label_volume_data_len_l
    #    result_label_volume_data_list.append(['Eye_L', result_label_volume_data_eye_l])
    #    result_label_volume_data_eye_r = result_label_volume_data_eye_r + result_label_volume_data_len_r
    #    result_label_volume_data_list.append(['Eye_R', result_label_volume_data_eye_r])

    return result_label_volume_data_list

def __write_nii_series_file(series_dataset, dst_one_case_file_path):
    dst_one_case_root_path = os.path.dirname(dst_one_case_file_path)
    if not os.path.exists(dst_one_case_root_path):
        os.makedirs(dst_one_case_root_path)

    volume_data_in_ct_value = dcmio.get_image_volume_data_ctvalue(series_dataset)
    orientation_matrix = dcmio.__get_orientation_matrix(series_dataset)
    orientation_matrix[0, 0] = -orientation_matrix[0, 0]
    orientation_matrix[1, 1] = -orientation_matrix[1, 1]
    spacing_matrix = dcmio.__get_spacing_matrix(series_dataset)
    origin_matrix = dcmio.__get_origin_matrix(series_dataset)
    affine_matrix = np.r_[np.c_[orientation_matrix * spacing_matrix, origin_matrix], [[0, 0, 0, 1]]]
    nii_image = nib.Nifti1Image(np.transpose(volume_data_in_ct_value), affine_matrix)
    nib.save(nii_image, dst_one_case_file_path)
    pass

def __get_label_volume_data_merged(ct_dataset_list, rtstruct_dataset):
    dict_referenced_sop_instance_uid_index = {}
    contour_image_sequence = rtstruct_dataset.ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[0].ContourImageSequence
    slices = len(contour_image_sequence)
    for i in range(slices):
        dict_referenced_sop_instance_uid_index[contour_image_sequence[i].ReferencedSOPInstanceUID] = i

    # [roi_name, threshold, volume_data]
    dict_rest_roi = {}
    rows = ct_dataset_list[0].Rows
    cols = ct_dataset_list[0].Columns
    result_label_volume_data = np.zeros([slices, rows, cols], np.uint8)
    roi_list = dcmio.__get_roi_list_from_rs_file(rtstruct_dataset)
    for roi in roi_list:
        label_volume_data = dcmio.__create_label_volume(roi, dict_referenced_sop_instance_uid_index, ct_dataset_list)
        roi_name = roi[0]
        threshold_value = np.uint8(dict_roiname_and_threshold[roi_name])
        if roi_name == 'Eye_L' or roi_name == 'Len_L' or roi_name == 'Eye_R' or roi_name == 'Len_R':
            dict_rest_roi[roi_name] = [threshold_value, label_volume_data]
        else:
            result_label_volume_data += np.uint8(threshold_value * (label_volume_data / 255))
    threshold_value_eye_l = dict_rest_roi['Eye_L'][0]
    threshold_value_len_l = dict_rest_roi['Len_L'][0]
    threshold_value_eye_r = dict_rest_roi['Eye_R'][0]
    threshold_value_len_r = dict_rest_roi['Len_R'][0]
    label_volume_data_eye_l = dict_rest_roi['Eye_L'][1]
    label_volume_data_len_l = dict_rest_roi['Len_L'][1]
    label_volume_data_eye_r = dict_rest_roi['Eye_R'][1]
    label_volume_data_len_r = dict_rest_roi['Len_R'][1]
    result_label_volume_data += np.uint8(threshold_value_eye_l * ((label_volume_data_eye_l - label_volume_data_len_l) / 255))
    result_label_volume_data += np.uint8(threshold_value_len_l * (label_volume_data_len_l / 255))
    result_label_volume_data += np.uint8(threshold_value_eye_r * ((label_volume_data_eye_r - label_volume_data_len_r) / 255))
    result_label_volume_data += np.uint8(threshold_value_len_r * (label_volume_data_len_r / 255))

    return result_label_volume_data

def __write_nii_label_file(series_dataset_list, rtstruct_dataset_list, dst_one_case_label_file_path):
    dst_one_case_root_path = os.path.dirname(dst_one_case_label_file_path)
    if not os.path.exists(dst_one_case_root_path):
        os.makedirs(dst_one_case_root_path)

    label_volume_data_merged = __get_label_volume_data_merged(series_dataset_list, rtstruct_dataset_list)
    orientation_matrix = dcmio.__get_orientation_matrix(series_dataset_list)
    orientation_matrix[0, 0] = -orientation_matrix[0, 0]
    orientation_matrix[1, 1] = -orientation_matrix[1, 1]
    spacing_matrix = dcmio.__get_spacing_matrix(series_dataset_list)
    origin_matrix = dcmio.__get_origin_matrix(series_dataset_list)
    affine_matrix = np.r_[np.c_[orientation_matrix * spacing_matrix, origin_matrix], [[0, 0, 0, 1]]]
    nii_image = nib.Nifti1Image(np.transpose(label_volume_data_merged), affine_matrix)
    nib.save(nii_image, dst_one_case_label_file_path)
    pass

def nii_2_dicomrt(src_case_root_path, dst_case_root_path):
    pid_list = os.listdir(src_case_root_path)
    case_nums = len(pid_list)
    for i in range(case_nums):
        print('start: ' + pid_list[i])
        pid = int(pid_list[i])
        #pid += 50
        pid_str = str('%03d' % (pid))
        # dst_one_case_file_path = os.path.join(dst_case_root_path, pid_list[i])
        dst_one_case_file_path = os.path.join(dst_case_root_path, pid_str)
        one_case_root_path = os.path.join(src_case_root_path, pid_list[i])

        one_case_file_path = os.path.join(one_case_root_path, 'data.nii.gz')
        series_dataset = nib.load(one_case_file_path)
        dataset_list = __write_dicom_series_file_nii(series_dataset, dst_one_case_file_path)

        one_label_file_path = os.path.join(one_case_root_path, 'label.nii.gz')
        label_volume_data_list = __create_label_volume_data_list_nii(one_label_file_path)

        one_case_dicom_data = [pid_list[i], dataset_list, dst_one_case_file_path]
        dcmio.write_rtstruct(one_case_dicom_data, label_volume_data_list)
        print('end: ' + pid_list[i])
    pass

def dicomrt_2_nii(src_case_root_path, dst_case_root_path):
    pid_list = os.listdir(src_case_root_path)
    case_nums = len(pid_list)
    for i in range(case_nums):
        print('start: ' + pid_list[i])

        src_one_case_root_path = os.path.join(src_case_root_path, pid_list[i])

        file_meta, series_dataset_list = dcmio.read_one_series(src_one_case_root_path, modality='CT')
        dst_one_case_seriesfile_path = os.path.join(dst_case_root_path, pid_list[i], 'data.nii.gz')
        __write_nii_series_file(series_dataset_list, dst_one_case_seriesfile_path)

        file_meta, rtstruct_dataset_list = dcmio.read_one_series(src_one_case_root_path, modality='RTSTRUCT')
        one_case_rtstruct_file_path = os.path.join(dst_case_root_path, pid_list[i], 'label.nii.gz')
        __write_nii_label_file(series_dataset_list, rtstruct_dataset_list, one_case_rtstruct_file_path)
        #__write_nii_label_file(series_dataset_list, rtstruct_dataset_list[0], one_case_rtstruct_file_path)
        print('end: ' + pid_list[i])
    pass
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def __generate_sop_instance_info_nrrd(series_dataset):
    sop_instance_info_list = []
    slice_thikness = series_dataset[1]['space directions'][2][2]
    x, y, z = series_dataset[1]['space origin']
    volume_size = series_dataset[1]['sizes']
    sop_instance_uid_base = pydicom.uid.generate_uid()
    for i in range(volume_size[2]):
        last_3_num = np.uint16(sop_instance_uid_base[-4:])
        last_3_num += i
        sop_instance_uid = sop_instance_uid_base[:-4] + str(last_3_num)
        instance_number = i + 1
        slice_location = z + slice_thikness * i
        image_position = '%f\\%f\\%f' % (x, y, slice_location)
        sop_instance_info = [instance_number, sop_instance_uid, slice_location, image_position]
        sop_instance_info_list.append(sop_instance_info)
    return sop_instance_info_list

def __create_pixel_data_array_nrrd(series_dataset, slice_index):
    volume_size = series_dataset[1]['sizes']
    dst_one_slice_data = np.zeros([volume_size[1], volume_size[0]], dtype=np.int16)
    src_one_slice_data = series_dataset[0][:, :, slice_index]
    for i in range(volume_size[1]):
            dst_one_slice_data[i, :] = src_one_slice_data[:, i] + 1024
    bytes_array = dst_one_slice_data.tobytes()
    return bytes_array

def __write_dicom_series_file_nrrd(series_dataset, dst_one_case_file_path):
    print('start: write dicom series')
    if not os.path.exists(dst_one_case_file_path):
        os.makedirs(dst_one_case_file_path)
    dataset_list = []

    sop_instance_info_list = __generate_sop_instance_info_nrrd(series_dataset)
    study_instance_uid = pydicom.uid.generate_uid()
    series_instance_uid = pydicom.uid.generate_uid()
    frame_of_reference_uid = pydicom.uid.generate_uid()
    volume_size = series_dataset[1]['sizes']  # [width height slices]
    for j in range(volume_size[2]):
        sop_instance_uid = sop_instance_info_list[j][1]
        filename = os.path.join(dst_one_case_file_path, 'CT.' + sop_instance_uid + '.dcm')

        file_meta = Dataset()
        file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
        file_meta.MediaStorageSOPInstanceUID = sop_instance_uid
        file_meta.ImplementationClassUID = '1.2.246.352.70.2.1.7'

        ds = FileDataset(filename, {}, file_meta=file_meta, preamble=b"\0" * 128)

        ds.InstanceNumber = sop_instance_info_list[j][0]
        ds.SliceLocation = sop_instance_info_list[j][2]
        ds.ImagePositionPatient = sop_instance_info_list[j][3]

        ds.fix_meta_info()
        ds.is_little_endian = True
        ds.is_implicit_VR = True
        ds.StudyInstanceUID = study_instance_uid
        ds.SeriesInstanceUID = series_instance_uid
        ds.FrameOfReferenceUID = frame_of_reference_uid
        ds.SpecificCharacterSet = 'ISO_IR 100'
        ds.ImageType = 'ORIGINAL\\PRIMARY\\AXIAL\\HELIX'
        dt = datetime.datetime.now()
        datestr = dt.strftime('%Y%m%d')
        timestr1 = dt.strftime('%H%M%S.%f')
        timestr2 = dt.strftime('%H%M%S')

        ds.InstanceCreationDate = datestr
        ds.InstanceCreationTime = timestr1
        ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
        ds.SOPInstanceUID = sop_instance_uid
        ds.StudyDate = datestr
        ds.SeriesDate = datestr
        ds.ContentDate = datestr
        ds.StudyTime = timestr2
        ds.SeriesTime = timestr2
        ds.ContentTime = timestr2
        ds.AccessionNumber = '116939'
        ds.Modality = 'CT'
        ds.Manufacturer = 'Philips'
        ds.InstitutionName = 'Cancer Hosipital'
        ds.ReferringPhysicianName = '79309'
        ds.StationName = '-'
        ds.StudyDescription = '-'
        ds.PatientName = 'patientxxx'
        ds.PatientID = 'patientxxx'
        ds.PatientBirthDate = '19000101'
        ds.PatientBirthTime = '000000'
        ds.PatientAge = '119'
        ds.PatientSex = '-'
        ds.PatientSize = 0
        ds.PatientWeight = 0
        ds.PatientAddress = '-'
        ds.PatientComments = '-'
        ds.SliceThickness = series_dataset[1]['space directions'][2][2]
        ds.KVP = 1111111
        patient_position = series_dataset[1]['space']
        #if patient_position == 'left-posterior-superior':
        ds.PatientPosition = 'HFS'
        ds.StudyID = '-'
        ds.SeriesNumber = 1
        ds.ImageOrientationPatient = '1\\0\\0\\0\\1\\0'
        ds.SamplesPerPixel = 1
        ds.Rows = series_dataset[1]['sizes'][1]
        ds.Columns = series_dataset[1]['sizes'][0]
        spacing_x = series_dataset[1]['space directions'][0][0]
        spacing_y = series_dataset[1]['space directions'][1][1]
        ds.PixelSpacing = '%f\\%f' % (spacing_x, spacing_y)
        ds.BitsAllocated = 16
        ds.BitsStored = 12
        ds.HighBit = 11
        ds.PixelRepresentation = 0
        ds.WindowCenter = 60
        ds.WindowWidth = 350
        ds.RescaleIntercept = -1024
        ds.RescaleSlope = 1
        ds.PixelData = __create_pixel_data_array_nrrd(series_dataset, j)
        ds.save_as(filename, write_like_original=False)
        dataset_list.append(ds)
    print('end: write dicom series')
    return dataset_list

def __create_label_volume_data_list_nrrd(one_case_rtstruct_file_path):
    result_label_volume_data_list = []
    roi_file_name_list = os.listdir(one_case_rtstruct_file_path)
    for roi_file_name in roi_file_name_list:
        one_roi_file_full_path = os.path.join(one_case_rtstruct_file_path, roi_file_name)
        label_dataset = nrrd.read(one_roi_file_full_path)

        volume_size = label_dataset[1]['sizes']
        one_label_volume = np.zeros([volume_size[2], volume_size[1], volume_size[0]], np.uint8)
        for i in range(volume_size[2]):
            for j in range(volume_size[1]):
                one_label_volume[i, j, :] = label_dataset[0][:, j, i] * 255

        roi_name = roi_file_name[0:-5]
        one_label_volume_data = [roi_name, one_label_volume]
        result_label_volume_data_list.append(one_label_volume_data)

    return result_label_volume_data_list

def nrrd_2_dicomrt(src_case_root_path, dst_case_root_path):
    pid_list = os.listdir(src_case_root_path)
    case_nums = len(pid_list)
    for i in range(case_nums):
        print('start: ' + pid_list[i])
        dst_one_case_file_path = os.path.join(dst_case_root_path, pid_list[i])
        one_case_root_path = os.path.join(src_case_root_path, pid_list[i])
        one_case_file_path = os.path.join(one_case_root_path, 'img.nrrd')

        series_dataset = nrrd.read(one_case_file_path)
        dataset_list = __write_dicom_series_file_nrrd(series_dataset, dst_one_case_file_path)

        one_case_rtstruct_file_path = os.path.join(one_case_root_path, 'structures')
        label_volume_data_list = __create_label_volume_data_list_nrrd(one_case_rtstruct_file_path)
        one_case_dicom_data = [pid_list[i], dataset_list, dst_one_case_file_path]
        dcmio.write_rtstruct(one_case_dicom_data, label_volume_data_list)
        print('end: ' + pid_list[i])

def dicomrt_2_nrrd(src_case_root_path, dst_case_root_path, modality):
    pass
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def __write_jpg_series_file(series_dataset, dst_one_case_root_path, ww, wl):
    if not os.path.exists(dst_one_case_root_path):
        os.makedirs(dst_one_case_root_path)

    volume_data_in_ct_value = dcmio.get_image_volume_data_ctvalue(series_dataset)
    result_volume_data_8bit = dcmio.set_ww_wl(volume_data_in_ct_value, ww, wl)
    slice_nums = result_volume_data_8bit.shape[0]
    for i in range(slice_nums):
        file_full_path = os.path.join(dst_one_case_root_path, '%03d.jpg' % (i))
        cv.imwrite(file_full_path, result_volume_data_8bit[i])
    pass

def __write_jpg_label_file(series_dataset_list, rtstruct_dataset, dst_one_case_root_path):
    if not os.path.exists(dst_one_case_root_path):
        os.makedirs(dst_one_case_root_path)

    dict_referenced_sop_instance_uid_index = {}
    contour_image_sequence = rtstruct_dataset.ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[0].ContourImageSequence
    slices = len(contour_image_sequence)
    for i in range(slices):
        dict_referenced_sop_instance_uid_index[contour_image_sequence[i].ReferencedSOPInstanceUID] = i

    rows = series_dataset_list[0].Rows
    cols = series_dataset_list[0].Columns
    roi_list = dcmio.__get_roi_list_from_rs_file(rtstruct_dataset)
    for roi in roi_list:
        label_volume_data = dcmio.__create_label_volume(roi, dict_referenced_sop_instance_uid_index, series_dataset_list)
        roi_name = roi[0]
        one_case_labels_path = os.path.join(dst_one_case_root_path, roi_name)
        if not os.path.exists(one_case_labels_path):
            os.makedirs(one_case_labels_path)
        for i in range(slices):
            one_label_slice_file_path = os.path.join(one_case_labels_path, '%03d.jpg' % (i))
            cv.imwrite(one_label_slice_file_path, label_volume_data[i])
    pass

def dicomrt_2_jpg(src_case_root_path, dst_case_root_path, ww, wl):
    pid_list = os.listdir(src_case_root_path)
    case_nums = len(pid_list)
    for i in range(case_nums):
        print('start: ' + pid_list[i])

        src_one_case_root_path = os.path.join(src_case_root_path, pid_list[i])

        file_meta, series_dataset_list = dcmio.read_one_series(src_one_case_root_path, modality='CT')
        dst_one_case_series_root_path = os.path.join(dst_case_root_path, pid_list[i], 'images')
        __write_jpg_series_file(series_dataset_list, dst_one_case_series_root_path, ww, wl)

        file_meta, rtstruct_dataset_list = dcmio.read_one_series(src_one_case_root_path, modality='RTSTRUCT')
        one_case_rtstruct_root_path = os.path.join(dst_case_root_path, pid_list[i], 'labels')
        __write_jpg_label_file(series_dataset_list, rtstruct_dataset_list[0], one_case_rtstruct_root_path)

        print('end: ' + pid_list[i])
    pass
#------------------------------------------------------------------------------