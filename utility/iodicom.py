import pydicom
from pydicom.sequence import Sequence
from pydicom.dataset import Dataset, FileDataset
import os
import numpy as np
import cv2 as cv
import datetime

def list_all_files_recursive(root_path):
    _files = []
    list = os.listdir(root_path)
    for i in range(0,len(list)):
           path = os.path.join(root_path, list[i])
           if os.path.isdir(path):
              _files.extend(list_all_files_recursive(path))
           if os.path.isfile(path):
              _files.append(path)
    return _files

#------------------------------------------------------------------------------
def __get_origin_matrix_normalized(dicom_dataset_list):
    origin_matrix = __get_origin_matrix(dicom_dataset_list)
    result_origin_matrix = np.r_[origin_matrix, np.matrix([1])]
    return result_origin_matrix

def __get_orientation_matrix_normalized(dicom_dataset_list):
    orientation_matrix = __get_orientation_matrix(dicom_dataset_list)
    new_cols = np.zeros([3, 1])
    temp_matrix = np.c_[orientation_matrix, new_cols]
    new_rows = np.array([[0, 0, 0, 1]])
    result_orientation_matrix = np.r_[temp_matrix, new_rows]
    return result_orientation_matrix

def __get_spacing_matrix_normalized(dicom_dataset_list):
    spacing_matrix = __get_spacing_matrix(dicom_dataset_list)
    new_cols = np.zeros([3, 1])
    temp_matrix = np.c_[spacing_matrix, new_cols]
    new_rows = np.array([[0, 0, 0, 1]])
    result_spacing_matrix = np.r_[temp_matrix, new_rows]
    return result_spacing_matrix

def __get_origin_matrix(dicom_dataset_list):
    first_dataset_item = dicom_dataset_list[0]
    image_position = first_dataset_item.ImagePositionPatient
    origin_matrix = np.matrix(image_position).T
    return origin_matrix

def __get_orientation_matrix(dicom_dataset_list):
    first_dataset_item = dicom_dataset_list[0]
    image_orientation = first_dataset_item.ImageOrientationPatient
    orientation_matrix = np.matrix(np.zeros((3, 3), dtype=np.float))
    orientation_matrix[0, 0] = image_orientation[0]
    orientation_matrix[1, 0] = image_orientation[1]
    orientation_matrix[2, 0] = image_orientation[2]
    orientation_matrix[0, 1] = image_orientation[3]
    orientation_matrix[1, 1] = image_orientation[4]
    orientation_matrix[2, 1] = image_orientation[5]
    orientation_matrix[0, 2] = image_orientation[1] * image_orientation[5] - image_orientation[2] * image_orientation[4]
    orientation_matrix[1, 2] = image_orientation[2] * image_orientation[3] - image_orientation[0] * image_orientation[5]
    orientation_matrix[2, 2] = image_orientation[0] * image_orientation[4] - image_orientation[1] * image_orientation[3]
    return orientation_matrix

def __get_spacing_matrix(dicom_dataset_list):
    first_dataset_item = dicom_dataset_list[0]
    pixel_spacing = first_dataset_item.PixelSpacing
    slice_thickness = first_dataset_item.SliceThickness
    spacing_matrix = np.matrix(np.diag([pixel_spacing[0], pixel_spacing[1], slice_thickness]))
    return spacing_matrix

def __transform_index_2_dicom_volume(dicom_dataset_list, point_list):
    origin_matrix = __get_origin_matrix(dicom_dataset_list)
    orientation_matrix = __get_orientation_matrix(dicom_dataset_list)
    spacing_matrix = __get_spacing_matrix(dicom_dataset_list)
    point_nums = len(point_list)
    result_point_list = []
    for i in range(point_nums):
        point_matrix = np.matrix(point_list[i]).T
        result_point = origin_matrix + orientation_matrix*spacing_matrix*point_matrix
        result_point_list.append(result_point.T)
    return result_point_list

def __transform_dicom_volume_2_index(dicom_dataset_list, point_list):
    origin_matrix = __get_origin_matrix(dicom_dataset_list)
    orientation_matrix = __get_orientation_matrix(dicom_dataset_list)
    spacing_matrix = __get_spacing_matrix(dicom_dataset_list)
    point_nums = len(point_list)
    result_point_list = []
    for i in range(point_nums):
        point_matrix = np.matrix(point_list[i]).T
        result_point = spacing_matrix.I * orientation_matrix.T * (point_matrix - origin_matrix)
        result_point_list.append(result_point.T)
    return result_point_list

def __get_referenced_sop_instance_uids(src_dicomdataset_list):
    num = len(src_dicomdataset_list)
    dict_instance_num_uids = {}
    for i in range(num):
        dict_instance_num_uids[src_dicomdataset_list[i].InstanceNumber] = \
            src_dicomdataset_list[i].SOPInstanceUID

    return dict_instance_num_uids

def __get_contour_list(src_dicomdataset_list, label_volume_data):
    dict_instance_num_uids = __get_referenced_sop_instance_uids(src_dicomdataset_list)

    width = src_dicomdataset_list[0].Columns
    height = src_dicomdataset_list[0].Rows
    current_height = len(label_volume_data[0])
    current_width = len(label_volume_data[0][0])
    threshold_val = 127
    contour_list = []
    label_slices = len(label_volume_data)
    for i in range(label_slices):
        one_image_data = np.reshape(label_volume_data[i], (current_height, current_width))
        one_image_data = cv.resize(one_image_data, (width, height), interpolation=cv.INTER_CUBIC)
        cv.threshold(one_image_data, threshold_val, 255, cv.THRESH_BINARY, one_image_data)
        contours, hierarchy = cv.findContours(one_image_data, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        contour_nums = len(contours)
        if contour_nums > 20:
            continue
        for j in range(contour_nums):
            one_contour = contours[j]
            point_list = []
            for k in range(len(one_contour)):
                one_point = np.matrix([one_contour[k][0][0], one_contour[k][0][1], i])
                point_list.append(one_point)
                result_point_list = __transform_index_2_dicom_volume(src_dicomdataset_list, point_list)
            # contour = [referenced_sop_instance_uid-point_list]
            referenced_sop_instance_uid = dict_instance_num_uids[i+1]
            contour = [referenced_sop_instance_uid, result_point_list]
            contour_list.append(contour)

    return contour_list

def get_roi_list(src_dicomdataset_list, label_volume_data_list):
    # roi=[roi_name-roi_number-roi_type-roi_display_olor-referenced_frame_of_reference_uid-contour_list]
    roi_list = []
    roi_nums = len(label_volume_data_list)
    for i in range(roi_nums):
        roi_name = label_volume_data_list[i][0]
        roi_number = i + 1
        roi_type = 'ORGAN'
        r_channel = np.random.randint(0, 255)
        g_channel = np.random.randint(0, 255)
        b_channel = np.random.randint(0, 255)
        roi_display_olor = '%d\\%d\\%d' % (r_channel, g_channel, b_channel)
        referenced_frame_of_reference_uid = src_dicomdataset_list[0].FrameOfReferenceUID
        contour_list = __get_contour_list(src_dicomdataset_list, label_volume_data_list[i][1])
        roi = [roi_name, roi_number, roi_type, roi_display_olor, referenced_frame_of_reference_uid, contour_list]
        roi_list.append(roi)
    return roi_list

def __write_file_meta(ds, sop_instance_uid):
    ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
    ds.is_little_endian = True
    ds.is_implicit_VR = True
    ds.file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.481.3'
    ds.file_meta.MediaStorageSOPInstanceUID = sop_instance_uid
    ds.fix_meta_info()

def __write_dicom_basic_info(ds, src_dicomdataset_list):
    first_slice_dataset = src_dicomdataset_list[0]
    ds.SOPClassUID = ds.file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
    ds.StudyDate = first_slice_dataset.StudyDate
    ds.StudyTime = first_slice_dataset.StudyTime
    ds.AccessionNumber = first_slice_dataset.AccessionNumber
    ds.Modality = 'RTSTRUCT'
    ds.Manufacturer = first_slice_dataset.Manufacturer
    ds.ReferringPhysicianName = first_slice_dataset.ReferringPhysicianName
    ReferencedStudySequenceItem = Dataset()
    ReferencedStudySequenceItem.ReferencedSOPClassUID = '1.2.840.10008.3.1.2.3.2'
    ReferencedStudySequenceItem.ReferencedSOPInstanceUID = first_slice_dataset.StudyInstanceUID
    ds.ReferencedStudySequence = Sequence([ReferencedStudySequenceItem])
    ds.PatientName = first_slice_dataset.PatientName
    ds.PatientID = first_slice_dataset.PatientID
    ds.PatientBirthDate = first_slice_dataset.PatientBirthDate
    ds.PatientSex = first_slice_dataset.PatientSex
    ds.PatientAge = first_slice_dataset.PatientAge
    # ds.PatientSize = first_slice_dataset.PatientSize
    ds.PatientWeight = first_slice_dataset.PatientWeight
    ds.PatientAddress = first_slice_dataset.PatientAddress
    #ds.PatientTelephoneNumber = first_slice_dataset.PatientTelephoneNumber
    ds.PatientComments = first_slice_dataset.PatientComments
    ds.StudyInstanceUID = first_slice_dataset.StudyInstanceUID
    ds.SeriesInstanceUID = pydicom.uid.generate_uid()
    ds.StudyID = first_slice_dataset.StudyID
    ds.SeriesNumber = first_slice_dataset.SeriesNumber
    ds.StructureSetLabel = 'RTStructureSet'
    ds.StructureSetName = 'RTStructureSet'
    dt = datetime.datetime.now()
    ds.StructureSetDate = dt.strftime('%Y%m%d')
    ds.StructureSetTime = dt.strftime('%H%M%S')

def __write_referenced_frame_of_reference_sequence(ds, src_dicomdataset_list):
    ds.ReferencedFrameOfReferenceSequence = Sequence()
    ReferencedFrameOfReferenceSequenceItem = Dataset()
    ds.ReferencedFrameOfReferenceSequence.append(ReferencedFrameOfReferenceSequenceItem)
    ReferencedFrameOfReferenceSequenceItem.FrameOfReferenceUID = src_dicomdataset_list[0].FrameOfReferenceUID
    ReferencedFrameOfReferenceSequenceItem.RTReferencedStudySequence = Sequence()
    RTReferencedStudySequenceItem = Dataset()
    ReferencedFrameOfReferenceSequenceItem.RTReferencedStudySequence.append(RTReferencedStudySequenceItem)
    RTReferencedStudySequenceItem.ReferencedSOPClassUID = '1.2.840.10008.3.1.2.3.2'
    RTReferencedStudySequenceItem.ReferencedSOPInstanceUID = ds.StudyInstanceUID
    RTReferencedStudySequenceItem.RTReferencedSeriesSequence = Sequence()
    RTReferencedSeriesSequenceItem = Dataset()
    RTReferencedStudySequenceItem.RTReferencedSeriesSequence.append(RTReferencedSeriesSequenceItem)
    RTReferencedSeriesSequenceItem.SeriesInstanceUID = src_dicomdataset_list[0].SeriesInstanceUID
    RTReferencedSeriesSequenceItem.ContourImageSequence = Sequence()
    #'''
    slices = len(src_dicomdataset_list)
    for i in range(slices):
        ContourImageSequenceItem = Dataset()
        ContourImageSequenceItem.ReferencedSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
        ContourImageSequenceItem.ReferencedSOPInstanceUID = src_dicomdataset_list[i].SOPInstanceUID
        RTReferencedSeriesSequenceItem.ContourImageSequence.append(ContourImageSequenceItem)
    #'''
    '''
    roi_nums = len(roi_list)
    for i in range(roi_nums):
        roi = roi_list[i]
        contour_list = roi[5]
        contour_nums = len(contour_list)
        for j in range(contour_nums):
            ContourImageSequenceItem = Dataset()
            ContourImageSequenceItem.ReferencedSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
            ContourImageSequenceItem.ReferencedSOPInstanceUID = contour_list[j][0]
            RTReferencedSeriesSequenceItem.ContourImageSequence.append(ContourImageSequenceItem)
    '''

def __write_structure_set_roi_sequence(ds, src_dicomdataset_list, roi_list):
    ds.StructureSetROISequence = Sequence()
    roi_nums = len(roi_list)
    for i in range(roi_nums):
        StructureSetROISequenceItem = Dataset()
        ds.StructureSetROISequence.append(StructureSetROISequenceItem)
        StructureSetROISequenceItem.ROINumber = i + 1
        StructureSetROISequenceItem.ReferencedFrameOfReferenceUID = src_dicomdataset_list[0].FrameOfReferenceUID
        StructureSetROISequenceItem.ROIName = roi_list[i][0]
        StructureSetROISequenceItem.ROIGenerationAlgorithm = 'MANUAL'

def __write_roi_contour_sequence(ds, roi_list):
    ds.ROIContourSequence = Sequence()
    roi_nums = len(roi_list)
    for i in range(roi_nums):
        ROIContourSequenceItem = Dataset()
        ds.ROIContourSequence.append(ROIContourSequenceItem)
        ROIContourSequenceItem.ROIDisplayColor = roi_list[i][3] # roi_display_color
        ROIContourSequenceItem.ContourSequence = Sequence()
        contour_list = roi_list[i][5]
        contour_nums = len(contour_list)
        for j in range(contour_nums):
            contour = contour_list[j]
            ContourSequenceItem = Dataset()
            ROIContourSequenceItem.ContourSequence.append(ContourSequenceItem)
            ContourSequenceItem.ContourImageSequence = Sequence()
            ContourImageSequenceItem = Dataset()
            ContourSequenceItem.ContourImageSequence.append(ContourImageSequenceItem)
            ContourImageSequenceItem.ReferencedSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
            ContourImageSequenceItem.ReferencedSOPInstanceUID = contour[0]
            ContourSequenceItem.ContourGeometricType = 'CLOSED_PLANAR'
            #point_nums = len(contour_item[2][0])
            point_nums = len(contour[1])
            ContourSequenceItem.NumberOfContourPoints = point_nums
            ContourSequenceItem.ContourNumber = j + 1
            result_point_list = contour[1]
            point_list_string = ''
            for k in range(point_nums):
                point_list_string += '%f\\%f\\%f' % (result_point_list[k][0, 0], result_point_list[k][0, 1], result_point_list[k][0, 2])
                if k != point_nums - 1:
                    point_list_string += '\\'
            #ContourSequenceItem.ContourData = '29.420486\\13.750359\\-265.5\\29.271103\\15.171627\\-265.5\\29.085907\\15.862774\\-265.5\\28.114929\\17.768417\\-265.5\\27.664627\\18.324493\\-265.5\\26.602615\\19.280743\\-265.5\\26.002518\\19.670452\\-265.5\\24.005814\\20.436909\\-265.5\\23.299103\\20.548843\\-265.5\\21.86998\\20.548843\\-265.5\\21.163269\\20.436909\\-265.5\\19.16658\\19.670452\\-265.5\\18.566467\\19.280746\\-265.5\\17.054153\\17.768417\\-265.5\\16.664444\\17.168327\\-265.5\\15.897995\\15.171627\\-265.5\\15.786057\\14.464905\\-265.5\\15.786057\\13.035805\\-265.5\\15.897995\\12.329084\\-265.5\\16.664444\\10.332386\\-265.5\\17.054153\\9.7322922\\-265.5\\18.010406\\8.6702652\\-265.5\\18.566467\\8.219965\\-265.5\\19.16658\\7.8302593\\-265.5\\19.804123\\7.5054169\\-265.5\\21.163269\\7.0638008\\-265.5\\21.86998\\6.9518661\\-265.5\\23.299103\\6.9518661\\-265.5\\24.005814\\7.0638008\\-265.5\\26.002518\\7.8302593\\-265.5\\26.602615\\8.2199688\\-265.5\\28.114929\\9.7322941\\-265.5\\28.504639\\10.332388\\-265.5\\28.829483\\10.969931\\-265.5\\29.085907\\11.637936\\-265.5\\29.383026\\13.035809\\-265.5'
            ContourSequenceItem.ContourData = point_list_string
        ROIContourSequenceItem.ReferencedROINumber = roi_list[i][1] # roi_number

def __write_rt_roi_observations_sequence(ds, roi_list):
    ds.RTROIObservationsSequence = Sequence()
    roi_nums = len(roi_list)
    for i in range(roi_nums):
        RTROIObservationsSequenceItem = Dataset()
        ds.RTROIObservationsSequence.append(RTROIObservationsSequenceItem)
        RTROIObservationsSequenceItem.ObservationNumber = roi_list[i][1]
        RTROIObservationsSequenceItem.ReferencedROINumber = roi_list[i][1]
        RTROIObservationsSequenceItem.ROIObservationLabel = roi_list[i][0]
        RTROIObservationsSequenceItem.RTROIInterpretedType = roi_list[i][2]

def __get_contour_list_from_rs_file(ContourSequence):
    contour_list = []
    contour_nums = len(ContourSequence)
    for i in range(contour_nums):
        ReferencedSOPInstanceUID = ContourSequence[i].ContourImageSequence[0].ReferencedSOPInstanceUID
        contour_data = ContourSequence[i].ContourData
        point_list = []
        point_nums = (len(contour_data) // 3)
        for j in range(point_nums):
            point_start_index = 3 * j
            one_point = np.matrix([contour_data[point_start_index], contour_data[point_start_index + 1],
                                   contour_data[point_start_index + 2]])
            point_list.append(one_point)
        contour = [ReferencedSOPInstanceUID, point_list]
        contour_list.append(contour)
    return contour_list

def __get_roi_list_from_rs_file(rtstruct_dataset):
    roi_list = []
    # roi=[roi_name-roi_number-roi_type-roi_display_olor-referenced_frame_of_reference_uid-contour_list]
    roi_nums = len(rtstruct_dataset.StructureSetROISequence)
    for i in range(roi_nums):
        roi_name = rtstruct_dataset.StructureSetROISequence[i].ROIName
        roi_number = rtstruct_dataset.StructureSetROISequence[i].ROINumber
        roi_type = rtstruct_dataset.RTROIObservationsSequence[i].RTROIInterpretedType
        roi_display_color = rtstruct_dataset.ROIContourSequence[i].ROIDisplayColor
        referenced_frame_of_reference_uid = rtstruct_dataset.StructureSetROISequence[
            i].ReferencedFrameOfReferenceUID
        contour_list = __get_contour_list_from_rs_file(rtstruct_dataset.ROIContourSequence[i].ContourSequence)
        roi = [roi_name, roi_number, roi_type, roi_display_color, referenced_frame_of_reference_uid, contour_list]
        roi_list.append(roi)

    return roi_list

def __get_roi_from_rs_file(roi_name, rtstruct_dataset):
    roi = []
    # roi=[roi_name-roi_number-roi_type-roi_display_olor-referenced_frame_of_reference_uid-contour_list]
    roi_nums = len(rtstruct_dataset.StructureSetROISequence)
    for i in range(roi_nums):
        if rtstruct_dataset.StructureSetROISequence[i].ROIName != roi_name:
            continue
        roi_number = rtstruct_dataset.StructureSetROISequence[i].ROINumber
        roi_type = rtstruct_dataset.RTROIObservationsSequence[i].RTROIInterpretedType
        roi_display_color = rtstruct_dataset.ROIContourSequence[i].ROIDisplayColor
        referenced_frame_of_reference_uid = rtstruct_dataset.StructureSetROISequence[
            i].ReferencedFrameOfReferenceUID
        contour_list = __get_contour_list_from_rs_file(rtstruct_dataset.ROIContourSequence[i].ContourSequence)
        roi.append(roi_name)
        roi.append(roi_number)
        roi.append(roi_type)
        roi.append(roi_display_color)
        roi.append(referenced_frame_of_reference_uid)
        roi.append(contour_list)

    return roi

def __create_label_volume(roi, dict_referenced_sop_instance_uid_index, ct_dataset_list):
    rows = ct_dataset_list[0].Rows
    cols = ct_dataset_list[0].Columns
    slices = len(dict_referenced_sop_instance_uid_index)
    label_volume_data = np.zeros([slices, rows, cols], np.uint8)
    contour_nums = len(roi[5])
    for i in range(contour_nums):
        contour = roi[5][i]
        referenced_sop_instance_uid = contour[0]
        result_point_list = __transform_dicom_volume_2_index(ct_dataset_list, contour[1])
        point_nums = len(result_point_list)
        points_array = np.zeros((point_nums, 2), dtype=np.int)
        for j in range(point_nums):
            points_array[j][0] = int(result_point_list[j][0][0, 0])
            points_array[j][1] = int(result_point_list[j][0][0, 1])
        points_array = points_array.reshape(-1, 1, 2)
        index = dict_referenced_sop_instance_uid_index[referenced_sop_instance_uid]
        label_volume_data[index] = cv.fillPoly(label_volume_data[index], [points_array], (255, 0, 0))
    return label_volume_data
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def set_ww_wl(image_data, ww, wl):
    # result_image_data = (255 / ww) * (image_data - wl + ww / 2)
    # for j in range(image_data.shape[0]):
    #     for k in range(image_data.shape[1]):
    #         if result_image_data[j, k] > 255:
    #             result_image_data[j, k] = 255
    #         if result_image_data[j, k] < 0:
    #             result_image_data[j, k] = 0
    # result_image_data = result_image_data.astype(np.uint8)
    result_image_data = (255 / ww) * (image_data - wl + ww / 2)
    result_image_data = np.clip(result_image_data, 0, 255)
    result_image_data = result_image_data.astype(np.uint8)

    return result_image_data

def get_dicom_series_file_list_sorted(root_dir, series_modality='CT'):
    slice_location_file_list = []

    file_list = list_all_files_recursive(root_dir)
    file_nums = len(file_list)
    for i in range(file_nums):
        dataset = pydicom.filereader.read_file(file_list[i], force=True)
        if dataset.Modality == series_modality:
            pos = dataset.SliceLocation
            slice_location_file_list.append([pos, file_list[i]])

    slice_location_file_list.sort()
    slices = len(slice_location_file_list)
    sorted_dicom_series_file_list = []
    for i in range(slices):
        sorted_dicom_series_file_list.append(slice_location_file_list[i][1])

    return sorted_dicom_series_file_list, slices

def get_dicom_rtstruct_file_list(root_dir):
    sorted_dicom_rtstruct_file_list = []

    file_list = list_all_files_recursive(root_dir)
    file_nums = len(file_list)
    for i in range(file_nums):
        dataset = pydicom.filereader.read_file(file_list[i])
        if dataset.Modality == 'RTSTRUCT':
            sorted_dicom_rtstruct_file_list.append(file_list[i])

    return sorted_dicom_rtstruct_file_list

def read_all_series(case_root_path, modality='CT'):
    case_list = os.listdir(case_root_path)
    case_nums = len(case_list)
    all_case_data_list = []
    for i in range(case_nums):
        one_case_path = os.path.join(case_root_path, case_list[i])
        sorted_dicom_series_file_list, slices = get_dicom_series_file_list_sorted(one_case_path, series_modality=modality)
        file_meta = pydicom.filereader.read_file_meta_info(os.path.join(case_root_path, sorted_dicom_series_file_list[0]))
        dataset_list = []
        for i in range(slices):
            one_dataset = pydicom.filereader.read_file(os.path.join(case_root_path, sorted_dicom_series_file_list[i]))
            dataset_list.append(one_dataset)
        all_case_data_list.append([case_list[i], file_meta, dataset_list, one_case_path])
    return all_case_data_list

# def normalize_training_data(training_data, real_slices):
#     max_val_array = np.zeros([real_slices, 1], np.int)
#     min_val_array = np.zeros([real_slices, 1], np.int)
#     for i in range(real_slices):
#         temp_max_val = training_data[i].max()
#         temp_min_val = training_data[i].min()
#         max_val_array[i] = temp_max_val
#         min_val_array[i] = temp_min_val
#     max_val = max_val_array.max()
#     min_val = min_val_array.min()
#     # assert max_val - min_val != 0
#     if max_val - min_val != 0:
#         training_data = training_data / (max_val - min_val)
#     return training_data
#
# def normalize_volume_data(volume_data):
#     max_val = max(volume_data)
#     min_val = min(volume_data)
#     assert max_val - min_val != 0
#     volume_data = volume_data / (max_val - min_val)
#     return volume_data

def read_one_series(one_case_path, modality='CT'):
    if modality == 'RTSTRUCT':
        dicom_rtstruct_file_list = get_dicom_rtstruct_file_list(one_case_path)
        file_meta_rtstruct = pydicom.filereader.read_file_meta_info(dicom_rtstruct_file_list[0])
        dataset_rtstruct = pydicom.filereader.read_file(dicom_rtstruct_file_list[0])
        return file_meta_rtstruct, dataset_rtstruct
    else:
        sorted_dicom_series_file_list, slices = get_dicom_series_file_list_sorted(one_case_path, series_modality=modality)
        file_meta_image = pydicom.filereader.read_file_meta_info(sorted_dicom_series_file_list[0])
        list_dataset_image = []
        for i in range(slices):
            one_dataset = pydicom.filereader.read_file(sorted_dicom_series_file_list[i])
            list_dataset_image.append(one_dataset)
        return file_meta_image, list_dataset_image

def get_image_volume_data_ctvalue(dataset_list):
    slices = len(dataset_list)
    volume_data_array = np.zeros([slices, dataset_list[0].Rows, dataset_list[0].Columns], np.int16)
    for i in range(slices):
        volume_data_array[i] = dataset_list[i].pixel_array.astype(np.int16)
        rescale_intercept = dataset_list[i].RescaleIntercept
        rescale_slope = dataset_list[i].RescaleSlope
        volume_data_array[i] = volume_data_array[i] * rescale_slope + rescale_intercept
    return volume_data_array

# def create_training_image_data(image_volume_data, image_depth, image_height, image_width, image_channels):
#     image_data = np.zeros([1, image_depth, image_height, image_width, image_channels], dtype=np.float)
#     slices = len(image_volume_data)
#     for i in range(slices):
#         one_image_data = cv.resize(image_volume_data[i], (image_width, image_height), interpolation=cv.INTER_CUBIC)
#         one_image_data = np.reshape(one_image_data, (image_height, image_width, 1))
#         image_data[0, i] = one_image_data.astype(np.float)
#     image_data[0] = normalize_training_data(image_data[0], len(image_data[0]))
#     return image_data

def get_image_volume_data_8bit(dataset_list, ww, wl):
    return set_ww_wl(get_image_volume_data_ctvalue(dataset_list), ww, wl)

def get_label_volume_data(roi_name, ct_dataset_list, rtstruct_dataset):
    dict_referenced_sop_instance_uid_index = {}
    contour_image_sequence = rtstruct_dataset.ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[0].ContourImageSequence
    slices = len(contour_image_sequence)
    for i in range(slices):
        dict_referenced_sop_instance_uid_index[contour_image_sequence[i].ReferencedSOPInstanceUID] = i

    roi = __get_roi_from_rs_file(roi_name, rtstruct_dataset)

    label_volume_data = __create_label_volume(roi, dict_referenced_sop_instance_uid_index, ct_dataset_list)

    return label_volume_data

def write_rtstruct(src_dicom_data, label_volume_data_list):
    sop_instance_uid = pydicom.uid.generate_uid()
    filename = os.path.join(src_dicom_data[2], 'RS.' + sop_instance_uid + '.dcm')

    file_meta = Dataset()
    ds = FileDataset(filename, {}, file_meta=file_meta, preamble=b"\0" * 128)

    __write_file_meta(ds, sop_instance_uid)

    src_dicomdataset_list = src_dicom_data[1]
    __write_dicom_basic_info(ds, src_dicomdataset_list)
    roi_list = get_roi_list(src_dicomdataset_list, label_volume_data_list)
    __write_referenced_frame_of_reference_sequence(ds, src_dicomdataset_list)
    __write_structure_set_roi_sequence(ds, src_dicomdataset_list, roi_list)
    __write_roi_contour_sequence(ds, roi_list)
    __write_rt_roi_observations_sequence(ds, roi_list)

    ds.save_as(filename, write_like_original=False)
    print('end: write rtstructure')

def get_one_series_image_volume_data_raw(
        image_data, one_case_path, image_height, image_width, batch_index, modality='CT'):
    file_meta, ct_dataset_list = read_one_series(one_case_path, modality)
    image_volume_data = get_image_volume_data_ctvalue(ct_dataset_list)
    image_slices = len(image_volume_data)
    for j in range(image_slices):
        one_image_data = cv.resize(image_volume_data[j], (image_width, image_height), interpolation=cv.INTER_CUBIC)
        one_image_data = np.reshape(one_image_data, (image_height, image_width, 1))
        image_data[batch_index, j] = one_image_data
    return file_meta, ct_dataset_list, image_data

def get_one_series_image_volume_data_8bit(
        image_data, one_case_path, image_height, image_width, batch_index, ww, wl, modality='CT'):
    file_meta, ct_dataset_list = read_one_series(one_case_path, modality)
    image_volume_data = get_image_volume_data_8bit(ct_dataset_list, ww, wl)
    slices = len(image_volume_data)
    for j in range(slices):
        one_image_data = set_ww_wl(image_volume_data[j], ww, wl)
        one_image_data = cv.resize(one_image_data, (image_width, image_height), interpolation=cv.INTER_CUBIC)
        one_image_data = np.reshape(one_image_data, (image_height, image_width, 1))
        image_data[batch_index, j] = one_image_data
    origin_image_volume_data = set_ww_wl(image_volume_data, ww, wl)
    return file_meta, ct_dataset_list, image_data, origin_image_volume_data

def get_one_series_label_volume_data(
        label_data, one_case_path, image_height, image_width, batch_index, roi_name, ct_dataset_list):
    file_meta, rtstruct_dataset_list = read_one_series(one_case_path, 'RTSTRUCT')
    label_volume_data = get_label_volume_data(roi_name, ct_dataset_list, rtstruct_dataset_list[0])
    slices = len(label_volume_data)
    assert slices == len(ct_dataset_list)
    for j in range(slices):
        one_label_data = label_volume_data[j]
        one_label_data = cv.resize(one_label_data, dsize=(image_width, image_height),interpolation=cv.INTER_CUBIC)
        cv.threshold(one_label_data, 127, 255, cv.THRESH_BINARY, one_label_data)
        one_label_data = np.reshape(one_label_data, (image_height, image_width, 1))
        label_data[batch_index, j] = one_label_data.astype(np.float)
    return file_meta, rtstruct_dataset_list, label_data, slices
#------------------------------------------------------------------------------
