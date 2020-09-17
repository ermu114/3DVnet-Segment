import utility.iodicom as dcmio
import utility.parameters as param
import data.dataset as ds
import predict.predictor_interface as prdct

predictor = prdct.IPredictor()

def initialize(model_root_path):
    print('initializing...')
    predictor.initialize(model_root_path)

def uninitialize():
    print('uninitializing...')
    predictor.uninitialize()

def predict(one_case_path, modality):
    data_source = ds.Dataset(one_case_path, modality)
    params = param.dict_supper_parameters
    list_label_volume_data = []
    for roi_name in params.keys():
        print('Predicting %s' % (roi_name))
        result_label_volume_data = predictor.predict(roi_name, data_source)
        list_label_volume_data.append([roi_name, result_label_volume_data])
    list_dicom_dataset = data_source.get_dicom_dataset_list()
    roi_list = dcmio.get_roi_list(list_dicom_dataset, list_label_volume_data)
    result_roi_list = []
    roi_nums = len(roi_list)
    for i in range(roi_nums):
        roi = roi_list[i]
        roi_name = roi[0]
        contour_list = roi[5]
        result_contour_list = []
        contour_nums = len(contour_list)
        for j in range(contour_nums):
            contour = contour_list[j]
            point_list = contour[1]
            result_point_list = []
            point_nums = len(point_list)
            # if point_nums < 3:
            #     continue;
            for k in range(point_nums):
                point = point_list[k]
                result_point = [point[0,0], point[0,1], point[0,2]]
                result_point_list.append(result_point)
            result_contour_list.append(result_point_list)
        result_roi_list.append([roi_name, result_contour_list])
    return result_roi_list

