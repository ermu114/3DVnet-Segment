import predict.predictor as predictor_vnet3d
import os
import utility.iodicom as dcmio

def test_predict_sol():
    case_root_path = 'D:\\tmp\\MIS\\train'
    model_root_path = 'D:\\tmp\\MIS\\model'
    loc_root_path = 'D:\\tmp\\MIS\\location'
    # one_case_path =  'D:\\tmp\\MIS\\test\\009'
    modality = 'MR'
    root_path = 'D:\\tmp\\MIS\\test'
    pid_list = os.listdir(root_path)
    for i in range(len(pid_list)):
        one_case_path = os.path.join(root_path, pid_list[i])
        # predictor_vnet3d.predict_sol(modality, case_root_path, model_root_path, loc_root_path)
        result_volume_data_list, dataset_list = predictor_vnet3d.predict_segmentation(modality, one_case_path, model_root_path)
        pid = os.path.basename(one_case_path)
        one_case_data = [pid, dataset_list, one_case_path]
        dcmio.write_rtstruct(one_case_data, result_volume_data_list)
    pass

test_predict_sol()