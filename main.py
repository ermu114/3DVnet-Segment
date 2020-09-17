import train.trainer as trainer
import predict.predictor as vnet3d_predictor
import utility.iodicom as dcmio
import os
import data.preprocessing_test as t

dir_base = 'D:\\MLDO\\'
g_modality = 'CT'

def train_test():
    dicom_case_root_path = dir_base + 'datasets\\dicomrt\\HaN_OAR\\train'
    jpg_case_root_path = dir_base + 'datasets\\jpg\\HaN_OAR\\train'
    log_root_path = dir_base + 'log'
    model_root_path = dir_base + 'model'
    loc_root_path = dir_base + 'location'
    vnet3d_trainer = trainer.VNet3DTrainer(dicom_case_root_path, model_root_path, log_root_path)
    #vnet3d_trainer.train_location(jpg_case_root_path, g_modality)
    #vnet3d_predictor.predict_location(g_modality, dicom_case_root_path, model_root_path, loc_root_path)
    vnet3d_trainer.train_segmentation(g_modality, loc_root_path)
    pass

def predict_test():
    case_root_path = dir_base + 'datasets\\dicomrt\\HaN_OAR\\test'
    model_root_path = dir_base + 'model'
    pid_list = os.listdir(case_root_path)
    for i in range(len(pid_list)):
        one_case_path = os.path.join(case_root_path, pid_list[i])
        result_volume_data_list, dataset_list = vnet3d_predictor.predict_segmentation(g_modality, one_case_path, model_root_path)
        pid = os.path.basename(one_case_path)
        one_case_data = [pid, dataset_list, one_case_path]
        dcmio.write_rtstruct(one_case_data, result_volume_data_list)
    pass

if __name__ == '__main__':
    #t.start_preproc_4_train_loc()
    #train_test()
    predict_test()

    print('end')

