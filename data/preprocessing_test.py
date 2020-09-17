import data.preprocessing as preprocessor
import os
import utility.parameters as param

def start_preproc_4_train_loc():
    src_root_path = 'D:\\MLDO\\datasets\\dicomrt\\HaN_OAR\\train'
    dst_root_path = 'D:\\MLDO\\datasets\\jpg\\HaN_OAR\\train'

    pid_list = os.listdir(src_root_path)
    oar_list = param.dict_supper_parameters.keys()

    for roi_name in oar_list:
        for pid in pid_list:
            print('%s processing %s : %s' % (preprocessor.get_cur_time(), roi_name, pid))
            one_case_src_path = os.path.join(src_root_path, pid)
            image_volume_data, list_dicom_dataset = preprocessor.get_one_image_volume_data(one_case_src_path)
            label_volume_data = preprocessor.get_one_label_volume_data(one_case_src_path, roi_name, list_dicom_dataset)
            image_volume, label_volume = preprocessor.preprocessing_4_train_loc(image_volume_data, list_dicom_dataset, label_volume_data, roi_name)
            ono_case_dst_path = os.path.join(dst_root_path, roi_name, pid)
            preprocessor.save_images(image_volume, os.path.join(ono_case_dst_path, 'images'))
            preprocessor.save_images(label_volume, os.path.join(ono_case_dst_path, 'labels'))
            pass
    pass

# start_preproc_4_train_loc()