import utility.parameters as param
import os
import tensorflow as tf
import data.dataset as dataset
import cv2 as cv
import numpy as np

# def __predict(model_full_path, list_train_data):
#     dict_predictions = {}
#     tf.reset_default_graph()
#     graph = tf.Graph()
#     with tf.Session(graph=graph) as sess:
#         with tf.gfile.GFile(model_full_path + '.pb', 'rb') as f:
#             graph_def = tf.GraphDef()
#             graph_def.ParseFromString(f.read())
#             output_result, x_input, y_label, drop = tf.import_graph_def(
#                 graph_def, return_elements=['output/result:0', 'x_input:0', 'y_label:0', 'dropout:0'])
#
#             case_nums = len(list_train_data)
#             for i in range(case_nums):
#                 training_data_image, pid, real_slices = list_train_data[i]
#                 y_dummy = training_data_image
#                 prediction = sess.run(output_result,
#                                       feed_dict={x_input: training_data_image, y_label: y_dummy, drop: 1})
#                 dict_predictions.update({pid: prediction})
#     return dict_predictions

def __predict(model_full_path, list_one_case_train_data):
    tf.reset_default_graph()
    graph = tf.Graph()
    with tf.Session(graph=graph) as sess:
        with tf.gfile.GFile(model_full_path + '.pb', 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            output_result, x_input, y_label, drop = tf.import_graph_def(
                graph_def, return_elements=['output/result:0', 'x_input:0', 'y_label:0', 'dropout:0'])

            list_predictions = []
            batches = len(list_one_case_train_data)
            for i in range(batches):
                x = list_one_case_train_data[i]
                y_dummy = x
                prediction = sess.run(output_result, feed_dict={x_input: x, y_label: y_dummy, drop: 1})
                list_predictions.append(prediction)
    return list_predictions

def __get_data_source_sol(case_root_path, roi_name, modality):
    list_case_path = []
    pid_list = os.listdir(case_root_path)
    for pid in pid_list:
        list_case_path.append(os.path.join(case_root_path, pid))

    depth = param.dict_supper_parameters[roi_name].depth
    height = param.dict_supper_parameters[roi_name].height
    width = param.dict_supper_parameters[roi_name].width
    format = param.dict_supper_parameters[roi_name].sol_format
    ww = param.dict_supper_parameters[roi_name].ww
    wl = param.dict_supper_parameters[roi_name].wl
    se_size = param.dict_supper_parameters[roi_name].se_size
    data_source = dataset.DatasetLocation(
        list_case_path=list_case_path,
        format=format,
        roi_name=roi_name,
        depth=depth,
        height=height,
        width=width,
        ww=ww,
        wl=wl,
        se_size=se_size,
        modality=modality)
    return data_source

def predict_location(modality, case_root_path, model_root_path, loc_root_path):
    list_case_path = []
    pid_list = os.listdir(case_root_path)
    for pid in pid_list:
        list_case_path.append(os.path.join(case_root_path, pid))

    oar_list = param.dict_supper_parameters.keys()
    for key in oar_list:
        roi_name = key
        location_model_file_full_path = os.path.join(model_root_path, 'location', roi_name, roi_name + '_location_model')

        data_source_sol = dataset.Dataset4PredictLocation(
            list_case_path=list_case_path,
            roi_name=roi_name,
            modality=modality)
        list_train_data = data_source_sol.get_all_case_training_data()
        tf.reset_default_graph()
        graph = tf.Graph()
        with tf.Session(graph=graph) as sess:
            with tf.gfile.GFile(location_model_file_full_path + '.pb', 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                output_result, x_input, y_label, drop = tf.import_graph_def(
                    graph_def, return_elements=['output/result:0', 'x_input:0', 'y_label:0', 'dropout:0'])

                case_nums = len(list_train_data)
                for i in range(case_nums):
                    list_one_case_train_data, pid, real_slices = list_train_data[i]
                    print('predicting %s: %s' % (roi_name, pid))
                    list_predictions = []
                    for j in range(len(list_one_case_train_data)):
                        x = list_one_case_train_data[j]
                        y_dummy = x
                        prediction = sess.run(output_result,
                                              feed_dict={x_input: x, y_label: y_dummy, drop: 1})
                        list_predictions.append(prediction)
                    result_label_volume_data = data_source_sol.postprocessing(pid, real_slices, list_predictions)
                    label_file_root_path = os.path.join(loc_root_path, roi_name, pid)
                    if not os.path.exists(label_file_root_path):
                        os.makedirs(label_file_root_path)
                    for j in range(result_label_volume_data.shape[0]):
                        one_label_data = result_label_volume_data[j]
                        cv.imwrite(os.path.join(label_file_root_path, '%03d.jpg' % (j)), one_label_data)
        ### data_source_sol.clear()
    pass

def predict_segmentation(modality, one_case_path, model_root_path):
    global dicom_dataset
    count = 0
    result_volume_data_list = []
    oar_list = param.dict_supper_parameters.keys()
    for key in oar_list:
        roi_name = key
        location_model_file_full_path = os.path.join(model_root_path, 'location', roi_name,
                                                     roi_name + '_location_model')
        data_source = dataset.Dataset4PredictSegmenation(
            one_case_path=one_case_path,
            roi_name=roi_name,
            modality=modality)
        list_one_case_train_data, pid, real_slices = data_source.get_training_data()
        print('predicting %s: %s' % (roi_name, pid))
        list_predictions = __predict(location_model_file_full_path, list_one_case_train_data)
        label_volume_data_loc = data_source.postprocessing_loc(pid, real_slices, list_predictions)
        # for j in range(label_volume_data_loc.shape[0]):
        #     cv.imwrite('D:\\MLDO\\tmp\\0-%s.%d.jpg' % (pid, j), label_volume_data_loc[j])

        list_one_case_train_data.clear()
        train_data_box, pid, box_depth = data_source.get_training_data_box(label_volume_data_loc)
        list_one_case_train_data.append(train_data_box)
        location_model_file_full_path = os.path.join(model_root_path, 'segmentation', roi_name, roi_name + '_segmentation_model')
        list_predictions = __predict(location_model_file_full_path, list_one_case_train_data)
        result_label_volume_data = data_source.postprocessing_seg(pid, list_predictions)
        if count == 0:
            dicom_dataset = data_source.get_dicom_dataset()
        count = count + 1
        result_volume_data_list.append([roi_name + '_AI', result_label_volume_data])
    return result_volume_data_list, dicom_dataset


