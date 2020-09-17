import os
import utility.parameters as param
import network.net_vnet3d as net
import tensorflow as tf
import data.dataset as dataset
import time
from tensorflow.python.framework import graph_util
import numpy as np

class VNet3DTrainer(object):
    def __init__(
            self,
            case_root_path,
            model_root_path,
            log_root_path):
        self.__list_case_path = []
        pid_list = os.listdir(case_root_path)
        for pid in pid_list:
            self.__list_case_path.append(os.path.join(case_root_path, pid))
        self.__model_root_path = model_root_path
        self.__log_root_path = log_root_path
        pass

    def train_location(self, case_root_path, modality):
        oar_list = param.dict_supper_parameters.keys()
        for key in oar_list:
            roi_name = key
            epoch = param.dict_supper_parameters[key].epochs
            lr = param.dict_supper_parameters[key].lr
            dropout = param.dict_supper_parameters[key].dropout
            depth = param.dict_supper_parameters[key].depth
            height = param.dict_supper_parameters[key].height
            width = param.dict_supper_parameters[key].width

            model = net.VNet3D(depth, height, width, channels=1, kernel_param=param.kernel_shape_parameters1)
            dst_model_file_path = os.path.join(self.__model_root_path, 'location', roi_name)
            dataset.make_dirs([self.__log_root_path, dst_model_file_path])
            model_file_full_path = os.path.join(dst_model_file_path, roi_name + '_location_model')
            data_source = dataset.Dataset4TrainLocation(
                roi_name=roi_name,
                case_root_path = case_root_path,
                modality=modality)
            self.__train_location(model, epoch, lr, model_file_full_path, dropout, roi_name, data_source)
            pass

    def get_patch(self, image_volume_data, label_volume_data):
        pass

    def __train_location(self, model, epoch, lr, model_file_full_path, dropout, roi_name, data_source, log_path=''):
        tensor_x_input = model.get_x_input()
        tensor_y_label = model.get_y_label()
        tensor_accuracy = model.get_accuracy()
        tensor_drop = model.get_drop()
        tensor_learning_rate = model.get_learning_rate()
        tensor_cost = model.get_cost()
        train_op = tf.train.AdamOptimizer(tensor_learning_rate).minimize(tensor_cost)
        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(init)
            decay_step = 20
            print('roi: %s' % (roi_name))
            for i in range(epoch):
                learning_rate = self.__get_constant_decay_learning_rate(i, lr, decay_step)
                train_case_nums = data_source.get_case_nums()
                for j in range(train_case_nums):
                    # image_volume_data, label_volume_data, pid, real_slices = data_source.next_batch()
                    # loss_train, accuracy_train, _ = sess.run(
                    #     [tensor_cost, tensor_accuracy, train_op],
                    #     feed_dict={tensor_x_input: image_volume_data, tensor_y_label: label_volume_data,
                    #                tensor_learning_rate: learning_rate, tensor_drop: dropout})
                    # timestr = time.strftime('%H:%M:%S', time.localtime())
                    # print('%10s %10s: [epochs: %04d, iterations: %04d] ====> [t_loss: %.4f, t_accuracy: %.4f]' % (
                    #     timestr, pid, i, j, loss_train, accuracy_train))
                    list_train_data, pid, real_slices = data_source.next_batch()
                    for k in range(len(list_train_data)):
                        x = list_train_data[k][0]
                        y = list_train_data[k][1]
                        if np.all(y[0] == 0):
                            continue
                        loss_train, accuracy_train, _ = sess.run(
                            [tensor_cost, tensor_accuracy, train_op],
                            feed_dict={tensor_x_input: x, tensor_y_label: y,
                                       tensor_learning_rate: learning_rate, tensor_drop: dropout})
                        timestr = time.strftime('%H:%M:%S', time.localtime())
                        print('%10s %10s: [epochs: %04d, iterations: %04d:%02d] ====> [loss: %.4f, accuracy: %.4f]' % (
                            timestr, pid, i, j, k, loss_train, accuracy_train))

                    # for k in range(11):
                    #     a = np.zeros((1, 32, 64, 48, 1), np.float)
                    #     b = np.zeros((1, 32, 64, 48, 1), np.float)
                    #     a[0] = image_volume_data[0][16*k:32+16*k]
                    #     b[0] = label_volume_data[0][16*k:32+16*k]
                    #     loss_train, accuracy_train, _ = sess.run(
                    #         [tensor_cost, tensor_accuracy, train_op],
                    #         feed_dict={tensor_x_input: a, tensor_y_label: b,
                    #                    tensor_learning_rate: learning_rate, tensor_drop: dropout})
                    #     timestr = time.strftime('%H:%M:%S', time.localtime())
                    #     print('%10s %10s: [epochs: %04d, iterations: %04d] ====> [t_loss: %.4f, t_accuracy: %.4f]' % (
                    #         timestr, pid, i, j, loss_train, accuracy_train))

            saver = tf.train.Saver()
            saver.save(sess, model_file_full_path)
            self.__export_pb_model(model_file_full_path)
        pass

    def train_segmentation(self, modality, label_4_location_root_path):
        oar_list = param.dict_supper_parameters.keys()
        for key in oar_list:
            roi_name = key
            epoch = param.dict_supper_parameters[key].sos_epochs
            lr = param.dict_supper_parameters[key].lr
            dropout = param.dict_supper_parameters[key].dropout
            depth = param.dict_supper_parameters[key].bbox[0]
            height = param.dict_supper_parameters[key].bbox[1]
            width = param.dict_supper_parameters[key].bbox[2]

            model = net.VNet3D(depth, height, width, channels=1, kernel_param=param.kernel_shape_parameters2)
            dst_model_file_path = os.path.join(self.__model_root_path, 'segmentation', roi_name)
            dataset.make_dirs([self.__log_root_path, dst_model_file_path])
            model_file_full_path = os.path.join(dst_model_file_path, roi_name + '_segmentation_model')
            data_source = dataset.Dataset4TrainSegmenation(
                roi_name=roi_name,
                list_case_path=self.__list_case_path,
                loc_root_path=label_4_location_root_path,
                modality=modality)
            self.__train_segmentation(model, epoch, lr, model_file_full_path, dropout, roi_name, data_source)
        pass

    def __train_segmentation(self, model, epoch, lr, model_file_full_path, dropout, roi_name, data_source, log_path=''):
        tensor_x_input = model.get_x_input()
        tensor_y_label = model.get_y_label()
        tensor_accuracy = model.get_accuracy()
        tensor_drop = model.get_drop()
        tensor_learning_rate = model.get_learning_rate()
        tensor_cost = model.get_cost()
        train_op = tf.train.AdamOptimizer(tensor_learning_rate).minimize(tensor_cost)
        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(init)
            decay_step = 20
            print('roi: %s' % (roi_name))
            for i in range(epoch):
                learning_rate = self.__get_constant_decay_learning_rate(i, lr, decay_step)
                train_case_nums = data_source.get_case_nums()
                for j in range(train_case_nums):
                    image_volume_data, label_volume_data, pid, real_slices = data_source.next_batch()
                    loss_train, accuracy_train, _ = sess.run(
                        [tensor_cost, tensor_accuracy, train_op],
                        feed_dict={tensor_x_input: image_volume_data, tensor_y_label: label_volume_data,
                                   tensor_learning_rate: learning_rate, tensor_drop: dropout})
                    timestr = time.strftime('%H:%M:%S', time.localtime())
                    print('%10s %10s: [epochs: %04d, iterations: %04d] ====> [loss: %.4f, accuracy: %.4f]' % (
                        timestr, pid, i, j, loss_train, accuracy_train))
            saver = tf.train.Saver()
            saver.save(sess, model_file_full_path)
            self.__export_pb_model(model_file_full_path)
        pass

    def __export_pb_model(self, export_path):
        saver = tf.train.import_meta_graph(export_path + '.meta', clear_devices=True)
        graph = tf.get_default_graph()
        input_graph_def = graph.as_graph_def()

        with tf.Session() as sess:
            saver.restore(sess, export_path)
            output_graph_def = graph_util.convert_variables_to_constants(
                sess=sess,
                input_graph_def=input_graph_def,
                output_node_names=['output/result', 'x_input', 'y_label', 'dropout'])
            with tf.gfile.GFile(export_path + '.pb', 'wb') as f:
                f.write(output_graph_def.SerializeToString())
                # print("%d ops in the final graph." % len(output_graph_def.node))
                # for op in graph.get_operations():
                #     print(op.name, op.values())
        os.remove(export_path + '.meta')
        os.remove(export_path + '.index')
        os.remove(export_path + '.data-00000-of-00001')
        base = os.path.dirname(export_path + '.meta')
        os.remove(os.path.join(base, 'checkpoint'))

    def __get_constant_decay_learning_rate(self, epoch, learning_rate, decay_constant):
        if epoch != 0 and epoch % decay_constant == 0:
            learning_rate = learning_rate / 10
        return learning_rate


