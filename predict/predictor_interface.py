import utility.parameters as param
import tensorflow as tf
import os
import cv2 as cv

class IPredictor(object):
    def __init__(self):
        self.__dict_loc_model_param = {}
        self.__dict_seg_model_param = {}
        pass

    def initialize(self, model_root_path):
        self.__model_root_path = model_root_path
        supper_params = param.dict_supper_parameters
        list_roi_names = supper_params.keys()
        for roi_name in list_roi_names:
            model_full_path = os.path.join(self.__model_root_path, 'location', roi_name, roi_name + '_location_model')
            tf.reset_default_graph()
            graph = tf.Graph()
            with graph.as_default():
                sess = tf.Session(graph=graph)
                with tf.gfile.GFile(model_full_path + '.pb', 'rb') as f:
                    graph_def = tf.GraphDef()
                    graph_def.ParseFromString(f.read())
                    output_result, x_input, y_label, drop = tf.import_graph_def(
                        graph_def, return_elements=['output/result:0', 'x_input:0', 'y_label:0', 'dropout:0'])
                    self.__dict_loc_model_param.update({roi_name: [sess, graph, output_result, x_input, y_label, drop]})

            model_full_path = os.path.join(self.__model_root_path, 'segmentation', roi_name, roi_name + '_segmentation_model')
            tf.reset_default_graph()
            graph = tf.Graph()
            with graph.as_default():
                sess = tf.Session(graph=graph)
                with tf.gfile.GFile(model_full_path + '.pb', 'rb') as f:
                    graph_def = tf.GraphDef()
                    graph_def.ParseFromString(f.read())
                    output_result, x_input, y_label, drop = tf.import_graph_def(
                        graph_def, return_elements=['output/result:0', 'x_input:0', 'y_label:0', 'dropout:0'])
                    self.__dict_seg_model_param.update({roi_name: [sess, graph, output_result, x_input, y_label, drop]})
        pass

    def uninitialize(self):
        for roi_name in self.__dict_loc_model_param.keys():
            sess = self.__dict_loc_model_param[roi_name][0]
            sess.close()
        for roi_name in self.__dict_seg_model_param.keys():
            sess = self.__dict_seg_model_param[roi_name][0]
            sess.close()
        tf.reset_default_graph()
        pass

    def predict(self, roi_name, data_source):
        sess_loc, graph_loc, output_result_loc, x_input_loc, y_label_loc, drop_loc = self.__dict_loc_model_param[roi_name]
        image_volume_data = data_source.get_training_image_data_loc(roi_name)
        y_dummy = image_volume_data
        prediction1 = sess_loc.run(output_result_loc, feed_dict={x_input_loc: image_volume_data, y_label_loc: y_dummy, drop_loc: 1})
        label_volume_data_loc = data_source.postprocessing_loc(roi_name, prediction1)
        # for i in range(label_volume_data_loc.shape[0]):
        #     cv.imwrite('D:\\tmp\\MIS\\other1\\%s.1.%d.jpg' % (roi_name, i), label_volume_data_loc[i])
        image_volume_data_box, real_box = data_source.get_training_data_box(roi_name, label_volume_data_loc)

        sess_seg, graph_seg, output_result_seg, x_input_seg, y_label_seg, drop_seg = self.__dict_seg_model_param[roi_name]
        y_dummy = image_volume_data_box
        prediction2 = sess_seg.run(output_result_seg, feed_dict={x_input_seg: image_volume_data_box, y_label_seg: y_dummy, drop_seg: 1})
        result_label_volume_data = data_source.postprocessing_seg(roi_name, prediction2, real_box)
        # for i in range(label_volume_data_loc.shape[0]):
        #     cv.imwrite('D:\\tmp\\MIS\\other1\\%s.2.%d.jpg' % (roi_name, i), result_label_volume_data[i])
        return result_label_volume_data

    pass