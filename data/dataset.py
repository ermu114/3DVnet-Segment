import numpy as np
import os
import cv2 as cv
import utility.iodicom as dcmio
import utility.data_processor as processor
import time
import utility.parameters as param
import SimpleITK as sitk
import datetime
import sys


g_bbox_padding = 15
# 默认重采样层厚
g_slice_thickness = 3.0
g_xy_padding = 0

g_tmp_dir_base = 'D:\\MLDO\\tmp'

def list_all_files_recursive(root_path):
    _files = []
    list = os.listdir(root_path)
    for i in range(0, len(list)):
        path = os.path.join(root_path, list[i])
        if os.path.isdir(path):
            _files.extend(list_all_files_recursive(path))
        if os.path.isfile(path):
            _files.append(path)
    return _files


def make_dirs(dir_list):
    for dir in dir_list:
        if not os.path.exists(dir):
            os.makedirs(dir)

class DICOMInfo(object):
    def __init__(
            self,
            shape,
            first_dataset_item,
            box=None):
        self.shape = shape
        self.box = box
        self.first_dataset_item = first_dataset_item
        pass

def save_images(volume, pid, prev, path):
    path = os.path.join(path, pid)
    if not os.path.exists(path):
        os.makedirs(path)
    slices = volume.shape[0]
    for i in range(slices):
        cv.imwrite(os.path.join(path, '%s.%03d.jpg' % (prev, i)), volume[i])
    pass

def print_cur_time():
    dt = datetime.datetime.now()
    timestr1 = dt.strftime('%H%M%S.%f')
    print(timestr1)
    pass

def image_resample(first_dataset_item, volume_data, new_thickness):
    slice_thickness = first_dataset_item.SliceThickness
    if slice_thickness != new_thickness:
        sitk_image = processor.create_itk_image(first_dataset_item, volume_data)
        volume_data_resampled = processor.image_resample(sitk_image, new_thickness)
    else:
        volume_data_resampled = volume_data
    return volume_data_resampled

def set_ww_wl(volume_data, roi_name, modality, pid, format):
    ww = param.dict_supper_parameters[roi_name].ww
    wl = param.dict_supper_parameters[roi_name].wl
    if format == 'dicom8bit':
        result = dcmio.set_ww_wl(volume_data, ww, wl)
    else:
        result = volume_data
    return result

def label_dilate(label_volume_data, se_size):
    if se_size != 0:
        result = processor.image_dilate(label_volume_data, se_size)
    else:
        result = label_volume_data
    return result

def get_one_image_volume_data(one_case_path, modality):
    _, list_dataset_image = dcmio.read_one_series(one_case_path, modality)
    image_volume_data = dcmio.get_image_volume_data_ctvalue(list_dataset_image)
    return image_volume_data, list_dataset_image

def get_one_label_volume_data(one_case_path, roi_name, list_dataset_image):
    _, dataset_rtstruct = dcmio.read_one_series(one_case_path, 'RTSTRUCT')
    label_volume_data = dcmio.get_label_volume_data(roi_name, list_dataset_image, dataset_rtstruct)
    return label_volume_data

def get_label_volume_data_loc(loc_root_path, roi_name, pid, shape):
    label_volume_data_loc = np.zeros(shape, np.uint8)
    loc_file_list = list_all_files_recursive(os.path.join(loc_root_path, roi_name, pid))
    loc_file_list.sort()
    assert len(loc_file_list) == label_volume_data_loc.shape[0]
    for i in range(len(loc_file_list)):
        label_volume_data_loc[i] = cv.imread(loc_file_list[i], cv.IMREAD_GRAYSCALE)
    return label_volume_data_loc

def read_one_series(path):
    list_slices = os.listdir(path)
    list_slices.sort()
    slices = len(list_slices)
    first_slice = cv.imread(os.path.join(path, list_slices[0]), cv.IMREAD_GRAYSCALE)
    volume_data = np.zeros((slices, first_slice.shape[1], first_slice.shape[0]), dtype=np.uint8)
    for i in range(slices):
        volume_data[i] = cv.imread(os.path.join(path, list_slices[i]), cv.IMREAD_GRAYSCALE)
    return volume_data

class Dataset4TrainLocation(object):
    '''
    用于训练location模型使用
    '''
    def __init__(
            self,
            roi_name,
            case_root_path,
            modality = 'CT'
    ):
        self.__roi_name = roi_name
        self.__list_case_path = []
        pid_list = os.listdir(os.path.join(case_root_path, roi_name))
        for pid in pid_list:
            self.__list_case_path.append(os.path.join(case_root_path, roi_name, pid))
        self.__format = param.dict_supper_parameters[roi_name].sol_format
        self.__depth = param.dict_supper_parameters[roi_name].depth
        self.__height = param.dict_supper_parameters[roi_name].height
        self.__width = param.dict_supper_parameters[roi_name].width
        self.__ww = param.dict_supper_parameters[roi_name].ww
        self.__wl = param.dict_supper_parameters[roi_name].wl
        self.__se_size = param.dict_supper_parameters[roi_name].se_size
        self.__modality = modality
        self.__list_train_data = []
        self.__index = 0
        # self.__dict_all_case_info = {}
        self.create_all_case_training_data()

    def create_all_case_training_data(self):
        '''
        将所有训练病例数据读取、处理之后，创建成网络训练所需要的格式: [batchsize, depth, height, widht, channel] 5维数据格式
        batchsize=1， channel=1， depth为超参里面设置的depth，height、width也为超参里面设置的宽和高
        例如：'Spinal_Cord': Parameter(..., depth=64, height=96, width=96,
        :return:
        '''
        for one_case_path in self.__list_case_path:
            timestr = time.strftime('%H:%M:%S', time.localtime())
            print('%10s loading %s' % (timestr, one_case_path))
            training_data_image, training_data_label, pid, real_slices = self.create_one_case_training_data(one_case_path)
            self.__list_train_data.append([training_data_image, training_data_label, pid, real_slices])
        pass

    def create_one_case_training_data(self, one_case_path):
        '''
        创建一个病例的训练格式数据
        :param one_case_path:
        :return: 病例ID即pid（也就是病例所在文件夹的名字，如001,002,...）;real_slices:此病例实际的切片数
        '''
        # 1-读取dicom-rt
        pid, image_volume_data_org, label_volume_data_org = self.get_one_case_volume_data(one_case_path)

        # 2-对image和label进行预处理
        image_volume_data_processed, label_volume_data_processed = self.preprocessing(image_volume_data_org, label_volume_data_org)
        real_slices = image_volume_data_processed.shape[0]

        # 3-将预处理之后的数据修改数据格式，主要是增加了一个channel维度[depth, height, width] -> [depth, height, width, 1]
        image_volume_data_reshaped = np.reshape(image_volume_data_processed, (real_slices, self.__height, self.__width, 1))

        # 4-修改image数据格式：[depth, height, width, 1] -> [1, depth, height, width, 1]
        training_data_image = np.zeros([1, real_slices, self.__height, self.__width, 1], dtype=np.float)
        training_data_image[0] = image_volume_data_reshaped

        # 5-修改label数据格式：[depth, height, width, 1] -> [1, depth, height, width, 1]
        label_volume_data_reshaped = np.reshape(label_volume_data_processed, (real_slices, self.__height, self.__width, 1))
        training_data_label = np.zeros([1, real_slices, self.__height, self.__width, 1], dtype=np.float)
        training_data_label[0] = label_volume_data_reshaped

        return training_data_image, training_data_label, pid, real_slices

    def get_one_case_volume_data(self, one_case_path):
        '''
        读取dicom和structure
        :param one_case_path:
        :return:
        '''
        pid = os.path.basename(one_case_path)
        image_volume_data = read_one_series(os.path.join(one_case_path, 'images'))
        # save_images(image_volume_data, pid, 'image')
        label_volume_data = read_one_series(os.path.join(one_case_path, 'labels'))
        # save_images(label_volume_data, pid, 'label')

        return pid, image_volume_data, label_volume_data

    def preprocessing(self, image_volume_data, label_volume_data):
        '''
        对一个病例的image和label进行数据预处理
        :param image_volume_data:
        :param label_volume_data:
        :param pid:
        :return:
        '''

        # 1-如果输入图像宽和高不相等，则变换为相等，如：512x512 --> 512x512; 192x256 --> 256x256; 1024x512 --> 1024x1024
        shape = image_volume_data.shape
        image_volume = processor.image_padding(image_volume_data, shape[1], shape[2])

        # 2-缩放图像至指定的大小，大小为超参数，人为指定，例如：
        # 'Spinal_Cord': Parameter(... depth=64, height=96, width=96
        image_volume = processor.image_resize_3d(image_volume, self.__height, self.__width)

        # 3-如果输入label图像宽和高不相等，则变换为相等，如：512x512 --> 512x512; 192x256 --> 256x256; 1024x512 --> 1024x1024
        shape = label_volume_data.shape
        label_volume= processor.image_padding(label_volume_data, shape[1], shape[2])

        # 4-缩放label图像至指定的大小，大小为超参数，人为指定，例如：
        # 'Spinal_Cord': Parameter(... depth=64, height=96, width=96
        label_volume = processor.image_resize_3d(label_volume, self.__height, self.__width)

        # 5-对label图像进行二值化阈值处理，小于127设为0，大于127设为255
        label_volume = processor.image_threshold_binaray_3d(label_volume, 127)
        # save_images(label_volume_data_dilated, pid, '03.label')

        # 6-将label图像规范化处理：[0 ~ 255] --> [0 ~ 1]
        label_volume = processor.normalize_image_data(label_volume)

        return image_volume, label_volume

    def get_case_nums(self):
        '''
        获取总共的病例数目
        :return:
        '''
        return len(self.__list_train_data)

    def get_all_case_training_data(self):
        '''
        获取所有训练格式的数据，链表的每一项格式为:
        [training_data_image, training_data_label, pid, real_slices]
        而training_data_image和training_data_label的格式均为：
        [1, real_slices, self.__height, self.__width, 1]
        :return:
        '''
        return self.__list_train_data

    def next_batch(self):
        '''
        训练过程中，每一个iteration get一个病例数据
        :return:
        '''
        # 如果当前游标索引值大于所有病例数目时，要把训练数据顺序打乱
        if self.__index >= len(self.__list_train_data):
            np.random.shuffle(self.__list_train_data)
            self.__index = 0

        # 获取一个病例数据
        training_data_image, training_data_label, pid, real_slices = self.__list_train_data[self.__index]

        # 取patch操作，例如：
        # 一个病例有120个slice（假设索引号从0开始，那么就是0~119），但是超参里面设置的是depth=64, height=96, width=96,此时
        # 取0~63, 8~71, 16~79, 24~87,....，然后将这些高度为depth的子序列存入链表list_train_data中
        list_train_data = []
        start_index = 0
        while True:
            image_data = np.zeros((1, self.__depth, self.__height, self.__width, 1), np.float)
            label_data = np.zeros((1, self.__depth, self.__height, self.__width, 1), np.float)
            end_index = start_index + self.__depth
            if end_index <= real_slices:
                image_data[0] = training_data_image[0][start_index:end_index]
                label_data[0] = training_data_label[0][start_index:end_index]
                list_train_data.append([image_data, label_data])
            elif end_index > real_slices and end_index < real_slices + self.__depth:
                image_data[0][0:real_slices-start_index] = training_data_image[0][start_index:real_slices]
                label_data[0][0:real_slices-start_index] = training_data_label[0][start_index:real_slices]
            else:
                break
            start_index += 8

        self.__index += 1
        return list_train_data, pid, real_slices

class Dataset4PredictLocation(object):
    '''
    用于预测location，预测结果输出为jpg图片，存放在location文件夹内，这些jpg图像用于抠图bounding_box，从而训练segmentation模型使用
    '''
    def __init__(
            self,
            roi_name,
            list_case_path,
            modality = 'CT'):
        self.__roi_name = roi_name
        self.__list_case_path = list_case_path
        self.__format = param.dict_supper_parameters[roi_name].sol_format
        self.__depth = param.dict_supper_parameters[roi_name].depth
        self.__height = param.dict_supper_parameters[roi_name].height
        self.__width = param.dict_supper_parameters[roi_name].width
        self.__ww = param.dict_supper_parameters[roi_name].ww
        self.__wl = param.dict_supper_parameters[roi_name].wl
        self.__se_size = param.dict_supper_parameters[roi_name].se_size
        self.__modality = modality
        self.__list_train_data = []
        self.__index = 0
        self.__dict_all_case_info = {}
        self.create_all_case_training_data()
        pass

    # 将trainlocation中的所有训练数据创建为网络模型需要的数据格式
    def create_all_case_training_data(self):
        for one_case_path in self.__list_case_path:
            timestr = time.strftime('%H:%M:%S', time.localtime())
            print('%10s loading %s' % (timestr, one_case_path))
            # 创建一个病例的训练格式数据
            training_data_image, pid, real_slices = self.create_one_case_training_data(one_case_path)

            # 取patch操作，和Dataset4TrainLocation::next_batch中取patch操作一样，只是这里没有label数据，只有image数据
            list_one_case_train_data = []
            start_index = 0
            while True:
                image_data = np.zeros((1, self.__depth, self.__height, self.__width, 1), np.float)
                end_index = start_index + self.__depth
                if end_index <= real_slices:
                    image_data[0] = training_data_image[0][start_index:end_index]
                    list_one_case_train_data.append(image_data)
                elif end_index > real_slices and end_index < real_slices + self.__depth:
                    image_data[0][0:real_slices - start_index] = training_data_image[0][start_index:real_slices]
                    list_one_case_train_data.append(image_data)
                else:
                    break
                start_index += self.__depth
            self.__list_train_data.append([list_one_case_train_data, pid, real_slices])
        pass

    def create_one_case_training_data(self, one_case_path):
        '''
        创建一个病例的训练格式数据，其中包含了数据预处理
        :param one_case_path:
        :return:
        '''
        # 1-读取dicom图像
        pid, image_volume_data_org = self.get_one_case_volume_data(one_case_path)

        # 2 - 数据预处理
        image_volume_data_processed = self.preprocessing(image_volume_data_org, pid)
        real_slices = image_volume_data_processed.shape[0]

        # 3-修改形状，主要是扩展了一个channel维度，由[depth, height, width] --> [depth, height, width, 1]
        image_volume_data_reshaped = np.reshape(image_volume_data_processed, (real_slices, self.__height, self.__width, 1))
        training_data_image = np.zeros([1, real_slices, self.__height, self.__width, 1], dtype=np.float)
        training_data_image[0] = image_volume_data_reshaped

        return training_data_image, pid, real_slices

    def get_one_case_volume_data(self, one_case_path):
        image_volume_data, list_dataset_image = get_one_image_volume_data(one_case_path, self.__modality)

        pid = os.path.basename(one_case_path)
        self.__dict_all_case_info.update({pid : DICOMInfo(image_volume_data.shape, list_dataset_image[0])})

        return pid, image_volume_data

    def preprocessing(self, image_volume_data, pid):
        # 获取第一个slice的dicom信息，例如：spacing、orgin、thickness等参数，后面重采样使用
        first_dataset_item = self.__dict_all_case_info[pid].first_dataset_item

        # 1-图像重采样，层厚设置为默认3mm
        image_volume_data_resampled = image_resample(first_dataset_item, image_volume_data, g_slice_thickness)

        # 2-设置窗宽窗位
        image_volume_data_wwwl = set_ww_wl(image_volume_data_resampled, self.__roi_name, self.__modality, pid, self.__format)

        # 3-如果输入图像宽和高不相等，则变换为相等，如：512x512 --> 512x512; 192x256 --> 256x256; 1024x512 --> 1024x1024
        image_volume_data_padded = processor.image_padding(
            image_volume_data_wwwl, image_volume_data_wwwl.shape[1], image_volume_data_wwwl.shape[2])
        # save_images(image_volume_data_padded, pid, '02.image')

        # 4-缩放图像至指定的大小，大小为超参数，人为指定，例如：
        # 'Spinal_Cord': Parameter(... depth=64, height=96, width=96
        image_volume_data_resized = processor.image_resize_3d(image_volume_data_padded, self.__height, self.__width)
        result_image_volume_data = image_volume_data_resized

        return result_image_volume_data

    def get_case_nums(self):
        return len(self.__list_train_data)

    def get_all_case_training_data(self):
        return self.__list_train_data

    def postprocessing(self, pid, real_slices, list_predictions):
        '''
        图像后处理操作。由于训练的时候采用的是将一个病人的volume数据patch成多个depth x height x width的数据进行训练的，
        因此预测的时候也是生成多个patch，然后将每个patch合并还原为一个volume
        :param pid:
        :param real_slices:
        :param list_predictions:
        :return:
        '''
        # 将多个patch合并
        batches = len(list_predictions)
        label_volume = np.zeros((batches*self.__depth, self.__height, self.__width, 1), np.float)
        for i in range(len(list_predictions)):
            label_volume[i*self.__depth:i*self.__depth + self.__depth] = list_predictions[i]

        # 修改数据预测结果数据形状，主要是去掉了channel维度：[depth, height, width, 1] --> [depth, height, width]
        label_volume_data_location = np.reshape(label_volume, (batches*self.__depth, self.__height, self.__width))

        # 由于Dataset4TrainLocation::preprocessing中最后一步将label数据进行了normalization操作，即将0~255归一化到0~1区间了，
        # 所以这里要乘以255还原回去，并且将数据类型有float修改为uint8
        label_volume_data_location = np.clip(label_volume_data_location * 255, 0, 255).astype(np.uint8)
        # save_images(label_volume_data_location, pid, '02.label')

        # 预处理的流程先后顺序为：resample；image_padding；image_resize，
        # 所以这里的流程应该和预处理正好相反：image_resize；image_padding；resample
        # 预处理：A --> B --> C
        # 后处理：iC-->iB-->iA

        # 1- 反向image_resize
        org_height = self.__dict_all_case_info[pid].shape[1]
        org_width = self.__dict_all_case_info[pid].shape[2]
        length = max(org_height, org_width)
        label_volume_resized = processor.image_resize_3d(label_volume_data_location, length, length)

        # 2-反向image_padding
        image_ipadding = label_volume_resized[:, :, (org_height - org_width) // 2:(length - (org_height - org_width) // 2)]

        # 3-反向重采样
        # 对预测生成的label进行反重采样，预处理里面将所有层厚都采样成了3mm后，这个地方要变换回原始的层厚
        # 例如：输入数据层厚为1mm，预处理后就变成了3mm，然后进行预测，预测完了之后的图像是3mm，然后再变换回原来的1mm
        org_thickness = self.__dict_all_case_info[pid].first_dataset_item.SliceThickness
        self.__dict_all_case_info[pid].first_dataset_item.SliceThickness = g_slice_thickness
        label_volume_data_resampled = image_resample(
            self.__dict_all_case_info[pid].first_dataset_item, image_ipadding[0:real_slices], org_thickness)

        return label_volume_data_resampled

class Dataset4TrainSegmenation(object):
    '''
    用于训练segmentation模型
    '''
    def __init__(
            self,
            list_case_path,
            loc_root_path,
            roi_name,
            modality = 'CT'):
        self.__roi_name = roi_name
        self.__list_case_path = list_case_path
        self.__loc_root_path = loc_root_path
        self.__sos_format = param.dict_supper_parameters[roi_name].sos_format
        self.__depth = param.dict_supper_parameters[roi_name].bbox[0]
        self.__height = param.dict_supper_parameters[roi_name].bbox[1]
        self.__width = param.dict_supper_parameters[roi_name].bbox[2]
        self.__ww = param.dict_supper_parameters[roi_name].ww
        self.__wl = param.dict_supper_parameters[roi_name].wl
        self.__se_size = param.dict_supper_parameters[roi_name].se_size
        self.__modality = modality
        self.__list_train_data = []
        self.__index = 0
        self.__dict_all_case_info = {}
        self.create_all_case_training_data()
        pass

    def create_all_case_training_data(self):
        '''
        创建训练格式的数据，包括图像预处理
        :return:
        '''
        for one_case_path in self.__list_case_path:
            print('loading %s' % (one_case_path))
            training_data_image, training_data_label, pid, real_slices = self.create_one_case_training_data(one_case_path)
            self.__list_train_data.append([training_data_image, training_data_label, pid, real_slices])
        pass

    def create_one_case_training_data(self, one_case_path):
        image_volume_data_org, label_volume_data_org, pid = self.get_one_case_volume_data(one_case_path)

        # 读取Dataset4PredictLocation中预测生成的用于抠图的jpg定位图像（位于D:\MLDO\location文件夹下），将此类图像记为label_loc
        label_volume_data_loc = get_label_volume_data_loc(self.__loc_root_path, self.__roi_name, pid, label_volume_data_org.shape)

        # 对待训练的image数据、label数据利用label_loc进行预处理操作，处理之后的数据为image boundingbox数据和相应的label boudingbox数据
        image_volume_data_processed, label_volume_data_processed = \
            self.preprocessing(pid, image_volume_data_org, label_volume_data_org, label_volume_data_loc)

        # 对处理之后image和label数据增加channel维度
        image_volume_data_reshaped = np.reshape(image_volume_data_processed, (self.__depth, self.__height, self.__width, 1))
        training_data_image = np.zeros([1, self.__depth, self.__height, self.__width, 1], dtype=np.float)
        training_data_image[0] = image_volume_data_reshaped[0:self.__depth]

        label_volume_data_reshaped = np.reshape(label_volume_data_processed, (self.__depth, self.__height, self.__width, 1))
        training_data_label = np.zeros([1, self.__depth, self.__height, self.__width, 1], dtype=np.float)
        training_data_label[0] = label_volume_data_reshaped[0:self.__depth]

        return training_data_image, training_data_label, pid, self.__depth

    def preprocessing(self, pid, image_volume_data, label_volume_data, label_volume_data_loc):
        first_dataset_item = self.__dict_all_case_info[pid].first_dataset_item

        ## 1-label_loc
        # 1.1-对lable_loc定位图像进行重采样，默认层厚3mm
        label_volume_data_loc_resampled = image_resample(first_dataset_item, label_volume_data_loc, g_slice_thickness)
        # 1.2-对lable_loc定位图像进行宽和高调整为相等
        label_volume_data_loc_shape_fixed = processor.image_padding(
            label_volume_data_loc_resampled,
            label_volume_data_loc_resampled.shape[1],
            label_volume_data_loc_resampled.shape[2])
        # 1.3-根据label_loc定位图像获取boundingbox
        if self.__roi_name == 'Len_L' or self.__roi_name == 'Len_R' or \
                self.__roi_name == 'Optic_Nerve_L' or self.__roi_name == 'Optic_Nerve_R' or self.__roi_name == 'Pituitary':
            box = processor.get_bounding_box(
                label_volume_data_loc_shape_fixed,
                self.__depth, xy_padding=g_xy_padding, z_padding=4)
        else:
            box = processor.get_bounding_box(label_volume_data_loc_shape_fixed, self.__depth)
        self.__dict_all_case_info[pid].box = box

        ## 2-image
        # 2.1-对image数据进行重采样，默认层厚为3mm
        image_volume_data_resampled = image_resample(first_dataset_item, image_volume_data, g_slice_thickness)
        # 对重采样之后的image数据设置window
        image_volume_data_wwwl = set_ww_wl(image_volume_data_resampled, self.__roi_name, self.__modality, pid, self.__sos_format)
        # save_images(image_volume_data_wwwl, pid, '02image')

        # 2.2-对image数据进行宽和高调整为相等
        image_volume_data_shape_fixed = processor.image_padding(
            image_volume_data_wwwl, image_volume_data_wwwl.shape[1], image_volume_data_wwwl.shape[2])
        # 2.3-根据boudingbox抠图，注意：get_bounding_box返回的图像宽和高总是相等的
        image_volume_data_box = np.zeros([self.__depth, box[3]-box[2], box[5]-box[4]], np.uint8)
        image_volume_data_box[0:box[1]-box[0]] = image_volume_data_shape_fixed[box[0]: box[1], box[2]: box[3], box[4]: box[5]]
        # save_images(image_volume_data_box, pid, '03image')

        # 2.4-由于get_bounding_box返回的图像宽和高总是相等的，所以将抠图之后的image bbox data直接进行缩放
        image_volume_data_resized = processor.image_resize_3d(image_volume_data_box, self.__height, self.__width)
        save_images(image_volume_data_resized, pid, 'image', os.path.join(g_tmp_dir_base, self.__roi_name))
        result_image_volume_data = image_volume_data_resized

        # 为了观察理解，可以将预处理之后的图像保存
        # save_images(result_image_volume_data, pid, 'image_bbox')

        ## 3-label
        # 3.1-对label数据进行重采样
        label_volume_data_resampled = image_resample(first_dataset_item, label_volume_data, g_slice_thickness)
        # save_images(label_volume_data_resampled, pid, '02label')

        # 3.2-对label数据进行宽和高调整为相等
        label_volume_data_shape_fixed = processor.image_padding(
            label_volume_data_resampled, label_volume_data_resampled.shape[1], label_volume_data_resampled.shape[2])

        # 3.3-根据boudingbox抠图，注意：get_bounding_box返回的图像宽和高总是相等的
        label_volume_data_box = np.zeros([self.__depth, box[3]-box[2], box[5]-box[4]], np.uint8)
        label_volume_data_box[0:box[1]-box[0]] = label_volume_data_shape_fixed[box[0]: box[1], box[2]: box[3], box[4]: box[5]]

        # 3.4-由于get_bounding_box返回的图像宽和高总是相等的，所以将抠图之后的image bbox data直接进行缩放
        label_volume_data_resized = processor.image_resize_3d(label_volume_data_box, self.__height, self.__width)
        save_images(label_volume_data_resized, pid, 'label', os.path.join(g_tmp_dir_base, self.__roi_name))

        # 为了观察理解，可以将预处理之后的图像保存
        # save_images(label_volume_data_resized, pid, 'label_bbox')

        # 对label数据归一化到0~1区间
        result_label_volume_data = processor.normalize_image_data(label_volume_data_resized)

        return result_image_volume_data, result_label_volume_data

    def get_one_case_volume_data(self, one_case_path):
        image_volume_data, list_dataset_image = get_one_image_volume_data(one_case_path, self.__modality)
        label_volume_data = get_one_label_volume_data(one_case_path, self.__roi_name, list_dataset_image)

        pid = os.path.basename(one_case_path)
        self.__dict_all_case_info.update({pid : DICOMInfo(image_volume_data.shape, list_dataset_image[0])})
        return image_volume_data, label_volume_data, pid

    def next_batch(self):
        if self.__index >= len(self.__list_train_data):
            np.random.shuffle(self.__list_train_data)
            self.__index = 0
        training_data_image, training_data_label, pid, real_slices = self.__list_train_data[self.__index]
        self.__index += 1
        return training_data_image, training_data_label, pid, real_slices

    def get_case_nums(self):
        return len(self.__list_train_data)

    def get_all_case_training_data(self):
        return self.__list_train_data

class Dataset4PredictSegmenation(object):
    '''
    用于从给定dicom数据预测最终勾画结果，整个过程分两步，记做：stage A 和stage B
    '''
    def __init__(
            self,
            roi_name,
            one_case_path,
            modality = 'CT'):
        self.__roi_name = roi_name
        self.__one_case_path = one_case_path
        self.__sol_format = param.dict_supper_parameters[roi_name].sol_format
        self.__sos_format = param.dict_supper_parameters[roi_name].sos_format
        self.__depth = param.dict_supper_parameters[roi_name].depth
        self.__height = param.dict_supper_parameters[roi_name].height
        self.__width = param.dict_supper_parameters[roi_name].width
        self.__ww = param.dict_supper_parameters[roi_name].ww
        self.__wl = param.dict_supper_parameters[roi_name].wl
        self.__se_size = param.dict_supper_parameters[roi_name].se_size
        self.__predifined_box_size = param.dict_supper_parameters[roi_name].bbox
        self.__modality = modality
        self.__image_volume_data = None
        self.__list_dataset_image = None
        self.__training_data_image = None

        self.__org_volume_shape = None
        self.__resampled_shape = None
        self.__padded_shape = None

        self.__crop_box = None

        self.initialize()
        pass

    def initialize(self):

        # 1-读取dicom数据，并将图像信息和dicom信息保存
        self.__image_volume_data, self.__list_dataset_image = get_one_image_volume_data(self.__one_case_path, self.__modality)

        # 2-根据路径获取到文件夹名字，也就是pid
        pid = os.path.basename(self.__one_case_path)

        # 3-对image数据预处理
        image_volume_data_processed = self.preprocessing_loc(pid)
        real_slices = image_volume_data_processed.shape[0]

        # 4-对处理后的数据增加channel维度
        image_volume_data_reshaped = np.reshape(image_volume_data_processed, (real_slices, self.__height, self.__width, 1))

        # 5-创建训练、预测所需格式的5维数据[batchsize=1, depth, height, width, channel=1], 并保存
        training_data_image = np.zeros([1, real_slices, self.__height, self.__width, 1], dtype=np.float)
        training_data_image[0] = image_volume_data_reshaped
        self.__training_data_image = training_data_image
        pass

    def preprocessing_loc(self, pid):
        '''
        stage A过程的预处理
        :param pid:
        :return:
        '''
        image_volume = self.__image_volume_data
        self.__org_volume_shape = self.__image_volume_data.shape

        # 1-重采样，默认层厚3mm
        first_dataset_item = self.__list_dataset_image[0]
        image_volume = image_resample(first_dataset_item, image_volume, g_slice_thickness)
        self.__resampled_shape = image_volume.shape

        # 2-设置window
        image_volume = set_ww_wl(image_volume, self.__roi_name, self.__modality, pid, self.__sol_format)
        save_images(image_volume, pid, '666', os.path.join(g_tmp_dir_base, self.__roi_name))

        # 3-如果image数据长和宽不相等则调整为相等，否则不做处理
        shape = image_volume.shape
        image_volume = processor.image_padding(image_volume, shape[1], shape[2])
        self.__padded_shape = image_volume.shape
        # save_images(image_volume, pid, '02.image', os.path.join(g_tmp_dir_base, self.__roi_name))

        # 4-根据超参数缩放图像
        result_image_volume_data = processor.image_resize_3d(image_volume, self.__height, self.__width)
        # save_images(image_volume, pid, '03.image', os.path.join(g_tmp_dir_base, self.__roi_name))

        return result_image_volume_data

    def get_training_data(self):
        '''
        取patch操作，另见Dataset4PredictLocation::create_all_case_training_data中的patch操作原理
        :return:
        '''
        real_slices = self.__training_data_image.shape[1]
        pid = os.path.basename(self.__one_case_path)
        list_one_case_train_data = []
        start_index = 0
        while True:
            image_data = np.zeros((1, self.__depth, self.__height, self.__width, 1), np.float)
            end_index = start_index + self.__depth
            if end_index <= real_slices:
                image_data[0] = self.__training_data_image[0][start_index:end_index]
                list_one_case_train_data.append(image_data)
            elif end_index > real_slices and end_index < real_slices + self.__depth:
                image_data[0][0:real_slices - start_index] = self.__training_data_image[0][start_index:real_slices]
                list_one_case_train_data.append(image_data)
            else:
                break
            start_index += self.__depth
        return list_one_case_train_data, pid, real_slices

    def postprocessing_loc(self, pid, real_slices, list_predictions):
        '''
        stage A过程，当预测出location图像后的后处理，部分过程参考Dataset4PredictLocation中的后处理过程
        :param pid:
        :param real_slices:
        :param list_predictions:
        :return:
        '''
        batches = len(list_predictions)
        label_volume = np.zeros((batches*self.__depth, self.__height, self.__width, 1), np.float)
        for i in range(len(list_predictions)):
            label_volume[i*self.__depth:i*self.__depth + self.__depth] = list_predictions[i]
        label_volume = np.reshape(label_volume, (batches*self.__depth, self.__height, self.__width))
        label_volume = np.clip(label_volume * 255, 0, 255).astype(np.uint8)

        label_volume = label_volume[0:real_slices]
        # save_images(label_volume, '00', '1', os.path.join(g_tmp_dir_base, self.__roi_name))

        '''
        preprocessing_loc中的预处理过程为：resample;image_padding;image_resize
        因此，这里的后处理过程为：image_resize;image_padding;resample
        '''
        # 1-反向image_resize
        shape = self.__padded_shape
        label_volume = processor.image_resize_3d(label_volume, shape[1], shape[2])
        # save_images(label_volume, '00', '2', os.path.join(g_tmp_dir_base, self.__roi_name))

        # 2-反向image_padding
        shape = self.__resampled_shape
        label_volume = processor.image_ipadding(label_volume, shape[1], shape[2])
        # save_images(label_volume, '00', '3', os.path.join(g_tmp_dir_base, self.__roi_name))

        # 3-反向重采样
        first_dataset_item = self.__list_dataset_image[0]
        org_thickness = first_dataset_item.SliceThickness
        first_dataset_item.SliceThickness = g_slice_thickness
        label_volume = image_resample(first_dataset_item, label_volume, org_thickness)
        first_dataset_item.SliceThickness = org_thickness
        save_images(label_volume, pid, '753', os.path.join(g_tmp_dir_base, self.__roi_name))

        return label_volume

    def get_training_data_box(self, label_volume_data_loc):
        '''
        根据postprocessing_loc返回的label_loc进行抠图
        :param label_volume_data_loc:
        :return:
        '''
        # 1-根据label_loc对image进行抠图等预处理操作
        image_volume = self.preprocessing_seg(label_volume_data_loc)
        # save_images(image_volume_data_processed, '00', '5', os.path.join(g_tmp_dir_base, self.__roi_name))

        # 2-增加channel维度
        image_volume = np.reshape(image_volume, (self.__predifined_box_size[0], self.__predifined_box_size[1], self.__predifined_box_size[2], 1))
        training_data_image_box = np.zeros([1, self.__predifined_box_size[0], self.__predifined_box_size[1], self.__predifined_box_size[2], 1], dtype=np.float)
        training_data_image_box[0] = image_volume

        pid = os.path.basename(self.__one_case_path)
        return training_data_image_box, pid, self.__depth

    def preprocessing_seg(self, label_volume_data_loc):
        image_volume_data = self.__image_volume_data
        first_dataset_item = self.__list_dataset_image[0]
        pid = os.path.basename(self.__one_case_path)

        ## 1-label_loc定位图像处理
        # 1-对label_loc定位图像重采样
        label_volume_data_loc_resampled = image_resample(first_dataset_item, label_volume_data_loc, g_slice_thickness)

        # 2-对于label_loc定位图像，如果宽高不相等，则调整为相等
        label_volume_data_loc_shape_fixed = processor.image_padding(
            label_volume_data_loc_resampled, label_volume_data_loc_resampled.shape[1], label_volume_data_loc_resampled.shape[2])

        ## 2-根据label_loc定位图像对image数据处理
        # 2.1-重采样，默认层厚3mm
        image_volume_data_resampled = image_resample(first_dataset_item, image_volume_data, g_slice_thickness)
        self.__resampled_shape = image_volume_data_resampled.shape

        # 2.2-设置window
        image_volume_data_wwwl = set_ww_wl(image_volume_data_resampled, self.__roi_name, self.__modality, pid, self.__sos_format)

        # 2.3-如果image数据长和宽不相等则调整为相等，否则不作处理
        image_volume_data_shape_fixed = processor.image_padding(
            image_volume_data_wwwl, image_volume_data_wwwl.shape[1], image_volume_data_wwwl.shape[2])
        self.__padded_shape = image_volume_data_shape_fixed.shape

        # 2.4-根据label_loc定位图像对image数据进行抠图
        if self.__roi_name == 'Len_L' or self.__roi_name == 'Len_R' or self.__roi_name == 'Optic_Nerve_L' \
                or self.__roi_name == 'Optic_Nerve_R' or self.__roi_name == 'Pituitary':
            box = processor.get_bounding_box(label_volume_data_loc_shape_fixed, self.__predifined_box_size[0], xy_padding=g_xy_padding, z_padding=4)
        else:
            box = processor.get_bounding_box(label_volume_data_loc_shape_fixed, self.__predifined_box_size[0])
        self.__crop_box = box
        image_volume_data_box = np.zeros([self.__predifined_box_size[0], box[3]-box[2], box[5]-box[4]], np.uint8)
        image_volume_data_box[0:box[1]-box[0]] = image_volume_data_shape_fixed[box[0]: box[1], box[2]: box[3], box[4]: box[5]]
        save_images(image_volume_data_box, pid, '798', os.path.join(g_tmp_dir_base, self.__roi_name))

        # 2.5-根据超参（如bbox=[96, 128, 128]）进行缩放
        image_volume_data_resized = processor.image_resize_3d(image_volume_data_box, self.__predifined_box_size[1], self.__predifined_box_size[2])
        save_images(image_volume_data_resized, pid, '795', os.path.join(g_tmp_dir_base, self.__roi_name))

        return image_volume_data_resized

    def postprocessing_seg(self, pid, list_predictions):
        label_volume_data_box = list_predictions[0][0]
        label_volume_data_box_reshaped = np.reshape(
            label_volume_data_box,
            (label_volume_data_box.shape[0], label_volume_data_box.shape[1], label_volume_data_box.shape[2]))
        label_volume_data_box_reshaped = np.clip(label_volume_data_box_reshaped * 255, 0, 255).astype(np.uint8)

        '''
        preprocessing_seg预处理流程为：image_resample；image_padding；crop; image_resize
        因此，这里后处理的流程为：image_resize；crop; image_padding；image_resample
        '''

        # 1-反向resize
        box = self.__crop_box
        label_volume_data_box_resized = processor.image_resize_3d(label_volume_data_box_reshaped, box[3] - box[2], box[5] - box[4])

        # 2-反向crop
        shape = self.__padded_shape
        label_volume_data_icroped = np.zeros(shape, dtype=np.uint8)
        label_volume_data_icroped[box[0]: box[1], box[2]: box[3], box[4]: box[5]] = label_volume_data_box_resized[0:box[1] - box[0]]

        # 3-反向pimage_padding
        shape = self.__resampled_shape
        label_volume_data_ipadded = processor.image_ipadding(label_volume_data_icroped, shape[1], shape[2])

        # 4-反向重采样
        label_volume_data = np.zeros(self.__image_volume_data.shape, np.uint8)
        first_dataset_item = self.__list_dataset_image[0]
        org_thickness = first_dataset_item.SliceThickness
        # first_dataset_item.SliceThickness = 1.0
        first_dataset_item.SliceThickness = g_slice_thickness
        label_volume_data_resampled = image_resample(first_dataset_item, label_volume_data_ipadded, org_thickness)
        first_dataset_item.SliceThickness = org_thickness

        # label_volume_data_shape_fixed = processor.image_padding(
        #     label_volume_data_resampled1, label_volume_data_resampled1.shape[1], label_volume_data_resampled1.shape[2])
        #
        # end_z = min(label_volume_data_shape_fixed.shape[0] - 1, box[0] + label_volume_data_box_resized.shape[0])
        # label_volume_data_shape_fixed[box[0]:end_z, box[2]:box[3], box[4]:box[5]] = label_volume_data_box_resized[0:end_z - box[0]]
        #
        # first_dataset_item.SliceThickness = g_slice_thickness
        # label_volume_data_resampled2 = image_resample(first_dataset_item, label_volume_data_shape_fixed, org_thickness)
        # first_dataset_item.SliceThickness = org_thickness
        #
        # org_height = first_dataset_item.Rows
        # org_width = first_dataset_item.Columns
        # length = max(org_height, org_width)
        # result = label_volume_data_resampled2[:, :, (org_height - org_width) // 2:(length - (org_height - org_width) // 2)]

        if self.__roi_name == 'Len_L' or self.__roi_name == 'Len_R' or self.__roi_name == 'Optic_Nerve_L' or self.__roi_name == 'Optic_Nerve_R':
            # result = processor.image_dilate(label_volume_data_resampled, 1)
            result = label_volume_data_resampled
        else:
            result = processor.image_erode(label_volume_data_resampled, 1)
            result = processor.image_keep_max_region(result)
            result = processor.image_dilate(result, 1)

        return result

    def get_dicom_dataset(self):
        return self.__list_dataset_image

