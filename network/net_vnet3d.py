import os
import tensorflow as tf
import numpy as np

class VNet3D(object):

    def __init__(self, depth, height, width, channels, kernel_param, cost_name = 'dice coefficient'):
        tf.reset_default_graph()
        self.__depth = depth
        self.__height = height
        self.__width = width
        self.__channels = channels
        self.__x_input = tf.placeholder(
            'float',
            shape=[None, self.__depth, self.__height, self.__width, self.__channels],
            name='x_input'
        )
        self.__y_label = tf.placeholder(
            'float',
            shape=[None, self.__depth, self.__height, self.__width, self.__channels],
            name='y_label'
        )
        self.__learning_rate = tf.placeholder('float', name='learning_rate')
        self.__drop = tf.placeholder('float', name='dropout')
        self.__y_prediction = self.__create_net(kernel_param)
        self.__cost = self.__get_cost(cost_name, 'loss')
        self.__accuracy = 1 - self.__get_cost(cost_name, 'accuracy')
        pass

    def get_learning_rate(self):
        return self.__learning_rate

    def get_cost(self):
        return self.__cost

    def get_accuracy(self):
        return self.__accuracy

    def get_drop(self):
        return self.__drop

    def get_x_input(self):
        return self.__x_input

    def get_y_label(self):
        return self.__y_label

    def __get_cost(self, cost_name, tensor_name = 'loss'):
        Z, H, W, C = self.__y_label.get_shape().as_list()[1:]
        if cost_name == "dice coefficient":
            smooth = 1e-5
            pred_flat = tf.reshape(self.__y_prediction, [-1, H * W * C * Z])
            true_flat = tf.reshape(self.__y_label, [-1, H * W * C * Z])
            intersection = 2 * tf.reduce_sum(pred_flat * true_flat, axis=1) + smooth
            denominator = tf.reduce_sum(pred_flat * pred_flat, axis=1) + tf.reduce_sum(true_flat * true_flat, axis=1) + smooth
            loss = 1 - tf.reduce_mean(intersection / denominator, name=tensor_name)
            return loss

    def __weight_xavier_init(
            self,
            shape,
            n_inputs,
            n_outputs,
            active_function='sigmoid',
            uniform=True,
            variable_name=None):
        with tf.device('/cpu:0'):
            if active_function == 'sigmoid':
                if uniform:
                    init_range = tf.sqrt(6.0 / (n_inputs + n_outputs))
                    initial = tf.random_uniform(shape, -init_range, init_range)
                    return tf.get_variable(name=variable_name, initializer=initial, trainable=True)
                else:
                    stddev = tf.sqrt(2.0 / (n_inputs + n_outputs))
                    initial = tf.truncated_normal(shape, mean=0.0, stddev=stddev)
                    return tf.get_variable(name=variable_name, initializer=initial, trainable=True)
            elif active_function == 'relu':
                if uniform:
                    init_range = tf.sqrt(6.0 / (n_inputs + n_outputs)) * np.sqrt(2)
                    initial = tf.random_uniform(shape, -init_range, init_range)
                    return tf.get_variable(name=variable_name, initializer=initial, trainable=True)
                else:
                    stddev = tf.sqrt(2.0 / (n_inputs + n_outputs)) * np.sqrt(2)
                    initial = tf.truncated_normal(shape, mean=0.0, stddev=stddev)
                    return tf.get_variable(name=variable_name, initializer=initial, trainable=True)
            elif active_function == 'tan':
                if uniform:
                    init_range = tf.sqrt(6.0 / (n_inputs + n_outputs)) * 4
                    initial = tf.random_uniform(shape, -init_range, init_range)
                    return tf.get_variable(name=variable_name, initializer=initial, trainable=True)
                else:
                    stddev = tf.sqrt(2.0 / (n_inputs + n_outputs)) * 4
                    initial = tf.truncated_normal(shape, mean=0.0, stddev=stddev)
                    return tf.get_variable(name=variable_name, initializer=initial, trainable=True)

    def __bias_variable(self, kernel_shape, variable_name=None):
        with tf.device('/cpu:0'):
            initial = tf.constant(0.1, shape=kernel_shape)
            return tf.get_variable(name=variable_name, initializer=initial, trainable=True)

    def __conv3d(self, input, kernel, stride=1):
        conv3d = tf.nn.conv3d(input, kernel, strides=[1, stride, stride, stride, 1], padding='SAME')
        return conv3d

    def __normalizationlayer(
            self,
            x,
            image_depth=None,
            image_height=None,
            image_width=None,
            norm_type=None,
            G=16,
            esp=1e-5,
            scope=None):
        """
        :param x:input data with shap of[batch,height,width,channel]
        :param is_train:flag of normalizationlayer,True is training,False is Testing
        :param height:in some condition,the data height is in Runtime determined,such as through deconv layer and conv2d
        :param width:in some condition,the data width is in Runtime determined
        :param image_z:
        :param norm_type:normalization type:support"batch","group","None"
        :param G:in group normalization,channel is seperated with group number(G)
        :param esp:Prevent divisor from being zero
        :param scope:normalizationlayer scope
        :return:
        """
        with tf.name_scope(scope + norm_type):
            if norm_type == None:
                output = x
            elif norm_type == 'batch':
                output = tf.contrib.layers.batch_norm(x, center=True, scale=True)
            elif norm_type == "group":
                # tranpose:[bs,z,h,w,c]to[bs,c,z,h,w]following the paper
                x = tf.transpose(x, [0, 4, 1, 2, 3])
                N, C, Z, H, W = x.get_shape().as_list()
                G = min(G, C)
                if H == None and W == None and Z == None:
                    Z, H, W = image_depth, image_height, image_width
                x = tf.reshape(x, [-1, G, C // G, Z, H, W])
                mean, var = tf.nn.moments(x, [2, 3, 4, 5], keep_dims=True)
                x = (x - mean) / tf.sqrt(var + esp)
                gama = tf.get_variable(scope + norm_type + 'group_gama', [C], initializer=tf.constant_initializer(1.0))
                beta = tf.get_variable(scope + norm_type + 'group_beta', [C], initializer=tf.constant_initializer(0.0))
                gama = tf.reshape(gama, [1, C, 1, 1, 1])
                beta = tf.reshape(beta, [1, C, 1, 1, 1])
                output = tf.reshape(x, [-1, C, Z, H, W]) * gama + beta
                # tranpose:[bs,c,z,h,w]to[bs,z,h,w,c]following the paper
                output = tf.transpose(output, [0, 2, 3, 4, 1])
            return output

    def __conv_bn_relu_drop(self, x, kernel_shape, depth=None, height=None, width=None, scope=None):
        with tf.name_scope(scope):
            w = self.__weight_xavier_init(
                shape=kernel_shape,
                n_inputs=kernel_shape[0] * kernel_shape[1] * kernel_shape[2] * kernel_shape[3],
                n_outputs=kernel_shape[-1],
                active_function='relu',
                variable_name=scope + '_conv_w')
            b = self.__bias_variable([kernel_shape[-1]], variable_name=scope + '_conv_b')
            conv = self.__conv3d(x, w) + b
            conv = self.__normalizationlayer(
                conv,
                image_depth=depth,
                image_height=height,
                image_width=width,
                norm_type='group',
                scope=scope)
            conv = tf.nn.dropout(tf.nn.relu(conv), self.__drop)
            return conv

    def __resnet_add(self, x1, x2):
        if x1.get_shape().as_list()[4] != x2.get_shape().as_list()[4]:
            # Option A: Zero-padding
            residual_connection = x2 + tf.pad(
                x1,
                [[0, 0], [0, 0], [0, 0], [0, 0], [0, x2.get_shape().as_list()[4] - x1.get_shape().as_list()[4]]])
        else:
            residual_connection = x2 + x1
        return residual_connection

    def __down_sampling(self, x, kernel_shape, depth=None, height=None, width=None, scope=None):
        with tf.name_scope(scope):
            w = self.__weight_xavier_init(
                shape=kernel_shape,
                n_inputs=kernel_shape[0] * kernel_shape[1] * kernel_shape[2] * kernel_shape[3],
                n_outputs=kernel_shape[-1],
                active_function='relu', variable_name=scope + 'w')
            b = self.__bias_variable([kernel_shape[-1]], variable_name=scope + 'b')
            conv = self.__conv3d(x, w, 2) + b
            conv = self.__normalizationlayer(
                conv,
                image_depth=depth,
                image_height=height,
                image_width=width,
                norm_type='group',
                scope=scope)
            conv = tf.nn.dropout(tf.nn.relu(conv), self.__drop)
            return conv

    def __gating_signal3d(self, x, kernel_shape, depth=None, height=None, width=None, scope=None):
        """this is simply 1x1x1 convolution, bn, activation,Gating Signal(Query)
        """
        with tf.name_scope(scope):
            w = self.__weight_xavier_init(
                shape=kernel_shape,
                n_inputs=kernel_shape[0] * kernel_shape[1] * kernel_shape[2] * kernel_shape[3],
                n_outputs=kernel_shape[-1],
                active_function='relu',
                variable_name=scope + 'conv_w')
            b = self.__bias_variable([kernel_shape[-1]], variable_name=scope + 'conv_b')
            conv = self.__conv3d(x, w) + b
            conv = self.__normalizationlayer(
                conv,
                image_depth=depth,
                image_height=height,
                image_width=width,
                norm_type='group',
                scope=scope)
            conv = tf.nn.relu(conv)
            return conv

    def __upsample3d(self, x, scale_factor, scope=None):
        ''''
        X shape is [nsample,dim,rows, cols, channel]
        out shape is[nsample,dim*scale_factor,rows*scale_factor, cols*scale_factor, channel]
        '''
        x_shape = tf.shape(x)
        k = tf.ones([scale_factor, scale_factor, scale_factor, x_shape[-1], x_shape[-1]])
        # note k.shape = [dim,rows, cols, depth_in, depth_output]
        output_shape = tf.stack(
            [x_shape[0], x_shape[1] * scale_factor, x_shape[2] * scale_factor, x_shape[3] * scale_factor, x_shape[4]])
        upsample = tf.nn.conv3d_transpose(value=x, filter=k, output_shape=output_shape,
                                          strides=[1, scale_factor, scale_factor, scale_factor, 1],
                                          padding='SAME', name=scope)
        return upsample

    def __attention_gated_block(
            self,
            x,
            g,
            inputfilters,
            outputfilters,
            scale_factor,
            depth=None,
            height=None,
            width=None,
            scope=None):
        """
        take g which is the spatially smaller signal, do a conv to get the same number of feature channels as x (bigger spatially)
        do a conv on x to also get same feature channels (theta_x)
        then, upsample g to be same size as x add x and g (concat_xg) relu, 1x1x1 conv, then sigmoid then upsample the final -
        this gives us attn coefficients
        :param x:
        :param g:
        :param inputfilters:
        :param outfilters:
        :param scale_factor:2
        :param scope:
        :return:
        """
        with tf.name_scope(scope):
            kernel_x_shape = (1, 1, 1, inputfilters, outputfilters)
            Wx = self.__weight_xavier_init(
                shape=kernel_x_shape,
                n_inputs=kernel_x_shape[0] * kernel_x_shape[1] * kernel_x_shape[2] * kernel_x_shape[3],
                n_outputs=kernel_x_shape[-1],
                active_function='relu',
                variable_name=scope + 'conv_Wx')
            Bx = self.__bias_variable([kernel_x_shape[-1]], variable_name=scope + 'conv_Bx')
            theta_x = self.__conv3d(x, Wx, scale_factor) + Bx

            kernel_g_shape = (1, 1, 1, inputfilters, outputfilters)
            Wg = self.__weight_xavier_init(
                shape=kernel_g_shape,
                n_inputs=kernel_g_shape[0] * kernel_g_shape[1] * kernel_g_shape[2] * kernel_g_shape[3],
                n_outputs=kernel_g_shape[-1],
                active_function='relu',
                variable_name=scope + 'conv_Wg')
            Bg = self.__bias_variable([kernel_g_shape[-1]], variable_name=scope + 'conv_Bg')
            phi_g = self.__conv3d(g, Wg) + Bg

            add_xg = self.__resnet_add(theta_x, phi_g)
            act_xg = tf.nn.relu(add_xg)

            kernel_psi_shape = (1, 1, 1, outputfilters, 1)
            Wpsi = self.__weight_xavier_init(
                shape=kernel_psi_shape,
                n_inputs=kernel_psi_shape[0] * kernel_psi_shape[1] * kernel_psi_shape[2] * kernel_psi_shape[3],
                n_outputs=kernel_psi_shape[-1],
                active_function='relu',
                variable_name=scope + 'conv_Wpsi')
            Bpsi = self.__bias_variable([kernel_psi_shape[-1]], variable_name=scope + 'conv_Bpsi')
            psi = self.__conv3d(act_xg, Wpsi) + Bpsi
            sigmoid_psi = tf.nn.sigmoid(psi)

            upsample_psi = self.__upsample3d(sigmoid_psi, scale_factor=scale_factor, scope=scope + "resampler")

            # Attention: upsample_psi * x
            # upsample_psi = layers.Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=4),
            #                              arguments={'repnum': outfilters})(upsample_psi)
            gat_x = tf.multiply(upsample_psi, x)
            kernel_gat_x_shape = (1, 1, 1, outputfilters, outputfilters)
            Wgatx = self.__weight_xavier_init(
                shape=kernel_gat_x_shape,
                n_inputs=kernel_gat_x_shape[0] * kernel_gat_x_shape[1] * kernel_gat_x_shape[2] * kernel_gat_x_shape[3],
                n_outputs=kernel_gat_x_shape[-1],
                active_function='relu',
                variable_name=scope + 'conv_Wgatx')
            Bgatx = self.__bias_variable([kernel_gat_x_shape[-1]], variable_name=scope + 'conv_Bgatx')
            gat_x_out = self.__conv3d(gat_x, Wgatx) + Bgatx

            gat_x_out = self.__normalizationlayer(
                gat_x_out,
                image_depth=depth,
                image_height=height,
                image_width=width,
                norm_type='group',
                scope=scope)

        return gat_x_out

    def __deconv3d(self, x, w, samefeature=False, depth=False):
        """
        depth flag:False is z axis is same between input and output,true is z axis is input is twice than output
        """
        x_shape = tf.shape(x)
        if depth:
            if samefeature:
                output_shape = tf.stack([x_shape[0], x_shape[1] * 2, x_shape[2] * 2, x_shape[3] * 2, x_shape[4]])
            else:
                output_shape = tf.stack([x_shape[0], x_shape[1] * 2, x_shape[2] * 2, x_shape[3] * 2, x_shape[4] // 2])
            deconv = tf.nn.conv3d_transpose(x, w, output_shape, strides=[1, 2, 2, 2, 1], padding='SAME')
        else:
            if samefeature:
                output_shape = tf.stack([x_shape[0], x_shape[1] * 2, x_shape[2] * 2, x_shape[3], x_shape[4]])
            else:
                output_shape = tf.stack([x_shape[0], x_shape[1] * 2, x_shape[2] * 2, x_shape[3], x_shape[4] // 2])
            deconv = tf.nn.conv3d_transpose(x, w, output_shape, strides=[1, 2, 2, 1, 1], padding='SAME')
        return deconv

    def __deconv_relu(self, x, kernel_shape, samefeture=False, scope=None):
        with tf.name_scope(scope):
            w = self.__weight_xavier_init(
                shape=kernel_shape,
                n_inputs=kernel_shape[0] * kernel_shape[1] * kernel_shape[2] * kernel_shape[-1],
                n_outputs=kernel_shape[-2],
                active_function='relu',
                variable_name=scope + 'w')
            b = self.__bias_variable([kernel_shape[-2]], variable_name=scope + 'b')
            conv = self.__deconv3d(x, w, samefeture, True) + b
            conv = tf.nn.relu(conv)
            return conv

    def __crop_and_concat(self, x1, x2):
        x1_shape = tf.shape(x1)
        x2_shape = tf.shape(x2)
        # offsets for the top left corner of the crop
        offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, (x1_shape[3] - x2_shape[3]) // 2, 0]
        size = [-1, x2_shape[1], x2_shape[2], x2_shape[3], -1]
        x1_crop = tf.slice(x1, offsets, size)
        self.offsets = offsets
        self.size = size
        return tf.concat([x1_crop, x2], 4)

    def __conv_sigmoid(self, x, kernel_shape, scope=None, name=None):
        with tf.name_scope(scope):
            w = self.__weight_xavier_init(
                shape=kernel_shape,
                n_inputs=kernel_shape[0] * kernel_shape[1] * kernel_shape[2] * kernel_shape[3],
                n_outputs=kernel_shape[-1],
                active_function='sigmoid',
                variable_name=scope + 'w')
            b = self.__bias_variable([kernel_shape[-1]], variable_name=scope + 'b')
            conv = self.__conv3d(x, w) + b
            conv = tf.nn.sigmoid(conv, name=name)
            return conv

    def __create_net(self, kernel_param):
        layer1 = self.__conv_bn_relu_drop(x=self.__x_input, kernel_shape=kernel_param[0], scope='layer1')
        layer1 = self.__resnet_add(self.__x_input, layer1)
        down1 = self.__down_sampling(x=layer1, kernel_shape=kernel_param[2], scope='down1')

        # layer2
        layer2 = self.__conv_bn_relu_drop(x=down1, kernel_shape=kernel_param[3], scope='layer2_1')
        layer2 = self.__conv_bn_relu_drop(x=layer2, kernel_shape=kernel_param[4], scope='layer2_2')
        layer2 = self.__resnet_add(x1=down1, x2=layer2)
        down2 = self.__down_sampling(x=layer2, kernel_shape=kernel_param[5], scope='down2')

        # layer3
        layer3 = self.__conv_bn_relu_drop(x=down2, kernel_shape=kernel_param[6], scope='layer3_1')
        layer3 = self.__conv_bn_relu_drop(x=layer3, kernel_shape=kernel_param[7], scope='layer3_2')
        layer3 = self.__conv_bn_relu_drop(x=layer3, kernel_shape=kernel_param[8], scope='layer3_3')
        layer3 = self.__resnet_add(x1=down2, x2=layer3)
        down3 = self.__down_sampling(x=layer3, kernel_shape=kernel_param[9], scope='down3')

        # layer4
        layer4 = self.__conv_bn_relu_drop(x=down3, kernel_shape=kernel_param[10], scope='layer4_1')
        layer4 = self.__conv_bn_relu_drop(x=layer4, kernel_shape=kernel_param[11], scope='layer4_2')
        layer4 = self.__conv_bn_relu_drop(x=layer4, kernel_shape=kernel_param[12], scope='layer4_3')
        layer4 = self.__resnet_add(x1=down3, x2=layer4)
        down4 = self.__down_sampling(x=layer4, kernel_shape=kernel_param[13], scope='down4')

        # layer5
        layer5 = self.__conv_bn_relu_drop(x=down4, kernel_shape=kernel_param[14], scope='layer5_1')
        layer5 = self.__conv_bn_relu_drop(x=layer5, kernel_shape=kernel_param[15], scope='layer5_2')
        layer5 = self.__conv_bn_relu_drop(x=layer5, kernel_shape=kernel_param[16], scope='layer5_3')
        layer5 = self.__resnet_add(x1=down4, x2=layer5)

        g1 = self.__gating_signal3d(layer5, kernel_shape=kernel_param[17], scope='g1')
        attn1 = self.__attention_gated_block(layer4, g1, kernel_param[17][4], kernel_param[17][4],
                                             scale_factor=2, scope='attn1')

        deconv1 = self.__deconv_relu(x=layer5, kernel_shape=kernel_param[18], scope='deconv1')

        # layer6
        layer6 = self.__crop_and_concat(attn1, deconv1)
        _, Z, H, W, _ = attn1.get_shape().as_list()
        layer6 = self.__conv_bn_relu_drop(x=layer6, kernel_shape=kernel_param[19], depth=Z, height=H,
                                          width=W, scope='layer6_1')
        layer6 = self.__conv_bn_relu_drop(x=layer6, kernel_shape=kernel_param[20], depth=Z, height=H,
                                          width=W, scope='layer6_2')
        layer6 = self.__conv_bn_relu_drop(x=layer6, kernel_shape=kernel_param[21], depth=Z, height=H,
                                          width=W, scope='layer6_3')
        layer6 = self.__resnet_add(x1=deconv1, x2=layer6)

        g2 = self.__gating_signal3d(layer6, kernel_shape=kernel_param[22], scope='g2')
        attn2 = self.__attention_gated_block(layer3, g2, kernel_param[22][4], kernel_param[22][4],
                                             scale_factor=2, scope='attn2')

        deconv2 = self.__deconv_relu(x=layer6, kernel_shape=kernel_param[23], scope='deconv2')

        layer7 = self.__crop_and_concat(attn2, deconv2)
        _, Z, H, W, _ = attn2.get_shape().as_list()
        layer7 = self.__conv_bn_relu_drop(x=layer7, kernel_shape=kernel_param[24], depth=Z, height=H,
                                          width=W, scope='layer7_1')
        layer7 = self.__conv_bn_relu_drop(x=layer7, kernel_shape=kernel_param[25], depth=Z, height=H,
                                          width=W, scope='layer7_2')
        layer7 = self.__resnet_add(x1=deconv2, x2=layer7)

        g3 = self.__gating_signal3d(layer7, kernel_shape=kernel_param[26], scope='g3')
        attn3 = self.__attention_gated_block(layer2, g3, kernel_param[26][4], kernel_param[26][4],
                                             scale_factor=2, scope='attn3')

        deconv3 = self.__deconv_relu(x=layer7, kernel_shape=kernel_param[27], scope='deconv3')

        layer8 = self.__crop_and_concat(attn3, deconv3)
        _, Z, H, W, _ = attn3.get_shape().as_list()
        layer8 = self.__conv_bn_relu_drop(x=layer8, kernel_shape=kernel_param[28], depth=Z, height=H,
                                          width=W, scope='layer8_1')
        layer8 = self.__conv_bn_relu_drop(x=layer8, kernel_shape=kernel_param[29], depth=Z, height=H,
                                          width=W, scope='layer8_2')
        layer8 = self.__conv_bn_relu_drop(x=layer8, kernel_shape=kernel_param[30], depth=Z, height=H,
                                          width=W, scope='layer8_3')
        layer8 = self.__resnet_add(x1=deconv3, x2=layer8)

        g4 = self.__gating_signal3d(layer8, kernel_shape=kernel_param[31], scope='g4')
        attn4 = self.__attention_gated_block(layer1, g4, kernel_param[31][4], kernel_param[31][4],
                                             scale_factor=2, scope='attn4')

        deconv4 = self.__deconv_relu(x=layer8, kernel_shape=kernel_param[32], scope='deconv4')

        layer9 = self.__crop_and_concat(attn4, deconv4)
        _, Z, H, W, _ = attn4.get_shape().as_list()
        layer9 = self.__conv_bn_relu_drop(x=layer9, kernel_shape=kernel_param[33], depth=Z, height=H,
                                          width=W, scope='layer9_1')
        layer9 = self.__conv_bn_relu_drop(x=layer9, kernel_shape=kernel_param[34], depth=Z, height=H,
                                          width=W, scope='layer9_2')
        layer9 = self.__conv_bn_relu_drop(x=layer9, kernel_shape=kernel_param[35], depth=Z, height=H,
                                          width=W, scope='layer9_3')
        layer9 = self.__resnet_add(x1=deconv4, x2=layer9)

        # output_map = self.__conv_sigmoid(x=layer9, kernel_shape=kernel_shape_parameters[32], scope='output')
        output_map = self.__conv_sigmoid(x=layer9, kernel_shape=kernel_param[36], scope='output',
                                         name='result')

        return output_map