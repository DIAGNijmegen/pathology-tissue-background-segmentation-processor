"""
Densely Connected Convolutional Networks.
"""

from . import tensorflowmodelbase as dmstensorflowmodelbase
import tensorflow as tf

#----------------------------------------------------------------------------------------------------

class DenseNet(dmstensorflowmodelbase.TensorFlowModelBase):
    """DenseNet model implementation in TensorFlow."""

    def __init__(self, name=None, description=None):
        """
        Initialize model parameters.

        Args:
            name (str): Name of the model.
            description (str): Description of the model.
        """

        # Initialize base class.
        #
        super().__init__(name, description)

        # Initialize members.
        #
        self.__input_shape = None      # Input shape: (channels, rows, cols).
        self.__valid_padding = False   # Use valid padding only.
        self.__pool_initial = False    # Pool initial convolutional layer.
        self.__pool_final = False      # Use global average pooling before the last convolutional layer.
        self.__growth_rate = 0         # Number of filters to add per convolutional block.
        self.__classes = 0             # Number of output classes.
        self.__layers_per_block = []   # Number of layers in each block.
        self.__bottleneck_rate = 0.0   # Bottleneck layer filter count control.
        self.__weight_decay = 0.0      # L2 lambda for loss.
        self.__compression = 0.0       # Reduction of the number of filters to the transition block.
        self.__dropout_rate = 0.0      # Probability of dropout on the dropout layers.
        self._channels_first = False   # If true, the network expect patches shaped BCHW, otherwise BHWC.

    def configure(self, input_shape, valid_padding, pool_initial, pool_final, growth_rate, classes, layers_per_block, bottleneck_rate, weight_decay, compression, dropout_rate):
        """
        Configure the network.

        Args:
            input_shape (tuple): Input shape (height, width, channels).
            valid_padding (bool): Only use valid padding. Introduces cropping layers.
            pool_initial (bool): As a first layer use a 7x7 convolution with 2x2 stride, and a 3x3 max pooling with 2x2 stride, if true. A single 3x3 convolution with 1x1 stride otherwise.
            pool_final (bool): Add global average pooling before the final convolutional layer.
            growth_rate (int): Filter count growth rate. The initial convolution produces 2 * growth_rate filters, all other 3x3 convolutions in the dense blocks produces this amount.
            classes (int): Number of classes.
            layers_per_block (list): Number of 3x3 convolutions in a dense block.
            bottleneck_rate (float): The bottleneck 1x1 convolution layers produce growth_rate * bottleneck_rate filters. (In the article it is fixed to 4.) If the rate is larger than 0.0
                then the bottleneck layers are added.
            weight_decay (float): L2 regularization weight.
            compression (float): Reduction of the number of filters in the transition blocks.
            dropout_rate (float): Dropout probability. If the probability is larger than 0 than the dropout layers are added.
        """

        self.__input_shape = input_shape
        self.__valid_padding = valid_padding
        self.__pool_initial = pool_initial
        self.__pool_final = pool_final
        self.__growth_rate = growth_rate
        self.__classes = classes
        self.__layers_per_block = layers_per_block
        self.__bottleneck_rate = bottleneck_rate
        self.__weight_decay = weight_decay
        self.__compression = compression
        self.__dropout_rate = dropout_rate

    def _customdata(self):
        """
        Get a map of custom data structure to add to the dictionary of exported values.

        Returns:
            dict: Dictionary of values to add to the exported data.
        """

        return {'input_shape': self.__input_shape,
                'valid_padding': self.__valid_padding,
                'pool_initial': self.__pool_initial,
                'pool_final': self.__pool_final,
                'growth_rate': self.__growth_rate,
                'classes': self.__classes,
                'layers_per_block': self.__layers_per_block,
                'bottleneck_rate': self.__bottleneck_rate,
                'weight_decay': self.__weight_decay,
                'compression': self.__compression,
                'dropout_rate': self.__dropout_rate}

    def _setcustomdata(self, data_maps):
        """
        Set the class specific data.

        Args:
            data_maps (dict): Custom data map.
        """

        self.__input_shape = data_maps.get('input_shape', None)
        self.__valid_padding = data_maps.get('valid_padding', False)
        self.__pool_initial = data_maps.get('pool_initial', False)
        self.__pool_final = data_maps.get('pool_final', False)
        self.__growth_rate = data_maps.get('growth_rate', 0)
        self.__classes = data_maps.get('classes', 0)
        self.__layers_per_block = data_maps.get('layers_per_block', [])
        self.__bottleneck_rate = data_maps.get('bottleneck_rate', 0.0)
        self.__weight_decay = data_maps.get('weight_decay', 0.0)
        self.__compression = data_maps.get('compression', 0.0)
        self.__dropout_rate = data_maps.get('dropout_rate', 0.0)

    def _restoremodelparameters(self, parameters):
        """
        Custom restore function. Makes sure that the output tensors are correctly set, even after a load.

        Args:
            parameters (dict): Parameters of the model.
        """

        super()._restoremodelparameters(parameters)

        # Explicitly reset the train and validate functions of the internal model
        # TODO this is a workaround for https://github.com/fchollet/keras/issues/8468
        #
        self._model_instance.train_function = None
        self._model_instance.test_function = None

        # if self._model_instance.optimizer:
        #     self._model_instance.metrics_tensors.append(self._model_instance.outputs[0])
        #     self._model_instance.metrics_names.append('predictions')
        #
        #     l2_loss = sum(self._model_instance.losses)
        #     self._model_instance.metrics_tensors.append(l2_loss)
        #     self._model_instance.metrics_names.append('l2 loss')

    def __initconvolution(self, inputs, shape, training):
        """

        Args:
            inputs:
            shape:
            training:

        Returns:

        """

        data_format = 'channels_first' if self._channels_first else 'channels_last'
        batch_norm_axis = 1 if self._channels_first else -1
        filter_count = 2 * self.__growth_rate
        padding = 'valid' if self.__valid_padding else 'same'

        # Calculate the filters and stride of the initial convolutional layer.
        #
        if self.__pool_initial:
            init_filters = (7, 7)
            init_strides = (2, 2)
        else:
            init_filters = (3, 3)
            init_strides = (1, 1)

        # Add initial convolutional layer.
        #
        init_conv = tf.keras.layers.Conv2D(name='init_conv',
                                           filters=filter_count,
                                           kernel_size=init_filters,
                                           strides=init_strides,
                                           padding=padding,
                                           use_bias=False,
                                           data_format=data_format,
                                           kernel_initializer='he_normal',
                                           kernel_regularizer=tf.keras.regularizers.l2(self.__weight_decay))

        outputs = init_conv(inputs=inputs)
        shape = init_conv.compute_output_shape(input_shape=shape)

        # Add initial pooling layer.
        #
        if self.__pool_initial:
            batch_norm_init = tf.keras.layers.BatchNormalization(name='init_batch_norm', axis=batch_norm_axis)
            outputs = batch_norm_init(inputs=outputs, training=training)
            shape = batch_norm_init.compute_output_shape(input_shape=shape)

            non_linearity_init = tf.keras.layers.Activation(name='init_non_lin', activation='relu')
            outputs = non_linearity_init(inputs=outputs)
            shape = non_linearity_init.compute_output_shape(input_shape=shape)

            pool_init = tf.keras.layers.MaxPooling2D(name='init_max_pool', pool_size=(3, 3), strides=(2, 2), padding=padding, data_format=data_format)
            outputs = pool_init(inputs=outputs)
            shape = pool_init.compute_output_shape(input_shape=shape)

        return outputs, shape

    def __convolutionalblock(self, name, inputs, shape, filter_count, training):
        """

        Args:

        Returns:

        """

        data_format = 'channels_first' if self._channels_first else 'channels_last'
        batch_norm_axis = 1 if self._channels_first else -1
        padding = 'valid' if self.__valid_padding else 'same'

        # Add batch norm layer.
        #
        batch_norm = tf.keras.layers.BatchNormalization(name='{name}_batch_norm'.format(name=name), axis=batch_norm_axis)
        outputs = batch_norm(inputs=inputs, training=training)
        shape = batch_norm.compute_output_shape(input_shape=shape)

        # Add bottleneck 1x1 convolutional layer.
        #
        if 0.0 < self.__bottleneck_rate:
            bottleneck_filter_count = int(filter_count * self.__bottleneck_rate)

            non_linearity_bottleneck = tf.keras.layers.Activation(name='{name}_btn_non_lin'.format(name=name), activation='relu')
            outputs = non_linearity_bottleneck(inputs=outputs)
            shape = non_linearity_bottleneck.compute_output_shape(input_shape=shape)

            conv_1x1_bottleneck = tf.keras.layers.Conv2D(name='{name}_btn_conv'.format(name=name),
                                                         filters=bottleneck_filter_count,
                                                         kernel_size=(1, 1),
                                                         padding='valid',
                                                         use_bias=False,
                                                         data_format=data_format,
                                                         kernel_initializer='he_normal',
                                                         kernel_regularizer=tf.keras.regularizers.l2(self.__weight_decay))

            outputs = conv_1x1_bottleneck(inputs=outputs)
            shape = conv_1x1_bottleneck.compute_output_shape(input_shape=shape)

            batch_norm_bottleneck = tf.keras.layers.BatchNormalization(name='{name}_btn_batch_norm'.format(name=name), axis=batch_norm_axis)
            outputs = batch_norm_bottleneck(inputs=outputs, training=training)
            shape = batch_norm_bottleneck.compute_output_shape(input_shape=shape)

        # Add 3x3 convolutional layer and dropout.
        #
        non_linearity = tf.keras.layers.Activation(name='{name}_non_lin'.format(name=name), activation='relu')
        outputs = non_linearity(inputs=outputs)

        conv_3x3 = tf.keras.layers.Conv2D(name='{name}_conv'.format(name=name),
                                          filters=filter_count,
                                          kernel_size=(3, 3),
                                          padding=padding,
                                          use_bias=False,
                                          data_format=data_format,
                                          kernel_initializer='he_normal',
                                          kernel_regularizer=tf.keras.regularizers.l2(self.__weight_decay))

        outputs = conv_3x3(inputs=outputs)
        shape = conv_3x3.compute_output_shape(input_shape=shape)

        if 0.0 < self.__dropout_rate:
            dropout = tf.keras.layers.SpatialDropout2D(name='{name}_dropout'.format(name=name), rate=self.__dropout_rate, data_format=data_format)
            outputs = dropout(inputs=outputs, training=training)
            shape = dropout.compute_output_shape(input_shape=shape)

        return outputs, shape

    def __denseblock(self, name, inputs, shape, layer_count, training):
        """

        Args:
            layer_count:

        Returns:

        """

        data_format = 'channels_first' if self._channels_first else 'channels_last'
        concat_axis = 1 if self._channels_first else -1
        row_index = 2 if self._channels_first else 1
        col_index = 3 if self._channels_first else 2

        # Build dense block.
        #
        outputs = inputs
        for index in range(layer_count):
            # Add convolutional block.
            #
            block_outputs, block_shape = self.__convolutionalblock(name='{name}_block_{index}'.format(name=name, index=index),
                                                                   inputs=outputs,
                                                                   shape=shape,
                                                                   filter_count=self.__growth_rate,
                                                                   training=training)

            # Add cropping in case of valid padding.
            #
            if self.__valid_padding:
                crop_shape = (int((shape[row_index] - block_shape[row_index]) // 2), int((shape[col_index] - block_shape[col_index]) // 2))
                crop = tf.keras.layers.Cropping2D(cropping=crop_shape, data_format=data_format)
                outputs = crop(inputs=outputs)
                shape = crop.compute_output_shape(input_shape=shape)

            # Concatenate the output of the current convolutional block to the output of the previous blocks.
            #
            concatenate = tf.keras.layers.Concatenate(name='{name}_block_{index}_concat'.format(name=name, index=index), axis=concat_axis)
            outputs = concatenate(inputs=[outputs, block_outputs])
            shape = concatenate.compute_output_shape(input_shape=[shape, block_shape])

        return outputs, shape

    def __transitionblock(self, name, inputs, shape, filter_count, training):
        """

        Args:
            inputs:
            shape:
            filter_count:
            training:

        Returns:

        """

        data_format = 'channels_first' if self._channels_first else 'channels_last'
        batch_norm_axis = 1 if self._channels_first else -1

        # Add batch norm layer.
        #
        batch_norm = tf.keras.layers.BatchNormalization(name='{name}_batch_norm'.format(name=name), axis=batch_norm_axis)
        outputs = batch_norm(inputs=inputs, training=training)
        shape = batch_norm.compute_output_shape(input_shape=shape)

        # Add 1x1 convolutional layer.
        #
        non_linearity = tf.keras.layers.Activation(name='{name}_non_lin'.format(name=name), activation='relu')
        outputs = non_linearity(inputs=outputs)
        shape = non_linearity.compute_output_shape(input_shape=shape)

        conv_1x1 = tf.keras.layers.Conv2D(name='{name}_conv'.format(name=name),
                                          filters=filter_count,
                                          kernel_size=(1, 1),
                                          padding='valid',
                                          use_bias=False,
                                          data_format=data_format,
                                          kernel_initializer='he_normal',
                                          kernel_regularizer=tf.keras.regularizers.l2(self.__weight_decay))

        outputs = conv_1x1(inputs=outputs)
        shape = conv_1x1.compute_output_shape(input_shape=shape)

        # Add average pooling.
        #
        avg_pool = tf.keras.layers.AveragePooling2D(name='{name}_avg_pool'.format(name=name), pool_size=(2, 2), padding='valid', data_format=data_format)
        outputs = avg_pool(inputs=outputs)
        shape = avg_pool.compute_output_shape(input_shape=shape)

        return outputs, shape

    def __finalconvolution(self, inputs, shape, training):
        """

        Args:
            inputs:
            shape:
            training:

        Returns:

        """

        data_format = 'channels_first' if self._channels_first else 'channels_last'
        batch_norm_axis = 1 if self._channels_first else -1
        pool_size = (int(shape[2]), int(shape[3])) if self._channels_first else (int(shape[1]), int(shape[2]))

        # Add batch norm and non-linearity after the last block.
        #
        batch_norm_last_block = tf.keras.layers.BatchNormalization(name='final_batch_norm', axis=batch_norm_axis)
        outputs = batch_norm_last_block(inputs=inputs, training=training)
        shape = batch_norm_last_block.compute_output_shape(input_shape=shape)

        non_linearity_last_block = tf.keras.layers.Activation(name='final_non_lin', activation='relu')
        outputs = non_linearity_last_block(inputs=outputs)
        shape = non_linearity_last_block.compute_output_shape(input_shape=shape)

        # Add global average pooling.
        #
        if self.__pool_final:
            final_conv_size = (1, 1)

            final_pool = tf.keras.layers.AveragePooling2D(name='final_avg_pool', pool_size=pool_size, strides=(1, 1), padding='valid', data_format=data_format)
            outputs = final_pool(inputs=outputs)
            shape = final_pool.compute_output_shape(input_shape=shape)
        else:
            final_conv_size = pool_size

        # Add final convolutional layer (instead of the fully connected layer).
        #
        final_conv = tf.keras.layers.Conv2D(name='final_conv',
                                            filters=self.__classes,
                                            kernel_size=final_conv_size,
                                            strides=(1, 1),
                                            padding='valid',
                                            use_bias=True,
                                            activation='softmax',
                                            data_format=data_format,
                                            kernel_initializer='he_normal',
                                            kernel_regularizer=tf.keras.regularizers.l2(self.__weight_decay))

        outputs = final_conv(inputs=outputs)
        shape = final_conv.compute_output_shape(input_shape=shape)

        # Reshape the output to final (batch, classes) shape.
        #
        if training:
            reshape = tf.keras.layers.Reshape(name='final_reshape', target_shape=(self.__classes,))
            outputs = reshape(inputs=outputs)
            shape = reshape.compute_output_shape(input_shape=shape)

        return outputs, shape

    def build(self, training=True):
        """Build the network instance with the pre-configured parameters."""

        if not self._compiled:
            # Calculate the number of filters after each block using compression to reduce the number of inputs to the transition blocks.
            #
            filter_count = 2 * self.__growth_rate
            filter_counts_after_blocks = [filter_count]
            for i in range(1, len(self.__layers_per_block)):
                filter_counts_after_blocks.append(int((filter_counts_after_blocks[i - 1] + (self.__growth_rate * self.__layers_per_block[i - 1])) * self.__compression))
            filter_counts_after_blocks = filter_counts_after_blocks[1:]

            # Create input layer.
            #
            inputs = tf.keras.Input(name='input', shape=self.__input_shape)
            shape = (None,) + self.__input_shape

            # Add initial convolutional layer.
            #
            outputs, shape = self.__initconvolution(inputs=inputs, shape=shape, training=training)

            # Add dense blocks and transition blocks.
            #
            for block_index in range(len(self.__layers_per_block)):
                outputs, shape = self.__denseblock(name='dense_{index}'.format(index=block_index),
                                                   inputs=outputs,
                                                   shape=shape,
                                                   layer_count=self.__layers_per_block[block_index],
                                                   training=training)

                if block_index + 1 < len(self.__layers_per_block):
                    outputs, shape = self.__transitionblock(name='trans_{index}'.format(index=block_index),
                                                            inputs=outputs, shape=shape,
                                                            filter_count=filter_counts_after_blocks[block_index],
                                                            training=training)

            # Add final convolutional layer.
            #
            outputs, shape = self.__finalconvolution(inputs=outputs, shape=shape, training=training)

            # Build model.
            #
            self._model_instance = tf.keras.models.Model(inputs=inputs, outputs=outputs)
            self._model_instance.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

            self._compiled = True

            # self._model_instance.metrics_tensors.append(self._model_instance.outputs[0])
            # self._model_instance.metrics_names.append('predictions')
            #
            # l2_loss = sum(self._model_instance.losses)
            # self._model_instance.metrics_tensors.append(l2_loss)
            # self._model_instance.metrics_names.append('l2 loss')
            #
            # self._addcustommetric(name='errors')
