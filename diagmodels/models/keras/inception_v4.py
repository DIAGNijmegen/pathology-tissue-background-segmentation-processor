"""
Inception v4 implementation using Keras.

Paper: https://arxiv.org/pdf/1602.07261.pdf

Addaptation of source to our library.

Source https://github.com/kentsommer/keras-inceptionV4/blob/master/inception_v4.py
"""

from . import kerasmodelbase as dmskerasmodelbase
from ...errors import modelerrors as dmsmodelerrors

import numpy as np
from keras import models, layers, optimizers, regularizers, initializers
import keras.backend as K
from keras.utils.data_utils import get_file

#----------------------------------------------------------------------------------------------------

class Inceptionv4(dmskerasmodelbase.KerasModelBase):
    """This class is the base class for all network model classes implemented using Keras"""

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
        self.__input_shape = None               # Input shape: (channels, rows, cols).
        self.__classes = 0                      # Number of output classes.
        self.__l2_lambda = 0.00004              # L2 lambda for loss.
        self.__dropout_prob = 0.2               # Probability of dropout on the dropout layers.
        self._channels_first = False            # If true, the network expect patches shaped BCHW, otherwise BHWC.
        self.__load_weights = False              # load predefined weights
        self._model_predict_model = None        # predict Model of model

    def configure(self, input_shape, classes, dropout_prob, l2_lambda, channels_first, load_weights):
        """
        Save the network configuration parameters.

        Args:
            input_shape (tuple): Number of channels and size of the individual input images. (channels, rows, cols)
            classes (int): Number of output classes.
            dropout_prob (float): Dropout probability.
            l2_lambda (float): L2 loss lambda.
            channels_first (bool): If true, the network expect patches shaped BCHW, otherwise BHWC.
            load_weights (bool): If true, the network will load a pre-trained network.


        Raises:
            InvalidInputShapeError: The input shape is not valid.
            InvalidModelClassCountError: The number of classes is not valid.
            InvalidDropoutProbabilityError: The dropout probability is out of (0.0, 1.0) bounds.
        """

        # Check input parameters.
        #
        if len(input_shape) != 3 or min(input_shape) <= 0:
            raise dmsmodelerrors.InvalidInputShapeError(input_shape)

        if classes <= 1:
            raise dmsmodelerrors.InvalidModelClassCountError(classes)

        if dropout_prob < 0.0 or 1.0 <= dropout_prob:
            raise dmsmodelerrors.InvalidDropoutProbabilityError(dropout_prob)

        if channels_first:
            raise dmsmodelerrors.NotSupportedChannelOrder('Channels first')

        # Save parameters.
        #
        self.__input_shape = input_shape
        self.__classes = classes
        self.__l2_lambda = l2_lambda
        self.__load_weights = load_weights
        self._channels_first = channels_first

        # Set the correct channel order for the backend
        #
        self.setchannelsorder(channels_first)

    def _customdata(self):
        """
        Get a map of custom data structure to add to the dictionary of exported values.

        Returns:
            dict: Dictionary of values to add to the exported data.
        """

        # Collect custom data.
        #
        return {'classes': self.__classes,
                'dropout_prob': self.__dropout_prob,
                'l2_lambda': self.__l2_lambda,
                'load_weights': self.__load_weights,
                'channel_first': True if K.image_data_format() == 'channels_first' else False}

    def _setcustomdata(self, data_maps):
        """
        Set the class specific data.

        Args:
            data_maps (dict): Custom data map.
        """

        # Configure custom data.
        #
        self._classes = data_maps.get('classes', 0)
        self.__dropout_prob = data_maps.get('dropout_prob', 0.0)
        self.__l2_lambda = data_maps.get('l2_lambda', 0.0)
        self.__load_weights = data_maps.get('load_weights', False)
        self._channels_first = data_maps.get('channel_first', False)

    def build(self):
        """Build the network instance with the pre-configured parameters."""

        inputs, outputs_base = self.__networkdefinitionbase()

        self._model_instance = models.Model(inputs, outputs_base, name='inception_v4')

        if self.__load_weights:
            # TODO: fix this, because now it gives an error. 
            # Load pre-trained network
            #
            self._model_instance = self.__loadbasemodelparams(self._model_instance)

        outputs = self.__networkdefinition(outputs_base)

        optimizer = optimizers.SGD(lr=1e-3, momentum=0.9, nesterov=True)

        self._model_instance = models.Model(inputs, outputs)

        self._model_instance.compile(optimizer=optimizer, loss=['categorical_crossentropy'], metrics=['accuracy'])

        self._compiled = True

        # Retrieve output shape of the intermediate output layer to reshape predictions.
        #
        self.__output_shape = self._model_instance.get_layer('output').output_shape

        self._model_instance.metrics_tensors += self._model_instance.outputs
        self._model_instance.metrics_names += ['predictions']

        self._addcustommetric(name='errors')

        self._model_predict_model = models.Model([self._model_instance.layers[0].input],
                                                 [self._model_instance.layers[-2].output])

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

        self._model_instance.metrics_tensors += self._model_instance.outputs
        self._model_instance.metrics_names += ['predictions']

        self._model_predict_model = models.Model([self._model_instance.layers[0].input],
                                                 [self._model_instance.layers[-2].output])

    def update(self, x, y, *args, **kwargs):
        """
        Update the network.

        Args:
            x (numpy.ndarray): an array containing image data.
            y (numpy.ndarray ): an array containing label data.

        Returns:
            dict: Output of the update function.
        """

        output = super().update(x, y)
        output['errors'] = np.max((y - output['predictions']), axis=1)

        if len(output['errors'].shape) == 1:
            output['errors'] = np.expand_dims(output['errors'], axis=0)

        return output

    def predict(self, x, *args, **kwargs):
        """
        Use the network for evaluation. Get the layer before the reshape (so N, H, W, C) instead of (H, C)
        Args:
            x (numpy.ndarray or list of numpy.ndarray): contains image data.
        Returns:
            dict: Output of the evaluation function.
        """

        return {"predictions": self._model_predict_model.predict_on_batch(x)}

    @property
    def inputshape(self):
        """
        Return input shape of unet

        Returns:
            list: input shape
        """
        if not self._compiled:
            raise dmsmodelerrors.ModelNotCompiledError()

        if not self.__input_shape:
            self.__input_shape = self._model_instance.input_shape

        return self.__input_shape

    @property
    def outputshape(self):
        """
        Return output shape of unet

        Returns:
            list: Output shape

        Raises:
            ModelNotCompiledError
        """
        if not self._compiled:
            raise dmsmodelerrors.ModelNotCompiledError()

        if not self.__output_shape:
            self.__output_shape = self._model_instance.get_layer('output').output_shape

        return self.__output_shape

    def __conv2d_bn(self, inputs, nr_filters, num_rows, num_cols, padding='same', strides=(1, 1)):
        """
        Utility function to apply conv + BN.
        (Slightly modified from https://github.com/fchollet/keras/blob/master/keras/applications/inception_v3.py)
        """
        if self._channels_first:
            channel_axis = 1
        else:
            channel_axis = -1

        x = layers.Convolution2D(nr_filters, (num_rows, num_cols),
                          strides=strides,
                          padding=padding,
                          use_bias=False,
                          kernel_regularizer=regularizers.l2(self.__l2_lambda),
                          kernel_initializer=initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='normal', seed=None))(inputs)
        x = layers.BatchNormalization(axis=channel_axis, momentum=0.9997, scale=False)(x)
        x = layers.Activation('relu')(x)

        return x

    def __blocka(self, inputs):
        """
        Define inception block A.
        """

        if self._channels_first:
            channel_axis = 1
        else:
            channel_axis = -1

        branch_0 = self.__conv2d_bn(inputs, 96, 1, 1)

        branch_1 = self.__conv2d_bn(inputs, 64, 1, 1)
        branch_1 = self.__conv2d_bn(branch_1, 96, 3, 3)

        branch_2 = self.__conv2d_bn(inputs, 64, 1, 1)
        branch_2 = self.__conv2d_bn(branch_2, 96, 3, 3)
        branch_2 = self.__conv2d_bn(branch_2, 96, 3, 3)

        branch_3 = layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(inputs)
        branch_3 = self.__conv2d_bn(branch_3, 96, 1, 1)

        x = layers.merge.concatenate([branch_0, branch_1, branch_2, branch_3], axis=channel_axis)
        return x

    def __blockreductiona(self, inputs):
        """
        Define inception reduction block A.
        """

        if self._channels_first:
            channel_axis = 1
        else:
            channel_axis = -1

        branch_0 = self.__conv2d_bn(inputs, 384, 3, 3, strides=(2, 2), padding='valid')

        branch_1 = self.__conv2d_bn(inputs, 192, 1, 1)
        branch_1 = self.__conv2d_bn(branch_1, 224, 3, 3)
        branch_1 = self.__conv2d_bn(branch_1, 256, 3, 3, strides=(2, 2), padding='valid')

        branch_2 = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(inputs)

        x = layers.merge.concatenate([branch_0, branch_1, branch_2], axis=channel_axis)
        return x

    def __blockb(self, inputs):
        """
        Define inception block B.
        """

        if self._channels_first:
            channel_axis = 1
        else:
            channel_axis = -1

        branch_0 = self.__conv2d_bn(inputs, 384, 1, 1)

        branch_1 = self.__conv2d_bn(inputs, 192, 1, 1)
        branch_1 = self.__conv2d_bn(branch_1, 224, 1, 7)
        branch_1 = self.__conv2d_bn(branch_1, 256, 7, 1)

        branch_2 = self.__conv2d_bn(inputs, 192, 1, 1)
        branch_2 = self.__conv2d_bn(branch_2, 192, 7, 1)
        branch_2 = self.__conv2d_bn(branch_2, 224, 1, 7)
        branch_2 = self.__conv2d_bn(branch_2, 224, 7, 1)
        branch_2 = self.__conv2d_bn(branch_2, 256, 1, 7)

        branch_3 = layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(inputs)
        branch_3 = self.__conv2d_bn(branch_3, 128, 1, 1)

        x = layers.merge.concatenate([branch_0, branch_1, branch_2, branch_3], axis=channel_axis)
        return x

    def __blockreductionb(self, inputs):
        """
        Define inception reduction block B.
        """

        if self._channels_first:
            channel_axis = 1
        else:
            channel_axis = -1

        branch_0 = self.__conv2d_bn(inputs, 192, 1, 1)
        branch_0 = self.__conv2d_bn(branch_0, 192, 3, 3, strides=(2, 2), padding='valid')

        branch_1 = self.__conv2d_bn(inputs, 256, 1, 1)
        branch_1 = self.__conv2d_bn(branch_1, 256, 1, 7)
        branch_1 = self.__conv2d_bn(branch_1, 320, 7, 1)
        branch_1 = self.__conv2d_bn(branch_1, 320, 3, 3, strides=(2, 2), padding='valid')

        branch_2 = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(inputs)

        x = layers.merge.concatenate([branch_0, branch_1, branch_2], axis=channel_axis)
        return x

    def __blockc(self, inputs):
        """
        Define inception block C.
        """

        if self._channels_first:
            channel_axis = 1
        else:
            channel_axis = -1

        branch_0 = self.__conv2d_bn(inputs, 256, 1, 1)

        branch_1 = self.__conv2d_bn(inputs, 384, 1, 1)
        branch_10 = self.__conv2d_bn(branch_1, 256, 1, 3)
        branch_11 = self.__conv2d_bn(branch_1, 256, 3, 1)
        branch_1 = layers.merge.concatenate([branch_10, branch_11], axis=channel_axis)

        branch_2 = self.__conv2d_bn(inputs, 384, 1, 1)
        branch_2 = self.__conv2d_bn(branch_2, 448, 3, 1)
        branch_2 = self.__conv2d_bn(branch_2, 512, 1, 3)
        branch_20 = self.__conv2d_bn(branch_2, 256, 1, 3)
        branch_21 = self.__conv2d_bn(branch_2, 256, 3, 1)
        branch_2 = layers.merge.concatenate([branch_20, branch_21], axis=channel_axis)

        branch_3 = layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(inputs)
        branch_3 = self.__conv2d_bn(branch_3, 256, 1, 1)

        x = layers.merge.concatenate([branch_0, branch_1, branch_2, branch_3], axis=channel_axis)
        return x

    def __inceptionbase(self, inputs):
        """
        Define inception base
        """

        if K.image_data_format() == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1

        net = self.__conv2d_bn(inputs, 32, 3, 3, strides=(2, 2), padding='valid')
        net = self.__conv2d_bn(net, 32, 3, 3, padding='valid')
        net = self.__conv2d_bn(net, 64, 3, 3)

        branch_0 = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(net)

        branch_1 = self.__conv2d_bn(net, 96, 3, 3, strides=(2, 2), padding='valid')

        net = layers.merge.concatenate([branch_0, branch_1], axis=channel_axis)

        branch_0 = self.__conv2d_bn(net, 64, 1, 1)
        branch_0 = self.__conv2d_bn(branch_0, 96, 3, 3, padding='valid')

        branch_1 = self.__conv2d_bn(net, 64, 1, 1)
        branch_1 = self.__conv2d_bn(branch_1, 64, 1, 7)
        branch_1 = self.__conv2d_bn(branch_1, 64, 7, 1)
        branch_1 = self.__conv2d_bn(branch_1, 96, 3, 3, padding='valid')

        net = layers.merge.concatenate([branch_0, branch_1], axis=channel_axis)

        branch_0 = self.__conv2d_bn(net, 192, 3, 3, strides=(2, 2), padding='valid')
        branch_1 = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(net)

        net = layers.merge.concatenate([branch_0, branch_1], axis=channel_axis)

        # 35 x 35 x 384
        # 4 x Inception-A blocks
        for idx in range(4):
            net = self.__blocka(net)

        # 35 x 35 x 384
        # Reduction-A block
        net = self.__blockreductiona(net)

        # 17 x 17 x 1024
        # 7 x Inception-B blocks
        for idx in range(7):
            net = self.__blockb(net)

        # 17 x 17 x 1024
        # Reduction-B block
        net = self.__blockreductionb(net)

        # 8 x 8 x 1536
        # 3 x Inception-C blocks
        for idx in range(3):
            net = self.__blockc(net)

        return net

    def __networkdefinitionbase(self):
        """
        Network definition.

        Returns
            Tensor: The network architecture.

        """

        # Set the correct input shape for channels first or last
        # Note that the network should be trained with a patch_size of 299x299
        #
        if self._channels_first:
            inputs = layers.Input(shape=[3, None, None], name='input')
        else:
            inputs = layers.Input(shape=[None, None, 3], name='input')

        net = self.__inceptionbase(inputs)

        return inputs, net

    def __loadbasemodelparams(self, model):
        """
        Function to load a pre-trained inception V4 network
        Cannot use model.layers.pop() since model is not of Sequential() type
        # The method below works since pre-trained weights are stored in layers but not in the model

        """

        path_to_params='https://github.com/kentsommer/keras-inceptionV4/releases/download/2.1/inception-v4_weights_tf_dim_ordering_tf_kernels_notop.h5'

        weights_path = get_file(
            'inception-v4_weights_tf_dim_ordering_tf_kernels_notop.h5',
            path_to_params,
            cache_subdir='models',
            md5_hash='9296b46b5971573064d12e4669110969')
        model.load_weights(weights_path, by_name=True)

        return model

    def __networkdefinition(self, network):
        """
        Network definition.

        Returns final output.
        """

        # Addapt original to make it fully convolutional
        #
        net = layers.AveragePooling2D((8, 8), padding='valid', strides=(1, 1))(network)
        net = layers.Dropout(self.__dropout_prob)(net)
        output = layers.Conv2D(self.__classes, (1, 1), activation='softmax', name='output',
                               kernel_initializer='he_normal')(net)

        # Reshape to get (batch_size, n_classes) as output
        #
        flatten = layers.Reshape((self.__classes,), name='output_layer')(output)

        return flatten

    def getreconstructioninformation(self, input_shape=None):
        # TODO: fix reconstruction information, because the network_lost is equal to the input size (shouldn't be the case).
        #

        if self._model_predict_model is None:
            raise dmsmodelerrors.MissingNetworkError()

        selected_layers = [self._model_predict_model.layers[-1]]

        for current_layer in reversed(self._model_predict_model.layers[:-1]):
            if isinstance(selected_layers[-1].input, list):
                current_input = selected_layers[-1].input[0]
            else:
                current_input = selected_layers[-1].input
            if current_layer.output == current_input:
                selected_layers.append(current_layer)
        selected_layers = [l for l in selected_layers if not any(isinstance(l, layer) for layer in [layers.InputLayer,
                                                                                                    layers.Dropout,
                                                                                                    layers.Cropping2D,
                                                                                                    layers.BatchNormalization,
                                                                                                    layers.Concatenate,
                                                                                                    layers.Reshape,
                                                                                                    layers.Activation])]

        if input_shape is None:
            input_shape = selected_layers[0].input_shape
        if len(input_shape) == 3:
            input_shape = (1,) + tuple(input_shape)

        return self._getreconstructioninformationforlayers(input_shape, list(reversed(selected_layers)))