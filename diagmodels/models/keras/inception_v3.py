# -*- coding: utf-8 -*-
"""Inception V3 model for Keras.

Note that the input image format for this model is different than for
the VGG16 and ResNet models (299x299 instead of 224x224),
and that the input preprocessing function is also different (same as Xception).

# Reference

- [Rethinking the Inception Architecture for Computer Vision](http://arxiv.org/abs/1512.00567)

"""
import warnings

from keras.models import Model
from keras import layers, optimizers
from keras.layers import Activation, Dense, Input, BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Dropout
from keras.engine.topology import get_source_inputs
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.applications.imagenet_utils import _obtain_input_shape

from . import kerasmodelbase as dmskerasmodelbase
from ...errors import modelerrors as dmsmodelerrors

WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.5/inception_v3_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.5/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'


def conv2d_bn(x,
              filters,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1),
              name=None):
    """Utility function to apply conv + BN.

    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.

    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    if K.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3
    x = Conv2D(
        filters, (num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=False,
        name=conv_name)(x)
    x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    x = Activation('relu', name=name)(x)
    return x


class InceptionV3(dmskerasmodelbase.KerasModelBase):
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
        self.__input_shape = (None, None, 3)  # Default
        self.__classes = 1000                 # Number of output classes, default.
        self.__preload_weights = False        # Whether to use weights from ImageNet
        self.__include_top = True             # Whether to include the final fully connected layers
        self.__use_aux_classifier = False     # Whether to use an auxiliary classifier
        self.__depth_multiplier = 1.0         # Multiplies the number of filters per layer with this number
        self.__min_depth = 16                 # Minimum number of filters per layer
        self.__prediction_model = None        # Fully convolutional model for the predict function

    def configure(self, input_shape, classes, include_top=False,
                  preload_weights=False, use_aux_classifier=True, depth_multiplier=1.0, min_depth=16, channels_first=False):
        """
        Save the network configuration parameters.

        Args:
            input_shape (tuple of ints): Shape of the input patches
            classes (int): Number of output classes.
            include_top (bool): Whether to include the original classification layers of the network
            preload_weights (bool): Whether to use weights from ImageNet to initialize model
            use_aux_classifier (bool): Whether to use an auxiliary classifier within the network
            depth_multiplier (float): Multiplies the number of filters per layer with this number
            min_depth (int): Minimum number of filters per layer
            channels_first (bool): If true, the network expect patches shaped BCHW, otherwise BHWC.


        Raises:
            InvalidModelClassCountError: The number of classes is not valid.
        """

        # Check input parameters.
        #

        if classes <= 1:
            raise dmsmodelerrors.InvalidModelClassCountError(classes)

        if input_shape[0] is not None and input_shape[1] is not None:
            if len(input_shape) < 3 or input_shape[0] < 139 or input_shape[1] < 139:
                raise dmsmodelerrors.InvalidInputShapeError(input_shape)

        if preload_weights and include_top and classes != 1000:
            raise ValueError('If using pre-trained weights with `include_top`'
                             ' as true, `classes` should be 1000')

        if preload_weights and use_aux_classifier:
            raise ValueError('Pre-trained weights cannot be combined with the auxiliary classifier')

        if channels_first:
            raise dmsmodelerrors.NotSupportedChannelOrder('Channels first')

        # Save parameters.
        #
        self.__classes = classes
        self.__include_top = include_top
        self.__preload_weights = preload_weights
        self.__use_aux_classifier = use_aux_classifier
        self.__depth_multiplier = depth_multiplier
        self.__min_depth = min_depth

    def build(self):
        """Build the network instance with the pre-configured parameters."""
        optimizer = optimizers.Adam(lr=0.025, epsilon=1.0)

        weights_parameter = "imagenet" if self.__preload_weights else None
        self._model_instance = KerasInceptionV3(self.__include_top, weights_parameter, self.__depth_multiplier,
                                                self.__min_depth, input_shape=self.__input_shape,
                                                classes=self.__classes, use_aux_classifier=self.__use_aux_classifier)
        if self.__use_aux_classifier:
            self._model_instance.compile(optimizer=optimizer, loss={'predictions': 'categorical_crossentropy',
                                                                    'aux_predictions': 'categorical_crossentropy'},
                                         metrics={'predictions': 'accuracy', 'aux_predictions': 'accuracy'})
        else:
            self._model_instance.compile(optimizer=optimizer, loss='categorical_crossentropy',
                                         metrics=['accuracy'])
        try:
            self.__prediction_model = Model(self._model_instance.inputs[0], self._model_instance.get_layer("predictions_4d").output)
        except ValueError:
            self.__prediction_model = self._model_instance
        self._compiled = True

    def _restoremodelparameters(self, parameters):

        super()._restoremodelparameters(parameters)

        try:
            self.__prediction_model = Model(self._model_instance.inputs[0],
                                            self._model_instance.get_layer("predictions_4d").output)
        except ValueError:
            self.__prediction_model = self._model_instance
        self.compile = True

    def update(self, x, y, sample_weight=None, class_weight=None, *args, **kwargs):
        """
        Update the network.

        Args:
            x (numpy.ndarray): an array containing image data.
            y (numpy.ndarray ): an array containing label data.
            sample_weight (numpy.ndarray): an array containing weights for samples to increase their contribution to the loss
            class_weight (numpy.ndarray): an array containing weights for classes to increase their contribution to the loss

        Returns:
            dict: Output of the update function.
        """

        if self.__use_aux_classifier:
            output_values = self._model_instance.train_on_batch(x, {'predictions': y, 'aux_predictions': y},
                                                                sample_weight=sample_weight,
                                                                class_weight=class_weight)
            output_values = dict(zip(self._model_instance.metrics_names, output_values))
            output_values["acc"] = output_values['predictions_acc']
        else:
            output_values =  super().update(x, y, sample_weight=sample_weight, class_weight=class_weight, *args, **kwargs)
        return output_values

    def getreconstructioninformation(self, input_shape=None):
        if self.__prediction_model is None:
            raise dmsmodelerrors.MissingNetworkError()

        selected_layers = [self.__prediction_model.layers[-1]]

        for current_layer in reversed(self.__prediction_model.layers[:-1]):
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

    def validate(self, x, y, sample_weight=None, *args, **kwargs):
        """
        Validate the network.

        Args:
            x (numpy.ndarray or list of numpy.ndarray): contains image data.
            y (numpy.ndarray or list of numpy.ndarray ): contains label data.
            sample_weights (numpy.ndarray): an array containing weights for samples to increase their contribution to the loss
        Returns:
            dict: Output of the validation function.
        """
        if self.__use_aux_classifier:
            output_values = self._model_instance.evaluate(x, {'predictions': y, 'aux_predictions': y}, sample_weight=sample_weight, verbose=0)
            output_values = dict(zip(self._model_instance.metrics_names, output_values))
            output_values["acc"] = output_values['predictions_acc']
        else:
            output_values = super().update(x, y, sample_weight=sample_weight, *args, **kwargs)
        return output_values

    def predict(self, x, *args, **kwargs):
        """
        Use the network for evaluation.

        Args:
            x (numpy.ndarray or list of numpy.ndarray): contains image data.

        Returns:
            list: Output of the evaluation function.
        """

        # Run the model on the set of patches.
        #
        return {"predictions" : self.__prediction_model.predict_on_batch(x)}

def KerasInceptionV3(include_top=True,
                     weights='imagenet',
                     depth_multiplier = 1.0,
                     min_depth = 16,
                     input_tensor=None,
                     input_shape=None,
                     pooling=None,
                     classes=1000,
                     use_aux_classifier=True):
    """Instantiates the Inception v3 architecture.

    Optionally loads weights pre-trained
    on ImageNet. Note that when using TensorFlow,
    for best performance you should set
    `image_data_format='channels_last'` in your Keras config
    at ~/.keras/keras.json.
    The model and the weights are compatible with both
    TensorFlow and Theano. The data format
    convention used by the model is the one
    specified in your Keras config file.
    Note that the default input image size for this model is 299x299.

    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization)
            or 'imagenet' (pre-training on ImageNet).
        depth_multiplier: multiplies the number of filters per layer with this value. Only works when weights = None
        min_depth: minimum number of filters per layer. Only works when weights = None
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(299, 299, 3)` (with `channels_last` data format)
            or `(3, 299, 299)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 139.
            E.g. `(150, 150, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
        use_aux_classifier: whether to use the auxiliary classifier half-way
            through the network.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')

    if weights == 'imagenet' and use_aux_classifier:
        raise ValueError('If using `weights` as imagenet auxiliary classifier cannot be used.')

    if weights == 'ímagenet':
        depth = lambda d: d
    else:
        if depth_multiplier <= 0:
            raise ValueError('depth_multiplier is not greater than zero.')
        depth = lambda d: max(int(d * depth_multiplier), min_depth)

    # Determine proper input shape
    input_shape = _obtain_input_shape(
        input_shape,
        default_size=299,
        min_size=139,
        data_format=K.image_data_format(),
        require_flatten=False,
        weights=weights)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3

    x = conv2d_bn(img_input, depth(32), 3, 3, strides=(2, 2), padding='valid')
    x = conv2d_bn(x, depth(32), 3, 3, padding='valid')
    x = conv2d_bn(x, depth(64), 3, 3)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv2d_bn(x, depth(80), 1, 1, padding='valid')
    x = conv2d_bn(x, depth(192), 3, 3, padding='valid')
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # mixed 0, 1, 2: 35 x 35 x 256
    branch1x1 = conv2d_bn(x, depth(64), 1, 1)

    branch5x5 = conv2d_bn(x, depth(48), 1, 1)
    branch5x5 = conv2d_bn(branch5x5, depth(64), 5, 5)

    branch3x3dbl = conv2d_bn(x, depth(64), 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, depth(96), 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, depth(96), 3, 3)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, depth(32), 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed0')

    # mixed 1: 35 x 35 x 256
    branch1x1 = conv2d_bn(x, depth(64), 1, 1)

    branch5x5 = conv2d_bn(x, depth(48), 1, 1)
    branch5x5 = conv2d_bn(branch5x5, depth(64), 5, 5)

    branch3x3dbl = conv2d_bn(x, depth(64), 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, depth(96), 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, depth(96), 3, 3)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, depth(64), 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed1')

    # mixed 2: 35 x 35 x 256
    branch1x1 = conv2d_bn(x, depth(64), 1, 1)

    branch5x5 = conv2d_bn(x, depth(48), 1, 1)
    branch5x5 = conv2d_bn(branch5x5, depth(64), 5, 5)

    branch3x3dbl = conv2d_bn(x, depth(64), 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, depth(96), 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, depth(96), 3, 3)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, depth(64), 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed2')

    # mixed 3: 17 x 17 x 768
    branch3x3 = conv2d_bn(x, depth(384), 3, 3, strides=(2, 2), padding='valid')

    branch3x3dbl = conv2d_bn(x, depth(64), 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, depth(96), 3, 3)
    branch3x3dbl = conv2d_bn(
        branch3x3dbl, depth(96), 3, 3, strides=(2, 2), padding='valid')

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.concatenate(
        [branch3x3, branch3x3dbl, branch_pool], axis=channel_axis, name='mixed3')

    # mixed 4: 17 x 17 x 768
    branch1x1 = conv2d_bn(x, depth(192), 1, 1)

    branch7x7 = conv2d_bn(x, depth(128), 1, 1)
    branch7x7 = conv2d_bn(branch7x7, depth(128), 1, 7)
    branch7x7 = conv2d_bn(branch7x7, depth(192), 7, 1)

    branch7x7dbl = conv2d_bn(x, depth(128), 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, depth(128), 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, depth(128), 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, depth(128), 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, depth(192), 1, 7)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, depth(192), 1, 1)
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=channel_axis,
        name='mixed4')

    # mixed 5, 6: 17 x 17 x 768
    for i in range(2):
        branch1x1 = conv2d_bn(x, depth(192), 1, 1)

        branch7x7 = conv2d_bn(x, depth(160), 1, 1)
        branch7x7 = conv2d_bn(branch7x7, depth(160), 1, 7)
        branch7x7 = conv2d_bn(branch7x7, depth(192), 7, 1)

        branch7x7dbl = conv2d_bn(x, depth(160), 1, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, depth(160), 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, depth(160), 1, 7)
        branch7x7dbl = conv2d_bn(branch7x7dbl, depth(160), 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, depth(192), 1, 7)

        branch_pool = AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, depth(192), 1, 1)
        x = layers.concatenate(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            axis=channel_axis,
            name='mixed' + str(5 + i))

    aux_x = x

    # mixed 7: 17 x 17 x 768
    branch1x1 = conv2d_bn(x, depth(192), 1, 1)

    branch7x7 = conv2d_bn(x, depth(192), 1, 1)
    branch7x7 = conv2d_bn(branch7x7, depth(192), 1, 7)
    branch7x7 = conv2d_bn(branch7x7, depth(192), 7, 1)

    branch7x7dbl = conv2d_bn(x, depth(192), 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, depth(192), 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, depth(192), 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, depth(192), 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, depth(192), 1, 7)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, depth(192), 1, 1)
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=channel_axis,
        name='mixed7')

    # mixed 8: 8 x 8 x 1280
    branch3x3 = conv2d_bn(x, depth(192), 1, 1)
    branch3x3 = conv2d_bn(branch3x3, depth(320), 3, 3,
                          strides=(2, 2), padding='valid')

    branch7x7x3 = conv2d_bn(x, depth(192), 1, 1)
    branch7x7x3 = conv2d_bn(branch7x7x3, depth(192), 1, 7)
    branch7x7x3 = conv2d_bn(branch7x7x3, depth(192), 7, 1)
    branch7x7x3 = conv2d_bn(
        branch7x7x3, depth(192), 3, 3, strides=(2, 2), padding='valid')

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.concatenate(
        [branch3x3, branch7x7x3, branch_pool], axis=channel_axis, name='mixed8')

    # mixed 9: 8 x 8 x 2048
    for i in range(2):
        branch1x1 = conv2d_bn(x, depth(320), 1, 1)

        branch3x3 = conv2d_bn(x, depth(384), 1, 1)
        branch3x3_1 = conv2d_bn(branch3x3, depth(384), 1, 3)
        branch3x3_2 = conv2d_bn(branch3x3, depth(384), 3, 1)
        branch3x3 = layers.concatenate(
            [branch3x3_1, branch3x3_2], axis=channel_axis, name='mixed9_' + str(i))

        branch3x3dbl = conv2d_bn(x, depth(448), 1, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, depth(384), 3, 3)
        branch3x3dbl_1 = conv2d_bn(branch3x3dbl, depth(384), 1, 3)
        branch3x3dbl_2 = conv2d_bn(branch3x3dbl, depth(384), 3, 1)
        branch3x3dbl = layers.concatenate(
            [branch3x3dbl_1, branch3x3dbl_2], axis=channel_axis)

        branch_pool = AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, depth(192), 1, 1)
        x = layers.concatenate(
            [branch1x1, branch3x3, branch3x3dbl, branch_pool],
            axis=channel_axis,
            name='mixed' + str(9 + i))
    if include_top:
        # Classification block
        if weights == 'imagenet':
            x = GlobalAveragePooling2D(name='avg_pool')(x)
            x = Dropout(0.2)(x)
            x = Dense(classes, activation='softmax', name='predictions')(x)
        else:
            x = AveragePooling2D((8, 8), strides=(1, 1), name='avg_pool')(x)
            x = Dropout(0.2)(x)
            x = Conv2D(classes, (1, 1), activation='softmax', name='predictions_4d')(x)
            x = GlobalAveragePooling2D(name="predictions")(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    if use_aux_classifier:
        aux_x = AveragePooling2D((5, 5), strides=(3, 3), padding='valid')(aux_x)
        aux_x = conv2d_bn(aux_x, depth(128), 1, 1)
        aux_x = conv2d_bn(aux_x, depth(768), 5, 5, padding="valid")
        aux_x = GlobalAveragePooling2D(name='aux_avg_pool')(aux_x)
        aux_x = Dense(classes, activation='softmax', name='aux_predictions')(aux_x)
        model = Model(inputs, [x, aux_x], name='inception_v3')
    else:
        model = Model(inputs, x, name='inception_v3')

    # load weights
    if weights == 'imagenet':
        if K.image_data_format() == 'channels_first':
            if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image data format convention '
                              '(`image_data_format="channels_first"`). '
                              'For best performance, set '
                              '`image_data_format="channels_last"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')
        if include_top:
            weights_path = get_file(
                'inception_v3_weights_tf_dim_ordering_tf_kernels.h5',
                WEIGHTS_PATH,
                cache_subdir='models',
                file_hash='9a0d58056eeedaa3f26cb7ebd46da564')
        else:
            weights_path = get_file(
                'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5',
                WEIGHTS_PATH_NO_TOP,
                cache_subdir='models',
                file_hash='bcbd6486424b2319ff4ef7d526e38f63')
        model.load_weights(weights_path)
    return model
