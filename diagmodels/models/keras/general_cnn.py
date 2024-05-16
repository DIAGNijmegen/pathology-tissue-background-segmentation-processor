"""
Naturenet implementation using Keras.
"""

from . import kerasmodelbase as dmskerasmodelbase
from ...errors import modelerrors as dmsmodelerrors

import numpy as np
from keras import models, layers, optimizers, regularizers
import keras.backend as K

#----------------------------------------------------------------------------------------------------

class general_FCNN(dmskerasmodelbase.KerasModelBase):
    """This class is the base class for all network model classes implemented using Keras."""

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
        self.__depth = 4                        # Number of conv blocks in the network
        self.__branching_factor = 0             # Filter count branching factor.
        self.__batch_norm = False               # Batch normalization.
        self.__dropout_count = 0                # Number of dropout layers to add.
        self.__dropout_prob = 0.0               # Probability of dropout on the dropout layers.
        self.__l2_lambda = 0.0                  # L2 lambda for loss.
        self._channels_first = False            # If true, the network expect patches shaped BCHW, otherwise BHWC.
        self._model_predict_model = None        # predict Model of model

    def configure(self, input_shape, depth, classes, branching_factor, batch_norm, dropout_count, dropout_prob, l2_lambda, channels_first):
        """
        Save the network configuration parameters.

        Args:
            input_shape (tuple): Number of channels and size of the individual input images. (channels, rows, cols)
            classes (int): Number of output classes.
            branching_factor (int): Branching factor. The number of filters on the first layer is 2 on the power of branching factor.
            batch_norm (bool): Use batch normalization.
            dropout_count (int): Number of dropout layers.
            dropout_prob (float): Dropout probability.
            l2_lambda (float): L2 loss lambda.
            channels_first (bool): If true, the network expect patches shaped BCHW, otherwise BHWC.

        Raises:
            InvalidInputShapeError: The input shape is not valid.
            InvalidModelClassCountError: The number of classes is not valid.
            InvalidBranchingFactorError: The branching factor is not valid.
            InvalidDropoutLayerCountError: The number of dropout layers is not valid.
            InvalidDropoutProbabilityError: The dropout probability is out of (0.0, 1.0) bounds.
        """

        # Check input parameters.
        #
        if len(input_shape) != 3 or min(input_shape) <= 0:
            raise dmsmodelerrors.InvalidInputShapeError(input_shape)

        if classes <= 1:
            raise dmsmodelerrors.InvalidModelClassCountError(classes)

        if branching_factor < 0:
            raise dmsmodelerrors.InvalidBranchingFactorError(branching_factor)

        if dropout_count < 0 or 2 < dropout_count:
            raise dmsmodelerrors.InvalidDropoutLayerCountError(dropout_count)

        if dropout_prob < 0.0 or 1.0 <= dropout_prob:
            raise dmsmodelerrors.InvalidDropoutProbabilityError(dropout_prob)

        # TODO: remove this if we update the keras model, see https://github.com/keras-team/keras/issues/10382
        if channels_first:
            raise dmsmodelerrors.NotSupportedChannelOrder('Channels first')

        # Save parameters.
        #
        self.__input_shape = input_shape
        self.__classes = classes
        self.__branching_factor = branching_factor
        self.__batch_norm = batch_norm
        self.__dropout_count = dropout_count
        self.__dropout_prob = dropout_prob
        self.__l2_lambda = l2_lambda
        self.__depth = depth
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
                'branching_factor': self.__branching_factor,
                'depth': self.__depth,
                'batch_norm': self.__batch_norm,
                'dropout_count': self.__dropout_count,
                'dropout_prob': self.__dropout_prob,
                'l2_lambda': self.__l2_lambda,
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
        self.__branching_factor = data_maps.get('branching_factor', 0)
        self.__depth = data_maps.get('depth', 0)
        self.__batch_norm = data_maps.get('batch_norm', False)
        self.__dropout_count = data_maps.get('dropout_count', 0)
        self.__dropout_prob = data_maps.get('dropout_prob', 0.0)
        self.__l2_lambda = data_maps.get('l2_lambda', 0.0)
        self._channels_first = data_maps.get('channel_first', False)

    def build(self):
        """Build the network instance with the pre-configured parameters."""

        inputs, outputs = self.__networkdefinition()
        optimizer = optimizers.Adam(lr=0.0001)

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

    def __conv_block(self, input, filters, pool, activation, l2, shape):
        """
        Build a single convolution block of the fcnn.

        Args:
            input: Network input at this point.
            filters (int): Number of filters.
            activation (string|callable): Activation function for the convolution layers.
            l2 (Regularizer): Regularizer object

        """

        net = layers.Conv2D(filters, kernel_size=3, activation=activation, padding='valid',
                            kernel_regularizer=l2, kernel_initializer='he_normal')(input)

        shape -= (2, 2)

        if self.__batch_norm:
            net = layers.BatchNormalization(axis=1 if self._channels_first else -1)(net)

        net = layers.Conv2D(filters, kernel_size=3, activation=activation, padding='valid',
                            kernel_regularizer=l2, kernel_initializer='he_normal')(net)

        shape -= (2, 2)

        if self.__batch_norm:
            net = layers.BatchNormalization(axis=1 if self._channels_first else -1)(net)

        if pool == 'maxpool':
            net = layers.MaxPool2D(pool_size=2, strides=2)(net)
            shape //= 2

        elif pool == 'avg':
            net = layers.AveragePooling2D(pool_size=2, strides=2)(net)
            shape //= 2

        return net, shape


    def __level_block(self, input, filters, depth, activation, l2):
        """


        Args:
            net: Current network object.
            filters (int): Number of filters on this level.
            depth (int): Depth at this stage.
            activation (string|callable): Activation function for layers.
            l2 (Regularizer): Regularizer object
        """

        shape = np.asanyarray(self.__input_shape[0:2])

        for indx in range(depth):

            if indx == 0:
                net = layers.Conv2D(filters, kernel_size=5, activation=activation, padding='valid',
                                    kernel_regularizer=l2, kernel_initializer='he_normal')(input)

                # Lost this layer
                #
                shape -= (4, 4)

                if self.__batch_norm:
                    net = layers.BatchNormalization(axis=1 if self._channels_first else -1)(net)

                net = layers.Conv2D(filters, kernel_size=5, activation=activation, padding='valid',
                                    kernel_regularizer=l2, kernel_initializer='he_normal')(net)

                # Lost this layer
                #
                shape -= (4, 4)

                if self.__batch_norm:
                    net = layers.BatchNormalization(axis=1 if self._channels_first else -1)(net)
                net = layers.MaxPool2D(pool_size=2, strides=2)(net)

                # Lost this layer
                #
                shape //= 2

            else:
                # build conv block
                #
                filters *= 2

                ## apply average pooling to the last block
                #
                if indx == (depth - 1):
                    net, shape = self.__conv_block(input=net, filters=filters, pool='avg',
                                                   activation=activation, l2=l2, shape=shape)

                else:
                    net, shape = self.__conv_block(input=net, filters=filters, pool='maxpool',
                                                   activation=activation, l2=l2, shape=shape)

        if self.__dropout_prob > 0 and self.__dropout_count > 2:
            # Dropout per channel.
            #
            self.__dropout_count -= 1
            noise_shape = (None, 1, 1, filters) if not self._channels_first else (None, filters, 1, 1)
            net = layers.Dropout(self.__dropout_prob, noise_shape=noise_shape)(net)

        filters *= 2
        net = layers.Conv2D(filters, kernel_size=(int(shape[0]), int(shape[1])), activation=activation, padding='valid',
                            kernel_regularizer=l2, kernel_initializer='he_normal')(net)

        if self.__batch_norm:
            net = layers.BatchNormalization(axis=1 if self._channels_first else -1)(net)

        if self.__dropout_prob > 0 and self.__dropout_count > 1:
            # Dropout per channel.
            #
            self.__dropout_count -= 1
            noise_shape = (None, 1, 1, filters)
            net = layers.Dropout(self.__dropout_prob, noise_shape=noise_shape)(net)

        filters = filters // 2
        net = layers.Conv2D(filters, kernel_size=1, activation=activation, padding='valid', kernel_regularizer=l2,
                            kernel_initializer='he_normal')(net)

        if self.__batch_norm:
            net = layers.BatchNormalization(axis=1 if self._channels_first else -1)(net)

        return net

    def __networkdefinition(self):
        """
        Network definition.

        Returns
            Tensor: The network architecture.
        """

        # Set the correct input shape for channels first or last
        #
        if self._channels_first:
            input = layers.Input(shape=[3, None, None], name='input')
        else:
            input = layers.Input(shape=[None, None, 3], name='input')

        l2 = regularizers.l2(self.__l2_lambda)

        net = self.__level_block(input, 2 ** self.__branching_factor, self.__depth, 'relu', l2)

        output = layers.Conv2D(self.__classes, 1, activation='softmax', name='output', kernel_regularizer=l2,
                               kernel_initializer='he_normal')(net)

        # Reshape to get (batch_size, n_classes) as output
        #
        flatten = layers.Reshape((self.__classes,), name='output_layer')(output)

        return input, flatten
