"""
Naturenet implementation using Keras.
"""

from . import kerasmodelbase as dmskerasmodelbase
from ...errors import modelerrors as dmsmodelerrors

import keras
import numpy as np

#----------------------------------------------------------------------------------------------------

class NatureNet(dmskerasmodelbase.KerasModelBase):
    """NatureNet model implementation in Keras."""

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
        self.__input_shape = None     # Input shape: (channels, rows, cols).
        self.__classes = 0            # Number of output classes.
        self.__branching_factor = 0   # Filter count branching factor.
        self.__batch_norm = False     # Batch normalization.
        self.__dropout_count = 0      # Number of dropout layers to add.
        self.__dropout_prob = 0.0     # Probability of dropout on the dropout layers.
        self.__l2_lambda = 0.0        # L2 lambda for loss.
        self._channels_first = False  # If true, the network expect patches shaped BCHW, otherwise BHWC.

    def configure(self, input_shape, classes, branching_factor, batch_norm, dropout_count, dropout_prob, l2_lambda, channels_first):
        """
        Save the network configuration parameters.

        Args:
            input_shape (tuple): Number of channels and size of the individual input images. (rows, cols, channels)
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
            InvalidDimensionOrder: The dimension order is not valid.

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
                'batch_norm': self.__batch_norm,
                'dropout_count': self.__dropout_count,
                'dropout_prob': self.__dropout_prob,
                'l2_lambda': self.__l2_lambda}

    def _setcustomdata(self, data_maps):
        """
        Set the class specific data.

        Args:
            data_maps (dict): Custom data map.
        """

        # Configure custom data.
        #
        self.__classes = data_maps.get('classes', 0)
        self.__branching_factor = data_maps.get('branching_factor', 0)
        self.__batch_norm = data_maps.get('batch_norm', False)
        self.__dropout_count = data_maps.get('dropout_count', 0)
        self.__dropout_prob = data_maps.get('dropout_prob', 0.0)
        self.__l2_lambda = data_maps.get('l2_lambda', 0.0)
        self._channels_first = data_maps.get('channel_first', False)

    def _networkdefinition(self):
        """
        Network definition.

        Returns
            Tensor: The network architecture.
        """

        # Create the network with free input shape.
        #
        unfixed_input_shape = (None,) * (len(self.__input_shape) - 1) + (self.__input_shape[-1],)

        inputs = keras.layers.Input(shape=unfixed_input_shape)
        l2 = keras.regularizers.l2(self.__l2_lambda)

        # Convolutional layer 0.
        #
        filter_count_0 = 2 ** self.__branching_factor
        conv_layer_0 = keras.layers.Conv2D(filters=filter_count_0,
                                           kernel_size=5,
                                           activation='relu',
                                           kernel_initializer='he_normal',
                                           bias_initializer='zeros',
                                           padding='valid',
                                           kernel_regularizer=l2)

        network = conv_layer_0(inputs)

        if self.__batch_norm:
            batch_norm_layer_0 = keras.layers.BatchNormalization(axis=1 if self._channels_first else -1)
            network = batch_norm_layer_0(network)

        max_pool_layer_0 = keras.layers.MaxPool2D(pool_size=2, strides=2)
        network = max_pool_layer_0(network)

        # Convolutional layer 1.
        #
        filter_count_1 = 2 * filter_count_0
        conv_layer_1 = keras.layers.Conv2D(filters=filter_count_1,
                                           kernel_size=5,
                                           activation='relu',
                                           kernel_initializer='he_normal',
                                           bias_initializer='zeros',
                                           padding='valid',
                                           kernel_regularizer=l2)
        network = conv_layer_1(network)

        if self.__batch_norm:
            batch_norm_layer_1 = keras.layers.BatchNormalization(axis=1 if self._channels_first else -1)
            network = batch_norm_layer_1(network)

        max_pool_layer_1 = keras.layers.MaxPool2D(pool_size=2, strides=2)
        network = max_pool_layer_1(network)

        # Convolutional layer 2.
        #
        filter_count_2 = 2 * filter_count_1
        conv_layer_2 = keras.layers.Conv2D(filters=filter_count_2,
                                           kernel_size=3,
                                           activation='relu',
                                           kernel_initializer='he_normal',
                                           bias_initializer='zeros',
                                           padding='valid',
                                           kernel_regularizer=l2)
        network = conv_layer_2(network)

        if self.__batch_norm:
            batch_norm_layer_2 = keras.layers.BatchNormalization(axis=1 if self._channels_first else -1)
            network = batch_norm_layer_2(network)

        max_pool_layer_2 = keras.layers.MaxPool2D(pool_size=2, strides=2)
        network = max_pool_layer_2(network)

        # Convolutional layer 3.
        #
        filter_count_3 = filter_count_2
        conv_layer_3 = keras.layers.Conv2D(filters=filter_count_3,
                                           kernel_size=3,
                                           activation='relu',
                                           kernel_initializer='he_normal',
                                           bias_initializer='zeros',
                                           padding='valid',
                                           kernel_regularizer=l2)
        network = conv_layer_3(network)

        if self.__batch_norm:
            batch_norm_layer_3 = keras.layers.BatchNormalization(axis=1 if self._channels_first else -1)
            network = batch_norm_layer_3(network)

        # Convolutional layer 4.
        #
        filter_count_4 = filter_count_2 * 16
        conv_layer_4 = keras.layers.Conv2D(filters=filter_count_4,
                                           kernel_size=11,
                                           activation='relu',
                                           kernel_initializer='he_normal',
                                           bias_initializer='zeros',
                                           padding='valid',
                                           kernel_regularizer=l2)

        network = conv_layer_4(network)

        if self.__batch_norm:
            batch_norm_layer_4 = keras.layers.BatchNormalization(axis=1 if self._channels_first else -1)
            network = batch_norm_layer_4(network)

        if self.__dropout_count > 1:
            dropout_layer_4 = keras.layers.SpatialDropout2D(rate=self.__dropout_prob)
            network = dropout_layer_4(network)

        # Convolutional layer 5.
        #
        filter_count_5 = filter_count_4 // 2
        conv_layer_5 = keras.layers.Conv2D(filters=filter_count_5,
                                           kernel_size=1,
                                           activation='relu',
                                           kernel_initializer='he_normal',
                                           bias_initializer='zeros',
                                           padding='valid',
                                           kernel_regularizer=l2)
        network = conv_layer_5(network)

        if self.__batch_norm:
            batch_norm_layer_5 = keras.layers.BatchNormalization(axis=1 if self._channels_first else -1)
            network = batch_norm_layer_5(network)

        if self.__dropout_count > 0:
            dropout_layer_5 = keras.layers.SpatialDropout2D(rate=self.__dropout_prob)
            network = dropout_layer_5(network)

        # Last layer uses softmax as the activation.
        #
        conv_layer_6 = keras.layers.Conv2D(filters=self.__classes,
                                           kernel_size=1,
                                           activation='softmax',
                                           kernel_initializer='he_normal',
                                           bias_initializer='zeros',
                                           padding='valid')

        network = conv_layer_6(network)

        # Reshape to get (batch_size, n_classes) as output.
        #
        reshape_layer_6 = keras.layers.Reshape(target_shape=(self.__classes,), name='output_layer')
        network = reshape_layer_6(network)

        return inputs, network

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

        if self._model_instance.optimizer:
            self._model_instance.metrics_tensors.append(self._model_instance.outputs[0])
            self._model_instance.metrics_names.append('predictions')

            l2_loss = sum(self._model_instance.losses)
            self._model_instance.metrics_tensors.append(l2_loss)
            self._model_instance.metrics_names.append('l2 loss')

            self._addcustommetric(name='errors')

    def build(self):
        """Build the network instance with the pre-configured parameters."""

        if not self._compiled:
            inputs, outputs = self._networkdefinition()
            optimizer = keras.optimizers.Adam(lr=0.0001)

            self._model_instance = keras.models.Model(inputs, outputs)
            self._model_instance.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

            self._model_instance.metrics_tensors.append(self._model_instance.outputs[0])
            self._model_instance.metrics_names.append('predictions')

            l2_loss = sum(self._model_instance.losses)
            self._model_instance.metrics_tensors.append(l2_loss)
            self._model_instance.metrics_names.append('l2 loss')

            self._addcustommetric(name='errors')

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

        # Get the output of the network.
        #
        output = super().update(x=x, y=y, sample_weight=sample_weight, class_weight=class_weight, *args, **kwargs)

        # Calculate the errors.
        #
        errors = y - output['predictions']
        errors = np.amax(a=errors, axis=1)

        # Multiply the errors with the weights.
        #
        if sample_weight is not None:
            errors = errors * sample_weight / sample_weight.sum()

        # Add errors the the output.
        #
        output['errors'] = errors

        return output
