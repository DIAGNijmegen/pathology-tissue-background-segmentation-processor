"""
Nature-net network model.
"""

from . import lasagnemodelbase as dmsmodelbase

from ...errors import modelerrors as dmsmodelerrors

import lasagne
import lasagne.layers as L
import theano
import theano.tensor as T

#----------------------------------------------------------------------------------------------------

class NatureNet(dmsmodelbase.LasagneModelBase):
    """Nature-net network model."""

    def __init__(self, name=None, description=None, levels=None):
        """
        Initialize model parameters.

        Args:
            name (str): Name of the model.
            description (str): Description of the model.
            levels (list): Processing levels.

        Raises:
            InvalidImageLevel: The configured level is negative.
        """

        # Initialize base class.
        #
        super().__init__(name, description)

        # Initialize members.
        #
        self.__branching_factor = 0  # Filter count branching factor.
        self.__batch_norm = False    # Batch normalization.
        self.__dropout_count = 0     # Number of dropout layers to add.
        self.__dropout_prob = 0.0    # Probability of dropout on the dropout layers.

    def configure(self, input_shape, classes, branching_factor, batch_norm, dropout_count, dropout_prob, l2_lambda, channels_first):
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

        if not channels_first:
            raise dmsmodelerrors.NotSupportedChannelOrder('Channels last')

        # Save parameters.
        #
        self._input_shape = input_shape
        self._classes = classes
        self.__branching_factor = branching_factor
        self.__batch_norm = batch_norm
        self.__dropout_count = dropout_count
        self.__dropout_prob = dropout_prob
        self._l2_lambda = l2_lambda

        # Clear existing instances.
        #
        self.clear()
        self._model_instance = self._buildnetworkfromdefinition()

    def _customdata(self):
        """
        Get a map of custom data structure to add to the dictionary of exported values.

        Returns:
            dict: Dictionary of values to add to the exported data.
        """

        # Collect custom data.
        #
        return {'classes': self._classes,
                'branching_factor': self.__branching_factor,
                'batch_norm': self.__batch_norm,
                'dropout_count': self.__dropout_count,
                'dropout_prob': self.__dropout_prob,
                'l2_lambda': self._l2_lambda}

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
        self.__batch_norm = data_maps.get('batch_norm', False)
        self.__dropout_count = data_maps.get('dropout_count', 0)
        self.__dropout_prob = data_maps.get('dropout_prob', 0.0)
        self.__l2_lambda = data_maps.get('l2_lambda', 0.0)

    def _buildnetworkfromdefinition(self):
        """
        Construct the network layers.

        Returns:
            lasagne.layers.Layer: Output layer.
        """

        # Create the input (the patches) and targets (the class labels) variables.
        #
        input_tensor = T.ftensor4('patches')

        # Input layer.
        #
        input_shape = (None, ) + self._input_shape
        input_layer_name = 'input'
        network = {input_layer_name: L.InputLayer(shape=input_shape, input_var=input_tensor, name=input_layer_name)}

        # Convolutional layer 0.
        #
        convolution_layer_0_name = 'convolution_0'
        convolution_layer_0_filter_count = 2 ** self.__branching_factor
        network[convolution_layer_0_name] = L.Conv2DLayer(incoming=network[input_layer_name],
                                                          name=convolution_layer_0_name,
                                                          num_filters=convolution_layer_0_filter_count,
                                                          filter_size=5,
                                                          pad='valid',
                                                          W=lasagne.init.HeNormal(gain='relu'),
                                                          nonlinearity=lasagne.nonlinearities.rectify)

        # Add batch normalization.
        #
        if self.__batch_norm:
            network[convolution_layer_0_name] = L.batch_norm(network[convolution_layer_0_name])

        # Max pooling layer 0.
        #
        pooling_layer_0_name = 'pool_0'
        network[pooling_layer_0_name] = L.MaxPool2DLayer(incoming=network[convolution_layer_0_name], name=pooling_layer_0_name, pool_size=2, stride=2)

        # Convolutional layer 1.
        #
        convolution_layer_1_name = 'convolution_1'
        convolution_layer_1_filter_count = convolution_layer_0_filter_count * 2
        network[convolution_layer_1_name] = L.Conv2DLayer(incoming=network[pooling_layer_0_name],
                                                          name=convolution_layer_1_name,
                                                          num_filters=convolution_layer_1_filter_count,
                                                          filter_size=5,
                                                          pad='valid',
                                                          W=lasagne.init.HeNormal(gain='relu'),
                                                          nonlinearity=lasagne.nonlinearities.rectify)

        # Add batch normalization.
        #
        if self.__batch_norm:
            network[convolution_layer_1_name] = L.batch_norm(network[convolution_layer_1_name])

        # Max pooling layer 1.
        #
        pooling_layer_1_name = 'pool_1'
        network[pooling_layer_1_name] = L.MaxPool2DLayer(incoming=network[convolution_layer_1_name], name=pooling_layer_1_name, pool_size=2, stride=2)

        # Convolutional layer 2.
        #
        convolution_layer_2_name = 'convolution_2'
        convolution_layer_2_filter_count = convolution_layer_1_filter_count * 2
        network[convolution_layer_2_name] = L.Conv2DLayer(incoming=network[pooling_layer_1_name],
                                                          name=convolution_layer_2_name,
                                                          num_filters=convolution_layer_2_filter_count,
                                                          filter_size=3,
                                                          pad='valid',
                                                          W=lasagne.init.HeNormal(gain='relu'),
                                                          nonlinearity=lasagne.nonlinearities.rectify)

        # Add batch normalization.
        #
        if self.__batch_norm:
            network[convolution_layer_2_name] = L.batch_norm(network[convolution_layer_2_name])

        # Max pooling layer 2.
        #
        pooling_layer_2_name = 'pool_2'
        network[pooling_layer_2_name] = L.MaxPool2DLayer(incoming=network[convolution_layer_2_name], name=pooling_layer_2_name, pool_size=2, stride=2)

        # Convolutional layer 3.
        #
        convolution_layer_3_name = 'convolution_3'
        convolution_layer_3_filter_count = convolution_layer_2_filter_count
        network[convolution_layer_3_name] = L.Conv2DLayer(incoming=network[pooling_layer_2_name],
                                                          name=convolution_layer_3_name,
                                                          num_filters=convolution_layer_3_filter_count,
                                                          filter_size=3,
                                                          pad='valid',
                                                          W=lasagne.init.HeNormal(gain='relu'),
                                                          nonlinearity=lasagne.nonlinearities.rectify)

        # Add batch normalization.
        #
        if self.__batch_norm:
            network[convolution_layer_3_name] = L.batch_norm(network[convolution_layer_3_name])

        # Convolutional layer 4.
        #
        convolution_layer_4_name = 'convolution_4'
        convolution_layer_4_filter_count = convolution_layer_3_filter_count * 16
        network[convolution_layer_4_name] = L.Conv2DLayer(incoming=network[convolution_layer_3_name],
                                                          name=convolution_layer_4_name,
                                                          num_filters=1024,
                                                          filter_size=11,
                                                          pad='valid',
                                                          W=lasagne.init.HeNormal(gain='relu'),
                                                          nonlinearity=lasagne.nonlinearities.rectify)

        # Add batch normalization.
        #
        if self.__batch_norm:
            network[convolution_layer_4_name] = L.batch_norm(network[convolution_layer_4_name])

        # Add dropout layer.
        #
        if 1 < self.__dropout_count:
            dropout_layer_4_name = 'dropout_4'
            dropout_layer_4_params = {'name': dropout_layer_4_name, 'p': self.__dropout_prob, 'rescale': True}
            convolution_layer_5_incoming = dropout_layer_4_name
            network[dropout_layer_4_name] = L.dropout_channels(incoming=network[convolution_layer_4_name], **dropout_layer_4_params)
        else:
            convolution_layer_5_incoming = convolution_layer_4_name

        # Convolutional layer 5.
        #
        convolution_layer_5_name = 'convolution_5'
        convolution_layer_5_filter_count = convolution_layer_4_filter_count // 2
        network[convolution_layer_5_name] = L.Conv2DLayer(incoming=network[convolution_layer_5_incoming],
                                                          name=convolution_layer_5_name,
                                                          num_filters=convolution_layer_5_filter_count,
                                                          filter_size=1,
                                                          pad='valid',
                                                          W=lasagne.init.HeNormal(gain='relu'),
                                                          nonlinearity=lasagne.nonlinearities.rectify)

        # Add batch normalization.
        #
        if self.__batch_norm:
            network[convolution_layer_5_name] = L.batch_norm(network[convolution_layer_5_name])

        # Add dropout layer.
        #
        if 0 < self.__dropout_count:
            dropout_layer_5_name = 'dropout_5'
            dropout_layer_5_params = {'name': dropout_layer_5_name, 'p': self.__dropout_prob, 'rescale': True}
            output_incoming = dropout_layer_5_name
            network[dropout_layer_5_name] = L.dropout_channels(incoming=network[convolution_layer_5_name], **dropout_layer_5_params)
        else:
            output_incoming = convolution_layer_5_name

        # Output layer.
        #
        output_layer_name = 'output'
        network[output_layer_name] = L.Conv2DLayer(incoming=network[output_incoming],
                                                   name=output_layer_name,
                                                   num_filters=self._classes,
                                                   filter_size=1,
                                                   pad='valid',
                                                   W=lasagne.init.HeNormal(gain='relu'),
                                                   nonlinearity=lasagne.nonlinearities.rectify)

        # Return the network object.
        #
        return network[output_layer_name]
