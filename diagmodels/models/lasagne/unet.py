"""
U-Net network model.
"""

from . import lasagnemodelbase as dmsmodelbase

from ...errors import modelerrors as dmsmodelerrors

import lasagne
import lasagne.layers as L
import theano
import theano.tensor as T

#----------------------------------------------------------------------------------------------------

class UNet(dmsmodelbase.LasagneModelBase):
    """U-Net network model."""

    def __init__(self, name=None, description=None, levels=None):
        """
        Initialize model parameters.

        Args:
            name (str): Name of the model.
            description (str): Description of the model.
            levels (list): Processing level.
        """

        # Initialize base class.
        #
        super().__init__(name=name, description=description)

        # Initialize members.
        #
        self.__output_shape = None   # Output shape: (rows, cols, channels
        self.__depth = 0             # Number of contraction steps.
        self.__classes = 0           # Number of output classes.
        self.__branching_factor = 0  # Filter count branching factor.
        self.__batch_norm = False    # Batch normalization.
        self.__dropout_count = 0     # Number of dropout layers to add.
        self.__dropout_prob = 0.0    # Probability of dropout on the dropout layers.
        self.__l2_lambda = 0.0       # L2 lambda for loss.
        self.__padding = None        # Padding type.
        weight_map_input = [T.ftensor4('weights')]
        self._setextrainputsforupdate(weight_map_input)    # Add inputs for the weight map
        self._setextrainputsforvalidate(weight_map_input)  # Add inputs for the weight map

    @staticmethod
    def outputsizeforinput(input_size, depth):
        """
        Calculate the size of the image at each layer based on the input size.

        Args:
            input_size (int): Size of the input.
            depth (int): Depth of the model.

        Returns:
            list: List of sizes in each depth.
        """

        # First contraction.
        #
        input_size_list = [input_size]
        input_size -= 4
        input_size_list.append(input_size)

        # Contraction steps.
        #
        for _ in range(depth - 1):
            input_size //= 2
            input_size -= 4
            input_size_list.append(input_size)

        # Expansion steps.
        #
        for _ in range(depth - 1):
            input_size *= 2
            input_size -= 4
            input_size_list.append(input_size)

        # Return result list.
        #
        return input_size_list

    def configure(self, input_shape, depth, classes, branching_factor, batch_norm, dropout_count, dropout_prob, l2_lambda, padding, channels_first):
        """
        Set the network configuration parameters.

        Args:
            input_shape (tuple): Number of channels and size of the individual input images. (channels, rows, cols)
            depth (int): Depth of the network.
            classes (int): Number of output classes.
            branching_factor (int): Branching factor. The number of filters on the first layer is 2 on the power of branching factor.
            batch_norm (bool): Use batch normalization.
            dropout_count (int): Number of dropout layers.
            dropout_prob (float): Dropout probability.
            l2_lambda (float): L2 loss lambda.
            padding (string): Value for the padding.
            channels_first (bool): If true, the network expect patches shaped BCHW, otherwise BHWC.

        Raises:
            InvalidInputShapeError: The input shape is not valid.
            InvalidModelDepthError: The model depth is not valid.
            InvalidModelClassCountError: The number of classes is not valid.
            InvalidBranchingFactorError: The branching factor is not valid.
            InvalidDropoutLayerCountError: The number of dropout layers is not valid.
            InvalidDropoutProbabilityError: The dropout probability is out of (0.0, 1.0) bounds.
            InvalidDimensionOrder: The dimension order is not valid.
        """

        # Check input parameters.
        #
        if len(input_shape) != 3:
            raise dmsmodelerrors.InvalidInputShapeError(input_shape)

        if not channels_first:
            raise dmsmodelerrors.NotSupportedChannelOrder('Channels last')

        if depth <= 0:
            raise dmsmodelerrors.InvalidModelDepthError(depth)

        if classes <= 1:
            raise dmsmodelerrors.InvalidModelClassCountError(classes)

        if branching_factor < 0:
            raise dmsmodelerrors.InvalidBranchingFactorError(branching_factor)

        if dropout_count < 0 or depth < dropout_count:
            raise dmsmodelerrors.InvalidDropoutLayerCountError(dropout_count)

        if 0 < dropout_count and (dropout_prob <= 0.0 or 1.0 <= dropout_prob):
            raise dmsmodelerrors.InvalidDropoutProbabilityError(dropout_prob)

        if padding not in ['same', 'valid']:
            raise dmsmodelerrors.NotSupportedPaddingError(padding, 'conv2d')

        # Save parameters.
        #
        self._input_shape = input_shape
        self.__depth = depth
        self.__classes = classes
        self.__branching_factor = branching_factor
        self.__batch_norm = batch_norm
        self.__dropout_count = dropout_count
        self.__dropout_prob = dropout_prob
        self.__l2_lambda = l2_lambda
        self.__padding = padding

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
        return {'depth': self.__depth,
                'classes': self.__classes,
                'branching_factor': self.__branching_factor,
                'batch_norm': self.__batch_norm,
                'dropout_count': self.__dropout_count,
                'dropout_prob': self.__dropout_prob,
                'l2_lambda': self.__l2_lambda,
                'padding': self.__padding}

    def _setcustomdata(self, data_maps):
        """
        Set the class specific data.

        Args:
            data_maps (dict): Custom data map.
        """

        # Configure custom data.
        #
        self.__depth = data_maps.get('depth', 0)
        self.__classes = data_maps.get('classes', 0)
        self.__branching_factor = data_maps.get('branching_factor', 0)
        self.__batch_norm = data_maps.get('batch_norm', False)
        self.__dropout_count = data_maps.get('dropout_count', 0)
        self.__dropout_prob = data_maps.get('dropout_prob', 0.0)
        self.__l2_lambda = data_maps.get('l2_lambda', 0.0)
        self.__padding = data_maps.get('padding', None)

    def _filters_for_depth(self, depth_index):
        """
        Get the number of filters for the given depth.

        Args:
            depth_index (int): Index of the depth.

        Returns:
            int: Number of filter for the given depth.
        """

        return 2 ** (self.__branching_factor + depth_index)

    def _append_contraction_step(self, network_instance, depth_index):
        """
        Append a contraction step to the network instance.

        Args:
            network_instance (dict): Network built so far.
            depth_index (int): Index of the current depth.
        """

        # Calculate parameters for this step.
        #
        filter_count = self._filters_for_depth(depth_index)                          # Calculate the number of filters at this depth.
        deepest_layer = (depth_index == self.__depth - 1)                            # There is no pooling at last contraction step.
        add_dropout_layer = (self.__depth - 1 - self.__dropout_count < depth_index)  # Add dropout to the end of the contraction steps.

        # Add convolutional layers with batch normalization.
        #
        convolution_layer_0_incoming = 'input' if depth_index == 0 else 'contraction_pool_{depth}'.format(depth=(depth_index - 1))
        convolution_layer_0_name = 'contraction_convolution_{depth}_0'.format(depth=depth_index)
        network_instance[convolution_layer_0_name] = L.Conv2DLayer(incoming=network_instance[convolution_layer_0_incoming],
                                                                   name=convolution_layer_0_name,
                                                                   num_filters=filter_count,
                                                                   filter_size=3,
                                                                   pad=self.__padding,
                                                                   W=lasagne.init.HeNormal(gain='relu'),
                                                                   nonlinearity=lasagne.nonlinearities.rectify)

        if self.__batch_norm:
            network_instance[convolution_layer_0_name] = L.batch_norm(network_instance[convolution_layer_0_name])

        convolution_layer_1_name = 'contraction_convolution_{depth}_1'.format(depth=depth_index)
        network_instance[convolution_layer_1_name] = L.Conv2DLayer(incoming=network_instance[convolution_layer_0_name],
                                                                   name=convolution_layer_1_name,
                                                                   num_filters=filter_count,
                                                                   filter_size=3,
                                                                   pad=self.__padding,
                                                                   W=lasagne.init.HeNormal(gain='relu'),
                                                                   nonlinearity=lasagne.nonlinearities.rectify)

        if self.__batch_norm:
            network_instance[convolution_layer_1_name] = L.batch_norm(network_instance[convolution_layer_1_name])

        # Add dropout layer if necessary.
        #
        if add_dropout_layer:
            dropout_layer_name = 'contraction_dropout_{depth}'.format(depth=depth_index)
            dropout_layer_params = {'name': dropout_layer_name, 'p': self.__dropout_prob, 'rescale': True}
            network_instance[dropout_layer_name] = L.dropout_channels(incoming=network_instance[convolution_layer_1_name], **dropout_layer_params)
        else:
            dropout_layer_name = None  # Just to suppress warning.

        # Add pooling layer if necessary.
        #
        if not deepest_layer:
            pooling_layer_incoming_name = dropout_layer_name if add_dropout_layer else convolution_layer_1_name
            pooling_layer_name = 'contraction_pool_{depth}'.format(depth=depth_index)
            network_instance[pooling_layer_name] = L.MaxPool2DLayer(incoming=network_instance[pooling_layer_incoming_name], name=pooling_layer_name, pool_size=2, stride=2)

    def _append_expansion_step(self, network_instance, depth_index):
        """
        Append a expansion step to the network instance.

        Args:
            network_instance (dict): Network built so far.
            depth_index (int): Index of the current depth.
        """

        # Calculate parameters for this step.
        #
        filter_count = self._filters_for_depth(depth_index)                     # Calculate the number of filters at this depth.
        deepest_layer = (depth_index == self.__depth - 2)                       # There is no pooling at last contraction step.
        from_dropout = (deepest_layer and 0 < self.__dropout_count)             # Check if the incoming connection from the contraction step is dropout.
        bridge_dropout = self.__depth - 1 - self.__dropout_count < depth_index  # Check if the contraction side used dropout on the corresponding step.

        # Add upconvolution layer with batch normalization.
        #
        if deepest_layer:
            upconvolution_layer_incoming = 'contraction_dropout_{depth}'.format(depth=(depth_index + 1)) if from_dropout else 'contraction_convolution_{depth}_1'.format(depth=(depth_index + 1))
        else:
            upconvolution_layer_incoming = 'expansion_convolution_{depth}_1'.format(depth=(depth_index + 1))
        upconvolution_layer_name = 'expansion_upconvolution_{depth}'.format(depth=depth_index)
        network_instance[upconvolution_layer_name] = L.TransposedConv2DLayer(incoming=network_instance[upconvolution_layer_incoming],
                                                                             name=upconvolution_layer_name,
                                                                             num_filters=filter_count,
                                                                             filter_size=2,
                                                                             stride=2,
                                                                             crop='valid',
                                                                             W=lasagne.init.HeNormal(gain='relu'),
                                                                             nonlinearity=lasagne.nonlinearities.rectify)

        if self.__batch_norm:
            network_instance[upconvolution_layer_name] = L.batch_norm(network_instance[upconvolution_layer_name])

        # Add bridge connection from the contraction side.
        #
        bridge_layer_incoming = 'contraction_dropout_{depth}'.format(depth=depth_index) if bridge_dropout else 'contraction_convolution_{depth}_1'.format(depth=depth_index)
        bridge_layer_name = 'expansion_bridge_{depth}'.format(depth=depth_index)
        network_instance[bridge_layer_name] = L.ConcatLayer(incomings=[network_instance[upconvolution_layer_name], network_instance[bridge_layer_incoming]],
                                                            name=bridge_layer_name,
                                                            axis=1,
                                                            cropping=[None, None, 'center', 'center'])

        # Acc convolutional layers with batch normalization.
        #
        convolution_layer_0_name = 'expansion_convolution_{depth}_0'.format(depth=depth_index)
        network_instance[convolution_layer_0_name] = L.Conv2DLayer(incoming=network_instance[bridge_layer_name],
                                                                   name=convolution_layer_0_name,
                                                                   num_filters=filter_count,
                                                                   filter_size=3,
                                                                   pad=self.__padding,
                                                                   W=lasagne.init.HeNormal(gain='relu'),
                                                                   nonlinearity=lasagne.nonlinearities.rectify)

        if self.__batch_norm:
            network_instance[convolution_layer_0_name] = L.batch_norm(network_instance[convolution_layer_0_name])

        convolution_layer_1_name = 'expansion_convolution_{depth}_1'.format(depth=depth_index)
        network_instance[convolution_layer_1_name] = L.Conv2DLayer(incoming=network_instance[convolution_layer_0_name],
                                                                   name=convolution_layer_1_name,
                                                                   num_filters=filter_count,
                                                                   filter_size=3,
                                                                   pad=self.__padding,
                                                                   W=lasagne.init.HeNormal(gain='relu'),
                                                                   nonlinearity=lasagne.nonlinearities.rectify)

        if self.__batch_norm:
            network_instance[convolution_layer_1_name] = L.batch_norm(network_instance[convolution_layer_1_name])

    def _metrics(self, target, output, l2_loss):
        """
        Get the metrics calculation: loss, accuracy and prediction.

        Args:
            target (theano.tensor.ftensor4): Target classes tensor.
            output (theano.tensor.ftensor4): Network output tensor.
            l2_loss (theano.tensor.fscalar): L2 loss.

        Returns:
            theano.tensor.fscalar, theano.tensor.fscalar, theano.tensor.fvector, theano.tensor.itensor4: Loss, accuracy, errors, prediction.
        """

        # Store the original network output shape for later shape recoveries.
        #
        output_shape = output.shape

        # Flatten parameters.
        #
        output_flat = output.dimshuffle(1, 0, 2, 3).flatten(ndim=2).dimshuffle(1, 0)
        target_flat = target.flatten(ndim=1)
        weight_flat = self._extra_inputs['update'][0].flatten(ndim=1)

        # Convert target to one-hot.
        #
        target_flat_one_hot = lasagne.utils.one_hot(target_flat, self.__classes)

        # Softmax output, original paper may have used a sigmoid but here we opt for softmax, as this also works for multiclass segmentation.
        #
        prediction_flat = lasagne.nonlinearities.softmax(output_flat)

        # Calculate loss.
        #
        epsilon = 1e-8
        loss = lasagne.objectives.categorical_crossentropy(T.clip(prediction_flat, epsilon, 1.0 - epsilon), target_flat_one_hot)
        loss = loss * weight_flat
        loss = loss.mean()
        loss += l2_loss

        # Pixel-wise accuracy.
        #
        weight_sum = T.sum(weight_flat)
        accuracy = T.sum(T.eq(T.argmax(prediction_flat, axis=1), target_flat) * weight_flat) / weight_sum

        # Reshape prediction to its original shape.
        #
        prediction_shape = [output_shape[0], output_shape[2], output_shape[3], output_shape[1]]
        prediction = prediction_flat.reshape(shape=prediction_shape, ndim=4).dimshuffle(0, 3, 1, 2)

        # Patch-wise errors.
        #
        errors_flat = (T.max(target_flat_one_hot - prediction_flat, axis=1) * weight_flat) / weight_sum
        errors_shape = [output_shape[0], output_shape[2], output_shape[3]]
        errors = T.sum(errors_flat.reshape(shape=errors_shape, ndim=3), axis=[1, 2])

        # Return loss, accuracy, errors and prediction.
        #
        return loss, accuracy, errors, prediction

    def _defineevaluationfunction(self, output_layer, network_input):
        """
        Define the evaluation function.

        Args:
            output_layer (theano.layers.Layer): Output layer of the network.
            network_input (list): List of input tensors of the network.

        Returns:
            function: Network evaluation function.
        """

        # Get the output from the network.
        #
        network_output = L.get_output(output_layer, deterministic=True)

        # Calculate shape and shuffling directives.
        #
        output_shape = network_output.shape
        recon_shape = [output_shape[0], output_shape[2], output_shape[3], output_shape[1]]

        # Flatten the output to all data, class value dimensions, apply softmax and restore the result to its original shape.
        #
        network_output_flat = network_output.dimshuffle(1, 0, 2, 3).flatten(ndim=2).dimshuffle(1, 0)
        network_prediction = lasagne.nonlinearities.softmax(network_output_flat).reshape(shape=recon_shape, ndim=4).dimshuffle(0, 3, 1, 2)

        # Return the evaluation function.
        #
        return theano.function(inputs=[theano.In(tensor_item) for tensor_item in network_input], outputs=[theano.Out(network_prediction)], allow_input_downcast=True)

    def _definefunctions(self, output_layer, input_tensors):
        """
        Define the update, validation and evaluation functions.

        Args:
            output_layer (lasagne.layers.Layer): Output layer of the network.
            input_tensors (list): List of input tensors.

        Returns:
            function, function, function: Update, validation and evaluation functions.
        """

        # Symbolic input variables.
        #
        target_tensor = T.ftensor4('labels')
        learning_rate_scalar = T.fscalar('learning_rate')

        # Define the functions.
        #
        update_func = self._defineupdatefunction(output_layer=output_layer, network_input=input_tensors,
                                                 target=target_tensor, learning_rate=learning_rate_scalar)
        validation_func = self._definevalidationfunction(output_layer=output_layer, network_input=input_tensors,
                                                         target=target_tensor)
        evaluation_func = self._defineevaluationfunction(output_layer=output_layer, network_input=input_tensors)

        # Return constructed functions.
        #
        return update_func, validation_func, evaluation_func

    def _buildnetworkfromdefinition(self):
        """
        Construct the network instance layers.

        Returns:
            lasagne.layers.Layer: Output layer.
        """

        # Symbolic input variable.
        #
        input_tensor = T.ftensor4('patches')

        # Input layer.
        #
        input_shape = self._input_shape
        input_layer_name = 'input'
        network = {input_layer_name: L.InputLayer(shape=input_shape, input_var=input_tensor, name=input_layer_name)}

        # Contraction side of U.
        #
        for depth_index in range(self.__depth):
            self._append_contraction_step(network_instance=network, depth_index=depth_index)

        # Expansion side of U.
        #
        for depth_index in reversed(range(self.__depth - 1)):
            self._append_expansion_step(network_instance=network, depth_index=depth_index)

        # Output layer.
        #
        output_layer_name = 'output'
        network[output_layer_name] = L.Conv2DLayer(incoming=network['expansion_convolution_0_1'],
                                                   name=output_layer_name,
                                                   num_filters=self.__classes,
                                                   filter_size=1,
                                                   pad='valid',
                                                   nonlinearity=None)

        # Return the network object.
        #
        return network[output_layer_name]

    @property
    def inputshape(self):
        """
        Return input shape of unet

        Returns:
            list: input shape
        """
        if not self._compiled:
            raise dmsmodelerrors.ModelNotCompiledError()

        if not self._input_shape:
            self._input_shape = self._model_instance.input_shape

        return self._input_shape

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
            self.__output_shape = self._model_instance.output_shape

        return self.__output_shape