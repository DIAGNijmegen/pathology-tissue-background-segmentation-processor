"""
Nature-net network model.
"""

from .. import modelbase as dmsmodelbase

from ...errors import modelerrors as dmsmodelerrors

import lasagne
import lasagne.layers as L
import theano
import theano.tensor as T
import numpy as np

#----------------------------------------------------------------------------------------------------

class LasagneModelBase(dmsmodelbase.ModelBase):
    """This class is the base class for all Lasagne network model classes."""

    def __init__(self, name=None, description=None):
        """
        Initialize model parameters.

        Args:
            name (str): Name of the model.
            description (str): Description of the model.

        Raises:
            InvalidImageLevel: The configured level is negative.
        """

        # Initialize base class.
        #
        super().__init__(name=name, description=description)

        self._learning_rate = np.array(0.001, dtype="float32")
        self._extra_inputs = {"update" : [], "validate" : [], "predict" : []}
        self._input_shape = None    # Input shape: (channels, rows, cols).
        self._classes = 0           # Number of output classes.
        self._l2_lambda = 0.0       # L2 lambda for loss.
        self.__optimizer_function = None  # Optimizer function
        self._update_function = None      # Update function.
        self._validation_function = None  # Validation function.
        self._evaluation_function = None  # Evaluation function.
        self.__output_offset = None        # Placeholder for the difference between the label patch and the network output shape in pixels.

    def _setextrainputsforupdate(self, input_tensors):
        """

        Args:
            input_tensors: (list of theano.tensor): Extra input tensors
        """

        self._extra_inputs['update'] = input_tensors

    def _setextrainputsforvalidate(self, input_tensors):
        """
-
        Args:
            input_tensors: (list of theano.tensor): Extra input tensors
        """

        self._extra_inputs['validate'] = input_tensors

    def _setextrainputsforpredict(self, input_tensors):
        """

        Args:
            input_tensors: (list of theano.tensor): Extra input tensors
        """

        self._extra_inputs['predict'] = input_tensors


    def updatelearningrate(self, learning_rate):
        """
        Args:
            learning_rate (float): New learning rate.
        """
        self._learning_rate = np.array(learning_rate, dtype="float32")

    def getnumberofoutputchannels(self):
        """
        Returns the output channels of the network
        """

        return self._classes

    def metricnames(self):
        """
        Get the list of metric names that the network returns.

        Returns:
            list: Metric names.
        """

        return ['loss', 'L2 loss', 'accuracy', 'errors', 'predictions']

    def _metrics(self, target, output, l2_loss):
        """
        Get the metrics calculation: loss, accuracy, errors and prediction.

        Args:
            target (theano.tensor.ivector or theano.tensor.fmatrix): Target classes tensor.
            output (theano.tensor.fmatrix): Network output tensor.
            l2_loss (theano.tensor.fscalar): L2 loss.

        Returns:
            theano.tensor.fscalar, theano.tensor.fscalar, theano.tensor.fvector, theano.tensor.itensor4: Loss, accuracy, errors, prediction.
        """

        # Flatten parameters.
        #
        output_flat = output.flatten(ndim=2)

        # Softmax output, original paper may have used a sigmoid but here we opt for softmax, as this also works for multiclass segmentation.
        #
        prediction = lasagne.nonlinearities.softmax(output_flat)

        # Calculate loss.
        #
        epsilon = 1e-8
        loss = lasagne.objectives.categorical_crossentropy(T.clip(prediction, epsilon, 1.0 - epsilon), target)
        loss = loss.mean()
        loss += l2_loss

        # Patch-wise accuracy.
        #
        accuracy = T.mean(T.eq(T.argmax(prediction, axis=1), T.argmax(target, axis=1)), dtype=theano.config.floatX)

        # Patch-wise errors.
        #
        errors = T.max(target - prediction, axis=1)

        # Return loss, accuracy, errors and prediction.
        #
        return loss, accuracy, errors, prediction

    def _defineupdatefunction(self, output_layer, network_input, target, learning_rate):
        """
        Define the update function.

        Args:
            output_layer (theano.layers.Layer): Output layer of the network.
            network_input (list): List of input tensors of the network.
            target (theano.tensor.var.TensorVariable): Classification target tensor.
            learning_rate (theano.tensor.var.TensorVariable): Learning rate scalar.

        Returns:
            function: Network update function.
        """

        # Get the network output and metrics calculation: loss and accuracy.
        #
        network_output = L.get_output(output_layer, deterministic=False)
        l2_loss = self._l2_lambda * lasagne.regularization.regularize_network_params(output_layer, lasagne.regularization.l2)
        loss, accuracy, errors, prediction = self._metrics(target, network_output, l2_loss)

        # Get the network parameters and the update function.
        #
        network_params = L.get_all_params(output_layer, trainable=True)
        self.__optimizer_function = lasagne.updates.adam(loss, network_params, learning_rate=learning_rate)

        # Construct the update function.
        #
        input_parameters = network_input + self._extra_inputs["update"] + [target, learning_rate]
        output_values = [loss, l2_loss, accuracy, errors, prediction]

        return theano.function(inputs=[theano.In(tensor_item) for tensor_item in input_parameters], outputs=[theano.Out(tensor_item) for tensor_item in output_values], updates=self.__optimizer_function, allow_input_downcast=True)

    def _definevalidationfunction(self, output_layer, network_input, target):
        """
        Define the validation function.

        Args:
            output_layer (theano.layers.Layer): Output layer of the network.
            network_input (list): List of input tensors of the network.
            target (theano.tensor.var.TensorVariable): Classification target tensor.

        Returns:
            function: Network validation function.
        """

        # Get the network output and metrics calculation: loss and accuracy.
        #
        network_output = L.get_output(output_layer, deterministic=True)
        l2_loss = self._l2_lambda * lasagne.regularization.regularize_network_params(output_layer, lasagne.regularization.l2)
        loss, accuracy, errors, prediction = self._metrics(target, network_output, l2_loss)

        # Construct the validation function.
        #
        input_parameters = network_input + self._extra_inputs["validate"] + [target]
        output_values = [loss, l2_loss, accuracy, errors, prediction]

        return theano.function(inputs=[theano.In(tensor_item) for tensor_item in input_parameters], outputs=[theano.Out(tensor_item) for tensor_item in output_values], updates=None, allow_input_downcast=True)

    def _defineevaluationfunction(self, output_layer, network_input):
        """
        Define the evaluation function.

        Args:
            output_layer (theano.layers.Layer): Output layer of the network.
            network_input (list): List of input tensors of the network.

        Returns:
            function: Network evaluation function.
        """

        # Get the output from the network and apply softmax.
        #
        input_parameters = network_input + self._extra_inputs["predict"]
        network_output = L.get_output(output_layer, deterministic=True)
        network_output_flat = network_output.flatten(ndim=2)
        network_prediction = lasagne.nonlinearities.softmax(network_output_flat)

        # Return the evaluation function.
        #
        return theano.function(inputs=[theano.In(tensor_item) for tensor_item in input_parameters],
                               outputs=[theano.Out(network_prediction)],
                               allow_input_downcast=True)

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
        target_tensor = T.fmatrix('labels')
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

    @classmethod
    def statisticnames(self):
        """
        Get the names of the statistics returned by the train function.

        Returns:
            list: List of dictionary keys in the ouptut of the train function.
        """

        return ["loss", "accuracy"]

    @property
    def dimensionorder(self):
        """
        Get the dimension order that is always batch-channel-height-width = 'bchw' for Lasagne.

        Returns:
            str: Channel order descriptor.
        """

        return 'bchw'

    def build(self):
        if self._model_instance:
            inputs = [layer.input_var for layer in L.get_all_layers(self._model_instance) if isinstance(layer, L.InputLayer)]
            (self._update_function,
             self._validation_function,
             self._evaluation_function) = self._definefunctions(self._model_instance, inputs)
            self._compiled = True

    def _modelparameters(self):
        if self._model_instance:
            if self.__optimizer_function:
                optimizer_params = [p.get_value() for p in self.__optimizer_function.keys()]
            else:
                optimizer_params = []
            trainable_params = L.get_all_param_values(self._model_instance, trainable = True)
            other_params = L.get_all_param_values(self._model_instance, trainable=False)
            return {"optimizer": optimizer_params, "trainable" : trainable_params, "other" : other_params}
        else:
            return {}

    def _restoremodelparameters(self, parameters):
        # If the model was not configured yet, build the network model and train functions.
        #
        if not self._model_instance:
            # TODO: Remove (or move) explicitly setting the input shape to another part of the code
            if not self._input_shape:
                self._input_shape = (None, 3, None, None)
            # TODO: _buildnetworkfromdefinition should be added as an abstract method to the lasagne modelbase.
            self._model_instance = self._buildnetworkfromdefinition()
            self.build()

        if self._compiled:
            if self.__optimizer_function:
                for p, value in zip(self.__optimizer_function.keys(), parameters["optimizer"]):
                    p.set_value(value)
            L.set_all_param_values(self._model_instance, parameters["trainable"], trainable=True)
            L.set_all_param_values(self._model_instance, parameters["other"], trainable=False)


    def _matchtargettonetwork(self, patches):
        """
        Slice label or weight patches based on the output size of the network. Values on the edges are discarded. Does nothing
        if the output size matches the label size.
        Args:
            patches (np.array): A numpy array containing the labels or weights
        Returns:
            np.array: The sliced labels
        """

        # Only slice multidimensional arrays.
        #
        if len(patches.shape) == 1:
            return patches

        # No need to reshape if the shape of the labels matches the output shape.
        #
        if self._model_instance.output_shape[1:] == patches.shape[1:]:
            return patches

        # Compute the offset based on the difference between input and output.
        #
        if self.__output_offset is None:
            # Returns an array with offset for [left, right, top, bottom].
            #
            self.__output_offset, _ = self.getreconstructioninformation()

        # Slice the labels based on the offset.
        #
        return patches[:, :, self.__output_offset[0]:-self.__output_offset[1], self.__output_offset[2]:-self.__output_offset[3]]

    def update(self, x, y, *args, **kwargs):
        """
        Update the network.

        Args:
            x (numpy.ndarray or list of numpy.ndarray): an array or list of arrays containing training data.
            y (numpy.ndarray or list of numpy.ndarray): an array or list of arrays containing label data.
            *args (list): Ordered arguments.
            **kwargs (dict): Keyword arguments.

        Returns:
            dict: Output of the update function.
        """

        if isinstance(x, np.ndarray):
            x = [x.astype(np.float32)]

        # Crop y if needed
        #
        y = self._matchtargettonetwork(y)

        if isinstance(y, np.ndarray):
            y = [y.astype(np.float32)]

        if 'sample_weight'in kwargs:
            sample_weight = self._matchtargettonetwork(kwargs['sample_weight']).astype(np.float32)
            input_arguments = x + y + [sample_weight] + [self._learning_rate]
        else:
            input_arguments = x + y + [self._learning_rate]

        return_values = self._update_function(*input_arguments)
        return dict(zip(['loss', 'L2 loss', 'accuracy', 'errors', 'predictions'], return_values))

    def validate(self, x, y, *args, **kwargs):
        """
        Validate the network.

        Args:
            x (numpy.ndarray or list of numpy.ndarray): an array or list of arrays containing training data.
            y (numpy.ndarray or list of numpy.ndarray): an array or list of arrays containing label data.
            *args (list): Ordered arguments.
            **kwargs (dict): Keyword arguments.

        Returns:
            dict: Output of the validation function.
        """
        if isinstance(x, np.ndarray):
            x = [x.astype(np.float32)]

        # Crop y if needed
        #
        y = self._matchtargettonetwork(y)

        if isinstance(y, np.ndarray):
            y = [y.astype(np.float32)]

        if 'sample_weight'in kwargs:
            sample_weight = self._matchtargettonetwork(kwargs['sample_weight']).astype(np.float32)
            input_arguments = x + y + [sample_weight]
        else:
            input_arguments = x + y

        return_values = self._validation_function(*input_arguments)
        return dict(zip(["loss", "l2_loss", "acc", "errors", "predictions"], return_values))

    def predict(self, x, *args, **kwargs):
        """
        Use the network for evaluation.

        Args:
            x (numpy.ndarray or list of numpy.ndarray): an array or list of arrays containing training data.
            *args (list): Ordered arguments.
            **kwargs (dict): Keyword arguments.

        Returns:
            dict: Output of the evaluation function.
        """
        if isinstance(x, np.ndarray):
            x = [x]

        return {"predictions": self._evaluation_function(*x)}

    def getreconstructioninformation(self, input_shape=None):
        if self._model_instance is None:
            raise dmsmodelerrors.MissingNetworkError()

        layers = L.get_all_layers(self._model_instance)
        # Check for layer compatiblity
        #
        for layer in layers:
            if not isinstance(layer, (L.InputLayer,
                                      L.Pool2DLayer,
                                      L.Conv2DLayer,
                                      L.NonlinearityLayer,
                                      L.DropoutLayer,
                                      L.BatchNormLayer,
                                      L.merge.ElemwiseSumLayer,
                                      L.merge.ConcatLayer,
                                      L.TransposedConv2DLayer)):
                raise dmsmodelerrors.NotSupportedLayerError(layer)

            if isinstance(layer, L.merge.ConcatLayer) and layer.cropping != [None, None, 'center', 'center']:
                # Only center cropping is supported for now, this means that the smallest input size is equal
                # to the output size of the concat layer.
                #
                raise dmsmodelerrors.NotSupportedPaddingError(str(layer.cropping), str(layer))

        # Retrieve all layers that perform an operation on the  (filter out other ones)
        #
        selected_layers = [l for l in layers if not any(isinstance(l, layer) for layer in [L.InputLayer,
                                                                                           L.special.NonlinearityLayer,
                                                                                           L.noise.DropoutLayer,
                                                                                           L.normalization.BatchNormLayer,
                                                                                           L.merge.ConcatLayer,
                                                                                           L.merge.ElemwiseSumLayer])]
        if input_shape is None:
            input_shape = layers[0].input_shape

        return self._getreconstructioninformationforlayers(input_shape, selected_layers)

    def _getreconstructioninformationforlayers(self, input_shape, layers):

        lost = []
        downsamples = []

        for l in layers:
            lost_this_layer = np.array([0, 0, 0, 0])  # left, right, top, bottom

            next_shape = np.array(l.get_output_shape_for(last_shape))

            # For the transposed convolutions the output size increases
            #
            if isinstance(l, L.TransposedConv2DLayer):
                # Currently only valid padding with a filter_size of 2 and stride 2 is supported
                #
                if l.crop != (0, 0) and l.crop != "valid":
                    raise dmsmodelerrors.NotSupportedPaddingError(str(l.crop), str(l))
                elif l.filter_size != (2, 2):
                    raise dmsmodelerrors.NotSupportedLayerConfigurationError('filter_size', str(l.filter_size), str(l))
                elif l.stride != (2, 2):
                    raise dmsmodelerrors.NotSupportedLayerConfigurationError('stride', str(l.stride), str(l))

                # A valid padding with filter size of 2 and stride of 2 results in a output size
                # that is double the size of the input image. No pixels are lost.
                downsamples.append(np.array([0.5, 0.5]))
            else:
                cur_stride = np.array(l.stride)
                up_shape = (next_shape[-2:] * cur_stride) - cur_stride + 1
                downsamples.append(cur_stride)

                if isinstance(l, L.Conv2DLayer):
                    if l.pad == (0, 0) or l.pad == "valid":
                        lost_this_layer[0] = (l.filter_size[0] - 1) // 2
                        lost_this_layer[2] = (l.filter_size[1] - 1) // 2
                        lost_this_layer[1] = last_shape[-1] - up_shape[-1] - lost_this_layer[0]
                        lost_this_layer[3] = last_shape[-2] - up_shape[-2] - lost_this_layer[2]
                    elif l.pad == "full":
                        lost_this_layer[0] = -((l.filter_size[0] - 1) // 2)
                        lost_this_layer[2] = -((l.filter_size[0] - 1) // 2)
                        lost_this_layer[1] = -(up_shape[-1] - last_shape[-1] + lost_this_layer[0])
                        lost_this_layer[3] = -(up_shape[-2] - last_shape[-2] + lost_this_layer[2])
                    elif l.pad == "same" or l.pad == (((l.filter_size[0] - 1) // 2), ((l.filter_size[1] - 1) // 2)):
                        lost_this_layer[1] = last_shape[-1] - up_shape[-1]
                        lost_this_layer[3] = last_shape[-2] - up_shape[-2]
                    else:
                        raise dmsmodelerrors.NotSupportedPaddingError(str(l.pad), str(l))

                elif isinstance(l, L.Pool2DLayer):
                    lost_this_layer[0] = (l.pool_size[0] - 1) // 2 - l.pad[0]
                    lost_this_layer[2] = (l.pool_size[1] - 1) // 2 - l.pad[1]
                    lost_this_layer[1] = last_shape[-1] - up_shape[-1] - lost_this_layer[0] - l.pad[-1]
                    lost_this_layer[3] = last_shape[-2] - up_shape[-2] - lost_this_layer[2] - l.pad[-2]

            last_shape = next_shape
            lost.append(lost_this_layer)

        # Convert to float for potential upsampling
        #
        downsamples = [np.array([float(x), float(y)]) for x, y in downsamples]
        lost = [x.astype(float) for x in lost]

        for i in range(1, len(downsamples)):
            downsamples[i] *= downsamples[i - 1]
            lost[i][0:2] *= downsamples[i - 1][0]
            lost[i][2:] *= downsamples[i - 1][1]

        # Sum up the lost pixels and convert to normal python int lists
        #
        return np.array(lost).sum(axis=0).astype(int).tolist(), downsamples[-1].astype(int).tolist()