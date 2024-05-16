"""
Base class for Keras network models.
"""

from .. import modelbase as dmsmodelbase
from ...errors import modelerrors as dmsmodelerrors
from ...utils import network as dmsnetwork

import keras
import keras.backend as K

import numpy as np

#----------------------------------------------------------------------------------------------------

class KerasModelBase(dmsmodelbase.ModelBase):
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
        self.__custom_metrics = set()  # Custom metric names.

    def updatelearningrate(self, learning_rate):
        """
        Update the learning rate.

        Args:
            learning_rate (float): New learning rate.
        """

        # Use the Keras backend to update the learning rate internally
        if self._model_instance:
            K.set_value(self._model_instance.optimizer.lr, learning_rate)

    def _addcustommetric(self, name):
        """
        Add a custom metric name to the list of available metrics.

        Args:
            name (str): Name of the metric.
        """

        self.__custom_metrics.add(name)

    def metricnames(self):
        """
        Get the list of metric names that the network returns.

        Returns:
            list: Metric names.
        """

        return ['accuracy' if name == 'acc' or name == 'weighted_acc' else name for name in self._model_instance.metrics_names] + list(self.__custom_metrics) if self._model_instance else []

    def _modelparameters(self):
        """
        Collect the parameters of the network to save.

        Returns:
            dict: Dictionary of parameters.
        """

        if self._model_instance:
            # Create a purely in-memory HF5 file for saving the model and save the model to the memory file.
            #
            model_hf5 = dmsnetwork.hf5_memory_file(name='kerasmodelbase.hf5', mode='w')
            self._model_instance.save(filepath=model_hf5, overwrite=True, include_optimizer=True)

            # Get the file content into a byte array and return it.
            #
            byte_stream = dmsnetwork.hf5_memory_file_image(hf5_file=model_hf5)
            model_hf5.close()

            return {'hf5_image': byte_stream}

    def _restoremodelparameters(self, parameters):
        """
        Restores the state of the model and the optimizer

        Args:
            parameters: Dictionary of parameters
        """

        # Load the stored byte stream to HF5 file.
        #
        hf5_file_name = '{module_name}.{class_name}.{function_name}.hf5'.format(module_name=self.__class__.__module__, class_name=self.__class__.__name__, function_name=self._modelparameters.__name__)
        hf5_file = dmsnetwork.load_bytes_to_hf5_memory_file(byte_stream=parameters['hf5_image'], name=hf5_file_name)

        # Set channels last for models that don't support channels first.
        #
        if hasattr(self, '_channels_first'):
            self.setchannelsorder(self._channels_first)

        # Load the stored model settings from the HF5 file.
        #
        self._model_instance = keras.models.load_model(filepath=hf5_file)
        self._compiled = True

        hf5_file.close()

    def setchannelsorder(self, channel_first):
        """
        Set the channels first dimension order for the network.

        Note! This is a glob setting, meaning if two instances are initiated simultaneous they should have the same data format.

        Args:
            channel_first (bool): Channels should be the first after the batch dimension.
        """

        K.set_image_data_format('channels_first' if channel_first else 'channels_last')

    def _customdata(self):
        """
        Add data format ('channels first' or 'channels last') to model.

        Returns:
            dict: Dictionary of values to add to the exported data.
        """

        # By default it returns an empty map.
        #
        return {'channel_order': K.image_data_format()}

    @property
    def dimensionorder(self):
        """
        Get the dimension order.

        Returns:
            str: Channel order descriptor.
        """

        return 'bchw' if K.image_data_format() == 'channels_first' else 'bhwc'

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

        output_values = self._model_instance.train_on_batch(x, y,
                                                            sample_weight=sample_weight,
                                                            class_weight=class_weight)

        output = dict(zip(self._model_instance.metrics_names, output_values))

        # Get accuracy from model output, favor weighted accuracy over the non-weighted variant.
        #
        output['accuracy'] = output.pop('weighted_acc') if 'weighted_acc' in output.keys() else output.pop('acc')

        return output

    def validate(self, x, y, sample_weight=None, *args, **kwargs):
        """
        Validate the network.

        Args:
            x (numpy.ndarray or list of numpy.ndarray): contains image data.
            y (numpy.ndarray or list of numpy.ndarray ): contains label data.
            sample_weight (numpy.ndarray): an array containing weights for samples to increase their contribution to the loss

        Returns:
            dict: Output of the validation function.
        """

        output_values = self._model_instance.evaluate(x, y, sample_weight=sample_weight, verbose=0)

        output = dict(zip(self._model_instance.metrics_names, output_values))

        # Get accuracy from model output, favor weighted accuracy over the non-weighted variant.
        #
        output['accuracy'] = output.pop('weighted_acc') if 'weighted_acc' in output.keys() else output.pop('acc')

        return output

    def predict(self, x, *args, **kwargs):
        """
        Use the network for evaluation.

        Args:
            x (numpy.ndarray or list of numpy.ndarray): contains image data.

        Returns:
            dict: Output of the evaluation function.
        """

        # Run the model on the set of patches.
        #
        return {"predictions": self._model_instance.predict_on_batch(x)}

    def getreconstructioninformation(self, input_shape=None):
        """
        Calculate the scale factor and padding to reconstruct the input shape.

        This function calculates the information needed to reconstruct the input image shape given the output of a layer. For each layer leading up to the
        output it will return the number of pixels lost/gained on all image edges.

        For transposed convolutions the function checks the stride and cropping method to compute the correct upsampling factor.

        Args:
            input_shape (sequence of ints): input_shape to calculate the reconstruction information for. Order should be (nr_channels, width, height)
            selected_layers (sequence of keras.layers): only calculate the reconstruction information for this specific
                set of layers.

        Returns:
            np.array: lost pixels
            np.array: downsample factor

        Raises:
            MissingNetworkError: The network is not defined.
        """

        if self._model_instance is None:
            raise dmsmodelerrors.MissingNetworkError()

        for layer in self._model_instance.layers:
            if not isinstance(layer, (keras.layers.InputLayer,
                                      keras.layers.MaxPool2D,
                                      keras.layers.Conv2D,
                                      keras.layers.Dropout,
                                      keras.layers.BatchNormalization,
                                      keras.layers.Concatenate,
                                      keras.layers.UpSampling2D,
                                      keras.layers.Conv2DTranspose,
                                      keras.layers.Cropping2D,
                                      keras.layers.Reshape,
                                      keras.layers.Activation,
                                      keras.layers.ReLU,
                                      keras.layers.LeakyReLU,
                                      keras.layers.PReLU,
                                      keras.layers.ELU,
                                      keras.layers.ThresholdedReLU,
                                      keras.layers.Softmax,
                                      keras.layers.AvgPool2D,
                                      keras.layers.GlobalAvgPool2D,
                                      keras.layers.GlobalMaxPool2D)):
                raise dmsmodelerrors.NotSupportedLayerError(layer)

        # Retrieve all layers that perform an operation on the image (filter out other ones)
        #
        selected_layers = [l for l in self._model_instance.layers if not any(isinstance(l, layer) for layer in [keras.layers.InputLayer,
                                                                                                                keras.layers.Dropout,
                                                                                                                keras.layers.Cropping2D,
                                                                                                                keras.layers.ReLU,
                                                                                                                keras.layers.LeakyReLU,
                                                                                                                keras.layers.PReLU,
                                                                                                                keras.layers.ELU,
                                                                                                                keras.layers.ThresholdedReLU,
                                                                                                                keras.layers.Softmax,
                                                                                                                keras.layers.BatchNormalization,
                                                                                                                keras.layers.Concatenate,
                                                                                                                keras.layers.Reshape,
                                                                                                                keras.layers.Activation])]
        if input_shape is None:
            input_shape = selected_layers[0].input_shape
        if len(input_shape) == 3:
            input_shape = (1,) + tuple(input_shape)
        return self._getreconstructioninformationforlayers(input_shape, selected_layers)

    def getnumberofoutputchannels(self):
        """
        Returns the output channels of the network
        """
        if self._model_instance:
            return int(self._model_instance.output.shape[-1])
        else:
            raise dmsmodelerrors.MissingNetworkError()

    def _getreconstructioninformationforlayers(self, input_shape, layers):

        lost = []
        downsamples = []
        last_shape = np.array([input_shape[1], input_shape[2]])
        import keras.layers
        for l in layers:
            lost_this_layer = np.array([0, 0, 0, 0], dtype=np.float32)  # left, right, top, bottom

            # For the transposed convolutions the output size increases
            #
            if isinstance(l, keras.layers.Conv2DTranspose):
                # Currently only valid padding with a filter_size of 2 and stride 2 is supported
                #
                # if l.padding != "valid":
                #    raise dmsmodelerrors.NotSupportedPaddingError(l.padding, str(l))
                if l.kernel_size != (2, 2):
                    raise dmsmodelerrors.NotSupportedLayerConfigurationError('filter_size', str(l.kernel_size), str(l))
                elif l.strides != (2, 2):
                    raise dmsmodelerrors.NotSupportedLayerConfigurationError('stride', str(l.strides), str(l))

                # A valid padding with filter size of 2 and stride of 2 results in a output size
                # that is double the size of the input image. No pixels are lost.
                next_shape = last_shape * 2.0
                downsamples.append([0.5, 0.5])
            elif isinstance(l, keras.layers.UpSampling2D):
                next_shape = last_shape * 2.0
                downsamples.append([0.5, 0.5])
            elif isinstance(l, keras.layers.GlobalAvgPool2D) or isinstance(l, keras.layers.GlobalMaxPool2D):

                next_shape = np.asarray([1, 1])
                downsamples.append([l.kernel_size, l.kernel_size])
            else:
                cur_stride = np.array(l.strides)

                if isinstance(l, keras.layers.Conv2D) or isinstance(l, keras.layers.MaxPool2D) or isinstance(l, keras.layers.AvgPool2D):
                    if isinstance(l, keras.layers.Conv2D):
                        kernel_size = l.kernel_size
                    else:
                        kernel_size = l.pool_size
                    if l.padding == "same":
                        next_shape = np.ceil(last_shape / cur_stride)
                    elif l.padding == "valid":
                        next_shape = np.floor((last_shape - kernel_size) / cur_stride) + 1
                    cutoff = (last_shape - kernel_size) % cur_stride
                    if l.padding == "valid":
                        lost_this_layer[0] = (kernel_size[1] - cur_stride[1]) / 2
                        lost_this_layer[2] = (kernel_size[0] - cur_stride[0]) / 2
                        lost_this_layer[1] = (kernel_size[1] - cur_stride[1]) / 2 + cutoff[0]
                        lost_this_layer[3] = (kernel_size[0] - cur_stride[0]) / 2 + cutoff[1]
                    elif l.padding == "same":
                        lost_this_layer[1] = cutoff[0]
                        lost_this_layer[3] = cutoff[1]
                    else:
                        raise dmsmodelerrors.NotSupportedPaddingError(str(l.padding), str(l))
                downsamples.append(cur_stride)

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
        lost_total = np.array(lost).sum(axis=0).astype(np.float32).tolist()
        lost_total[0::2] = np.floor(lost_total[0::2])
        lost_total[1::2] = np.ceil(lost_total[1::2])

        interpolation_lost = np.ceil((downsamples[-1] - 1) / 2)
        interpolation_lost = np.array([interpolation_lost[0], interpolation_lost[0], interpolation_lost[1], interpolation_lost[1]])

        return np.array(lost_total, dtype=np.int32), downsamples[-1], interpolation_lost

    def unfix(self):
        """Unfix input shape."""

        if self._model_instance:
            # If any of the changes has to be made the whole layer topology has to be rebuild.
            #
            old_input_shape = self._model_instance.layers[0].input_shape

            if any(size is not None for size in old_input_shape[:-1]) or isinstance(self._model_instance.layers[-1], keras.layers.Reshape):
                # Calculate the new input shape and construct a new input layer: the shape of the input layer also contains the batch size.
                #
                new_input_shape = (None,) * (len(old_input_shape) - 2) + (old_input_shape[-1],)
                new_input_layer = keras.layers.Input(shape=new_input_shape)
                stop_layer_index = len(self._model_instance.layers) - 1 if isinstance(self._model_instance.layers[-1], keras.layers.Reshape) else len(self._model_instance.layers)

                # Rebuild the the layer topology.
                #
                last_new_layer = new_input_layer
                for index in range(1, stop_layer_index):
                    # Deep copy layers.
                    #
                    layer_config = self._model_instance.layers[index].get_config()

                    # Layers with shape information are not compatible (e.g. Reshape layers).
                    #
                    if 'target_shape' in layer_config:
                        raise dmsmodelerrors.NotSupportedLayerError(self._model_instance.layers[index])

                    new_layer = keras.layers.deserialize(config={'class_name': self._model_instance.layers[index].__class__.__name__, 'config': layer_config})
                    last_new_layer = new_layer(last_new_layer)

                # Build new model.
                #
                new_model = keras.models.Model(inputs=new_input_layer, outputs=last_new_layer)

                # Sync parameters.
                #
                for index in range(1, len(new_model.layers)):
                    new_model.layers[index].set_weights(weights=self._model_instance.layers[index].get_weights())

                # Remove the old model instance and set the unfixed one.
                #
                self._model_instance = new_model
