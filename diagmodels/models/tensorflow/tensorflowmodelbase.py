"""
Base class for TensorFlow network models.
"""

from .. import modelbase as dmsmodelbase
from ...errors import modelerrors as dmsmodelerrors
from ...utils import network as dmsnetwork

import tensorflow as tf

import numpy as np

#----------------------------------------------------------------------------------------------------

class TensorFlowModelBase(dmsmodelbase.ModelBase):
    """This class is the base class for all network model classes implemented using TensorFlow."""

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
            tf.keras.backend.set_value(self._model_instance.optimizer.lr, learning_rate)

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
        self._model_instance = tf.keras.models.load_model(filepath=hf5_file)
        self._compiled = True

        hf5_file.close()

    def setchannelsorder(self, channel_first):
        """
        Set the channels first dimension order for the network.

        Note! This is a glob setting, meaning if two instances are initiated simultaneous they should have the same data format.

        Args:
            channel_first (bool): Channels should be the first after the batch dimension.
        """

        tf.keras.backend.set_image_data_format('channels_first' if channel_first else 'channels_last')

    def _customdata(self):
        """
        Add data format ('channels first' or 'channels last') to model.

        Returns:
            dict: Dictionary of values to add to the exported data.
        """

        # By default it returns an empty map.
        #
        return {'channel_order': tf.keras.backend.image_data_format()}

    @property
    def dimensionorder(self):
        """
        Get the dimension order.

        Returns:
            str: Channel order descriptor.
        """

        return 'bchw' if tf.keras.backend.image_data_format() == 'channels_first' else 'bhwc'

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

        Returns:
            np.array: lost pixels
            np.array: downsample factor

        Raises:
            MissingNetworkError: The network is not defined.
        """

        if self._model_instance is None:
            raise dmsmodelerrors.MissingNetworkError()

        for layer in self._model_instance.layers:
            if not isinstance(layer, (tf.keras.layers.InputLayer,
                                      tf.keras.layers.MaxPool2D,
                                      tf.keras.layers.Conv2D,
                                      tf.keras.layers.Dropout,
                                      tf.keras.layers.BatchNormalization,
                                      tf.keras.layers.Concatenate,
                                      tf.keras.layers.UpSampling2D,
                                      tf.keras.layers.Conv2DTranspose,
                                      tf.keras.layers.Cropping2D,
                                      tf.keras.layers.Reshape,
                                      tf.keras.layers.Activation,
                                      tf.keras.layers.ReLU,
                                      tf.keras.layers.LeakyReLU,
                                      tf.keras.layers.PReLU,
                                      tf.keras.layers.ELU,
                                      tf.keras.layers.ThresholdedReLU,
                                      tf.keras.layers.Softmax,
                                      tf.keras.layers.AvgPool2D,
                                      tf.keras.layers.GlobalAvgPool2D,
                                      tf.keras.layers.GlobalMaxPool2D)):
                raise dmsmodelerrors.NotSupportedLayerError(layer)

        # Retrieve all layers that perform an operation on the image (filter out other ones)
        #
        selected_layers = [l for l in self._model_instance.layers if not any(isinstance(l, layer) for layer in [tf.keras.layers.InputLayer,
                                                                                                                tf.keras.layers.Dropout,
                                                                                                                tf.keras.layers.Cropping2D,
                                                                                                                tf.keras.layers.ReLU,
                                                                                                                tf.keras.layers.LeakyReLU,
                                                                                                                tf.keras.layers.PReLU,
                                                                                                                tf.keras.layers.ELU,
                                                                                                                tf.keras.layers.ThresholdedReLU,
                                                                                                                tf.keras.layers.Softmax,
                                                                                                                tf.keras.layers.BatchNormalization,
                                                                                                                tf.keras.layers.Concatenate,
                                                                                                                tf.keras.layers.Reshape,
                                                                                                                tf.keras.layers.Activation])]
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

        for l in layers:
            lost_this_layer = np.array([0, 0, 0, 0], dtype=np.float32)  # left, right, top, bottom
            computed_shape = l.compute_output_shape(input_shape=(1, last_shape[0], last_shape[1], 1))
            next_shape = np.array([int(computed_shape[1]), int(computed_shape[2])])

            # For the transposed convolutions the output size increases
            #
            if isinstance(l, tf.keras.layers.Conv2DTranspose):
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
                downsamples.append([0.5, 0.5])
            elif isinstance(l, tf.keras.layers.UpSampling2D):
                downsamples.append([0.5, 0.5])
            elif isinstance(l, tf.keras.layers.GlobalAvgPool2D) or isinstance(l, tf.keras.layers.GlobalMaxPool2D):
                downsamples.append([l.kernel_size, l.kernel_size])
            else:
                cur_stride = np.array(l.strides)

                if isinstance(l, tf.keras.layers.Conv2D) or isinstance(l, tf.keras.layers.MaxPool2D) or isinstance(l, tf.keras.layers.AvgPool2D):
                    if isinstance(l, tf.keras.layers.Conv2D):
                        kernel_size = l.kernel_size
                    else:
                        kernel_size = l.pool_size

                    # cutoff = last_shape - next_shape * cur_stride

                    if l.padding == "valid":
                        cutoff = (last_shape - kernel_size) % cur_stride
                        lost_this_layer[0] = (kernel_size[1] - cur_stride[1]) / 2
                        lost_this_layer[2] = (kernel_size[0] - cur_stride[0]) / 2
                        lost_this_layer[1] = (kernel_size[1] - cur_stride[1]) / 2 + cutoff[0]
                        lost_this_layer[3] = (kernel_size[0] - cur_stride[0]) / 2 + cutoff[1]
                    elif l.padding == "same":
                        lost_this_layer[1] = 0
                        lost_this_layer[3] = 0
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

            if any(size is not None for size in old_input_shape[:-1]) or isinstance(self._model_instance.layers[-1], tf.keras.layers.Reshape):
                # Configure new input shape.
                #
                new_input_shape = (None,) + (None,) * (len(old_input_shape) - 2) + (old_input_shape[-1],)
                model_config = self._model_instance.get_config()
                model_config['layers'][0]['config']['batch_input_shape'] = new_input_shape

                # Remove the reshape layer from the bottom.
                #
                bottom_reshape = model_config['layers'].pop()
                for output_item in model_config['output_layers']:
                    if output_item[0] == bottom_reshape['name']:
                        output_item[0] = model_config['layers'][-1]['name']

                new_model = tf.keras.Model.from_config(config=model_config)
                new_model.set_weights(weights=self._model_instance.get_weights())

                # Remove the old model instance and set the unfixed one.
                #
                self._model_instance = new_model
