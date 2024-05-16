"""
Naturenet implementation using Keras.
"""

from . import pytorchmodelbase as dmspytorchmodelbase
from ...errors import modelerrors as dmsmodelerrors

import torch
from .deeplab.deeplab import DeepLab

#----------------------------------------------------------------------------------------------------

class DeepLabNet(dmspytorchmodelbase.PytorchModelBase):
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
        self.__backbone = 'resnet'
        self.__l2_lambda = 0.0        # L2 lambda for loss.
        self.__output_stride = 16
        self._channels_first = True   # If true, the network expect patches shaped BCHW, otherwise BHWC.

    def configure(self, input_shape, classes, backbone='resnet', output_stride=16, l2_lambda=1e-5):
        """
        Save the network configuration parameters.

        Args:
            input_shape (tuple): Number of channels and size of the individual input images. (rows, cols, channels)
            classes (int): Number of output classes.
            branching_factor (int): Branching factor. The number of filters on the first layer is 2 on the power of branching factor.
            backbone (string): 'resnet' | 'xception'

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

        # Save parameters.
        #
        self.__input_shape = input_shape
        self.__classes = classes

    def _customdata(self):
        """
        Get a map of custom data structure to add to the dictionary of exported values.

        Returns:
            dict: Dictionary of values to add to the exported data.
        """

        # Collect custom data.
        #
        return {'backbone': self.__backbone, 'num_classes': self.__num_classes}

    def _setcustomdata(self, data_maps):
        """
        Set the class specific data.

        Args:
            data_maps (dict): Custom data map.
        """

        # Configure custom data.
        self.__backbone = data_maps.get('backbone', 'resnet')

    def _networkdefinition(self):
        """
        Network definition.

        Returns
            Tensor: The network architecture.
        """
        return DeepLab(self.__backbone, self.__output_stride, self.__classes, sync_bn=False)

    def _restoremodelparameters(self, parameters):
        """
        Custom restore function. Makes sure that the output tensors are correctly set, even after a load.

        Args:
            parameters (dict): Parameters of the model.
        """

        super()._restoremodelparameters(parameters)

    def build(self):
        """Build the network instance with the pre-configured parameters."""

        if not self._compiled:
            self._model_instance = self._networkdefinition()
            self._model_instance.cuda()
            self._optimizer = torch.optim.Adam(self._model_instance.parameters(), lr=0.0001)

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

        return output

    def getreconstructioninformation(self, input_shape=None):
        return (0, 0, 0, 0), (1, 1), (0, 0, 0, 0)
