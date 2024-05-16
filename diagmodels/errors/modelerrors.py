"""
Network model related errors.
"""

from . import errorbase as dmserrorbase

#----------------------------------------------------------------------------------------------------

class DiagModelsModelError(dmserrorbase.DiagModelsError):
    """Error base class for all model errors."""

    def __init__(self, *args):
        """
        Initialize the object.

        Args:
            *args: Argument list.
        """

        # Initialize base class.
        #
        super().__init__(*args)

#----------------------------------------------------------------------------------------------------

class InvalidModelDataFormatError(DiagModelsModelError):
    """Raise when the loaded model data is in invalid format."""

    def __init__(self):
        """Initialize the object."""

        # Initialize base class.
        #
        super().__init__('Invalid model data format.')

#----------------------------------------------------------------------------------------------------

class InvalidImageLevel(DiagModelsModelError):
    """Raise when the configured image levels are invalid."""

    def __init__(self, image_levels):
        """
        Initialize the object.

        Args:
            image_levels (list): Image levels.
        """

        # Initialize base class.
        #
        super().__init__('Invalid image levels: {levels}.'.format(levels=image_levels))

        # Store custom data.
        #
        self.image_levels = image_levels

#----------------------------------------------------------------------------------------------------

class MissingNetworkError(DiagModelsModelError):
    """Raise when the network is used before being built."""

    def __init__(self):
        """Initialize the object."""

        # Initialize base class.
        #
        super().__init__('Missing network instance.')

#----------------------------------------------------------------------------------------------------

class MissingModelError(DiagModelsModelError):
    """Raise when the model used before being loaded."""

    def __init__(self):
        """Initialize the object."""

        # Initialize base class.
        #
        super().__init__('Missing network model.')

#----------------------------------------------------------------------------------------------------

class ModelNotCompiledError(DiagModelsModelError):
    """Raise when the model has not been compiled."""

    def __init__(self):
        """Initialize the object."""

        # Initialize base class.
        #
        super().__init__('Model is not compiled before use.')

#----------------------------------------------------------------------------------------------------

class MissingFunctionError(DiagModelsModelError):
    """Raise when any of the network functions are not defined but called."""

    def __init__(self, function_purpose):
        """
        Initialize the object.

        Args:
            function_purpose (str): Function purpose.
        """

        # Initialize base class.
        #
        super().__init__('Network function not defined: {purpose}'.format(purpose=function_purpose))

        # Store custom data.
        #
        self.function_purpose = function_purpose

#----------------------------------------------------------------------------------------------------

class ModelClassError(DiagModelsModelError):
    """Raise when loaded model does not match the type of the object."""

    def __init__(self, self_type, class_list):
        """
        Initialize the object.

        Args:
            self_type (str): Type descriptor of the object.
            class_list (str): Inheritance list of the creator class.
        """

        # Initialize base class.
        #
        super().__init__('Loading model to {this} does not match {other}.'.format(this=self_type, other=class_list))

        # Store custom data.
        #
        self.self_type = self_type
        self.class_list = class_list

#----------------------------------------------------------------------------------------------------

class MissingLearnableParameterError(DiagModelsModelError):
    """Raise when a learnable parameter is inaccessible in a layer object."""

    def __init__(self, class_name, param_name):
        """
        Initialize the object.

        Args:
            class_name (str): Name of the layer class.
            param_name (str): Name of the missing parameter.
        """

        # Initialize base class.
        #
        super().__init__('Learnable parameter {name} is missing from {id}.'.format(name=param_name, id=class_name))

        # Store custom data.
        #
        self.class_name = class_name
        self.param_name = param_name

#----------------------------------------------------------------------------------------------------

class MissingInitParameterError(DiagModelsModelError):
    """Raise when a parameter that is present in the __init__ cannot be collected from the object."""

    def __init__(self, class_name, param_name):
        """
        Initialize the object.

        Args:
            class_name (str): Name of the layer class.
            param_name (str): Name of the missing parameter.
        """

        # Initialize base class.
        #
        super().__init__('Parameter {name} is missing from {id}.'.format(name=param_name, id=class_name))

        # Store custom data.
        #
        self.class_name = class_name
        self.param_name = param_name

#----------------------------------------------------------------------------------------------------

class InputDimensionsMismatchError(DiagModelsModelError):
    """Raise when the number of the dimensions of the input layer does not match the unbind directive."""

    def __init__(self, layer_name, tensor_name, shape, unbind):
        """
        Initialize the object.

        Args:
            layer_name (str): Name of the layer.
            tensor_name (str): Name of the tensor.
            shape (tuple): Input shape.
            unbind (tuple): Unbind directives.
        """

        # Initialize base class.
        #
        super().__init__('Unbind directive {unbind} does not match the shape {shape} of the input layer {layer} with the {tensor} input tensor.'.format(unbind=unbind,
                                                                                                                                                        shape=shape,
                                                                                                                                                        layer=layer_name,
                                                                                                                                                        tensor=tensor_name))

        # Store custom data.
        #
        self.layer_name = layer_name
        self.tensor_name = tensor_name
        self.shape = shape
        self.unbind = unbind

#----------------------------------------------------------------------------------------------------

class TensorShapeMismatchError(DiagModelsModelError):
    """Raise when a tensor is found with different expected sizes."""

    def __init__(self, tensor_name, input_shape_0, input_shape_1):
        """
        Initialize the object.

        Args:
            tensor_name (str): Name of the tensor.
            input_shape_0 (tuple): Model input shape 0.
            input_shape_1 (tuple): Model input shape 1.
        """

        # Initialize base class.
        #
        super().__init__('Tensor {tensor} has ambiguous expected size: {shape0}, {shape1}'.format(tensor=tensor_name, shape0=input_shape_0, shape1=input_shape_1))

        # Store custom data.
        #
        self.tensor_name = tensor_name
        self.input_shape_0 = input_shape_0
        self.input_shape_1 = input_shape_1

#----------------------------------------------------------------------------------------------------

class InvalidInputShapeError(DiagModelsModelError):
    """Raise when the type model input shape is invalid."""

    def __init__(self, input_shape):
        """
        Initialize the object.

        Args:
            input_shape (tuple): Model input shape.
        """

        # Initialize base class.
        #
        super().__init__('Invalid model input shape: {shape}'.format(shape=input_shape))

        # Store custom data.
        #
        self.input_shape = input_shape

#----------------------------------------------------------------------------------------------------

class InvalidModelDepthError(DiagModelsModelError):
    """Raise when the depth of the U-Net network is invalid."""

    def __init__(self, depth):
        """
        Initialize the object.

        Args:
            depth (int): U-Net model depth.
        """

        # Initialize base class.
        #
        super().__init__('Invalid U-Net model depth: {depth}.'.format(depth=depth))

        # Store custom data.
        #
        self.depth = depth

#----------------------------------------------------------------------------------------------------

class InvalidModelClassCountError(DiagModelsModelError):
    """Raise when the number of output classes are invalid."""

    def __init__(self, classes):
        """
        Initialize the object.

        Args:
            classes (int): Number of output classes in the model.
        """

        # Initialize base class.
        #
        super().__init__('Invalid number of output classes: {count}.'.format(count=classes))

        # Store custom data.
        #
        self.classes = classes

#----------------------------------------------------------------------------------------------------

class InvalidBranchingFactorError(DiagModelsModelError):
    """Raise when the filter count branching factor is invalid."""

    def __init__(self, branching_factor):
        """
        Initialize the object.

        Args:
            branching_factor (int): Filter count branching factor.
        """

        # Initialize base class.
        #
        super().__init__('Invalid filter count branching factor: {factor}.'.format(factor=branching_factor))

        # Store custom data.
        #
        self.branching_factor = branching_factor

#----------------------------------------------------------------------------------------------------

class InvalidDropoutLayerCountError(DiagModelsModelError):
    """Raise when the number of dropout layers is invalid."""

    def __init__(self, dropout_count):
        """
        Initialize the object.

        Args:
            dropout_count (int): Number of dropout layers.
        """

        # Initialize base class.
        #
        super().__init__('Invalid dropout layer count: {count}.'.format(count=dropout_count))

        # Store custom data.
        #
        self.dropout_count = dropout_count

#----------------------------------------------------------------------------------------------------

class InvalidDropoutProbabilityError(DiagModelsModelError):
    """Raise when the number dropout probability is out of (0.0, 1.0) bounds."""

    def __init__(self, dropout_prob):
        """
        Initialize the object.

        Args:
            dropout_prob (float): Dropout probability.
        """

        # Initialize base class.
        #
        super().__init__('Dropout probability is out of (0.0, 1.0) bounds: {prob}.'.format(prob=dropout_prob))

        # Store custom data.
        #
        self.dropout_prob = dropout_prob

#----------------------------------------------------------------------------------------------------

class InvalidZoomFactorError(DiagModelsModelError):
    """Raise when the zoom factor of the InterpolationModel is invalid."""

    def __init__(self, zoom):
        """
        Initialize the object.

        Args:
            zoom (float): Zoom factor.
        """

        # Initialize base class.
        #
        super().__init__('Invalid zoom factor: {factor}.'.format(factor=zoom))

        # Store custom data.
        #
        self.zoom = zoom

#----------------------------------------------------------------------------------------------------

class InvalidInputRangeError(DiagModelsModelError):
    """Raise when the input value range is invalid."""

    def __init__(self, input_range):
        """
        Initialize the object.

        Args:
            input_range (tuple): Input value range.
        """

        # Initialize base class.
        #
        super().__init__('Invalid input value range: {range}.'.format(range=input_range))

        # Store custom data.
        #
        self.input_range = input_range

#----------------------------------------------------------------------------------------------------

class MissingDataKeyError(DiagModelsModelError):
    """Raise when the 'data' key is missing from the keyword arguments of the evaluate, update or validate functions."""

    def __init__(self, keys):
        """
        Initialize the object.

        Args:
            keys (list): Available keys.
        """

        # Initialize base class.
        #
        super().__init__('Missing \'data\' key from the keyword arguments: {keywords}.'.format(keywords=keys))

        # Store custom data.
        #
        self.keys = keys

#----------------------------------------------------------------------------------------------------

class DataLevelMismatchError(DiagModelsModelError):
    """Raise when the received levels of the data does not match the expected levels."""

    def __init__(self, expected_levels, available_levels):
        """
        Initialize the object.

        Args:
            expected_levels (list): Expected data levels.
            available_levels (list): Available data levels.
        """

        # Initialize base class.
        #
        super().__init__('The available {available} levels do not match the expected {expected} levels.'.format(available=available_levels, expected=expected_levels))

        # Store custom data.
        #
        self.expected_levels = expected_levels
        self.available_levels = available_levels

#----------------------------------------------------------------------------------------------------

class NotSupportedLayerError(DiagModelsModelError):
    """Raise when a layer is not supported by a function."""

    def __init__(self, layer):
        """
        Initialize the object.

        Args:
            layer: The unsupported layer.
        """

        # Initialize base class.
        #
        super().__init__('The layer {layer} is not supported in this function.'.format(layer=layer))

        # Store custom data.
        #
        self.layer = layer

#----------------------------------------------------------------------------------------------------

class NotSupportedPaddingError(DiagModelsModelError):
    """Raise when padding setting is not supported by a function."""

    def __init__(self, padding, layer):
        """
        Initialize the object.

        Args:
            padding (str): The not supported padding.
            layer (str): The layer.
        """

        # Initialize base class.
        #
        super().__init__('Padding {padding} from layer {layer} is not supported in this function.'.format(padding=padding, layer=layer))

        # Store custom data.
        #
        self.padding = padding
        self.layer = layer

#----------------------------------------------------------------------------------------------------

class InvalidReconstructionInformationError(DiagModelsModelError):
    """???"""

    def __init__(self, layer, in_shape, out_shape, recon_shape, lost_this_layer, stride):
        """
        Initialize the object.

        Args:
            layer (???): ???
            in_shape (???): ???
            out_shape (???): ???
            recon_shape (???): ???
            lost_this_layer (???): ???
            stride (???): ???
        """

        # Initialize base class.
        #
        message = 'Reconstruction resulted in an invalid image shape of {recon_sh} from layer {layer} with input shape {in_shape}, output shape {out_shape}, stride {stride}, and {lost} pixels lost .'
        super().__init__(message.format(recon_sh=recon_shape,
                                        layer=layer,
                                        in_shape=in_shape,
                                        out_shape=out_shape,
                                        stride=stride,
                                        lost=lost_this_layer))

        # Store custom data.
        #
        self.layer = layer
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.recon_shape = recon_shape
        self.lost_this_layer = lost_this_layer
        self.stride = stride

#----------------------------------------------------------------------------------------------------

class NotSupportedLayerConfigurationError(DiagModelsModelError):
    """Raise when a layer setting is not supported by a function."""

    def __init__(self, setting, value, layer):
        """
        Initialize the object.

        Args:
            setting (str): The name of the setting.
            value (str): The not support value.
            layer (str): The layer.
        """

        # Initialize base class.
        #
        super().__init__('Setting {setting} with value {value} from layer {layer} is not supported in this function.'.format(setting=setting, value=value, layer=layer))

        # Store custom data.
        #
        self.setting = setting
        self.value = value
        self.layer = layer

#----------------------------------------------------------------------------------------------------

class NotSupportedChannelOrder(DiagModelsModelError):
    """Raise when configured channel order is not supported by the model."""

    def __init__(self, channel_order):
        """
        Initialize the object.

        Args:
            channel_order (str): Channel order description.
        """

        # Initialize base class.
        #
        super().__init__('The \'{order}\' channel order is not supported with this model.'.format(order=channel_order))

        # Store custom data.
        #
        self.channel_order = channel_order

#----------------------------------------------------------------------------------------------------

class NotSupportedLoss(DiagModelsModelError):
    """Raise when model does not support the configured loss."""

    def __init__(self, loss):
        """
        Initialize the object.

        Args:
            loss (str): Loss.
        """

        # Initialize base class.
        #
        super().__init__('The \'{name}\' is not supported with this model.'.format(name=loss))

        # Store custom data.
        #
        self.loss = loss

#----------------------------------------------------------------------------------------------------

class InvalidNumberOfClassesForLoss(DiagModelsModelError):
    """Raise when the configured loss requires another amount of classes than defined."""

    def __init__(self, loss, classes):
        """
        Initialize the object.

        Args:
            classes (int): str.
        """

        # Initialize base class.
        #
        super().__init__('The \'{loss}\' loss with {classes} number of classes requires different activation function and is not implemented yet.'.format(loss=loss, classes=classes))

        # Store custom data.
        #
        self.classes = classes
