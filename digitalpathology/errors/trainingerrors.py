"""
Network training related errors.
"""

from . import errorbase as dpterrorbase

#----------------------------------------------------------------------------------------------------

class DigitalPathologyTrainingError(dpterrorbase.DigitalPathologyError):
    """Error base class for all training errors."""

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

class InvalidIterationLogPercentError(DigitalPathologyTrainingError):
    """Raise when the iteration log percent is out of (0.0, 1.0] bounds."""

    def __init__(self, iter_log_percent):
        """
        Initialize the object.

        Args:
            iter_log_percent (float): An iteration log entry is made before every iter_log_percent chunk of iterations.
        """

        # Initialize base class.
        #
        super().__init__('Iteration log percent {percent} is out of (0.0, 1.0] bounds.'.format(percent=iter_log_percent))

        # Store custom data.
        #
        self.iter_log_percent = iter_log_percent

#----------------------------------------------------------------------------------------------------

class InvalidNetworkObjectError(DigitalPathologyTrainingError):
    """Raise when the network object is invalid."""

    def __init__(self):
        """Initialize the object."""

        # Initialize base class.
        #
        super().__init__('Invalid network object.')

#----------------------------------------------------------------------------------------------------

class InvalidBatchGeneratorError(DigitalPathologyTrainingError):
    """Raise when the batch generator is not valid."""

    def __init__(self, generator_purpose):
        """
        Initialize the object.

        Args:
            generator_purpose (str): Batch generator purpose.
        """

        # Initialize base class.
        #
        super().__init__('Invalid {purpose} batch generator.'.format(purpose=generator_purpose))

        # Store custom data.
        #
        self.generator_purpose = generator_purpose

#----------------------------------------------------------------------------------------------------

class DimensionOrderMismatchError(DigitalPathologyTrainingError):
    """Raise when the input dimension order of the network model does not match the output dimension order of batch generator."""

    def __init__(self, model_dim_order, generator_dim_order, generator_purpose):
        """
        Initialize the object.

        Args:
            model_dim_order (str): Input dimension order of the network model.
            generator_dim_order (str): Output dimension order of the batch generator.
            generator_purpose (str): Batch generator purpose.
        """

        # Initialize base class.
        #
        super().__init__('The input \'{model}\' dimension order of the network model does not match the \'{gen}\' output dimension order of {pps} batch generator.'.format(model=model_dim_order,
                                                                                                                                                                           gen=generator_dim_order,
                                                                                                                                                                           pps=generator_purpose))

        # Store custom data.
        #
        self.model_dim_order = model_dim_order
        self.generator_dim_order = generator_dim_order
        self.generator_purpose = generator_purpose

#----------------------------------------------------------------------------------------------------

class InvalidFileSynchronizerError(DigitalPathologyTrainingError):
    """Raise when the statistics aggregator is invalid."""

    def __init__(self):
        """Initialize the object."""

        # Initialize base class.
        #
        super().__init__('Invalid statistics aggregator.')

#----------------------------------------------------------------------------------------------------

class InvalidStatAggregatorError(DigitalPathologyTrainingError):
    """Raise when the statistics aggregator is invalid."""

    def __init__(self):
        """Initialize the object."""

        # Initialize base class.
        #
        super().__init__('Invalid statistics aggregator.')

#----------------------------------------------------------------------------------------------------

class UnknownMetricNameError(DigitalPathologyTrainingError):
    """Raise when the required metric is not produced by the model."""

    def __init__(self,  metric_name, available_metrics):
        """
        Initialize the object.

        Args:
            metric_name (str): Required metric name.
            available_metrics (list): Available model metrics.
        """

        # Initialize base class.
        #
        super().__init__('The required \'{metric}\' metric is not in the {available} list of available model metrics.'.format(metric=metric_name, available=available_metrics))

        # Store custom data.
        #
        self.metric_name = metric_name
        self.available_metrics = available_metrics

#----------------------------------------------------------------------------------------------------

class InvalidAveragingLengthError(DigitalPathologyTrainingError):
    """Raise when the epoch averaging length is not valid."""

    def __init__(self, averaging_length):
        """
        Initialize the object.

        Args:
            averaging_length (int): Number of epochs to average over to get the actual metric.
        """

        # Initialize base class.
        #
        super().__init__('Invalid epoch averaging length: {length}.'.format(length=averaging_length))

        # Store custom data.
        #
        self.averaging_length = averaging_length

#----------------------------------------------------------------------------------------------------

class InvalidEpochCountError(DigitalPathologyTrainingError):
    """Raise when the epoch count is not valid."""

    def __init__(self, epoch_count):
        """
        Initialize the object.

        Args:
            epoch_count (int): Epoch count.
        """

        # Initialize base class.
        #
        super().__init__('Invalid epoch count: {epochs}.'.format(epochs=epoch_count))

        # Store custom data.
        #
        self.epoch_count = epoch_count

#----------------------------------------------------------------------------------------------------

class InvalidRepetitionCountError(DigitalPathologyTrainingError):
    """Raise when the repetition count is not valid."""

    def __init__(self,  purpose, repetition_count):
        """
        Initialize the object.

        Args:
            purpose (str): Iteration purpose.
            repetition_count (int): Repetition count.
        """

        # Initialize base class.
        #
        super().__init__('Invalid {purpose} repetition count: {repetitions}.'.format(purpose=purpose, repetitions=repetition_count))

        # Store custom data.
        #
        self.purpose = purpose
        self.repetition_count = repetition_count

#----------------------------------------------------------------------------------------------------

class InvalidIterationCountError(DigitalPathologyTrainingError):
    """Raise when the iteration count is not valid."""

    def __init__(self, purpose, iter_count):
        """
        Initialize the object.

        Args:
            purpose (str): Iteration purpose.
            iter_count (int): Iteration count.
        """

        # Initialize base class.
        #
        super().__init__('Invalid {purpose} iteration count: {iters}.'.format(purpose=purpose, iters=iter_count))

        # Store custom data.
        #
        self.purpose = purpose
        self.iter_count = iter_count

#----------------------------------------------------------------------------------------------------

class InvalidDifficultUpdateRatio(DigitalPathologyTrainingError):
    """Raise when the difficult example update ratio is not valid."""

    def __init__(self, update_ratio):
        """
        Initialize the object.

        Args:
            update_ratio (float): Difficult example update ratio.
        """

        # Initialize base class.
        #
        super().__init__('Invalid difficult example update ratio: {ratio}.'.format(ratio=update_ratio))

        # Store custom data.
        #
        self.update_ratio = update_ratio

#----------------------------------------------------------------------------------------------------

class InvalidBufferConfigurationError(DigitalPathologyTrainingError):
    """Raise when the buffer is configured to boosting without double buffering."""

    def __init__(self, buffer_mode_switch):
        """
        Initialize the object.

        Args:
            buffer_mode_switch (int): Epoch index where the buffer mode should be switched from the initial 'ring' mode to the configured one.
        """

        # Initialize base class.
        #
        super().__init__('Boosting enabled at epoch {epoch} without double buffering.'.format(epoch=buffer_mode_switch))

        # Store custom data.
        #
        self.buffer_mode_switch = buffer_mode_switch

#----------------------------------------------------------------------------------------------------

class InvalidDifficultThreshold(DigitalPathologyTrainingError):
    """Raise when the difficult example threshold is not valid."""

    def __init__(self, difficult_threshold):
        """
        Initialize the object.

        Args:
            difficult_threshold (float): Difficult example threshold.
        """

        # Initialize base class.
        #
        super().__init__('Invalid difficult example threshold: {threshold}.'.format(threshold=difficult_threshold))

        # Store custom data.
        #
        self.difficult_threshold = difficult_threshold

#----------------------------------------------------------------------------------------------------

class InvalidLearningRateError(DigitalPathologyTrainingError):
    """Raise when the learning rate is not valid."""

    def __init__(self, learning_rate):
        """
        Initialize the object.

        Args:
            learning_rate (float): Learning rate.
        """

        # Initialize base class.
        #
        super().__init__('Invalid learning rate: {rate}.'.format(rate=learning_rate))

        # Store custom data.
        #
        self.learning_rate = learning_rate

#----------------------------------------------------------------------------------------------------

class InvalidModelSavePathError(DigitalPathologyTrainingError):
    """Raise when the model save path is not valid."""

    def __init__(self, role):
        """
        Initialize the object.

        Args:
            role (str): Role: 'best' or 'last'.
        """

        # Initialize base class.
        #
        super().__init__('Missing {role} network save path.'.format(role=role))

#----------------------------------------------------------------------------------------------------

class InvalidStateSavePathError(DigitalPathologyTrainingError):
    """Raise when the training state save path is not valid."""

    def __init__(self, role):
        """
        Initialize the object.

        Args:
            role (str): Role: 'best' or 'last'.
        """

        # Initialize base class.
        #
        super().__init__('Missing {role} training state save path.'.format(role=role))

#----------------------------------------------------------------------------------------------------

class InvalidSyncTargetPathError(DigitalPathologyTrainingError):
    """Raise when the sync target path is not valid."""

    def __init__(self, path):
        """
        Initialize the object.

        Args:
            path (str, None): Sync target path.
        """

        # Initialize base class.
        #
        super().__init__('Invalid sync target path: \'{path}\'.'.format(path=path))

        # Store custom data.
        #
        self.path = path

#----------------------------------------------------------------------------------------------------

class FileSyncFailedError(DigitalPathologyTrainingError):
    """Raise when the synchronization of files failed."""

    def __init__(self, path_list):
        """
        Initialize the object.

        Args:
            path_list (list): List of failed target path.
        """

        # Initialize base class.
        #
        super().__init__('Failed sync attempts: {items}.'.format(items=path_list))

        # Store custom data.
        #
        self.path_list = path_list

#----------------------------------------------------------------------------------------------------

class ModelHashMismatchError(DigitalPathologyTrainingError):
    """Raise when the stored hash of the model in the state does not match the hash of the actual file."""

    def __init__(self, state_hash, model_hash):
        """
        Initialize the object.

        Args:
            state_hash (str): Hash stored in the state description.
            model_hash (str): Actual hash of the model file
        """

        # Initialize base class.
        #
        super().__init__('State and available model hash mismatch: \'{state}\' - \'{model}\'.'.format(state=state_hash, model=model_hash))

        # Store custom data.
        #
        self.state_hash = state_hash
        self.model_hash = model_hash

#----------------------------------------------------------------------------------------------------

class ErrorsNotInModelOutputError(DigitalPathologyTrainingError):
    """Raise when the network does not return the individual errors per patch despite it is required e.g. for boosting."""

    def __init__(self):
        """Initialize the object."""

        # Initialize base class.
        #
        super().__init__('The network does not return errors despite it is required.')
