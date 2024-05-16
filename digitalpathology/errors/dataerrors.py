"""
Data source configuration related errors.
"""

from . import errorbase as dpterrorbase

#----------------------------------------------------------------------------------------------------

class DigitalPathologyDataError(dpterrorbase.DigitalPathologyError):
    """Error base class for all data errors."""

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

class EmptyDataSetsError(DigitalPathologyDataError):
    """Raise when the data sets of the extractor are empty."""

    def __init__(self):
        """
        Initialize the object.
        """

        # Initialize base class.
        #
        super().__init__('Empty data sets: patch extractor not initialized properly.')

#----------------------------------------------------------------------------------------------------

class StatHashMismatchError(DigitalPathologyDataError):
    """Raise when the hash of the loaded mask mr-image does not match the hash of collected data."""

    def __init__(self, mask_path, mask_hash, stat_path, stat_hash):
        """
        Initialize the object.

        Args:
            mask_path (str): Mask file path.
            mask_hash (str): Mask SHA256 digest.
            stat_path (str): Stat file path.
            stat_hash (str): Stat SHA256 digest.
        """

        # Initialize base class.
        #
        super().__init__('Mask - stat hash mismatch: {mask} and {stat}.'.format(mask=mask_hash, stat=stat_hash))

        # Store custom data.
        #
        self.mask_path = mask_path
        self.mask_hash = mask_hash
        self.stat_path = stat_path
        self.stat_hash = stat_hash

#----------------------------------------------------------------------------------------------------

class InvalidDataSourceTypeError(DigitalPathologyDataError):
    """Raise when the type identifier in data source JSON or YAML file is unknown."""

    def __init__(self, type_id):
        """
        Initialize the object.

        Args:
            type_id (str): JSON type identifier keyword.
        """

        # Initialize base class.
        #
        super().__init__('Unknown data source JSON type identifier: {id}.'.format(id=type_id))

        # Store custom data.
        #
        self.type_id = type_id

#----------------------------------------------------------------------------------------------------

class MissingImageFileError(DigitalPathologyDataError):
    """Raise when the given image file is missing."""

    def __init__(self, image_file_path):
        """
        Initialize the object.

        Args:
            image_file_path (str): Image file path.
        """

        # Initialize base class.
        #
        super().__init__('Missing image file: {path}.'.format(path=image_file_path))

        # Store custom data.
        #
        self.image_file_path = image_file_path

#----------------------------------------------------------------------------------------------------

class MissingMaskAndStatFilesError(DigitalPathologyDataError):
    """Raise when the given mask and stat files are missing."""

    def __init__(self, mask_file_path, stat_file_path):
        """
        Initialize the object.

        Args:
            mask_file_path (str, None): Mask file path.
            stat_file_path (str, None): Digest file path.
        """

        # Initialize base class.
        #
        super().__init__('Missing mask and stat file.')

        # Store custom data.
        #
        self.mask_file_path = mask_file_path
        self.stat_file_path = stat_file_path

#----------------------------------------------------------------------------------------------------

class MaskLabelListMismatchError(DigitalPathologyDataError):
    """Raise when the label list in the mask or stats file and the configured label list does not match."""

    def __init__(self, file_path, file_labels, config_labels):
        """
        Initialize the object.

        Args:
            file_path (str): Mask or stats file path.
            file_labels (list): List of labels in the file.
            config_labels (list): List of configured labels.
        """

        # Initialize base class.
        #
        super().__init__('Mask label list does not match the configured: {path}; {file_list}; {config_list}.'.format(path=file_path, file_list=file_labels, config_list=config_labels))

        # Store custom data.
        #
        self.file_path = file_path
        self.file_labels = file_labels
        self.config_labels = config_labels

#----------------------------------------------------------------------------------------------------

class MissingPatchSamplersError(DigitalPathologyDataError):
    """Raise when there is no configured patch sampler  present."""

    def __init__(self):
        """ Initialize the object."""

        # Initialize base class.
        #
        super().__init__('Missing patch samplers.')

#----------------------------------------------------------------------------------------------------

class EmptyPatchSourceError(DigitalPathologyDataError):
    """Raise when list of patch source objects is empty."""

    def __init__(self):
        """ Initialize the object."""

        # Initialize base class.
        #
        super().__init__('Empty patch source list.')

#----------------------------------------------------------------------------------------------------

class FailedSourceSelectionError(DigitalPathologyDataError):
    """Raise when a label cannot be represented in a source set."""

    def __init__(self, labels, sources):
        """
        Initialize the object.

        Args:
            labels (list): Label list.
            sources (list): List of selected sources.
        """

        # Initialize base class.
        #
        super().__init__('Not all {labels} labels can be sampled from the current source selection: {sources}.'.format(labels=labels, sources=sources))

        # Store custom data.
        #
        self.labels = labels
        self.sources = sources
