"""
Statistics collection related errors.
"""

from . import errorbase as dpterrorbase

#----------------------------------------------------------------------------------------------------

class DigitalPathologyStatError(dpterrorbase.DigitalPathologyError):
    """Error base class for all statistics errors."""

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

class MissingStatInputPathError(DigitalPathologyStatError):
    """Raise when the input path for MaskStats is None."""

    def __init__(self):
        """Initialize the object."""

        # Initialize base class.
        #
        super().__init__('Missing input path mask statistics.')

#----------------------------------------------------------------------------------------------------

class StatReadingError(DigitalPathologyStatError):
    """Raise when the stat file cannot be opened."""

    def __init__(self, stat_path):
        """
        Initialize the object.

        Args:
            stat_path (str): Path of stat file to load.

        """

        # Initialize base class.
        #
        super().__init__('Cannot open stat for reading  : \'{path}\'.'.format(path=stat_path))

        # Store custom data.
        #
        self.stat_path = stat_path

#----------------------------------------------------------------------------------------------------

class MissingDataKeyError(DigitalPathologyStatError):
    """Raise when not all the necessary keys are in the loaded stats data."""

    def __init__(self, loaded_keys, necessary_keys):
        """
        Initialize the object.

        Args:
            loaded_keys (list): Loaded keys from the stat file.
            necessary_keys (list): Mandatory keys in a stat file.
        """

        # Initialize base class.
        #
        super().__init__('Not all the mandatory {mandatory} keys are present in the {loaded} list of loaded keys.'.format(mandatory=necessary_keys, loaded=loaded_keys))

        # Store custom data.
        #
        self.loaded_keys = loaded_keys
        self.necessary_keys = necessary_keys

#----------------------------------------------------------------------------------------------------

class MaskPixelSpacingMismatchError(DigitalPathologyStatError):
    """Raise when the loaded mask pixel spacing does not match the configured mask pixel spacing."""

    def __init__(self, loaded_mask_spacing, configured_mask_spacing):
        """
        Initialize the object.

        Args:
            loaded_mask_spacing (float): Pixel spacing of the loaded stats (micrometer).
            configured_mask_spacing (float): Configured pixel spacing (micrometer).
        """

        # Initialize base class.
        #
        super().__init__('The {loaded} loaded mask pixel spacing does not match the {config} configured mask pixel spacing.'.format(loaded=loaded_mask_spacing, config=configured_mask_spacing))

        # Store custom data.
        #
        self.loaded_mask_spacing = loaded_mask_spacing
        self.configured_mask_spacing = configured_mask_spacing

#----------------------------------------------------------------------------------------------------

class LabelListMismatchError(DigitalPathologyStatError):
    """Raise when the loaded label list does not match the configured label list."""

    def __init__(self, loaded_label_list, configured_label_list):
        """
        Initialize the object.

        Args:
            loaded_label_list (int): Label list loaded from .stat file.
            configured_label_list (int): Configured label list.
        """

        # Initialize base class.
        #
        super().__init__('The loaded label list {loaded} does not match the configured label list {config}.'.format(loaded=loaded_label_list, config=configured_label_list))

        # Store custom data.
        #
        self.loaded_label_list = loaded_label_list
        self.configured_label_list = configured_label_list

#----------------------------------------------------------------------------------------------------

class InconsistentDataError(DigitalPathologyStatError):
    """Raise when the loaded stat data does not have the expected structure."""

    def __init__(self, stat_path):
        """
        Initialize the object.

        Args:
            stat_path (str): Path of loaded stat file.

        """

        # Initialize base class.
        #
        super().__init__('Inconsistent stat data structure in \'{path}\'.'.format(path=stat_path))

        # Store custom data.
        #
        self.stat_path = stat_path

#----------------------------------------------------------------------------------------------------

class UnknownMaskInputFormatError(DigitalPathologyStatError):
    """Raise when the mask source descriptor type is unknown."""

    def __init__(self):
        """Initialize the object."""

        # Initialize base class.
        #
        super().__init__('Unknown mask source descriptor.')
