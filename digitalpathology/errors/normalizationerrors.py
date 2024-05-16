"""
Normalization related errors.
"""

from . import errorbase as dpterrorbase

#----------------------------------------------------------------------------------------------------

class DigitalPathologyNormalizationError(dpterrorbase.DigitalPathologyError):
    """Error base class for all normalization errors."""

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

class InvalidNormalizationRangeError(DigitalPathologyNormalizationError):
    """Raise when the source or target range for batch normalization is invalid."""

    def __init__(self, range_purpose, normalization_range):
        """
        Initialize the object.

        Args:
            range_purpose (str): Purpose of the range.
            normalization_range (tuple): Range.
        """

        # Initialize base class.
        #
        super().__init__('Invalid {purpose} normalization range: {range}.'.format(purpose=range_purpose, range=normalization_range))

        # Store custom data.
        #
        self.range_purpose = range_purpose
        self.normalization_range = normalization_range
