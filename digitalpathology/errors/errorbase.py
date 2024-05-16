"""
Base error for all errors in the package.
"""

class DigitalPathologyError(Exception):
    """Error base class for all the errors in this package."""

    def __init__(self, *args):
        """
        Initialize the object.

        Args:
            *args: Argument list.
        """

        # Initialize base class.
        #
        super().__init__(*args)
