"""
This file contains class describing an image patch source.
"""

class PatchSource(object):
    """This class can store image path, mask path, mask pixel spacing, stat path, and list of available labels."""

    def __init__(self, image_path, mask_path, stat_path, available_labels):
        """
        Initialize the object.

        Args:
            image_path (str): Path of the image file.
            mask_path (str, None): Path of the mask file.
            stat_path (str, None): Path of the stats file.
            available_labels (tuple): List of available labels in the mask to use.
        """

        # Initialize the base class.
        #
        super().__init__()

        # Initialize members.
        #
        self.__image = image_path                                 # Path of the image.
        self.__mask = mask_path if mask_path is not None else ''  # Path of the mask to use with the image.
        self.__stat = stat_path if stat_path is not None else ''  # Mask stat path.
        self.__labels = tuple(available_labels)                   # Labels to use.

    def __str__(self):
        """
        String conversion.

        Returns:
            str: String representation of the object.
        """

        return ';'.join((self.__image, self.__mask, self.__stat, str(self.__labels)))

    def __eq__(self, other):
        """
        Equals operator for comparing sources.

        Args:
            other (PatchSource): Other side of comparison.

        Returns:
            bool: True if the two objects describes the same data source, False otherwise.
        """

        if not isinstance(other, self.__class__):
            raise TypeError('\'==\' not supported between instances of \'{left}\' and \'{right}\''.format(left=self.__class__.__name__, right=type(other).__name__))

        return (self.__image, self.__mask, self.__stat, self.__labels) == (other.__image, other.__mask, other.__stat, other.__labels)

    def __lt__(self, right):
        """
        Less than operator for comparing sources.

        Args:
            right (PatchSource): Right side of comparison.

        Returns:
            bool: True if the two objects describes the same data source, False otherwise.
        """

        if not isinstance(right, self.__class__):
            raise TypeError('\'<\' not supported between instances of \'{this}\' and \'{other}\''.format(this=self.__class__.__name__, other=type(right).__name__))

        return (self.__image, self.__mask, self.__stat, self.__labels) < (right.__image, right.__mask, right.__stat, right.__labels)

    def __hash__(self):
        """
        Hash for comparing sources.

        Returns:
            int: Hash of this object.
        """

        return hash((self.__image, self.__mask, self.__stat, self.__labels))

    @property
    def image(self):
        """
        Get the image path.

        Returns:
            str: Image path.
        """

        return self.__image

    @property
    def mask(self):
        """
        Get the mask path.

        Returns:
            str: Mask path.
        """

        return self.__mask

    @property
    def stat(self):
        """
        Get the stat path.

        Returns:
            str: Stat path.
        """

        return self.__stat

    @property
    def labels(self):
        """
        Get the available labels.

        Returns:
            list: Available labels.
        """

        return self.__labels
