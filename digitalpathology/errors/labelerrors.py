"""
Label configuration related errors.
"""

from . import errorbase as dpterrorbase

#----------------------------------------------------------------------------------------------------

class DigitalPathologyLabelError(dpterrorbase.DigitalPathologyError):
    """Error base class for all label errors."""

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

class EmptyLabelMapError(DigitalPathologyLabelError):
    """Raise when the label value to label index is empty."""

    def __init__(self):
        """
        Initialize the object.
        """

        # Initialize base class.
        #
        super().__init__('Empty label value to index map.')

#----------------------------------------------------------------------------------------------------

class EmptyLabelListError(DigitalPathologyLabelError):
    """Raise when the list of labels to process is empty."""

    def __init__(self):
        """
        Initialize the object.
        """

        # Initialize base class.
        #
        super().__init__('Empty label list.')

#----------------------------------------------------------------------------------------------------

class InvalidLabelValueError(DigitalPathologyLabelError):
    """Raise when the list of labels contains invalid item."""

    def __init__(self, label_list):
        """
        Initialize the object.

        Args:
            label_list (list): List of labels.
        """

        # Initialize base class.
        #
        super().__init__('Invalid label in list: {labels}.'.format(labels=label_list))

        # Store custom data.
        #
        self.label_list = label_list
#----------------------------------------------------------------------------------------------------

class InvalidKeyLabelError(DigitalPathologyLabelError):
    """Raise when the the keys (mask labels) of labels dictionary contains invalid item."""

    def __init__(self, label_dictionary):
        """
        Initialize the object.

        Args:
            label_dictionary (dictionary): Dictionary mapping mask labels to training labels.
        """

        # Initialize base class.
        #
        super().__init__("Invalid key in label mapping dictionary: {map}.".format(map=label_dictionary))

        # Store custom data.
        #
        self.label_dictionary = label_dictionary

#----------------------------------------------------------------------------------------------------

class NonContinuousLabelListError(DigitalPathologyLabelError):
    """Raise when the the values (training labels) of labels dictionary contains invalid item."""

    def __init__(self, label_dictionary):
        """
        Initialize the object.

        Args:
            label_dictionary (dictionary): Dictionary mapping mask labels to training labels.
        """

        # Initialize base class.
        #
        super().__init__("Non-continuous value in label mapping dictionary: {map}.".format(map=label_dictionary))

        # Store custom data.
        #
        self.label_dictionary = label_dictionary

#----------------------------------------------------------------------------------------------------

class LabelValueMappingError(DigitalPathologyLabelError):
    """Raise when a label value cannot be mapped to an index."""

    def __init__(self, label_list, label_map):
        """
        Initialize the object.

        Args:
            label_list (list): List of label values.
            label_map (list): Label mapping.
        """

        # Initialize base class.
        #
        super().__init__('Not all labels from the {labels} label set can be mapped to index by the {map} mapping.'.format(labels=label_list, map=label_map))

        # Store custom data.
        #
        self.label_list = label_list
        self.label_map = label_map

#----------------------------------------------------------------------------------------------------

class NegativeLabelRatioError(DigitalPathologyLabelError):
    """Raise when the label distribution contains negative value."""

    def __init__(self, label_dist):
        """
        Initialize the object.

        Args:
            label_dist (dict): Label sampling distribution.
        """

        # Initialize base class.
        #
        super().__init__('Negative ratio value in label distribution: {dist}.'.format(dist=label_dist))

        # Store custom data.
        #
        self.label_dist = label_dist

#----------------------------------------------------------------------------------------------------

class AllZeroLabelRatiosError(DigitalPathologyLabelError):
    """Raise when the label distribution list contains only zeros."""

    def __init__(self, label_dist):
        """
        Initialize the object.

        Args:
            label_dist (dict): Label sampling distribution.
        """

        # Initialize base class.
        #
        super().__init__('All label ratios are zero: {dist}.'.format(dist=label_dist))

        # Store custom data.
        #
        self.label_dist = label_dist

#----------------------------------------------------------------------------------------------------

class CategoryRatioListMismatchError(DigitalPathologyLabelError):
    """Raise when the per image category ratios are not matching the list of image categories."""

    def __init__(self, category_list, category_dist):
        """
        Initialize the object.

        Args:
            category_list (list): List of image category ids.
            category_dist (dict): Label sampling distribution.
        """

        # Initialize base class.
        #
        super().__init__('The image category distribution {dist} does not match the available image categories {categories}.'.format(dist=category_dist, categories=category_list))

        # Store custom data.
        #
        self.category_list = category_list
        self.category_dist = category_dist

#----------------------------------------------------------------------------------------------------

class NegativeCategoryRatioError(DigitalPathologyLabelError):
    """Raise when the image category distribution contains negative value."""

    def __init__(self, category_dist):
        """
        Initialize the object.

        Args:
            category_dist (dict): Image category sampling distribution.
        """

        # Initialize base class.
        #
        super().__init__('Negative ratio value in image category distribution: {dist}.'.format(dist=category_dist))

        # Store custom data.
        #
        self.category_dist = category_dist

#----------------------------------------------------------------------------------------------------

class AllZeroCategoryRatiosError(DigitalPathologyLabelError):
    """Raise when the class distribution list contains only zeros."""

    def __init__(self, category_dist):
        """
        Initialize the object.

        Args:
            category_dist (dict): Image category sampling distribution.
        """

        # Initialize base class.
        #
        super().__init__('All image category ratios are zero: {dist}.'.format(dist=category_dist))

        # Store custom data.
        #
        self.category_dist = category_dist

#----------------------------------------------------------------------------------------------------

class LabelCountProbabilityMismatchError(DigitalPathologyLabelError):
    """Raise when the label count and classification error lists do not match."""

    def __init__(self, label_counts, label_errors):
        """
        Initialize the object.

        Args:
            label_counts (list): Label count list.
            label_errors (list): Label classification error list.
        """

        # Initialize base class.
        #
        super().__init__('Label count and classification error list size mismatch: {labels} - {errors}.'.format(labels=len(label_counts), errors=len(label_errors)))

        # Store custom data.
        #
        self.label_counts = label_counts
        self.label_errors = label_errors

#----------------------------------------------------------------------------------------------------

class UnknownLabelError(DigitalPathologyLabelError):
    """Raise when the label value is unknown."""

    def __init__(self, label, label_list):
        """
        Initialize the object.

        Args:
            label (int): Label value.
            label_list (list): Label list.
        """

        # Initialize base class.
        #
        super().__init__('Unknown label value {value} in {labels}.'.format(value=label, labels=label_list))

        # Store custom data.
        #
        self.label = label
        self.label_list = label_list

#----------------------------------------------------------------------------------------------------

class LabelListMismatchError(DigitalPathologyLabelError):
    """Raise when the number of requested labels does not match the number of available labels."""

    def __init__(self, available_labels, requested_labels):
        """
        Initialize the object.

        Args:
            available_labels (list): Available labels.
            requested_labels (list): Requested labels.
        """

        # Initialize base class.
        #
        super().__init__('The number of requested labels: {requested} does not match the available: {available}.'.format(requested=len(requested_labels), available=len(available_labels)))

        # Store custom data.
        #
        self.available_labels = available_labels
        self.requested_labels = requested_labels

#----------------------------------------------------------------------------------------------------

class NegativeLabeledItemCountError(DigitalPathologyLabelError):
    """Raise when the label count list contains negative value."""

    def __init__(self, pixel_counts):
        """
        Initialize the object.

        Args:
            pixel_counts (dict): Label value to pixel count map.
        """

        # Initialize base class.
        #
        super().__init__('Negative label count value: {count}.'.format(count=pixel_counts))

        # Store custom data.
        #
        self.pixel_counts = pixel_counts

#----------------------------------------------------------------------------------------------------

class AllZeroLabelCountsError(DigitalPathologyLabelError):
    """Raise when the label count list contains only zeros."""

    def __init__(self):
        """
        Initialize the object.
        """

        # Initialize base class.
        #
        super().__init__('All label counts are zero.')

#----------------------------------------------------------------------------------------------------

class LabelSourceConfigurationError(DigitalPathologyLabelError):
    """Raise when the currently opened images cannot source the selected label."""

    def __init__(self, labels, sources):
        """
        Initialize the object.

        Args:
            labels (list): Label list.
            sources (list): Used mask sources.
        """

        # Initialize base class.
        #
        super().__init__('Current source set cannot support all {labels} labels: {sources}.'.format(labels=labels, sources=sources))

        # Store custom data.
        #
        self.labels = labels
        self.sources = sources

#----------------------------------------------------------------------------------------------------

class LabelDistributionAndMappingMismatchError(DigitalPathologyLabelError):
    """Raise when the label distribution and label mapping don't have the same keys."""

    def __init__(self, label_dist, label_map):
        """
        Initialize the object.
        
        Args:
            label_dist (dict): Label ratios.
            label_map (dict): Label mapping.
        """

        # Initialize base class.
        #
        super().__init__('The {dist} label distribution keys does not match the keys of {mapping} label mapping.'.format(dist=label_dist, mapping=label_map))

        # Store custom data.
        #
        self.label_dist = label_dist
        self.label_map = label_map

#----------------------------------------------------------------------------------------------------

class InvalidLabelDistributionWithoutMappingError(DigitalPathologyLabelError):
    """Raise when label mapping is not defined but the label distribution keys are not a continuous interval starting with zero."""

    def __init__(self, label_dist):
        """
        Initialize the object.

        Args:
            label_dist (dict): Label ratios.
        """

        # Initialize base class.
        #
        super().__init__('The {dist} label distribution is not valid without label mapping.'.format(dist=label_dist))

        # Store custom data.
        #
        self.label_dist = label_dist
