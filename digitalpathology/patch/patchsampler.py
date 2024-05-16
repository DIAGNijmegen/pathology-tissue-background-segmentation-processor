"""
This file contains class for sampling patches from whole slide images.
"""

from . import patchsource as dptpatchsource

from ..errors import imageerrors as dptimageerrors
from ..errors import dataerrors as dptdataerrors
from ..errors import configerrors as dptconfigerrors
from ..image import imagereader as dptimagereader
from ..mask import randomizer as dptrandom
from ..mask import maskstats as dptmaskstats

import numpy as np
import scipy.ndimage
import scipy.misc
import os

#----------------------------------------------------------------------------------------------------

class PatchSampler(object):
    """This class can sample patches from an image considering various conditions like mask and probability."""

    def __init__(self, patch_source, create_stat, mask_spacing, spacing_tolerance, input_channels, label_mode):
        """
        Initialize the object: load the image, load a mask for the configured image and check compatibility, extract and store necessary mask data
        in memory for efficient patch extraction and initialize index randomizer.

        Args:
            patch_source (dptpatchsource.PatchSource): Image patch source descriptor with image and mask paths, mask level and labels to use.
            create_stat (bool): Allow missing stat file, and create it if necessary.
            mask_spacing (float, None): Pixel spacing of the mask to process (micrometer).
            spacing_tolerance (float): Pixel spacing tolerance (percentage).
            input_channels (list): Desired channels that are extracted for each patch.
            label_mode (str): Label generation mode. Accepted values:
                'central': Just return the label of the central pixel.
                'synthesize': Synthesize the label map from the mask statistics and zoom it to the appropriate level of pixel spacing.
                'load': Load the label map from the label image and zoom it to the appropriate level of pixel spacing if necessary.

        Raises:
            InvalidPixelSpacingToleranceError: The pixel spacing tolerance is not valid.
            UnknownLabelModeError: The label generation mode is unknown.
            MissingMaskImageError: The label generation mode is set to 'load' but the mask image is not present.
            StatShapeMismatchError: The shape of the loaded stat and the given mask file cannot be matched.
            StatSpacingMismatchError: The pixel spacing of the loaded stat and the given mask file cannot be matched.
            MissingMaskImageError: Stat is not configured but mask image is also not configured for stat calculation.
            UnfixableImageSpacingError: The missing spacing information of he mask image cannot be fixed.
            MaskLabelListMismatchError: The configured labels does not match the list of the labels collected by the stat object.
            ImageShapeMismatchError: The shape of the image cannot be matched the shape of the mask.

            DigitalPathologyConfigError: Configuration errors.
            DigitalPathologyImageError: Image errors.
            DigitalPathologyLabelError: Label errors.
            DigitalPathologyStatError: Stat errors.
        """

        # Initialize the base class.
        #
        super().__init__()

        # Initialize members.
        #
        self.__loader = None             # Image patch loader.
        self.__mask = None               # Mask patch loader.
        self.__stats = None              # Mask statistics.
        self.__rand = None               # Index randomizer object.
        self.__spacing_tolerance = 0.0   # Tolerance for finding a level for the given pixel spacing.
        self.__input_channel_count = 0   # Desired channels that are extracted for each patch.
        self.__label_mode = ''           # Label mode for returning labels.
        self.__stats_level = 0           # Level of the image where its dimensions match the mask.
        self.__stats_downsamplings = []  # Downsampling factors for each used level relative to the matching level.
        self.__stats_shifts = []         # Central pixel shift values for each used level relative to the matching level.
        self.__mask_level = []           # Level of mask to use of each image level for label map extraction.
        self.__mask_downsamplings = []   # Downsampling factors for each used level relative to the mask level to use.
        self.__mask_shifts = []          # Central pixel shift values for each used level relative to the mask level to use.

        # Configure parameters.
        #
        self.__setspacing(spacing_tolerance=spacing_tolerance)
        self.__openimage(image_path=patch_source.image, mask_path=patch_source.mask, input_channels=input_channels)
        self.__configuremode(label_mode=label_mode)
        self.__collectdata(stat_path=patch_source.stat, create_stat=create_stat, mask_spacing=mask_spacing, mask_labels=patch_source.labels)
        self.__initrandomizer()

    def __setspacing(self, spacing_tolerance):
        """
        Set the spacing tolerance.

        Args:
            spacing_tolerance (float): Pixel spacing tolerance (percentage).

        Raises:
            InvalidPixelSpacingToleranceError: The pixel spacing tolerance is not valid.
        """

        # The tolerance must be non-negative.
        #
        if spacing_tolerance < 0.0:
            raise dptconfigerrors.InvalidPixelSpacingToleranceError(spacing_tolerance)

        self.__spacing_tolerance = spacing_tolerance

    def __openimage(self, image_path, mask_path, input_channels):
        """
        Load an image.

        Args:
            image_path (str): Path of the image to load.
            mask_path (str, None): Path of the mask to load.
            input_channels (list): Desired channels that are extracted for each patch.

        Raises:
            DigitalPathologyConfigError: Configuration errors.
            DigitalPathologyImageError: Image errors.
        """

        # Open and configure the multi-resolution images.
        #
        self.__loader = dptimagereader.ImageReader(image_path=image_path, spacing_tolerance=self.__spacing_tolerance, input_channels=input_channels, cache_path=None)

        if mask_path:
            self.__mask = dptimagereader.ImageReader(image_path=mask_path, spacing_tolerance=self.__spacing_tolerance, input_channels=[0], cache_path=None)

        self.__input_channel_count = self.__loader.channels

    def __configuremode(self, label_mode):
        """
        Configure the label generation mode.

        Args:
            label_mode (str): Label generation mode.

        Raises:
            UnknownLabelModeError: The label generation mode is unknown.
            MissingMaskImageError: The label generation mode is set to 'load' but the mask image is not present.
        """

        # Check if the label patch generation mode is valid.
        #
        if label_mode not in ['central', 'synthesize', 'load']:
            raise dptconfigerrors.UnknownLabelModeError(label_mode)

        # Check if mask patch loader is ready if the label patches are loaded from file.
        #
        if label_mode == 'load' and not self.__mask:
            raise dptconfigerrors.MissingMaskImageError()

        # Save the output shapes and the label mode.
        #
        self.__label_mode = label_mode

    def __collectdata(self, stat_path, create_stat, mask_spacing, mask_labels):
        """
        Extract and store necessary mask data in memory for efficient patch extraction.

        Args:
            stat_path (str, None): Path of the mask stat to load.
            create_stat (bool): Allow missing stat file, and create it if necessary.
            mask_spacing (float): Pixel spacing of the mask to process (micrometer).
            mask_labels (list): List of mask labels to use. All other labels will be considered as non-labeled area.

        Raises:
            StatShapeMismatchError: The shape of the loaded stat and the given mask file cannot be matched.
            StatSpacingMismatchError: The pixel spacing of the loaded stat and the given mask file cannot be matched.
            MissingMaskImageError: Stat is not configured but mask image is also not configured for stat calculation.
            UnfixableImageSpacingError: The missing spacing information of he mask image cannot be fixed.
            MaskLabelListMismatchError: The configured labels does not match the list of the labels collected by the stat object.
            ImageShapeMismatchError: The shape of the image cannot be matched the shape of the mask.

            DigitalPathologyConfigError: Configuration errors.
            DigitalPathologyImageError: Image errors.
            DigitalPathologyLabelError: Label errors.
            DigitalPathologyStatError: Stat errors.
        """

        # Load the mask data.
        #
        if stat_path and (os.path.isfile(stat_path) or not create_stat):
            # Load the stats file from disk.
            #
            self.__stats = dptmaskstats.MaskStats(file=stat_path, mask_spacing=mask_spacing, spacing_tolerance=self.__spacing_tolerance, mask_labels=mask_labels)

            source_path = stat_path

            # Check if the loaded stats matches the label image (mask) shape.
            #
            if self.__mask:
                mask_level = self.__mask.level(spacing=mask_spacing)

                if self.__mask.shapes[mask_level] != self.__stats.shape:
                    raise dptimageerrors.StatShapeMismatchError(self.__stats.path, self.__stats.shape, self.__mask.path, self.__mask.shapes[mask_level], mask_spacing)

                if self.__spacing_tolerance < abs(self.__mask.spacings[mask_level] - self.__stats.spacing):
                    raise dptimageerrors.StatSpacingMismatchError(self.__stats.path, self.__stats.spacing, self.__mask.path, self.__mask.spacings, self.__spacing_tolerance)
        else:
            # Stat is not configured or it should be created dynamically. Check if the mask image path is set.
            #
            if not self.__mask:
                raise dptconfigerrors.MissingMaskImageError()

            # Mask data collection is also based on spacing. If the spacing information is missing from the mask file, assume that the spacing of the mask is exactly the same as of the
            # image file where the shape of the two files first match.
            #
            if any(mask_spacing is None for mask_spacing in self.__mask.spacings):
                for image_level in range(len(self.__loader.shapes)):
                    if self.__loader.shapes[image_level] == self.__mask.shapes[0]:
                        self.__mask.correct(spacing=self.__loader.spacings[image_level], level=0)
                        break

            # Check if the mask spacings are correct now: the spacings for all levels are valid.
            #
            if any(mask_spacing is None for mask_spacing in self.__mask.spacings):
                raise dptimageerrors.UnfixableImageSpacingError(self.__loader.path, self.__loader.shapes, self.__mask.path, self.__mask.shapes)

            # Collect statistics from mask data.
            #
            self.__stats = dptmaskstats.MaskStats(file=self.__mask, mask_spacing=mask_spacing, spacing_tolerance=self.__spacing_tolerance, mask_labels=mask_labels)

            source_path = self.__mask.path

            # Save the stat file if it should be dynamically created.
            #
            if stat_path and create_stat:
                self.__stats.save(file_path=stat_path)

        # Check if the labels match.
        #
        if not set(mask_labels) <= set(self.__stats.labels):
            raise dptdataerrors.MaskLabelListMismatchError(source_path, self.__stats.labels, mask_labels)

        # Find the level in the image where it matches the dimensions of the mask image on the used mask level.
        #
        matching_level = None
        for image_level in range(len(self.__loader.shapes)):
            if self.__loader.shapes[image_level] == self.__stats.shape and abs(self.__loader.spacings[image_level] - self.__stats.spacing) < self.__spacing_tolerance:
                matching_level = image_level
                break

        # Check if the stat shape matches the image at any level.
        #
        if matching_level is not None:
            # Calculate the downsampling factor between the mask level and each image level.
            #
            self.__stats_level = matching_level
            self.__stats_downsamplings = [self.__loader.downsamplings[matching_level] / level_downsampling for level_downsampling in self.__loader.downsamplings]
            self.__stats_shifts = [max(0, int(stats_downsampling) // 2 - 1) for stats_downsampling in self.__stats_downsamplings]
        else:
            raise dptimageerrors.ImageShapeMismatchError(self.__loader.path, self.__loader.shapes, self.__loader.spacings, self.__stats.path, self.__stats.shape, self.__stats.spacing)

        # Check if the image shape matches the mask shape on the given level.
        #
        if self.__mask:
            # Get the mask level to use by the pixel spacing.
            #
            mask_level = self.__mask.level(spacing=mask_spacing)

            if matching_level is not None and \
               self.__loader.shapes[matching_level] == self.__mask.shapes[mask_level] and \
               abs(self.__loader.spacings[matching_level] - self.__mask.spacings[mask_level]) < self.__spacing_tolerance:
                # Calculate the closest matching level from the mask for each level in the image and the downsampling level to zoom to the same level as the image.
                #
                level_diff = matching_level - mask_level
                self.__mask_level = [min(max(0, level - level_diff), len(self.__mask.shapes) - 1) for level in range(len(self.__loader.shapes))]
                self.__mask_downsamplings = [self.__loader.downsamplings[level] / self.__loader.downsamplings[self.__mask_level[level] + level_diff] for level in range(len(self.__loader.shapes))]
                self.__mask_shifts = [max(0, int(mask_downsampling) // 2 - 1) for mask_downsampling in self.__mask_downsamplings]
            else:
                raise dptimageerrors.ImageShapeMismatchError(self.__loader.path,
                                                             self.__loader.shapes,
                                                             self.__loader.spacings,
                                                             self.__mask.path,
                                                             self.__mask.shapes[mask_level],
                                                             self.__mask.spacings[mask_level])

    def __initrandomizer(self):
        """
        Initialize index randomizer.

        Raises:
            DigitalPathologyLabelError: Label errors.
        """

        # Create and initialize new randomizer object.
        #
        self.__rand = dptrandom.IndexRandomizer(pixel_counts=self.__stats.counts)

    def __randomizecoordinates(self, counts):
        """
        Randomize coordinates with the previously configured distribution. The results are in numpy notation (row, col).

        Args:
            counts (dict): Label value to label count to extract mapping.

        Returns:
            np.ndarray: List of randomized coordinates organized in a list per label.

        Raises:
            LabelListMismatchError: The available labels does not match the requested labels
        """

        # Collect random pixel indexes in ascending order.
        #
        per_label_random_indexes = self.__rand.randomindices(counts=counts)

        # Convert indexes to (row, col, label) coordinates.
        #
        coordinate_array = np.empty((sum(pixel_count for pixel_count in counts.values()), 3), dtype=np.int32)
        filled_up_index = 0
        for label in per_label_random_indexes:
            index_array = per_label_random_indexes[label]
            index_count = len(index_array)
            coordinate_array[filled_up_index:filled_up_index + index_count] = self.__stats.indextocoorindate(index_array=index_array, label=label)
            filled_up_index += index_count

        return coordinate_array

    def sample(self, counts, shapes):
        """
        Collect a batch of patches with the configured distribution from the opened image.

        Args:
            counts (dict): Label value to label count to extract mapping.
            shapes (dict): Dictionary mapping pixel spacings to (rows, columns) patch shape.

        Returns:
            dict: Collected batch of RGB patches and corresponding labels or crops of mask data with label indices (not label values) per level.

        Raises:
            PixelSpacingLevelError: There is no level found for the given pixel spacing and tolerance.
            LabelListMismatchError: The available labels does not match the requested labels
        """

        # Extract random coordinates and sort them for more efficient patch extraction.
        #
        valid_counts = {label: label_count for label, label_count in counts.items() if 0 < label_count}
        non_organized_coordinates = self.__randomizecoordinates(valid_counts)
        lex_sort_order = np.lexsort((non_organized_coordinates[:, 1], non_organized_coordinates[:, 0]))
        patch_coordinates = non_organized_coordinates[lex_sort_order]

        # Initialize output.
        #
        sum_count = sum(valid_counts.values())
        patch_dict = {spacing: {'patches': np.empty((sum_count, self.__input_channel_count) + shapes[spacing], dtype=np.uint8),
                                'labels': np.empty((sum_count,), dtype=np.uint8) if self.__label_mode == 'central' else np.empty((sum_count, 1) + shapes[spacing], dtype=np.uint8)}
                      for spacing in shapes}

        # Extract patch from each coordinate.
        #
        for sample_index in range(patch_coordinates.shape[0]):
            central_coordinate = patch_coordinates[sample_index, 0:2]

            # Extract patches of the image.
            #
            for spacing in shapes:
                level = self.__loader.level(spacing=spacing)
                patch_level_shape = shapes[spacing]
                patch_level_center = np.multiply(central_coordinate, self.__stats_downsamplings[level]).astype(int) + self.__stats_shifts[level]
                patch_level_start = patch_level_center - [(patch_level_shape[0] - 1) // 2, (patch_level_shape[1] - 1) // 2]

                patch_dict[spacing]['patches'][sample_index] = self.__loader.read(spacing=spacing,
                                                                                  row=patch_level_start[0],
                                                                                  col=patch_level_start[1],
                                                                                  height=patch_level_shape[0],
                                                                                  width=patch_level_shape[1])

            # Extract or synthesize patches of the mask.
            #
            for spacing in shapes:
                level = self.__loader.level(spacing=spacing)
                level_shape = shapes[spacing]

                if self.__label_mode == 'load':
                    # Load label patch from the mask image. First convert the coordinate from the level of the statistics (matching level) back to the target level
                    # of the image then convert both the coordinate and the patch shape from the target level of the image to the actual mask image level to use.
                    #
                    load_level = self.__mask_level[level]
                    load_level_center_image = np.multiply(central_coordinate, self.__stats_downsamplings[level]).astype(np.int32) + self.__stats_shifts[level]
                    load_level_shape_mask = np.multiply(level_shape, self.__mask_downsamplings[level]).astype(int)
                    load_level_center_mask = np.multiply(load_level_center_image, self.__mask_downsamplings[level]).astype(int) + self.__mask_shifts[level]
                    load_level_start_mask = np.subtract(load_level_center_mask, [(load_level_shape_mask[0] - 1) // 2, (load_level_shape_mask[1] - 1) // 2])

                    loaded_level_patch = self.__mask.read(spacing=self.__mask.spacings[load_level],
                                                          row=load_level_start_mask[0],
                                                          col=load_level_start_mask[1],
                                                          height=load_level_shape_mask[0],
                                                          width=load_level_shape_mask[1])

                    if np.array_equal(load_level_shape_mask, level_shape):
                        patch_dict[spacing]['labels'][sample_index] = loaded_level_patch.squeeze()
                    else:
                        patch_dict[spacing]['labels'][sample_index] = np.expand_dims(scipy.misc.imresize(arr=loaded_level_patch.squeeze(), size=level_shape, interp='nearest'), axis=0)

                elif self.__label_mode == 'synthesize':
                    # Extract label from the mask construct and zoom it to the appropriate size.
                    #
                    synthesize_level_shape = np.divide(level_shape, self.__stats_downsamplings[level]).astype(int)
                    synthesize_level_start = central_coordinate - [(synthesize_level_shape[0] - 1) // 2, (synthesize_level_shape[1] - 1) // 2]

                    # Construct the mask patch and resample it to the appropriate size. Since these are label values no other interpolation can be used than nearest neighbour.
                    #
                    synthesized_label_patch = self.__stats.construct(row=synthesize_level_start[0],
                                                                     col=synthesize_level_start[1],
                                                                     height=synthesize_level_shape[0],
                                                                     width=synthesize_level_shape[1])

                    patch_dict[spacing]['labels'][sample_index] = np.expand_dims(scipy.misc.imresize(arr=synthesized_label_patch, size=level_shape, interp='nearest'), axis=0)

                elif self.__label_mode == 'central':
                    # Just store the label of the central pixel.
                    #
                    patch_dict[spacing]['labels'][sample_index] = patch_coordinates[sample_index, 2]

        # Return the constructed patch, label collection.
        #
        return patch_dict
