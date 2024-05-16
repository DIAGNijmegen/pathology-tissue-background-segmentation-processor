"""
This file contains class for collecting and handling data from mask mr-image.
"""

from ..errors import configerrors as dptconfigerrors
from ..errors import labelerrors as dptlabelerrors
from ..errors import staterrors as dptstaterrors
from ..utils import serialize as dptserialize
from ..image import imagereader as dptimagereader

import numpy as np
import os

#----------------------------------------------------------------------------------------------------

class MaskStats(object):
    """This class can collect and handle statistics from mask mr-image."""

    def __init__(self, file, mask_spacing=None, spacing_tolerance=None, mask_labels=None):
        """
        Initialize the object and extract and store necessary mask data in memory for efficient patch extraction.

        Args:
            file (str, dptimagereader.ImageReader): Mask mr-image or stat file path. Files with '.stat' extensions are treated as stat file, everything else as mask mr-image file.
            mask_spacing (float, None): Pixel spacing of the mask to process (micrometer).
            spacing_tolerance (float, None): Pixel spacing tolerance (percentage).
            mask_labels (list, None): List of mask labels to use. All other labels will be considered as non-labeled area.

        Raises:
            MissingStatInputPathError: Missing mask or stat input file path.
            InvalidPixelSpacingToleranceError: The pixel spacing tolerance is not valid.
            StatReadingError: The specified stat cannot be opened for reading.
            MissingDataKeyError: Not all the mandatory keys are present in the data loaded from the stat file.
            MaskPixelSpacingMismatchError: The loaded mask pixel spacing does not match the configured mask pixel spacing.
            LabelListMismatchError: The loaded label list does not match the configured label list.
            InconsistentDataError: Inconsistent stat data structure.
            UnknownMaskInputFormatError: The format of the mask source is unknown.
            EmptyLabelListError: The list of labels to process is empty.
            InvalidLabelValueError: The list of labels contains value outside of [1, 255] interval.

            DigitalPathologyConfigError: Configuration errors.
            DigitalPathologyImageError: Image errors.
        """

        # Initialize the base class.
        #
        super().__init__()

        # Initialize members.
        #
        self.__mask_path = ''           # Path of the mask mr-image.
        self.__mask_hash = ''           # SHA256 hash digest of the mask file.
        self.__mask_spacing = 0.0       # Pixel spacing of the mask file at the processed level.
        self.__mask_shape = (0, 0)      # Shape of the mask at the processed level.
        self.__label_list = []          # List of labels that are processed.
        self.__label_map = None         # Label value to index map for all values.
        self.__invalid = 0              # Invalid label value.
        self.__tile_cache = []          # Stored tile cache.
        self.__tile_stats = None        # Collected tile statistics: label counts, cache index for each tile.
        self.__tile_row_counts = None   # Sum of pixels per label per tile row.
        self.__tile_size = 512          # Used tile size.
        self.__tile_length = 512 * 512  # Pre-calculated length of tile data.

        # Collect data from mr-image.
        #
        self.__initdata(file=file, mask_spacing=mask_spacing, spacing_tolerance=spacing_tolerance, mask_labels=mask_labels)

    def __initdata(self, file, mask_spacing, spacing_tolerance, mask_labels):
        """
        Initialize by either processing the mask file or loading the pre-processed information from a stat file.

        Args:
            file (str, dptimagereader.ImageReader): Mask mr-image or stat file path, or opened ImageReader object.
            mask_spacing (float, None): Pixel spacing of the mask to process (micrometer).
            spacing_tolerance (float, None): Pixel spacing tolerance (percentage).
            mask_labels (list, None): List of mask labels to use. All other labels will be considered as non-labeled area.

        Raises:
            MissingStatInputPathError: Missing mask or stat input file path.
            InvalidPixelSpacingToleranceError: The pixel spacing tolerance is not valid.
            MissingDataKeyError: Not all the mandatory keys are present in the data loaded from the stat file.
            MaskPixelSpacingMismatchError: The loaded mask pixel spacing does not match the configured mask pixel spacing.
            LabelListMismatchError: The loaded label list does not match the configured label list.
            InconsistentDataError: Inconsistent stat data structure.
            UnknownMaskInputFormatError: The format of the mask source is unknown.
            EmptyLabelListError: The list of labels to process is empty.
            InvalidLabelValueError: The list of labels contains value outside of [1, 255] interval.

            DigitalPathologyConfigError: Configuration errors.
            DigitalPathologyImageError: Image errors.
        """

        # The file path must be valid.
        #
        if file is None:
            raise dptstaterrors.MissingStatInputPathError()

        # Check if the given file is a stat file or a mask file.
        #
        if type(file) is str and os.path.splitext(file)[1].lower() == '.stat':
            # Load the pre-processed data from file.
            #
            self.__loadstat(file_path=file, mask_spacing=mask_spacing, spacing_tolerance=spacing_tolerance, mask_labels=mask_labels)
        else:
            # Build the internal data representation from the mask file.
            #
            self.__loadmask(mask=file, mask_spacing=mask_spacing, spacing_tolerance=spacing_tolerance, mask_labels=mask_labels)

    def __loadstat(self, file_path, mask_spacing, spacing_tolerance, mask_labels):
        """
        Load pre-processed statistics from file.

        Args:
            file_path (str): Target file path.
            mask_spacing (float, None): Pixel spacing of the mask to process (micrometer).
            spacing_tolerance (float, None): Pixel spacing tolerance (percentage).
            mask_labels (list, None): List of mask labels to use. All other labels will be considered as non-labeled area.

        Raises:
            InvalidPixelSpacingToleranceError: The pixel spacing tolerance is not valid.
            StatReadingError: The specified stat cannot be opened for reading.
            MissingDataKeyError: Not all the mandatory keys are present in the data loaded from the stat file.
            MaskPixelSpacingMismatchError: The loaded mask pixel spacing does not match the configured mask pixel spacing.
            LabelListMismatchError: The loaded label list does not match the configured label list.
            InconsistentDataError: Inconsistent stat data structure.
        """

        # The spacing tolerance must be non-negative if the pixel spacing is used.
        #
        if mask_spacing is not None and (spacing_tolerance is None or spacing_tolerance < 0.0):
            raise dptconfigerrors.InvalidPixelSpacingToleranceError(spacing_tolerance)

        # Load the data from file.
        #
        load_data = dptserialize.load_object(path=file_path)

        # Check if the loading was successful.
        #
        if load_data is None:
            raise dptstaterrors.StatReadingError(file_path)

        # Check the loaded data consistency (somewhat).
        #
        mandatory_keys = ['mask_path', 'mask_hash', 'mask_spacing', 'mask_shape', 'label_list', 'label_map', 'invalid', 'tile_cache', 'tile_stats', 'tile_row_counts', 'tile_size', 'tile_length']
        if type(load_data) != dict or not all(key in load_data for key in mandatory_keys):
            raise dptstaterrors.MissingDataKeyError(list(load_data.keys()) if type(load_data) == dict else [], mandatory_keys)

        # Check the configured parameters against the loaded ones.
        #
        if mask_spacing is not None and spacing_tolerance < abs(mask_spacing - load_data['mask_spacing']):
            raise dptstaterrors.MaskPixelSpacingMismatchError(load_data['mask_spacing'], mask_spacing)

        if mask_labels is not None and mask_labels != load_data['label_list']:
            raise dptstaterrors.LabelListMismatchError(load_data['label_list'], mask_labels)

        # Fill in the member variable values from file.
        #
        try:
            self.__mask_path = load_data['mask_path']
            self.__mask_hash = load_data['mask_hash']
            self.__mask_spacing = load_data['mask_spacing']
            self.__mask_shape = load_data['mask_shape']
            self.__label_list = load_data['label_list']
            self.__label_map = dptserialize.reconstruct_ndarray(content=load_data['label_map'])
            self.__invalid = load_data['invalid']
            self.__tile_cache = list(tuple(dptserialize.reconstruct_ndarray(content=tuple_item) for tuple_item in cache_item) if cache_item else None for cache_item in load_data['tile_cache'])
            self.__tile_stats = dptserialize.reconstruct_ndarray(content=load_data['tile_stats'])
            self.__tile_row_counts = dptserialize.reconstruct_ndarray(content=load_data['tile_row_counts'])
            self.__tile_size = load_data['tile_size']
            self.__tile_length = load_data['tile_length']

        except:
            # Something is wrong in the loaded data structure.
            #
            raise dptstaterrors.InconsistentDataError(file_path)

    def __loadmask(self, mask, mask_spacing, spacing_tolerance, mask_labels):
        """
        Extract and store necessary mask data in memory for efficient patch extraction.

        Args:
            mask (str, dptimagereader.ImageReader): Mask mr-image path, or opened ImageReader object.
            mask_spacing (float, None): Pixel spacing of the mask to process (micrometer).
            spacing_tolerance (float, None): Pixel spacing tolerance (percentage).
            mask_labels (list, None): List of mask labels to use. All other labels will be considered as non-labeled area.

        Raises:
            UnknownMaskInputFormatError: The format of the mask source is unknown.
            EmptyLabelListError: The list of labels to process is empty.
            InvalidLabelValueError: The list of labels contains value outside of [1, 255] interval.

            DigitalPathologyConfigError: Configuration errors.
            DigitalPathologyImageError: Image errors.
        """

        # Store the fingerprint of the mask image file.
        #
        if type(mask) is str:
            mask_image = dptimagereader.ImageReader(image_path=mask, spacing_tolerance=spacing_tolerance, input_channels=[0], cache_path=None)
        elif isinstance(mask, dptimagereader.ImageReader):
            mask_image = mask
        else:
            raise dptstaterrors.UnknownMaskInputFormatError()

        # Save the source path and the hash of the file.
        #
        self.__mask_path = mask_image.path
        self.__mask_hash = mask_image.hash()

        # Check and store the label list. The list must not be empty and must be all positive.
        #
        if not mask_labels:
            raise dptlabelerrors.EmptyLabelListError()

        if min(mask_labels) < 0 or np.iinfo(np.uint8).max <= max(mask_labels):
            raise dptlabelerrors.InvalidLabelValueError(mask_labels)

        # The spacing tolerance must be non-negative if the pixel spacing is used.
        #
        if spacing_tolerance is None or spacing_tolerance < 0.0:
            raise dptconfigerrors.InvalidPixelSpacingToleranceError(spacing_tolerance)

        # Find the appropriate mask level based on the pixel spacing.
        #
        mask_level = mask_image.level(spacing=mask_spacing)

        self.__label_list = mask_labels
        self.__invalid = len(mask_labels)
        self.__mask_spacing = mask_image.spacings[mask_level]
        self.__mask_shape = mask_image.shapes[mask_level]

        # Shape the tile cache.
        #
        label_count = len(self.__label_list)
        bins_array_len = max(self.__label_list) + 1
        tile_stats_shape = (self.__mask_shape[0] // self.__tile_size + min(1, self.__mask_shape[0] % self.__tile_size),
                            self.__mask_shape[1] // self.__tile_size + min(1, self.__mask_shape[1] % self.__tile_size),
                            label_count + 1)

        self.__tile_stats = np.zeros(tile_stats_shape, dtype=np.int32)
        tile_row_counts_shape = (tile_stats_shape[0], label_count)
        self.__tile_row_counts = np.zeros(tile_row_counts_shape, dtype=np.int64)
        self.__tile_cache = [None] * label_count

        # Prepare the bins for collecting all possible values.
        #
        self.__label_map = np.full((np.iinfo(np.uint8).max + 1,), fill_value=self.__invalid, dtype=np.uint8)
        for label_index in range(len(self.__label_list)):
            self.__label_map[self.__label_list[label_index]] = label_index

        # Collect data.
        #
        for y in range(0, tile_stats_shape[0]):
            mask_load_y = y * self.__tile_size

            for x in range(0, tile_stats_shape[1]):
                # Extract patch from image file.
                #
                mask_load_x = x * self.__tile_size
                mask_tile = mask_image.read(spacing=mask_spacing, row=mask_load_y, col=mask_load_x, height=self.__tile_size, width=self.__tile_size).squeeze()

                # Collect label counts from patch.
                #
                label_bins = np.bincount(mask_tile.reshape(-1), minlength=bins_array_len)
                self.__tile_stats[y, x, 0:label_count] = [label_bins[label_value] for label_value in self.__label_list]
                self.__tile_row_counts[y] += self.__tile_stats[y, x, 0:label_count]

                # Store patch if it is not a clean patch.
                #
                max_count = np.max(self.__tile_stats[y, x])
                if max_count == 0:
                    self.__tile_stats[y, x, -1] = -1
                elif max_count == self.__tile_length:
                    self.__tile_stats[y, x, -1] = np.argmax(self.__tile_stats[y, x])
                else:
                    # Calculate jump list inside the tile.
                    #
                    in_tile_row_counts = np.empty((self.__tile_size, label_count), dtype=np.int32)
                    for tile_y in range(0, mask_tile.shape[0]):
                        tile_label_bins = np.bincount(mask_tile[tile_y, :], minlength=bins_array_len)
                        in_tile_row_counts[tile_y] = [tile_label_bins[label_value] for label_value in self.__label_list]
                        if 0 < tile_y:
                            in_tile_row_counts[tile_y] += in_tile_row_counts[tile_y - 1]

                    self.__tile_stats[y, x, -1] = len(self.__tile_cache)
                    self.__tile_cache.append((in_tile_row_counts, mask_tile))

                # Convert the per tile label sums to row sums for fast searching.
                #
                if 0 < x:
                    self.__tile_stats[y, x, 0:label_count] += self.__tile_stats[y, x - 1, 0:label_count]

            if 0 < y:
                self.__tile_row_counts[y] += self.__tile_row_counts[y - 1]

        # Close the opened image.
        #
        if not isinstance(mask, dptimagereader.ImageReader):
            mask_image.close()

    def save(self, file_path):
        """
        Save the processed mask statistics to file.

        Args:
            file_path (str): Target file path.
        """

        # Construct the data to be saved.
        #
        save_data = {'mask_path': self.__mask_path,
                     'mask_hash': self.__mask_hash,
                     'mask_spacing': self.__mask_spacing,
                     'mask_shape': self.__mask_shape,
                     'label_list': self.__label_list,
                     'label_map': dptserialize.serialize_ndarray(array=self.__label_map),
                     'invalid': self.__invalid,
                     'tile_cache': list(tuple(dptserialize.serialize_ndarray(array=tuple_item) for tuple_item in cache_item) if cache_item else None for cache_item in self.__tile_cache),
                     'tile_stats': dptserialize.serialize_ndarray(array=self.__tile_stats),
                     'tile_row_counts': dptserialize.serialize_ndarray(array=self.__tile_row_counts),
                     'tile_size': self.__tile_size,
                     'tile_length': self.__tile_length}

        # Since the data structure only contains built-in python objects it is safe to save it.
        #
        dptserialize.save_object(content=save_data, path=file_path)

    @property
    def path(self):
        """
        Get the path of the source dump file or mask mr-image.

        Returns:
            str: Path of the source dump file or mask mr-image.
        """

        return self.__mask_path

    @property
    def hash(self):
        """
        Get the SHA256 hash digest of the mask file.

        Returns:
            str: SHA256 hash digest of the mask file.
        """

        return self.__mask_hash

    @property
    def spacing(self):
        """
        Get the pixel spacing of te mask at the processed level.

        Returns:
            float: Pixel spacing of the mask.
        """

        return self.__mask_spacing

    @property
    def shape(self):
        """
        Get the shape of the mask at the processed level.

        Returns:
            tuple: Shape of the mask at the processed level.
        """

        return self.__mask_shape

    @property
    def labels(self):
        """
        Get the list of valid (non empty) labels.

        Returns:
            list: List of valid labels.
        """

        pixel_counts = self.__tile_row_counts[-1, :].tolist()
        return [self.__label_list[label_index] for label_index in range(len(self.__label_list)) if 0 < pixel_counts[label_index]]

    @property
    def counts(self):
        """
        Get the number of pixels per label.

        Returns:
            dict: Per label pixel counts.
        """

        pixel_counts = self.__tile_row_counts[-1, :].tolist()
        return {self.__label_list[label_index]: pixel_counts[label_index] for label_index in range(len(self.__label_list)) if 0 < pixel_counts[label_index]}

    def indextocoorindate(self, index_array, label):
        """
        Convert labeled pixel indices to pixel coordinates for the given label.

        Args:
            index_array (np.ndarray): Array of labeled pixel indices.
            label (int): Label value.

        Returns:
            np.ndarray: Array of pixel (row, col, label) coordinates.

        Raises:
            UnknownLabelError: The label is not in the list of processed labels.
        """

        # The label index must be in the valid [0, label count) range.
        #
        if label not in self.__label_list:
            raise dptlabelerrors.UnknownLabelError(label, self.__label_list)

        # Sort the indexes for more efficient extraction.
        #
        index_array.sort()

        # Convert the indices to (row, col, label) coordinates.
        #
        coordinate_array = np.empty((len(index_array), 3), dtype=np.int32)
        label_index = self.__label_list.index(label)
        for sample_index in range(len(index_array)):
            # Identify tile row and update pixel index relative to the row.
            #
            pixel_index = index_array[sample_index]
            tile_array_row_index = np.searchsorted(self.__tile_row_counts[:, label_index], pixel_index, side='right')
            if 0 < tile_array_row_index:
                pixel_index -= self.__tile_row_counts[tile_array_row_index - 1, label_index]

            # Identify tile inside the row and update pixel index relative to the tile start.
            #
            tile_array_col_index = np.searchsorted(self.__tile_stats[tile_array_row_index, :, label_index], pixel_index, side='right')
            if 0 < tile_array_col_index:
                pixel_index -= self.__tile_stats[tile_array_row_index, tile_array_col_index - 1, label_index]
            tile_cache_index = self.__tile_stats[tile_array_row_index, tile_array_col_index, -1]

            # Look for the pixel_index-th occurrence of the given label.
            #
            if tile_cache_index < len(self.__label_list):
                in_tile_row_index = pixel_index // self.__tile_size
                in_tile_col_index = pixel_index % self.__tile_size
            else:
                containing_tile_row_counts, containing_tile = self.__tile_cache[tile_cache_index]
                in_tile_row_index = np.searchsorted(containing_tile_row_counts[:, label_index], pixel_index, side='right')
                if 0 < in_tile_row_index:
                    pixel_index -= containing_tile_row_counts[in_tile_row_index - 1, label_index]

                in_tile_col_index = -1
                containing_tile_row = containing_tile[in_tile_row_index]
                while 0 <= pixel_index:
                    in_tile_col_index += 1
                    if containing_tile_row[in_tile_col_index] == label:
                        pixel_index -= 1

            # Convert and store the result index.
            #
            coordinate_array[sample_index, :] = (tile_array_row_index * self.__tile_size + in_tile_row_index, tile_array_col_index * self.__tile_size + in_tile_col_index, label)

        return coordinate_array

    def construct(self, row, col, height, width):
        """
        Construct a label patch from the stored statistics.

        Args:
            row (int): Row index of upper left pixel.
            col (int): Col index of upper left pixel.
            height (int): Height of patch.
            width (int): Width of patch.

        Returns:
            np.ndarray: Array of constructed label patch with dtype=np.uint8.
        """

        # Prepare result.
        #
        mask_patch = np.empty((height, width), dtype=np.uint8)

        # Go through the patch by tiles and fill in the overlapping areas.
        #
        patch_row_index = 0
        while patch_row_index < height:
            # Calculate row index data.
            #
            global_row_index = row + patch_row_index                                              # Index if the current overlap row start index in the whole image.
            stats_row_index = global_row_index // self.__tile_size                                # Row index of the target tile in the tile array.
            tile_row_index = global_row_index % self.__tile_size                                  # Overlap start row index inside the target tile
            overlap_row_count = min(self.__tile_size - tile_row_index, height - patch_row_index)  # Overlap row count.

            patch_col_index = 0
            while patch_col_index < width:
                # Calculate column index data.
                #
                global_col_index = col + patch_col_index                                             # Index if the current overlap row start index in the whole image.
                stats_col_index = global_col_index // self.__tile_size                               # Column index of the target tile in the tile array.
                tile_col_index = global_col_index % self.__tile_size                                 # Overlap start column index inside the target tile
                overlap_col_count = min(self.__tile_size - tile_col_index, width - patch_col_index)  # Overlap column count.

                # Select the tile.
                #
                tile_cache_index = self.__tile_stats[stats_row_index, stats_col_index, -1] if stats_row_index < self.__tile_stats.shape[0] and stats_col_index < self.__tile_stats.shape[1] else -1
                if tile_cache_index < 0:
                    mask_patch[patch_row_index: patch_row_index + overlap_row_count, patch_col_index: patch_col_index + overlap_col_count] = 0
                elif tile_cache_index < len(self.__label_list):
                    mask_patch[patch_row_index: patch_row_index + overlap_row_count, patch_col_index: patch_col_index + overlap_col_count] = self.__label_list[tile_cache_index]
                else:
                    mask_patch[patch_row_index: patch_row_index + overlap_row_count,
                               patch_col_index: patch_col_index + overlap_col_count] = self.__tile_cache[tile_cache_index][1][tile_row_index: tile_row_index + overlap_row_count,
                                                                                                                              tile_col_index: tile_col_index + overlap_col_count]

                patch_col_index += overlap_col_count
            patch_row_index += overlap_row_count

        # Return the constructed patch.
        #
        return mask_patch
