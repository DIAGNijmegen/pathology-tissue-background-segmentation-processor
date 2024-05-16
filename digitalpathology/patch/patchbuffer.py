"""
This file contains class for buffering patches of whole slide images.
"""

from ..errors import configerrors as dptconfigerrors
from ..errors import buffererrors as dptbuffererrors

import numpy as np
import threading

#----------------------------------------------------------------------------------------------------

class PatchBuffer(object):
    """This class is a buffer of patches from a collection of whole slide images."""

    def __init__(self, shapes, input_channels, label_maps, weight_maps, cache_size, chunk_size):
        """
        Initialize the object.

        Args:
            shapes (dict): Dictionary mapping pixel spacing to (rows, columns) patch shape.
            label_maps (bool): Buffer labels maps too, or just the label of the central pixel.
            weight_maps (bool): Buffer weight maps too.
            cache_size (int): Number of patches to keep in memory.
            chunk_size (int): Chunk size for complete buffer transfer.

        Raises:
            EmptyPixelSpacingListError: The list of pixel spacings is empty.
            InvalidPatchShapeError: The given patch shape is not valid.
            InvalidCacheSizeError: The number of cached patches is not valid.
            EmptyChannelListError: The list of channels is empty.
            InvalidBufferChunkSizeError: The buffer transfer chunk size is not valid.
        """

        # Initialize the base class.
        #
        super().__init__()

        # Initialize members.
        #
        self.__read_index = 0       # Start index of the next batch to read from the buffer.
        self.__write_index = 0      # Start index of the next batch to write to the buffer.
        self.__buffer_size = 0      # Number of stored patches per level of pixel spacing.
        self.__chunk_size = 0       # Buffer transfer chunk size.
        self.__weight_maps = False  # Buffer weight maps too.
        self.__buffer = {}          # Buffer of RGB patches and labels per level of pixel spacing.
        self.__errors = None        # Patch classification errors.
        self.__read_order = None    # Index order to read the buffer.
        self.__write_order = None   # Index order to write the buffer.
        self.__object_lock = None   # Object lock to ensure thread safety.

        # Process the configured parameters.
        #
        self.__allocbuffer(shapes=shapes, input_channels=input_channels, label_maps=label_maps, weight_maps=weight_maps, cache_size=cache_size, chunk_size=chunk_size)

        # Create lock.
        #
        self.__object_lock = threading.Lock()

    def __allocbuffer(self, shapes, input_channels, label_maps, weight_maps, cache_size, chunk_size):
        """
        Allocate buffer area.

        Args:
            shapes (dict): Dictionary mapping pixel spacings to (rows, columns) patch shape.
            label_maps (bool): Extract labels maps too, or just the label of the central pixel.
            weight_maps (bool): Buffer weight maps too.
            cache_size (int): Number of patches to keep in memory.
            chunk_size (int): Chunk size for complete buffer transfer.

        Raises:
            EmptyPixelSpacingListError: The list of pixel spacings is empty.
            InvalidPatchShapeError: The given patch shape is not valid.
            InvalidCacheSizeError: The number of cached batches is not valid.
            EmptyChannelListError: The list of channels is empty.
            InvalidBufferChunkSizeError: The buffer transfer chunk size is not valid.
        """

        # Check if the list of pixel spacings and input channels to use is not empty.
        #
        if not shapes:
            raise dptconfigerrors.EmptyPixelSpacingListError()

        if not input_channels:
            raise dptconfigerrors.EmptyChannelListError()

        # Check patch size: it must be positive.
        #
        if any(shape[0] <= 0 or shape[1] <= 0 for shape in shapes.values()):
            raise dptconfigerrors.InvalidPatchShapeError(shapes)

        # Check the number of cached batches: it must be positive.
        #
        if cache_size < 1:
            raise dptbuffererrors.InvalidCacheSizeError(cache_size)

        # Check if channel inputs list is not empty nor contains duplicates.
        #
        if not input_channels:
            raise dptconfigerrors.EmptyChannelListError()

        # Check chunk size: it must be positive.
        #
        if chunk_size <= 0:
            raise dptbuffererrors.InvalidBufferChunkSizeError(chunk_size)

        # Initialize buffer pointers.
        #
        self.__read_index = 0
        self.__write_index = 0
        self.__buffer_size = cache_size
        self.__chunk_size = chunk_size
        self.__weight_maps = weight_maps

        # Allocate memory.
        #
        self.__buffer = {spacing: {} for spacing in shapes}
        for spacing in shapes:
            self.__buffer[spacing]['patches'] = np.zeros(shape=(cache_size, len(input_channels)) + shapes[spacing], dtype=np.uint8)
            self.__buffer[spacing]['labels'] = np.zeros(shape=(cache_size, 1) + shapes[spacing], dtype=np.uint8) if label_maps else np.zeros(shape=(cache_size,), dtype=np.uint8)

            if weight_maps:
                self.__buffer[spacing]['weights'] = np.zeros(shape=(cache_size, 1) + shapes[spacing], dtype=np.float32)

        # Initialize classification errors and read order.
        #
        self.__errors = np.zeros(self.__buffer_size, dtype=np.float64)
        self.__read_order = np.arange(self.__buffer_size)

        # Initialize write order.
        #
        self.__write_order = None

    def __chunksizes(self, count):
        """
        Calculate the chunk size list for transferring from another buffer to this.

        Args:
            count (int): Number of patches to transfer.

        Returns:
            list: Transfer chunk size list.
        """

        # Calculate chunk list.
        #
        chunk_sizes = [self.__chunk_size] * (count // self.__chunk_size)
        remainder = count % self.__chunk_size
        if remainder:
            chunk_sizes.append(remainder)

        return chunk_sizes

    @property
    def size(self):
        """
        Get the size of the buffer: the number of RGB patches it can hold.

        Returns:
            The number of RGB patches it can hold.
        """

        return self.__buffer_size if self.__buffer is not None else 0

    def histogram(self, bins):
        """
        Calculate the histogram of the errors.

        Args:
            bins (int): Number of bins to use.

        Returns:
            np.ndarray, np.ndarray: Histogram array and the bin maxes.
        """

        counts, edges = np.histogram(self.__errors, bins=bins, range=(0.0, 1.0))
        return counts, edges[1:]

    def shuffle(self):
        """Shuffle the buffer read order."""

        # Shuffle the content.
        #
        np.random.shuffle(self.__read_order)

        # Reset the read index.
        #
        self.__read_index = 0

    def batch(self, batch_size):
        """
        Get a batch from the buffer with the configured size.

        Args:
            batch_size (int): Number of patches to put in the batch.

        Returns:
            dict, np.ndarray: Dictionary of patches and labels per pixel spacing, and the array of indices for classification error update.

        Raises:
            InvalidBatchSizeError: The requested batch size is not valid.
            BatchSizeLargerThanCacheError: The batch size is larger than the number of cached patches.
        """

        # Check batch size it have to be at least 1 and not larger than the number of cached patches.
        #
        if batch_size < 1:
            raise dptconfigerrors.InvalidBatchSizeError(batch_size)

        if self.__buffer_size < batch_size:
            raise dptbuffererrors.BatchSizeLargerThanCacheError(batch_size, self.__buffer_size)

        # Acquire lock to ensure thread safety.
        #
        self.__object_lock.acquire()

        # Return a collection of patches according to the mode.
        #
        indices = np.arange(start=self.__read_index, stop=self.__read_index + batch_size, step=1)
        indices %= self.__buffer_size
        indices = self.__read_order[indices]
        self.__read_index = (self.__read_index + batch_size) % self.__buffer_size

        # Generate result.
        #
        if self.__weight_maps:
            patches = {spacing: {'patches': self.__buffer[spacing]['patches'][indices],
                                 'labels': self.__buffer[spacing]['labels'][indices],
                                 'weights': self.__buffer[spacing]['weights'][indices]} for spacing in self.__buffer}
        else:
            patches = {spacing: {'patches': self.__buffer[spacing]['patches'][indices],
                                 'labels': self.__buffer[spacing]['labels'][indices]} for spacing in self.__buffer}

        # Release lock.
        #
        self.__object_lock.release()

        return patches, indices

    def count(self, threshold):
        """
        Count the number of patches in the buffer with larger than the given classification error threshold.

        Args:
            threshold (float): Error threshold.

        Returns:
            int: number of patches with larger than the given classification error threshold.
        """

        return np.count_nonzero(self.__errors > threshold)

    def update(self, indices, errors):
        """
        Update the classification errors on the given indices. All errors must be in the [0.0, 1.0] range.

        Args:
            indices (np.ndarray): Indices of classification errors to update.
            errors (np.ndarray): Network classification errors.

        Raises:
            InvalidClassificationErrorError: If a classification error is out of the [0.0, 1.0] interval.
        """

        # Check errors values. They must be in [0.0, 1.0].
        #
        if errors.min() < 0.0 or 1.0 < errors.max():
            raise dptbuffererrors.InvalidClassificationErrorError(errors.min() if errors.min() < 0.0 else errors.max())

        # Update classification errors.
        #
        self.__errors[indices] = errors

    def partition(self, threshold=0.0):
        """
        Create write order by partitioning the patches into lower and higher-or-equal classification errors. The all subsequent push calls will work in the
        calculated order and overwrite the patches with lower than the threshold classification errors.

        Args:
            threshold (float): Threshold for partitioning.
        """

        if 0.0 < threshold:
            le_flags = self.__errors < threshold
            less_indices = np.flatnonzero(le_flags)
            ge_indices = np.flatnonzero(np.logical_not(le_flags))

            np.random.shuffle(less_indices)
            np.random.shuffle(ge_indices)

            self.__write_order = np.concatenate((less_indices, ge_indices), axis=0)
        else:
            self.__write_order = None

        # Reset the write index.
        #
        self.__write_index = 0

    def push(self, patches):
        """
        Push a collection of patches to the buffer. The patches are overwritten in a random order.

        Args:
            patches (dict): dictionary {spacing: {'patches': patch array, 'labels': label array} with the patch and label arrays. These arrays are both np.array types.

        Raises:
            SpacingMismatchError: The pushed pixel spacings do not match the buffered pixel spacings.
            LabelCountMismatchError: The number of patches and labels does not match.
            BufferOverflowError: The number of patches is larger than the size of the buffer.
            MissingWeightsError: Weights are missing from the pushed patches.
        """

        # Check if the pushed pixel spacings are the actually buffered pixel spacings.
        #
        if patches.keys() != self.__buffer.keys():
            raise dptbuffererrors.SpacingMismatchError(tuple(patches.keys()), tuple(self.__buffer.keys()))

        # Check data consistency: the number of patches and labels must be the same and less than the buffer size.
        #
        for spacing in patches:
            if patches[spacing]['patches'].shape[0] != patches[spacing]['labels'].shape[0]:
                raise dptbuffererrors.LabelCountMismatchError(spacing, patches[spacing]['patches'].shape[0], patches[spacing]['labels'].shape[0])

            if self.__buffer_size < patches[spacing]['patches'].shape[0]:
                raise dptbuffererrors.BufferOverflowError(spacing, self.__buffer_size, patches[spacing]['patches'].shape[0])

            if self.__weight_maps and 'weights' not in patches[spacing]:
                raise dptbuffererrors.MissingWeightsError(spacing)

        # Acquire lock to ensure thread safety.
        #
        self.__object_lock.acquire()

        # Check if the whole buffer is updated. If it is, take a shortcut and just copy the content of the source.
        #
        if next(iter(patches.values()))['patches'].shape[0] == self.__buffer_size:
            # The whole buffer is overwritten.
            #
            for spacing in self.__buffer:
                self.__buffer[spacing]['patches'][:] = patches[spacing]['patches']
                self.__buffer[spacing]['labels'][:] = patches[spacing]['labels']

                if self.__weight_maps:
                    self.__buffer[spacing]['weights'][:] = patches[spacing]['weights']

            # Set all classification errors to the default 1.0.
            #
            self.__errors[:] = 1.0
        else:
            # Store the collection of patches.
            #
            patch_count = next(iter(patches.values()))['patches'].shape[0]
            indices = np.arange(start=self.__write_index, stop=self.__write_index + patch_count, step=1)
            indices %= self.__buffer_size

            if self.__write_order is not None:
                indices = self.__write_order[indices]

            self.__write_index = (self.__write_index + patch_count) % self.__buffer_size

            # Write the data.
            #
            for spacing in self.__buffer:
                self.__buffer[spacing]['patches'][indices] = patches[spacing]['patches']
                self.__buffer[spacing]['labels'][indices] = patches[spacing]['labels']

                if self.__weight_maps:
                    self.__buffer[spacing]['weights'][indices] = patches[spacing]['weights']

            # Initialize the new classification errors to the default 1.0.
            #
            self.__errors[indices] = 1.0

        # Release lock.
        #
        self.__object_lock.release()

    def copy(self, other, count=0, threshold=0.0):
        """
        Copy given number of patches from the other buffer. The threshold is used for selecting patches to be skipped in this buffer
        if the number of patches to transfer is smaller than the size of this buffer.

        Args:
            other (PatchBuffer): Buffer to copy.
            count (int): Number of patches to transfer.
            threshold (float): Error threshold. If the number of transferred patches are smaller than the size of this buffer the patches
                with larger than threshold error are selected to be kept.

        Raises:
            SpacingMismatchError: The pushed pixel spacings do not match the buffered pixel spacings.
            LabelCountMismatchError: The number of patches and labels does not match.
            BufferOverflowError: The number of patches is larger than the size of the buffer.
            InvalidBatchSizeError: The requested batch size is not valid.
            BatchSizeLargerThanCacheError: The batch size is larger than the number of cached patches.
        """

        # Check the number of available patches.
        #
        if other.size < count:
            raise dptbuffererrors.BatchSizeLargerThanCacheError(count, other.__buffer_size)

        # Check if the whole other buffer is transferred.
        #
        if count == 0 or count == other.__buffer_size:
            # Acquire lock to ensure thread safety.
            #
            other.__object_lock.acquire()

            # Transfer the whole other buffer at once.
            #
            self.push(patches=other.__buffer)

            # Release lock.
            #
            other.__object_lock.release()
        else:
            # Partition the buffer for writing: the patches with lover than the threshold classification error will be overwritten first.
            #
            self.partition(threshold=threshold)

            # Transfer the other buffer in chunks.
            #
            chunk_sizes = self.__chunksizes(count=count)
            for size in chunk_sizes:
                patches, _ = other.batch(batch_size=size)
                self.push(patches=patches)

            # Clear the partitioning.
            #
            self.partition(threshold=0.0)
