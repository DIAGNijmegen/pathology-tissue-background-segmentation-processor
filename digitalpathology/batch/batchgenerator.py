"""
This file contains class for generating batches and buffering and augmenting them.
"""

from . import batchsampler as dptbatchsampler
from . import batchsamplerdaemon as dptbatchsamplerdaemon

from ..errors import labelerrors as dptlabelerrors
from ..errors import configerrors as dptconfigerrors
from ..errors import processerrors as dptprocesserrors
from ..errors import weighterrors as dptweighterrors
from ..patch import patchbuffer as dptpatchbuffer

import logging
import numpy as np
import queue
import sys
import threading
import math
import os

#----------------------------------------------------------------------------------------------------

class BatchGenerator(object):
    """
    This class is a batch generator class that extracts patches from a collection of whole slide images, buffers and augments them. This class supports generating patches at multiple levels of pixel
    spacing.
    """

    def __init__(self,
                 label_dist,
                 patch_shapes,
                 mask_spacing,
                 spacing_tolerance,
                 input_channels,
                 dimension_order,
                 label_mode,
                 patch_sources,
                 category_dist,
                 strict_selection,
                 create_stats,
                 main_buffer_size,
                 buffer_chunk_size=sys.maxsize,
                 read_buffer_size=0,
                 labels_one_hot=True,
                 batch_normalizer=None,
                 patch_augmenter=None,
                 label_mapper=None,
                 free_label_range=False,
                 patch_weight_mapper=None,
                 batch_weight_mapper=None,
                 multi_threaded=False,
                 sampler_process_count=0,
                 sampler_pool_size=sys.maxsize,
                 sampler_chunk_size=sys.maxsize,
                 join_timeout=60,
                 response_timeout=600,
                 poll_timeout=900,
                 name_tag=None):
        """
        Initialize the object.

        Args:
            label_dist (dict): Label sampling distribution mapping from label value to ratio in a single batch.
            patch_shapes (dict): Desired patch shapes (rows, cols) per pixel spacing.
            mask_spacing (float): Pixel spacing of the masks to process (micrometer).
            spacing_tolerance (float): Pixel spacing tolerance (percentage).
            input_channels (list): Desired channels that are extracted for each patch.
            dimension_order (str): Dimension order. 'BHWC' or 'BCHW' where b is batch, C is channels, H is height, and W is width.
            label_mode (str): Label generation mode. Accepted values:
                'central': Just return the label of the central pixel.
                'synthesize': Synthesize the label map from the mask statistics and zoom it to the appropriate level of pixel spacing.
                'load': Load the label map from the label image and zoom it to the appropriate level of pixel spacing if necessary.
            patch_sources (dict): Data source: map from image categories to PatchSource object sets.
            category_dist (dict): Image category sampling distribution mapping from image category to ratio in a single batch.
            strict_selection (bool): If true, every label that has higher than 0.0 ratio must be available in every source image selection.
            create_stats (bool): Allow missing stat files, and create them if necessary.
            main_buffer_size (int): Number of batches to keep in memory.
            buffer_chunk_size (int): Chunk size for complete buffer transfer.
            read_buffer_size (int): Double buffer mode. If larger than 0, all loaded patches will be accumulated here first and then transferred to the main patch
                buffer later with the transfer() command.
            labels_one_hot (bool): If set than labels will be converted to one hot encoding.
            batch_normalizer (batch.BatchNormalizer, None): Batch normalizer object.
            patch_augmenter (augmenters.AugmenterPool, None): Configured patch augmenter object.
            label_mapper (label.LabelMapper, None): Label value to label index mapper object.
            free_label_range (bool): If this flag is True, non-continuous label ranges, and ranges that do not start at 0 are also allowed.
            patch_weight_mapper (weight.WeightMapperBase, None): Patch based label weight map calculator object.
            batch_weight_mapper (weight.WeightMapperBase, None): Batch based label weight map calculator object.
            multi_threaded (bool): If true the batch sampler will be executed in a separate thread.
            sampler_process_count (int): Number of processes to use during extraction at once. 0 means that there will be no external worker processes.
            sampler_pool_size (int): Number of images to open or worker processes to spawn at once for extraction.
            sampler_chunk_size (int): Number of patches to read at once. If the number of patches is larger than the chunk size it will be re read in chunks.
            join_timeout (int): Seconds to wait for child processes to join.
            response_timeout (int): Seconds to wait for inter-process communication responses. 0 means no timeout.
            poll_timeout (int): Seconds a process waits for a message. Safeguard against hanging. 0 means no timeout.
            name_tag (str, None): Name tag of the generator for logging.

        Raises:
            InvalidDimensionOrderError: Invalid dimension order.
            LabelDistributionAndMappingMismatchError: Label distribution - label mapping key mismatch.
            InvalidLabelDistributionWithoutMappingError: The label distribution is not valid without label mapping.
            WeightMappingConfigurationError: Weight mapping is configured without label patch extraction.
            MissingLabelMapperForWeightMapperError: Weight mapping is configured without label mapping.
            WeightMapperLabelMapperClassesMismatchError: The network labels known by the label mapper and weight mapper does not match.
            EmptyPixelSpacingListError: The list of pixel spacings is empty.
            InvalidPixelSpacingInPatchShapesError: A pixel spacing is not valid.
            InvalidPatchShapeError: A patch shape is not valid.
            EmptyChannelListError: The list of channels is empty.
            DuplicateChannelError: There are multiples of a single channel in the channel list.
            InvalidTimeoutError: A timeout is not valid.
            InvalidSamplerChunkSizeError: Processing chunk size is not valid.

            DigitalPathologyBufferError: Buffer errors.
            DigitalPathologyConfigError: Configuration errors.
        """

        # Initialize the base class.
        #
        super().__init__()

        # Initialize members.
        #
        self.__logger = None               # Configured logger object.
        self.__name_tag = ''               # Name tag of the batch generator.
        self.__log_name_tag = ''           # Logging name tag of the batch generator for easy identification in logs.
        self.__label_count = 0             # Number of labels.
        self.__patch_shapes = {}           # Shape of extracted patches per pixel spacing.
        self.__mask_spacing = 0.0          # Mask pixel spacing to use.
        self.__input_channels = []         # List of channels used in a patch.
        self.__dimension_order = None      # Dimension order.
        self.__label_mode = ''             # Label generation mode.
        self.__batch_sampler = None        # Batch sampler.
        self.__command_queue = None        # Batch sampler command queue.
        self.__response_queue = None       # Batch sampler response queue.
        self.__main_patch_buffer = None    # Batch buffer dictionary.
        self.__read_patch_buffer = None    # Read buffer dictionary for double buffering.
        self.__worker_pids = []            # Last known list of worker process IDs sent back by the batch sampler.
        self.__one_hot_labels = False      # One hot encoding
        self.__batch_normalizer = None     # Batch normalizer object.
        self.__patch_weight_mapper = None  # Patch based label weight mapper object.
        self.__batch_weight_mapper = None  # Batch based label weight mapper object.
        self.__multi_threaded = False      # Batch sampler threading.
        self.__chunk_size = sys.maxsize    # Patch chunk size for sampling patches.
        self.__patch_augmenter = None      # Patch augmenter pool object.
        self.__label_dist = {}             # Label distribution ratio in a batch.
        self.__patch_sources = {}          # List of PatchSource objects per source category.
        self.__spacing_tolerance = 0.0     # Tolerance for finding a level for the given pixel spacing.
        self.__category_dist = {}          # Image category distribution ratio in a batch.
        self.__strict_selection = False    # Strict source item selection.
        self.__create_stats = False        # Create missing .stat files.
        self.__label_mapper = None         # Label value to label index mapper object.
        self.__free_label_range = False    # Free label mapping flag: no label range checking if True.
        self.__process_count = 0           # Number of worker processes to spawn.
        self.__pool_size = 0               # Number of images to use for extraction in a round.
        self.__response_timeout = 0        # Timeout seconds to wait for inter-thread communication responses.
        self.__join_timeout = 0            # Timeout seconds to wait for child processes to join.
        self.__poll_timeout = 0            # Timeout seconds to wait for inter-process communication requests.

        # Set instance name.
        #
        self.__setname(name_tag=name_tag)

        # Initialize logging.
        #
        self.__initlogging()

        # Process the configured parameters.
        #
        self.__setdimensionorder(dimension_order=dimension_order)
        self.__configurelabels(label_mapper=label_mapper, free_label_range=free_label_range, label_dist=label_dist, label_mode=label_mode, labels_one_hot=labels_one_hot)
        self.__setweightmappers(patch_weight_mapper=patch_weight_mapper, batch_weight_mapper=batch_weight_mapper)
        self.__createbuffer(main_buffer_size=main_buffer_size, read_buffer_size=read_buffer_size, patch_shapes=patch_shapes, input_channels=input_channels, chunk_size=buffer_chunk_size)

        self.__configuresampler(patch_sources=patch_sources,
                                mask_spacing=mask_spacing,
                                spacing_tolerance=spacing_tolerance,
                                category_dist=category_dist,
                                strict_selection=strict_selection,
                                create_stats=create_stats,
                                patch_augmenter=patch_augmenter,
                                process_count=sampler_process_count,
                                pool_size=sampler_pool_size,
                                join_timeout=join_timeout,
                                response_timeout=response_timeout,
                                poll_timeout=poll_timeout,
                                multi_threaded=multi_threaded,
                                chunk_size=sampler_chunk_size)

        self.__setbatchnormalizer(batch_normalizer=batch_normalizer)

    def __del__(self):
        """Destruct the object."""

        self.__stop()

    def __setname(self, name_tag):
        """
        Set instance name tag and name tag for logging.

        Args:
            name_tag (str, None): Name tag of the generator for logging.
        """

        # Configure the name tag.
        #
        self.__name_tag = name_tag if name_tag else ''
        self.__log_name_tag = '[{name}] '.format(name=name_tag) if name_tag else ''

    def __initlogging(self):
        """Initialize logging."""

        # Configure logging. This class relies on configured logging somewhere down on the hierarchy.
        #
        qualified_class_name = '{module_name}.{class_name}'.format(module_name=self.__class__.__module__, class_name=self.__class__.__name__)
        self.__logger = logging.getLogger(name=qualified_class_name)

        # Report own process and thread identifiers.
        #
        self.__logger.info('{tag}Batch generator initializing; process: {pid}; thread: {tid}'.format(tag=self.__log_name_tag, pid=os.getpid(), tid=threading.current_thread().ident))

    def __flushlogging(self):
        """Flush all log handlers."""

        for log_handler in self.__logger.handlers:
            log_handler.flush()

    def __setdimensionorder(self, dimension_order):
        """
        Set the dimension order.

        Args:
            dimension_order (str): Dimension order.

        Raises:
            InvalidDimensionOrderError: Invalid dimension order.
        """

        # Check if dimension order is correct.
        #
        lower_dimension_order = dimension_order.lower()
        if lower_dimension_order == 'bhwc' or lower_dimension_order == 'bchw':
            self.__dimension_order = lower_dimension_order
        else:
            self.__logger.error('Batch generator does not support \'{order}\' dimension order'.format(order=dimension_order))
            raise dptconfigerrors.InvalidDimensionOrderError(dimension_order)

    def __configurelabels(self, label_mapper, free_label_range, label_dist, label_mode, labels_one_hot):
        """
        Store the label configuration.

        Args:
            label_mapper (label.LabelMapper): Label mapper.
            free_label_range (bool): If this flag is True, non-continuous label ranges, and ranges that do not start at 0 are also allowed.
            label_dist (dict): Label sampling distribution mapping from label value to ratio in a single batch.
            label_mode (str): Label generation mode.
            labels_one_hot (bool): If set than labels will be converted to one hot encoding.

        Raises:
            LabelDistributionAndMappingMismatchError: Label distribution - label mapping key mismatch.
            InvalidLabelDistributionWithoutMappingError: The label distribution is not valid without label mapping.
        """

        if label_mapper is not None:
            # The label distribution keys must match the label mapper keys.
            #
            if not set(label_dist.keys()) == set(label_mapper.mapping):
                raise dptlabelerrors.LabelDistributionAndMappingMismatchError(label_dist, label_mapper.mapping)
        elif not free_label_range:
            # The label mapper can be invalid but then the extracted labels must be a continuous range starting from zero.
            #
            if not set(label_dist.keys()) == set(range(len(label_dist))):
                raise dptlabelerrors.InvalidLabelDistributionWithoutMappingError(label_dist)

        # Save the parameters.
        #
        self.__label_count = label_mapper.classes if label_mapper is not None else len(label_dist)
        self.__label_mode = label_mode
        self.__one_hot_labels = labels_one_hot
        self.__label_dist = label_dist
        self.__label_mapper = label_mapper
        self.__free_label_range = free_label_range

    def __setweightmappers(self, patch_weight_mapper, batch_weight_mapper):
        """
        Check and store the weight mapper objects.

        Args:
            patch_weight_mapper (weight.WeightMapperBase, None): Label weight map calculator object.

        Raises:
            WeightMappingConfigurationError: Weight mapping is configured without label patch extraction.
            MissingLabelMapperForWeightMapperError: Weight mapping is configured without label mapping.
            WeightMapperLabelMapperClassesMismatchError: The network labels known by the label mapper and weight mapper does not match.
        """

        if patch_weight_mapper is not None or batch_weight_mapper is not None:
            # Check the label extraction mode.
            #
            if self.__label_mode == 'central':
                raise dptweighterrors.WeightMappingConfigurationError(self.__label_mode)

            # Check if label mapper is configured. It is necessary for the valid pixel map generation.
            #
            if self.__label_mapper is None:
                raise dptweighterrors.MissingLabelMapperForWeightMapperError()

        if patch_weight_mapper is not None:
            # Check if the label mapper and the patch weight mapper has the same network labels.
            #
            if patch_weight_mapper.classes != self.__label_mapper.classes:
                raise dptweighterrors.WeightMapperLabelMapperClassesMismatchError(patch_weight_mapper.classes, self.__label_mapper.classes)

        if batch_weight_mapper is not None:
            # Check if the label mapper and the batch weight mapper has the same network labels.
            #
            if batch_weight_mapper.classes != self.__label_mapper.classes:
                raise dptweighterrors.WeightMapperLabelMapperClassesMismatchError(batch_weight_mapper.classes, self.__label_mapper.classes)

            # Check if the batch weight mapper is accompanied by a patch weight mapper.
            #
            if patch_weight_mapper is None:
                raise dptweighterrors.MissingPatchMapperWithBatchMapper()

        # Store the weight mapper.
        #
        self.__patch_weight_mapper = patch_weight_mapper
        self.__batch_weight_mapper = batch_weight_mapper

    def __createbuffer(self, main_buffer_size, read_buffer_size, patch_shapes, input_channels, chunk_size):
        """
        Initialize the batch buffer dictionaries.

        Args:
            main_buffer_size (int): Number of patches to keep in memory.
            read_buffer_size (int): Size of the read buffer. If larger than zero the double buffer mode is enabled.
            patch_shapes (dict): Desired patch shapes (rows, cols) per pixel spacing.
            input_channels (list) : Desired channels that are extracted for each patch.
            chunk_size (int): Chunk size for complete buffer transfer.

        Raises:
            EmptyPixelSpacingListError: The list of pixel spacings is empty.
            InvalidPixelSpacingInPatchShapesError: A pixel spacing is not valid.
            InvalidPatchShapeError: A patch shape is not valid.
            EmptyChannelListError: The list of channels is empty.
            DuplicateChannelError: There are multiples of a single channel in the channel list.

            DigitalPathologyBufferError: Buffer errors.
            DigitalPathologyConfigError: Configuration errors.
        """

        # The patch shapes cannot be empty.
        #
        if not patch_shapes:
            raise dptconfigerrors.EmptyPixelSpacingListError()

        if any(spacing <= 0.0 for spacing in patch_shapes):
            raise dptconfigerrors.InvalidPixelSpacingInPatchShapesError(patch_shapes)

        if any(shape[0] <= 0 or shape[1] <= 0 for shape in patch_shapes.values()):
            raise dptconfigerrors.InvalidPatchShapeError(patch_shapes)

        # Check if channel inputs list is not empty nor contains duplicates.
        #
        if not input_channels:
            raise dptconfigerrors.EmptyChannelListError()

        if len(input_channels) != len(set(input_channels)):
            raise dptconfigerrors.DuplicateChannelError(input_channels)

        # Save the patch shape.
        #
        self.__patch_shapes = {spacing: tuple(shape) for spacing, shape in patch_shapes.items()}

        # Save the input channels.
        #
        self.__input_channels = input_channels

        # Determine mode.
        #
        generate_label_maps = self.__label_mode != 'central'
        generate_weight_maps = self.__patch_weight_mapper is not None

        # Create batch buffer object.
        #
        self.__logger.info('{tag}Allocating batch buffers'.format(tag=self.__log_name_tag))

        main_log_message = '{tag}Main buffer: shapes: {shapes}; channels: {channels}; generate label maps: {maps}; generate weight maps: {weights}; cache size: {size}; chunk size: {chunk}'
        self.__logger.debug(main_log_message.format(tag=self.__log_name_tag,
                                                    shapes=self.__patch_shapes,
                                                    channels=self.__input_channels,
                                                    maps=generate_label_maps,
                                                    weights=generate_weight_maps,
                                                    size=main_buffer_size,
                                                    chunk=chunk_size))

        self.__main_patch_buffer = dptpatchbuffer.PatchBuffer(shapes=self.__patch_shapes,
                                                              input_channels=self.__input_channels,
                                                              label_maps=generate_label_maps,
                                                              weight_maps=generate_weight_maps,
                                                              cache_size=main_buffer_size,
                                                              chunk_size=chunk_size)

        # Create the second, read buffer.
        #
        if 0 < read_buffer_size:
            read_log_message = '{tag}Read buffer: shapes: {shapes}; channels: {channels}; generate label maps: {maps}; generate weight maps: {weights}; cache size: {size}; chunk size: {chunk}'
            self.__logger.debug(read_log_message.format(tag=self.__log_name_tag,
                                                        shapes=self.__patch_shapes,
                                                        channels=self.__input_channels,
                                                        maps=generate_label_maps,
                                                        weights=generate_weight_maps,
                                                        size=read_buffer_size,
                                                        chunk=chunk_size))

            self.__read_patch_buffer = dptpatchbuffer.PatchBuffer(shapes=self.__patch_shapes,
                                                                  input_channels=self.__input_channels,
                                                                  label_maps=generate_label_maps,
                                                                  weight_maps=generate_weight_maps,
                                                                  cache_size=read_buffer_size,
                                                                  chunk_size=chunk_size)

        else:
            self.__logger.debug('{tag}No read buffer'.format(tag=self.__log_name_tag))

    def __configuresampler(self,
                           patch_sources,
                           mask_spacing,
                           spacing_tolerance,
                           category_dist,
                           strict_selection,
                           create_stats,
                           patch_augmenter,
                           process_count,
                           pool_size,
                           join_timeout,
                           response_timeout,
                           poll_timeout,
                           multi_threaded,
                           chunk_size):
        """
        Create batch samplers.

        Args:
            patch_sources (dict): Data source: map from image categories to PatchSource object sets.
            mask_spacing (float): Pixel spacing of the masks to process (micrometer).
            spacing_tolerance (float): Pixel spacing tolerance (percentage).
            category_dist (dict): Image category sampling distribution mapping from image category to ratio in a single batch.
            strict_selection (bool): If true, every label that has higher than 0.0 ratio must be available in every source image selection.
            create_stats (bool): Allow missing stat files, and create them if necessary.
            patch_augmenter (augmenters.AugmenterPool, None): Configured patch augmenter object.
            process_count (int): Number of processes to use during extraction at once. 0 means that there will be no external worker processes.
            pool_size (int): Number of images to open or worker processes to spawn at once for extraction.
            join_timeout (int): Seconds to wait for child processes to join.
            response_timeout (int): Seconds to wait for inter-process communication responses. 0 means no timeout.
            poll_timeout (int): Seconds a process waits for a message. Safeguard against hanging. 0 means no timeout.
            multi_threaded (bool): If true the batch sampler will be executed in a separate thread.
            chunk_size (int): Number of patches to read at once. If the number of patches is larger than the chunk size it will be re read in chunks.

        Raises:
            InvalidTimeoutError: A timeout is not valid.
            InvalidSamplerChunkSizeError: Processing chunk size is not valid.
        """

        # Check the timeout: it must be non-negative.
        #
        if join_timeout < 0:
            raise dptprocesserrors.InvalidTimeoutError('join', join_timeout)

        if response_timeout < 0:
            raise dptprocesserrors.InvalidTimeoutError('response', response_timeout)

        if poll_timeout < 0:
            raise dptprocesserrors.InvalidTimeoutError('poll', poll_timeout)

        # Check chunk size: it must be positive.
        #
        if chunk_size <= 0:
            raise dptprocesserrors.InvalidSamplerChunkSizeError(chunk_size)

        # Save parameters.
        #
        self.__multi_threaded = multi_threaded
        self.__chunk_size = chunk_size

        # Finalize the patch augmenter pool.
        #
        self.__patch_augmenter = patch_augmenter
        if self.__patch_augmenter is not None:
            self.__patch_augmenter.distribute()

        # Save the parameters for the batch sampler object.
        #
        self.__patch_sources = patch_sources
        self.__mask_spacing = mask_spacing
        self.__spacing_tolerance = spacing_tolerance
        self.__category_dist = category_dist
        self.__strict_selection = strict_selection
        self.__create_stats = create_stats
        self.__process_count = process_count
        self.__pool_size = pool_size
        self.__response_timeout = response_timeout
        self.__join_timeout = join_timeout
        self.__poll_timeout = poll_timeout

    def __setbatchnormalizer(self, batch_normalizer):
        """
        Check and store the batch normalizer object.

        Args:
            batch_normalizer (batch.BatchNormalizer, None): Batch normalizer object.
        """

        # Store the batch normalizer. Extra checking could be done here later.
        #
        self.__batch_normalizer = batch_normalizer

    def __checkerrormessages(self):
        """
        Check response queue for error messages.

        Raises:
            ErrorThreadResponseError: An error received from the thread.
            UnknownThreadResponseError: Unknown message received from the thread.
        """

        # Consume all messages from the queue.
        #
        while not self.__response_queue.empty():
            # Get one message without waiting.
            #
            response_message = self.__response_queue.get_nowait()

            # Fail-safe: check if the queue was not empty.
            #
            if response_message:
                if response_message['response'] == 'error':
                    self.__logger.error('{tag}Error response from batch sampler: {response}'.format(tag=self.__log_name_tag, response=response_message))

                    raise dptprocesserrors.ErrorThreadResponseError(response_message['tid'], response_message['command'], response_message['exception'])
                else:
                    self.__logger.error('{tag}Unexpected response from batch sampler: {response}'.format(tag=self.__log_name_tag, response=response_message))

                    raise dptprocesserrors.UnknownThreadResponseError(self.__batch_sampler.ident if self.__multi_threaded else None, response_message)

    def __issuecommand(self, command):
        """
        Issue a command and wait for the response if necessary.

        Args:
            command (dict): Command map.

        Raises:
            ErrorThreadResponseError: An error received from the thread.
            UnknownThreadResponseError: Unknown message received from the thread.
            ThreadTerminatedError: The thread is not alive.
        """

        # Check if there is any non-processed message in the command queue.
        #
        self.__checkerrormessages()

        # Insert command to the queue if the batch sampler is alive.
        #
        if self.__batch_sampler.is_alive():
            self.__command_queue.put(command)
        else:
            raise dptprocesserrors.ThreadTerminatedError(self.__batch_sampler.ident)

    def __load(self, batch_size, target_buffer):
        """
        Load items to the target buffer.

        Args:
            batch_size (int): Number of patches to load.
            target_buffer (dptpatchbuffer.PatchBuffer): Target patch buffer.

        Raises:
            ErrorThreadResponseError: An error received from the thread.
            UnknownThreadResponseError: Unknown message received from the thread.
            ThreadTerminatedError: The thread is not alive.
            InvalidBatchSizeError: The requested batch size is not valid.
            MissingPatchSamplersError: There are no configured patch samplers.
            LabelSourceConfigurationError: Label selected without source mask.
            ProcessTerminatedError: Patch sampler process is terminated.
            ProcessResponseTimeoutError: The processes did not responded in time.

            DigitalPathologyBufferError: Buffer errors.
            DigitalPathologyConfigError: Configuration errors.
            DigitalPathologyImageError: Image errors.
            DigitalPathologyLabelError: Label errors.
        """

        # Calculate chunk list.
        #
        chunk_list = [self.__chunk_size] * (batch_size // self.__chunk_size)
        remainder = batch_size % self.__chunk_size
        if remainder:
            chunk_list.append(remainder)

        # Load patches to the buffer.
        #
        for chunk_size in chunk_list:
            if self.__multi_threaded:
                self.__issuecommand({'command': 'batch', 'count': chunk_size, 'buffer': target_buffer})
            else:
                patch_dict = self.__batch_sampler.batch(batch_size=chunk_size)
                target_buffer.push(patch_dict)

    def __labelstocategorical(self, batch):
        """
        Converts a class vector (integers) to binary class matrix (one hot representation). E.g. for use with categorical_crossentropy.

        Args:
            batch (dict): A {spacing: {'patches': patches, 'labels': labels, 'weights': weights}} dictionary with extracted patches, corresponding labels or label patches and weight maps.

        Returns:
             dict: Batch dictionary with one-hot labels.
        """

        # Process all levels of pixel spacing.
        #
        for spacing in batch:
            # Get the labels from this level of pixel spacing.
            # TODO: Now we do a transpose, in order to ensure that the ravel works. Fix this in a nicer way.
            #
            if len(batch[spacing]['labels'].shape) > 1:
                labels = batch[spacing]['labels'].transpose(0, 2, 3, 1)
                labels_shape = labels.shape[:-1]
            else:
                labels = batch[spacing]['labels']
                labels_shape = labels.shape

            # Ravel the label array to 1D.
            #
            labels = labels.ravel()

            # Convert to one-hot.
            #
            categorical = np.zeros(shape=(labels.shape[0], self.__label_count), dtype=np.float32)
            categorical[np.arange(labels.shape[0]), labels] = 1

            # Restore shape.
            #
            target_shape = (labels_shape[0], self.__label_count) if len(labels_shape) < 2 else labels_shape + (self.__label_count,)

            if len(labels_shape) > 1:
                batch[spacing]['labels'] = np.reshape(a=categorical, newshape=target_shape).transpose((0, 3, 1, 2))
            else:
                batch[spacing]['labels'] = np.reshape(a=categorical, newshape=target_shape)

        return batch

    def __stop(self):
        """Terminate and join all patch sampler processes and join the batch sampler thread."""

        try:
            # Terminate and join the sampler thread.
            #
            if self.__multi_threaded:
                # Terminate the thread.
                #
                if self.__batch_sampler.is_alive():
                    self.__issuecommand(command={'command': 'stop'})

                # Shut down the thread.
                #
                self.__batch_sampler.join(timeout=self.__response_timeout)
                if self.__batch_sampler.is_alive():
                    self.__batch_sampler.terminate()
            else:
                # Shut down the child processes in the batch sampler object.
                #
                self.__batch_sampler.stop()

        except:
            # Consume all exceptions during termination.
            #
            pass

    def __barrier(self):
        """
        Barrier function. Sends ping message to the batch sampler (the only message that is responds with a message) and waits for the answer. Blocking function.

        Raises:
            ThreadJobTimeoutError: The sampler thread did not finished its jobs in time.
            ThreadTerminatedError: The thread is not alive.
            ErrorThreadResponseError: An error received from the thread.
            UnknownThreadResponseError: Unknown message received from the thread.
        """

        if self.__multi_threaded:
            # Send the barrier ping message to see if all previous commands have been executed.
            #
            self.__issuecommand(command={'command': 'ping'})

            try:
                # Get the response message.
                #
                response_message = self.__response_queue.get(block=True, timeout=self.__response_timeout)

                # Check if the right message has arrived.
                #
                if response_message['response'] == 'pong':
                    self.__worker_pids = response_message['pids']
                else:
                    self.__logger.error('{tag}Unknown response from batch sampler: {response}'.format(tag=self.__log_name_tag, response=response_message))

                    raise dptprocesserrors.UnknownThreadResponseError(response_message['tid'], response_message)

            except queue.Empty as empty_queue_error:
                # No response in time.
                #
                self.__logger.error('{tag}Timeout, no response in {secs} seconds from batch sampler'.format(tag=self.__log_name_tag, secs=self.__response_timeout))

                raise dptprocesserrors.ThreadJobTimeoutError(self.__response_timeout, self.__batch_sampler.ident, empty_queue_error)

            # Check error messages in the response queue.
            #
            self.__checkerrormessages()

    @staticmethod
    def __orderdimensionstochannelslast(batch):
        """
        Reorder the batch-first dimensions in a batch to match the required 'BHWC' output dimension order.

        Args:
            batch (dict): A {level: {'patches': patches, 'labels': labels, 'weights': weights}} dictionary with extracted patches, corresponding labels or label patches and weight maps.

        Returns:
             dict: Batch dictionary with the required dimension orders.
        """

        # Process all levels.
        #
        for level in batch:
            batch[level]['patches'] = batch[level]['patches'].transpose(0, 2, 3, 1)

            if batch[level]['labels'].ndim == batch[level]['patches'].ndim:
                batch[level]['labels'] = batch[level]['labels'].transpose(0, 2, 3, 1)

            if 'weights' in batch[level]:
                batch[level]['weights'] = batch[level]['weights'].transpose(0, 2, 3, 1)

        return batch

    def start(self):
        """
        Instantiate a batch sampler.

        Raises:
            EmptyLabelListError: The label list is empty.
            NegativeLabelRatioError: There is a negative label ratio.
            AllZeroLabelRatiosError: All label ratios are zero.
            LabelDistributionAndMappingMismatchError: Label distribution - label mapping key mismatch.
            InvalidLabelDistributionWithoutMappingError: The label distribution is not valid without label mapping.
            WeightMappingConfigurationError: Weight mapping is configured without label patch extraction.
            MissingLabelMapperForWeightMapperError: Weight mapping is configured without label mapping.
            WeightMapperLabelMapperClassesMismatchError: The network labels known by the label mapper and weight mapper does not match.
            EmptyPixelSpacingListError: The list of pixel spacings is empty.
            InvalidPixelSpacingInPatchShapesError: A pixel spacing is not valid.
            InvalidPatchShapeError: A patch shape is not valid.
            InvalidMaskPixelSpacingError: The mask pixel spacing is not valid.
            InvalidPixelSpacingToleranceError: The pixel spacing tolerance is not valid.
            EmptyChannelListError: The list of channels is empty.
            DuplicateChannelError: There are multiples of a single channel in the channel list.
            EmptyPatchSourceError: The patch source collection is empty.
            MissingImageFileError: If any item in the data source points to a non existent image file.
            MissingMaskAndStatFilesError: If any item in the data source points to non existent mask and stat file.
            CategoryRatioListMismatchError: The image category ids in the image category distribution do not match the image category ids in the source list.
            NegativeCategoryRatioError: There is a negative image category ratio.
            AllZeroCategoryRatiosError: All image category ratios are zero.
            InvalidProcessCountError: The process count is not valid.
            InvalidPoolSizeError: The pool size is not valid.
            InvalidTimeoutError: A timeout is not valid.
            ProcessTerminatedError: A patch sampler process is unexpectedly terminated.
            UnexpectedProcessResponseError: Unexpected sampler process response received.
            ProcessResponseTimeoutError: The processes did not responded in time.

            DigitalPathologyConfigError: Configuration errors.
            DigitalPathologyDataError: Data errors.
            DigitalPathologyImageError: Image errors.
            DigitalPathologyLabelError: Label errors.
        """

        # Prevent the existence of multiple samplers.
        #
        if not self.__batch_sampler:
            self.__logger.info('{tag}Initializing batch sampler'.format(tag=self.__log_name_tag))

            if self.__multi_threaded:
                self.__logger.debug('{tag}Multi-threaded mode. Starting batch sampler thread'.format(tag=self.__log_name_tag))

                # Create communication channels.
                #
                self.__command_queue = queue.Queue()
                self.__response_queue = queue.Queue()

                # Instantiate thread.
                #
                thread_name = '{tag} batch sampler'.format(tag=self.__name_tag) if self.__name_tag else 'batch sampler'
                thread_kwargs = {'command_queue': self.__command_queue,
                                 'response_queue': self.__response_queue,
                                 'label_dist': self.__label_dist,
                                 'patch_shapes': self.__patch_shapes,
                                 'mask_spacing': self.__mask_spacing,
                                 'spacing_tolerance': self.__spacing_tolerance,
                                 'input_channels': self.__input_channels,
                                 'label_mode': self.__label_mode,
                                 'patch_sources': self.__patch_sources,
                                 'category_dist': self.__category_dist,
                                 'strict_selection': self.__strict_selection,
                                 'create_stats': self.__create_stats,
                                 'patch_augmenter': self.__patch_augmenter,
                                 'label_mapper': self.__label_mapper,
                                 'free_label_range': self.__free_label_range,
                                 'weight_mapper': self.__patch_weight_mapper,
                                 'process_count': self.__process_count,
                                 'pool_size': self.__pool_size,
                                 'join_timeout': self.__join_timeout,
                                 'response_timeout': self.__response_timeout,
                                 'poll_timeout': self.__poll_timeout,
                                 'name_tag': self.__name_tag}

                self.__batch_sampler = threading.Thread(target=dptbatchsamplerdaemon.batchsampler_daemon_loop, name=thread_name, kwargs=thread_kwargs)

                # Startup and initialize thread.
                #
                self.__batch_sampler.start()
                self.__issuecommand({'command': 'init'})
            else:
                self.__logger.debug('{tag}Non-threaded mode. Instantiating local batch sampler object'.format(tag=self.__log_name_tag))

                # Instantiate a batch sampler object in this thread.
                #
                self.__batch_sampler = dptbatchsampler.BatchSampler(label_dist=self.__label_dist,
                                                                    patch_shapes=self.__patch_shapes,
                                                                    mask_spacing=self.__mask_spacing,
                                                                    spacing_tolerance=self.__spacing_tolerance,
                                                                    input_channels=self.__input_channels,
                                                                    label_mode=self.__label_mode,
                                                                    patch_sources=self.__patch_sources,
                                                                    category_dist=self.__category_dist,
                                                                    strict_selection=self.__strict_selection,
                                                                    create_stats=self.__create_stats,
                                                                    patch_augmenter=self.__patch_augmenter,
                                                                    label_mapper=self.__label_mapper,
                                                                    free_label_range=self.__free_label_range,
                                                                    weight_mapper=self.__patch_weight_mapper,
                                                                    process_count=self.__process_count,
                                                                    pool_size=self.__pool_size,
                                                                    join_timeout=self.__join_timeout,
                                                                    response_timeout=self.__response_timeout,
                                                                    poll_timeout=self.__poll_timeout,
                                                                    name_tag=self.__name_tag)
        else:
            # The batch sampler object or thread already exits.
            #
            self.__logger.info('{tag}Re-initialization of the batch sampler skipped, it already exits'.format(tag=self.__log_name_tag))

    def stop(self):
        """
        Terminate and join all patch sampler processes and join the batch sampler thread. There is no coming back from this. This function leaves the
        object in an invalid state and all subsequent function calls will raise exceptions. The aim of this function is to shut down all threads and
        processes to make the program able to shut down a clean way. Unfortunately the python garbage collector does not necessarily destructs all
        objects before exit. This function eats all raised exceptions so if there are multiple BatchGenerator objects all can be shut down safely.
        """

        self.__logger.debug('{tag}Stopping the batch sampler'.format(tag=self.__log_name_tag))

        self.__stop()

    def ping(self):
        """
        Check if the batch sampler is alive and responding. Blocking function.

        Raises:
            ThreadJobTimeoutError: The sampler thread did not finished its jobs in time.
            ThreadTerminatedError: The thread is not alive.
            ErrorThreadResponseError: An error received from the thread.
            UnknownThreadResponseError: Unknown message received from the thread.
            ProcessTerminatedError: A patch sampler process is unexpectedly terminated.
        """

        self.__logger.debug('{tag}Checking the batch sampler'.format(tag=self.__log_name_tag))

        if self.__multi_threaded:
            self.__barrier()

            self.__logger.debug('{tag}The batch sampler is alive and responding'.format(tag=self.__log_name_tag))
        else:
            self.__batch_sampler.ping()

    def step(self):
        """
        Step the batch samplers for each purpose. In multi-threaded mode this is a non-blocking function.

        Raises:
            EmptyDataSetsError: The data sets are empty, the patch extractor is not initialized.
            FailedSourceSelectionError: A label cannot be represented in the source selection.
            ErrorThreadResponseError: An error received from the thread.
            UnknownThreadResponseError: Unknown message received from the thread.
            ThreadTerminatedError: The thread is not alive.
            ProcessTerminatedError: A patch sampler process is unexpectedly terminated.
            UnexpectedProcessResponseError: Unexpected sampler process response received.
            ProcessResponseTimeoutError: The processes did not responded in time.

            DigitalPathologyConfigError: Configuration errors.
            DigitalPathologyDataError: Data errors.
            DigitalPathologyImageError: Image errors.
            DigitalPathologyLabelError: Label errors.
            DigitalPathologyStatError: Stats errors.
        """

        self.__logger.debug('{tag}Stepping the batch sampler sources'.format(tag=self.__log_name_tag))

        if self.__multi_threaded:
            self.__issuecommand({'command': 'step'})
        else:
            self.__batch_sampler.step()

    def histogram(self, bins, raw=True):
        """
        Calculate the histogram of the classification errors.

        Args:
            bins (int): Number of bins to use.
            raw (bool): If true the raw counts and bins arrays are returned, otherwise a formatted bin string to count value dictionary.

        Returns:
            np.ndarray, np.ndarray: Histogram array and the bin edges.
            dict: Histogram formatted to dictionary to be added to pandas data frame.
        """

        # Get histogram data from buffer.
        #
        counts, edges = self.__main_patch_buffer.histogram(bins)

        # Format data if necessary.
        #
        if raw:
            return counts, edges
        else:
            precision = math.ceil(math.log10(float(bins)))
            header = '{{:.{precision}f}}'.format(precision=precision)
            keys = [header.format(float(item)) for item in edges]
            histogram = dict(list(zip(keys, counts)))
            return histogram

    @property
    def name(self):
        """
        Get the name of the batch generator.

        Returns:
            str: Name of the batch generator.
        """

        return self.__name_tag

    @property
    def labelcount(self):
        """
        Get the number of labels.

        Returns:
             int: Number of labels.
        """

        return self.__label_count

    @property
    def size(self):
        """
        Get the main buffer size.

        Returns:
            int: The main buffer size.
        """

        return self.__main_patch_buffer.size

    @property
    def multithreaded(self):
        """
        Get the multit-hreaded mode.

        Returns:
            bool: The multi-threaded flag.
        """

        return self.__multi_threaded

    @property
    def doublebuffered(self):
        """
        Check if double buffering is enabled.

        Returns:
            boot: True if a secondary (read) buffer is present, False otherwise.
        """

        return self.__read_patch_buffer is not None

    @property
    def maskspacing(self):
        """
        Get the used pixel spacing with masks.

        Returns:
            float: Used pixel spacing with masks.
        """

        return self.__mask_spacing

    @property
    def inputchannels(self):
        """
        Get the list of channels used in a patch.

        Returns:
            list: List of channel indices.
        """

        return self.__input_channels

    @property
    def dimensionorder(self):
        """
        Get the produced dimension order.

        Returns:
            str: Dimension order descriptor.
        """

        return self.__dimension_order

    @property
    def labeldistribution(self):
        """
        Get the label distribution ratios in a batch.

        Returns:
            dict: Label distribution ratios in a batch.
        """

        return self.__label_dist

    @property
    def workers(self):
        """
        Get the last known list of worker process IDs. It is refreshed in every ping-pong message exchange.

        Returns:
            list: List of worker process IDs.
        """

        return self.__worker_pids

    def batch(self, batch_size):
        """
        Generate one batch.

        Args:
            batch_size (int): Number of patches in the batch.

        Returns:
            (dict, np.ndarray): a {spacing: {'patches': patches, 'labels': labels, 'weights': weights}} dictionary with extracted patches, corresponding labels or label
                patches and weight maps if label patches are extracted and an array that contains the index of the extracted patches in the buffer for weight updates.

        Raises:
            InvalidBatchSizeError: The requested batch size is not valid.

            DigitalPathologyBufferError: Buffer errors.
        """

        # Check if batch size is valid.
        #
        if batch_size < 1:
            raise dptconfigerrors.InvalidBatchSizeError(batch_size)

        # Extract batch. The buffer returns patches in BCHW order.
        #
        batch_dict, indices = self.__main_patch_buffer.batch(batch_size=batch_size)

        # Map the weights based on batch statistics.
        #
        if self.__batch_weight_mapper:
            batch_dict = self.__batch_weight_mapper.calculate(patches=batch_dict)

        # Normalize the image patches and the label values to the proper range and data type.
        #
        if self.__batch_normalizer:
            batch_dict = self.__batch_normalizer.normalize(batch=batch_dict)

        # Convert the label values to one-hot representation.
        #
        if self.__one_hot_labels:
            batch_dict = self.__labelstocategorical(batch=batch_dict)

        # Convert the BHWC dimension order from the default BCHW.
        #
        if self.__dimension_order == 'bhwc':
            batch_dict = self.__orderdimensionstochannelslast(batch=batch_dict)

        return batch_dict, indices

    def update(self, indices, errors):
        """
        Update the classification errors on the given indices. All errors must be in the [0.0, 1.0] range.

        Args:
            indices (np.ndarray): Indices of errors to update.
            errors (np.ndarray): Classification errors.

        Raises:
            DigitalPathologyBufferError: Buffer errors.
        """

        # Update the classification errors in the buffer.
        #
        self.__main_patch_buffer.update(indices=indices, errors=errors)

    def count(self, threshold):
        """
        Count the number of patches in the buffer with larger than the given classification error.

        Args:
            threshold (float): Classification error threshold.

        Returns:
            int: number of patches with larger than the given error.
        """

        return self.__main_patch_buffer.count(threshold=threshold)

    def fill(self):
        """
        Fill the patch buffer directly with new patch data from the samplers. In multi-threaded mode this is a non-blocking function.

        Raises:
            ErrorThreadResponseError: An error received from the thread.
            UnknownThreadResponseError: Unknown message received from the thread.
            ThreadTerminatedError: The thread is not alive.
            InvalidBatchSizeError: The requested batch size is not valid.
            MissingPatchSamplersError: There are no configured patch samplers.
            LabelSourceConfigurationError: Label selected without source mask.
            ProcessTerminatedError: Patch sampler process is terminated.
            ProcessResponseTimeoutError: The processes did not responded in time.

            DigitalPathologyBufferError: Buffer errors.
            DigitalPathologyConfigError: Configuration errors.
            DigitalPathologyImageError: Image errors.
            DigitalPathologyLabelError: Label errors.
        """

        self.__logger.debug('{tag}Filling the batch buffer'.format(tag=self.__log_name_tag))

        # Fill the patch buffer.
        #
        self.__load(batch_size=self.__main_patch_buffer.size, target_buffer=self.__main_patch_buffer)

        # Shuffle the patch buffer.
        #
        self.__main_patch_buffer.shuffle()

    def load(self, batch_size=0):
        """
        Load given number of patches to the buffer. If double buffering is enabled and the requested batch size is zero than the read buffer will be
        re-filled with as many patches as it can hold. In multi-threaded mode this is a non-blocking function.

        Args:
            batch_size (int): Number of patches to load.

        Raises:
            BatchSizeOutOfBoundsError: The requested batch size is not valid.
            ErrorThreadResponseError: An error received from the thread.
            UnknownThreadResponseError: Unknown message received from the thread.
            ThreadTerminatedError: The thread is not alive.
            InvalidBatchSizeError: The requested batch size is not valid.
            MissingPatchSamplersError: There are no configured patch samplers.
            LabelSourceConfigurationError: Label selected without source mask.
            ProcessTerminatedError: Patch sampler process is terminated.
            ProcessResponseTimeoutError: The processes did not responded in time.

            DigitalPathologyBufferError: Buffer errors.
            DigitalPathologyConfigError: Configuration errors.
            DigitalPathologyImageError: Image errors.
            DigitalPathologyLabelError: Label errors.
        """

        # Check the batch size.
        #
        if self.__read_patch_buffer:
            if batch_size < 0 or self.__read_patch_buffer.size < batch_size:
                raise dptconfigerrors.BatchSizeOutOfBoundsError(batch_size, (0, self.__read_patch_buffer.size))
        else:
            if batch_size < 0 or self.__main_patch_buffer.size < batch_size:
                raise dptconfigerrors.BatchSizeOutOfBoundsError(batch_size, (0, self.__main_patch_buffer.size))

        # Check double buffering.
        #
        if self.__read_patch_buffer:
            # Calculate the actual batch size and load patches.
            #
            actual_batch_size = batch_size if 0 < batch_size else self.__read_patch_buffer.size
            self.__logger.debug('{tag}Loading {count} new patches to the read patch buffer'.format(tag=self.__log_name_tag, count=actual_batch_size))

            self.__load(batch_size=actual_batch_size, target_buffer=self.__read_patch_buffer)
        else:
            # Calculate the actual batch size and load patches.
            #
            actual_batch_size = batch_size if 0 < batch_size else self.__main_patch_buffer.size
            self.__logger.debug('{tag}Loading {count} new patches to the mean patch buffer'.format(tag=self.__log_name_tag, count=actual_batch_size))

            self.__load(batch_size=actual_batch_size, target_buffer=self.__main_patch_buffer)

    def transfer(self, batch_size=0, difficult_threshold=0.0):
        """
        Transfer loaded patches from the read buffer to the patch buffer. It has no effect if double buffering is not enabled. If the difficult
        threshold is set the patches with the lower than threshold error overwritten first, than the difficult ones.

        Args:
            batch_size (int): Number of patches to transfer.
            difficult_threshold (float): Error threshold for difficult patches.

        Raises:
            DigitalPathologyBufferError: Buffer errors.
            DigitalPathologyConfigError: Configuration errors.
        """

        # Check if read buffering is enabled. Copy the patches from the read buffer and shuffle the order.
        #
        if self.__read_patch_buffer:
            self.__logger.debug('{tag}Transferring {count} patches from read buffer content to main buffer with {threshold} threshold'.format(tag=self.__log_name_tag,
                                                                                                                                              count=batch_size if batch_size else 'all',
                                                                                                                                              threshold=difficult_threshold))

            self.__main_patch_buffer.copy(other=self.__read_patch_buffer, count=batch_size, threshold=difficult_threshold)
            self.__main_patch_buffer.shuffle()

    def wait(self):
        """
        Wait for all the issued jobs to finish in multi-threaded mode. Blocking function. Has no effect if the multi-threading is not enabled.

        Raises:
            ThreadJobTimeoutError: The sampler thread did not finished its jobs in time.
            ThreadTerminatedError: The thread is not alive.
            ErrorThreadResponseError: An error received from the thread.
            UnknownThreadResponseError: Unknown message received from the thread.
        """

        if self.__multi_threaded:
            self.__logger.debug('{tag}Waiting for the batch sampler to finish assigned tasks'.format(tag=self.__log_name_tag))

            self.__barrier()

            self.__logger.debug('{tag}Waiting done'.format(tag=self.__log_name_tag))
