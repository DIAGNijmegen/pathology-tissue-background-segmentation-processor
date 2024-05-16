"""
This file contains class for extracting patches from a collection of whole slide images.
"""

from ..errors import labelerrors as dptlabelerrors
from ..errors import configerrors as dptconfigerrors
from ..errors import dataerrors as dptdataerrors
from ..errors import weighterrors as dptweighterrors
from ..errors import processerrors as dptprocesserrors
from ..patch import patchsampler as dptpatchsampler
from ..patch import patchsamplerdaemon as dptpatchsamplerdaemon
from ..utils import population as dptpopulation

import logging
import numpy as np
import queue
import os
import sys
import random
import multiprocessing as mp
import threading

#----------------------------------------------------------------------------------------------------

class BatchSampler(object):
    """This class is a batch sampler class that extracts patches from a collection of whole slide images."""

    def __init__(self,
                 label_dist,
                 patch_shapes,
                 mask_spacing,
                 spacing_tolerance,
                 input_channels,
                 label_mode,
                 patch_sources,
                 category_dist,
                 strict_selection=False,
                 create_stats=False,
                 patch_augmenter=None,
                 label_mapper=None,
                 free_label_range=False,
                 weight_mapper=None,
                 process_count=0,
                 pool_size=sys.maxsize,
                 join_timeout=60,
                 response_timeout=600,
                 poll_timeout=900,
                 name_tag=''):
        """
        Initialize the object. Set the list of labels, store the patch shape, the ratio of data sets per purpose (e.g. 'training', 'validation' and 'testing'),
        add data source, configure the augmenter object and  set collection size that is the number of opened images of patch extraction at once per image category.

        Args:
            label_dist (dict): Label sampling distribution mapping from label value to ratio in a single batch.
            patch_shapes (dict): Dictionary mapping pixel spacings to (rows, columns) patch shape.
            mask_spacing (float): Pixel spacing of the masks to process (micrometer).
            spacing_tolerance (float): Pixel spacing tolerance (percentage).
            input_channels (list): Desired channels that are extracted for each patch.
            label_mode (str): Label generation mode. Accepted values:
                'central': Just return the label of the central pixel.
                'synthesize': Synthesize the label map from the mask statistics and zoom it to the appropriate level of pixel spacing.
                'load': Load the label map from the label image and zoom it to the appropriate level of pixel spacing if necessary.
            patch_sources (dict): Data source: map from image categories to PatchSource object sets.
            category_dist (dict): Image category sampling distribution mapping from image category to ratio in a single batch.
            strict_selection (bool): If true, every label that has higher than 0.0 ratio must be available in every source image selection.
            create_stats (bool): Allow missing stat files, and create them if necessary.
            patch_augmenter (augmenters.AugmenterPool, None): Configured patch augmenter object.
            label_mapper (label.LabelMapper, None): Label value to label index mapper object.
            free_label_range (bool): If this flag is True, non-continuous label ranges, and ranges that do not start at 0 are also allowed.
            weight_mapper (weight.WeightMapperBase, None): Label weight map calculator object.
            process_count (int): Number of processes to use during extraction at once. 0 means that there will be no external worker processes.
            pool_size (int): Number of images to open or worker processes to spawn at once for extraction.
            join_timeout (int): Seconds to wait for child processes to join.
            response_timeout (int): Seconds to wait for inter-process communication responses. 0 means no timeout.
            poll_timeout (int): Seconds a process waits for a message. Safeguard against hanging. 0 means no timeout.
            name_tag (str): Name tag of the sampler for logging.

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

        # Initialize the base class.
        #
        super().__init__()

        # Initialize members.
        #
        self.__name_tag = ''             # Name tag of the batch sampler.
        self.__log_name_tag = ''         # Logging name tag of the batch sampler for easy identification in logs.
        self.__data_source = {}          # List of PatchSource objects per source category.
        self.__category_dist = {}        # Image category distribution ratio in a batch.
        self.__label_dist = {}           # Label distribution ratio in a batch.
        self.__strict_selection = False  # Strict source item selection.
        self.__create_stats = False      # Create missing .stat files.
        self.__patch_shapes = {}         # Shape of extracted patches per pixel spacing.
        self.__mask_spacing = 0.0        # Mask pixel spacing to use.             mask_spacing (float): Pixel spacing of the masks to process (micrometer).
        self.__spacing_tolerance = 0.0   # Tolerance for finding a level for the given pixel spacing.
        self.__input_channels = []       # List of channels used in a patch
        self.__label_mode = ''           # Label generation mode.
        self.__process_count = 0         # Number of worker processes to spawn.
        self.__pool_size = sys.maxsize   # Number of images to use for extraction in a round.
        self.__patch_samplers = {}       # Patch sampler collection (either PatchSampler objects or host process identifiers).
        self.__patch_augmenter = None    # Patch augmenter pool object.
        self.__label_mapper = None       # Label value to label index mapper object.
        self.__free_label_range = False  # Free label mapping flag: no label range checking if True.
        self.__weight_mapper = None      # Label weight mapper object.
        self.__mp_context = None         # Multiprocessing context.
        self.__command_pipes = {}        # Pipes for the patch sampler processes.
        self.__worker_processes = {}     # Patch sampler worker processes.
        self.__response_queue = None     # Response queue.
        self.__join_timeout = 0          # Timeout seconds to wait for child processes to join.
        self.__response_timeout = 0      # Timeout seconds to wait for inter-process communication responses.
        self.__poll_timeout = 0          # Timeout seconds to wait for inter-process communication requests.

        # Set instance name.
        #
        self.__setname(name_tag=name_tag)

        # Initialize logging.
        #
        self.__initlogging()

        # Process the configured parameters.
        #
        self.__setlabels(label_dist=label_dist, label_mode=label_mode, strict_selection=strict_selection, label_mapper=label_mapper, free_label_range=free_label_range)
        self.__setweightmapper(weight_mapper=weight_mapper)
        self.__setpatchshape(patch_shapes=patch_shapes, mask_spacing=mask_spacing, spacing_tolerance=spacing_tolerance, input_channels=input_channels)
        self.__addsource(patch_sources=patch_sources, create_stats=create_stats, category_dist=category_dist, patch_augmenter=patch_augmenter)
        self.__configuremultiprocessing(process_count=process_count, pool_size=pool_size, join_timeout=join_timeout, response_timeout=response_timeout, poll_timeout=poll_timeout)

    def __del__(self):
        """Destruct object. Terminate all sampler processes and close the response queue."""

        # Terminate all sampler processes and wait for them to stop.
        #
        self.__stop()

        # Close the response queue.
        #
        if self.__response_queue:
            self.__response_queue.close()

    def __setname(self, name_tag):
        """
        Set instance name tag and name tag for logging.

        Args:
            name_tag (str): Name tag of the generator for logging.
        """

        # Configure the name tag.
        #
        self.__name_tag = name_tag
        self.__log_name_tag = '[{name}] '.format(name=name_tag) if name_tag else ''

    def __initlogging(self):
        """Initialize logging."""

        # Configure logging. This class relies on configured logging somewhere down on the hierarchy.
        #
        qualified_class_name = '{module_name}.{class_name}'.format(module_name=self.__class__.__module__, class_name=self.__class__.__name__)
        self.__logger = logging.getLogger(name=qualified_class_name)

        # Report own process and thread identifiers.
        #
        self.__logger.debug('{tag}Batch sampler initializing. process: {pid}; thread: {tid}'.format(tag=self.__log_name_tag, pid=os.getpid(), tid=threading.current_thread().ident))

    def __flushlogging(self):
        """Flush all log handlers."""

        for log_handler in self.__logger.handlers:
            log_handler.flush()

    def __setlabels(self, label_dist, label_mode, strict_selection, label_mapper, free_label_range):
        """
        The list of labels.

        Args:
            label_dist (dict): Label sampling distribution mapping from label value to ratio in a single batch.
            label_mode (str): Label generation mode.
            strict_selection (bool): If true, every label that has higher than 0.0 ratio must be available in every source image selection.
            label_mapper (label.LabelMapper, None): Label value to label index mapper object.
            free_label_range (bool): If this flag is True, non-continuous label ranges, and ranges that do not start at 0 are also allowed.

        Raises:
            EmptyLabelListError: The label list is empty.
            NegativeLabelRatioError: There is a negative label ratio.
            AllZeroLabelRatiosError: All label ratios are zero.
            LabelDistributionAndMappingMismatchError: Label distribution - label mapping key mismatch.
            InvalidLabelDistributionWithoutMappingError: The label distribution is not valid without label mapping.
        """

        # Check label list: it cannot be empty.
        #
        if not label_dist:
            raise dptlabelerrors.EmptyLabelListError()

        # All label ratios must be non-negative.
        #
        if any(label_ratio < 0.0 for label_ratio in label_dist.values()):
            raise dptlabelerrors.NegativeLabelRatioError(label_dist)

        # There should be at least one ratio with larger than 0.0 value.
        #
        if all(label_ratio == 0.0 for label_ratio in label_dist.values()):
            raise dptlabelerrors.AllZeroLabelRatiosError(label_dist)

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

        # Calculate the label weight sum for normalization.
        #
        label_dist_sum = sum(label_dist.values())

        # Save the normalized label distribution, the label generation mode, and the strict selection flag.
        #
        self.__label_dist = {label: weight / label_dist_sum for label, weight in label_dist.items()}
        self.__label_mode = label_mode
        self.__strict_selection = strict_selection

        # Store the label mapper object and the label range checking flag.
        #
        self.__label_mapper = label_mapper
        self.__free_label_range = free_label_range

    def __setweightmapper(self, weight_mapper):
        """
        Check and store the weight mapper object.

        Args:
            weight_mapper (weight.WeightMapperBase, None): Label weight map calculator object.

        Raises:
            WeightMappingConfigurationError: Weight mapping is configured without label patch extraction.
            MissingLabelMapperForWeightMapperError: Weight mapping is configured without label mapping.
            WeightMapperLabelMapperClassesMismatchError: The network labels known by the label mapper and weight mapper does not match.
        """

        if weight_mapper is not None:
            # Check the label extraction mode.
            #
            if self.__label_mode == 'central':
                raise dptweighterrors.WeightMappingConfigurationError(self.__label_mode)

            # Check if label mapper is configured. It is necessary for the valid pixel map generation.
            #
            if self.__label_mapper is None:
                raise dptweighterrors.MissingLabelMapperForWeightMapperError()

            # Check if the label mapper and the weight mapper has the same network labels.
            #
            if weight_mapper.classes != self.__label_mapper.classes:
                raise dptweighterrors.WeightMapperLabelMapperClassesMismatchError(weight_mapper.classes, self.__label_mapper.classes)

        # Store the weight mapper.
        #
        self.__weight_mapper = weight_mapper

    def __setpatchshape(self, patch_shapes, mask_spacing, spacing_tolerance, input_channels):
        """
        Store the patch shape.

        Args:
            patch_shapes (dict): Desired patch shapes {spacing: (rows, cols)} per pixel spacing.
            mask_spacing (float): Pixel spacing of the masks to process (micrometer).
            spacing_tolerance (float): Pixel spacing tolerance (percentage).
            input_channels (list): Desired channels that are extracted for each patch.

        Raises:
            EmptyPixelSpacingListError: The list of pixel spacings is empty.
            InvalidPixelSpacingInPatchShapesError: A pixel spacing is not valid.
            InvalidPatchShapeError: A patch shape is not valid.
            InvalidMaskPixelSpacingError: The mask pixel spacing is not valid.
            InvalidPixelSpacingToleranceError: The pixel spacing tolerance is not valid.
            EmptyChannelListError: The list of channels is empty.
            DuplicateChannelError: There are multiples of a single channel in the channel list.
        """

        # Check if the list of pixel spacings to use is not empty.
        #
        if not patch_shapes:
            raise dptconfigerrors.EmptyPixelSpacingListError()

        # The pixel spacings must be positive.
        #
        if any(spacing <= 0.0 for spacing in patch_shapes):
            raise dptconfigerrors.InvalidPixelSpacingInPatchShapesError(patch_shapes)

        # Check patch size: it must be positive.
        #
        if any(shape[0] <= 0 or shape[1] <= 0 for shape in patch_shapes.values()):
            raise dptconfigerrors.InvalidPatchShapeError(patch_shapes)

        # The mask spacing must be positive.
        #
        if mask_spacing <= 0.0:
            raise dptconfigerrors.InvalidMaskPixelSpacingError(mask_spacing)

        # The tolerance must be non-negative.
        #
        if spacing_tolerance < 0.0:
            raise dptconfigerrors.InvalidPixelSpacingToleranceError(spacing_tolerance)

        # Check if channel inputs list is not empty nor contains duplicates.
        #
        if not input_channels:
            raise dptconfigerrors.EmptyChannelListError()

        if len(input_channels) != len(set(input_channels)):
            raise dptconfigerrors.DuplicateChannelError(input_channels)

        # Configure the patch shapes.
        #
        self.__patch_shapes = patch_shapes
        self.__mask_spacing = mask_spacing
        self.__spacing_tolerance = spacing_tolerance
        self.__input_channels = input_channels

    def __addsource(self, patch_sources, create_stats, category_dist, patch_augmenter):
        """
        Add data source.

        Args:
            patch_sources (dict): Data source: map from categories to PatchSource object sets.
            create_stats (bool): Allow missing stat files, and create them if necessary.
            category_dist (dict): Image category sampling distribution mapping from image category to ratio in a single batch.
            patch_augmenter (dptaugmenterpool.AugmenterPool, None): Configured patch augmenter object.

        Raises:
            EmptyPatchSourceError: The patch source collection is empty.
            MissingImageFileError: If any item in the data source points to a non existent image file.
            MissingMaskAndStatFilesError: If any item in the data source points to non existent mask and stat file.
            CategoryRatioListMismatchError: The image category ids in the image category distribution do not match the image category ids in the source list.
            NegativeCategoryRatioError: There is a negative image category ratio.
            AllZeroCategoryRatiosError: All image category ratios are zero.
        """

        # Calculate total count of samples.
        #
        source_count = 0
        for category_id in patch_sources:
            source_count += len(patch_sources[category_id])

        # Check file existence.
        #
        for category_items in patch_sources.values():
            for source_item in category_items:
                if not os.path.isfile(source_item.image):
                    raise dptdataerrors.MissingImageFileError(source_item.image)
                if not os.path.isfile(source_item.mask) and not os.path.isfile(source_item.stat):
                    raise dptdataerrors.MissingMaskAndStatFilesError(source_item.mask, source_item.stat)

        # Check data source and purpose ratios.
        #
        if source_count == 0:
            raise dptdataerrors.EmptyPatchSourceError()

        # The list of categories and their distribution must match.
        #
        if not all(category_id in patch_sources for category_id in category_dist):
            raise dptlabelerrors.CategoryRatioListMismatchError(list(patch_sources.keys()), category_dist)

        # All label ratios must be non-negative.
        #
        if any(category_ratio < 0.0 for category_ratio in category_dist.values()):
            raise dptlabelerrors.NegativeCategoryRatioError(category_dist)

        # There should be at least one ratio with larger than 0.0 value.
        #
        if all(category_ratio == 0.0 for category_ratio in category_dist.values()):
            raise dptlabelerrors.AllZeroCategoryRatiosError(category_dist)

        # Store data. Apart from the sources build a quick search index for listing all the images that contains the given label index.
        #
        for category_id, item_set in patch_sources.items():
            # Convert set to ordered list.
            #
            item_list = list(item_set)

            # Store the source list.
            #
            self.__data_source[category_id] = {'items': item_list, 'search': {label: set() for label in self.__label_dist}}

            # Build the per label search index.
            #
            for item_index in range(len(item_list)):
                for label in self.__label_dist:
                    if label in item_list[item_index].labels:
                        self.__data_source[category_id]['search'][label].add(item_index)

        # Save the automatic stat file creation flag.
        #
        self.__create_stats = create_stats

        # Store the class distribution and normalize it to 1.0 sum.
        #
        category_dist_sum = sum(category_dist.values())
        self.__category_dist = {category_id: category_prob / category_dist_sum for category_id, category_prob in category_dist.items()}

        # Store the augmentation pool.
        #
        self.__patch_augmenter = patch_augmenter

    def __configuremultiprocessing(self, process_count, pool_size, join_timeout, response_timeout, poll_timeout):
        """
        Set the number of spawned processes at once, set the number of processors to use at once and initialize the response queue.

        Args:
            process_count (int): Number of processes to use during extraction at once. 0 means that there will be no external worker processes.
            pool_size (int): Number of images to open or worker processes to spawn at once for extraction.
            join_timeout (int): Seconds to wait for child processes to join.
            response_timeout (int): Seconds to wait for inter-process communication responses.
            poll_timeout (int): Seconds a process waits for a message. Safeguard against hanging.

        Raises:
            InvalidProcessCountError: The process count is not valid.
            InvalidPoolSizeError: The pool size is not valid.
            InvalidTimeoutError: A timeout is not valid.
        """

        # Check process count: it must be non-negative.
        #
        if process_count < 0:
            raise dptprocesserrors.InvalidProcessCountError(process_count)

        # Check the collection size: it must be positive.
        #
        if pool_size <= 0:
            raise dptprocesserrors.InvalidPoolSizeError(pool_size)

        # Check the timeouts: it must be non-negative.
        #
        if join_timeout < 0:
            raise dptprocesserrors.InvalidTimeoutError('join', join_timeout)

        if response_timeout < 0:
            raise dptprocesserrors.InvalidTimeoutError('response', response_timeout)

        if poll_timeout < 0:
            raise dptprocesserrors.InvalidTimeoutError('poll', poll_timeout)

        # Store the collection size.
        #
        self.__pool_size = pool_size

        # Configure the CPU core count to use
        #
        self.__process_count = process_count

        # The response queue and the timeout length is only required if the multiprocessing is enabled.
        #
        if 0 < process_count:
            self.__logger.debug('{tag}Multiprocessing enabled'.format(tag=self.__log_name_tag))

            # Create 'spawn' type multiprocessing context. It is not the default on Linux.
            #
            self.__mp_context = mp.get_context(method='spawn')

            # Create the response queue.
            #
            self.__response_queue = self.__mp_context.Queue()

            # Configure the timeouts.
            #
            self.__join_timeout = join_timeout
            self.__response_timeout = response_timeout
            self.__poll_timeout = poll_timeout

        else:
            self.__logger.debug('{tag}Multiprocessing disabled'.format(tag=self.__log_name_tag))

    def __startprocess(self):
        """
        Create a new worker process.

        Returns:
            int: Identifier of the new process.

        Raises:
            UnexpectedProcessResponseError: Unexpected sampler process response received.
            ProcessTerminatedError: A patch sampler process is unexpectedly terminated.
        """

        # Create new process.
        #
        pipe_cli, pipe_srv = self.__mp_context.Pipe(duplex=False)
        process_name = 'patch sampler process {index}'.format(index=len(self.__worker_processes))
        process_kwargs = {'command_pipe': pipe_cli, 'response_queue': self.__response_queue, 'poll_timeout': self.__poll_timeout}
        sampler_process = self.__mp_context.Process(target=dptpatchsamplerdaemon.patchsampler_daemon_loop, name=process_name, kwargs=process_kwargs)

        # Startup and initialize process.
        #
        sampler_process.daemon = True
        sampler_process.start()

        # Store the process and its pipe.
        #
        self.__command_pipes[sampler_process.pid] = pipe_srv
        self.__worker_processes[sampler_process.pid] = sampler_process

        self.__logger.debug('{tag}Started sampler process: {pid}'.format(tag=self.__log_name_tag, pid=sampler_process.pid))

        # Return the PID of the new process.
        #
        return sampler_process.pid

    def __checkprocess(self, process_id):
        """
        Check if the process is alive and send ping command to reset its timer.

        Args:
            process_id (int): Process identifier.

        Returns:
            bool: True if the process is alive.

        Raises:
            UnexpectedProcessResponseError: Unexpected sampler process response received.
        """

        # Check if the process is alive.
        #
        if self.__worker_processes[process_id].is_alive():

            # Ping the process to reset its wait timer.
            #
            self.__command_pipes[process_id].send({'command': 'ping'})
            response_message = self.__response_queue.get(block=True, timeout=self.__response_timeout)

            if response_message['response'] != 'pong':
                self.__logger.error('{tag}Checking sampler process {pid}: unexpected response: {response}'.format(tag=self.__log_name_tag,
                                                                                                                  pid=response_message.get('pid', None),
                                                                                                                  response=response_message))

                raise dptprocesserrors.UnexpectedProcessResponseError(response_message['pid'], 'ping', response_message)

            self.__logger.debug('{tag}Checking sampler process {pid}: alive and responding'.format(tag=self.__log_name_tag, pid=process_id))

            return True

        # The process it not alive.
        #
        self.__logger.error('{tag}Checking sampler process {pid}: process dead'.format(tag=self.__log_name_tag, pid=process_id))

        return False

    def __waituntildone(self, count, command):
        """
        Wait until the given number of 'done' messages arrive back from the spawned processes.

        Args:
            count (int): Number of response messages to wait for.
            command (str): The command to be finished.

        Raises:
            UnexpectedProcessResponseError: Unexpected sampler process response received.
            ProcessResponseTimeoutError: The processes did not responded in time.
        """

        self.__logger.debug('{tag}Waiting for confirmation for \'{command}\' command from {count} sampler process{plural}'.format(tag=self.__log_name_tag,
                                                                                                                                  command=command,
                                                                                                                                  count=count,
                                                                                                                                  plural='es' if 1 < count else ''))

        # Wait for all the responses.
        #
        try:
            for _ in range(count):
                # Wait for one response message.
                #
                response_message = self.__response_queue.get(block=True, timeout=self.__response_timeout)

                if response_message['response'] != 'done':
                    self.__logger.error('{tag}Unexpected response from {pid} sampler process for \'{command}\' command: {message}'.format(tag=self.__log_name_tag,
                                                                                                                                          pid=response_message.get('pid', None),
                                                                                                                                          command=command,
                                                                                                                                          message=response_message))

                    raise dptprocesserrors.UnexpectedProcessResponseError(response_message['pid'], command, response_message)
                else:
                    self.__logger.debug('{tag}Sampler process {pid} is done'.format(tag=self.__log_name_tag, pid=response_message.get('pid', None)))

        except queue.Empty as empty_queue_error:
            # No response in time.
            #
            self.__logger.error('{tag}Not all responses arrived in time for \'{command}\' command'.format(tag=self.__log_name_tag, command=command))

            raise dptprocesserrors.ProcessResponseTimeoutError(self.__response_timeout, empty_queue_error)

    def __terminateprocess(self, process_id):
        """
        Terminate the process. Close its connection and remove the process and the pipes from the dictionaries.

        Args:
            process_id (int): Process identifier.
        """

        self.__logger.debug('{tag}Stopping sampler process {pid}'.format(tag=self.__log_name_tag, pid=process_id))

        # Sent stop command.
        #
        if self.__worker_processes[process_id].is_alive():
            self.__command_pipes[process_id].send({'command': 'stop'})

        # Close the connections.
        #
        self.__command_pipes[process_id].close()

        # Wait for the process to stop. Terminate it forcefully if it does not exits in time.
        #
        self.__worker_processes[process_id].join(timeout=self.__join_timeout)
        if self.__worker_processes[process_id].is_alive():
            self.__worker_processes[process_id].terminate()

        # Remove process and pipe items from the dictionaries.
        #
        del self.__command_pipes[process_id]
        del self.__worker_processes[process_id]

        # Clear the patch sampler list.
        #
        deleted_source_items = [source_item for source_item in self.__patch_samplers if self.__patch_samplers[source_item] == process_id]
        for source_item in deleted_source_items:
            del self.__patch_samplers[source_item]

    def __samplesources(self):
        """
        Select a set of source items based on the categories and their available labels.

        Returns:
            set: Set of patch source items.

        Raises:
            EmptyDataSetsError: The data sets are empty, the patch extractor is not initialized.
            FailedSourceSelectionError: A label cannot be represented in the source selection.
        """

        # Calculate total count of samples and check if any data source is present.
        #
        source_count = sum(len(self.__data_source[category_id]['items']) for category_id in self.__category_dist)
        if source_count == 0:
            raise dptdataerrors.EmptyDataSetsError()

        # Distribute the image collection count (pool size) among the image categories.
        #
        category_population = min(source_count, self.__pool_size)
        category_minimum_counts = {category_id: 1 if len(self.__data_source) <= category_population and 0.0 < self.__category_dist[category_id] else 0 for category_id in self.__category_dist}
        category_distribution = {category_id: (self.__category_dist[category_id], category_minimum_counts[category_id], category_population) for category_id in self.__category_dist}
        category_source_count = dptpopulation.distribute_population(population=category_population, ratios=category_distribution)

        # Check if the target distribution is possible.
        #
        if any(len(self.__data_source[category_id]['items']) < category_source_count[category_id] for category_id in self.__category_dist):
            available_image_counts = {category_id: len(self.__data_source[category_id]['items']) for category_id in self.__category_dist}
            self.__logger.warning('{tag}The calculated {target} image distribution is not possible with {available} available images'.format(tag=self.__log_name_tag,
                                                                                                                                             target=category_source_count,
                                                                                                                                             available=available_image_counts))

        # Distribute the per category image collection count among the labels.
        #
        new_source_item_set = set()
        shuffled_label_list = list(self.__label_dist.keys())
        for category_id in self.__category_dist:
            # Check if there are images distributed into this category.
            #
            if category_source_count[category_id]:
                # Calculate per label image count.
                #
                label_population = category_source_count[category_id]
                label_minimum_counts = {label: 1 if len(self.__label_dist) <= label_population and 0.0 < self.__label_dist[label] else 0 for label in self.__label_dist}
                label_distribution = {label: (self.__label_dist[label], label_minimum_counts[label], label_population) for label in self.__label_dist}
                label_source_count = dptpopulation.distribute_population(population=label_population, ratios=label_distribution)

                # Calculate set of all available image indices.
                #
                data_source_category = self.__data_source[category_id]
                category_image_indices = set()
                available_image_indices = set(range(len(data_source_category['items'])))
                category_data_source_search = data_source_category['search']

                # Select source images.
                #
                random.shuffle(shuffled_label_list)
                for label in shuffled_label_list:
                    # Calculate the image list and select from it: the intersection of the images that contains this label and the images that has not been selected yet.
                    #
                    label_available_image_indices = available_image_indices.intersection(category_data_source_search[label])
                    label_selection = random.sample(population=label_available_image_indices, k=min(label_source_count[label], len(label_available_image_indices)))

                    # Add the selected images to the per category image index set and remove them from the set of not selected images.
                    #
                    category_image_indices.update(label_selection)
                    available_image_indices.difference_update(label_selection)

                # Fill up the rest randomly if there are available images left.
                #
                if available_image_indices:
                    category_image_indices.update(random.sample(population=available_image_indices, k=label_population - len(category_image_indices)))

                # Add the selected image index set to the result list.
                #
                for item_index in category_image_indices:
                    new_source_item_set.add(data_source_category['items'][item_index])

        # Add random images to the list if necessary.
        #
        if len(new_source_item_set) < category_population:
            # Collect all the images that has not been selected yet, but contains any of the necessary labels.
            #
            available_images = set()
            for category_id in self.__category_dist:
                data_source_category = self.__data_source[category_id]
                category_available_image_indices = set()
                for label in self.__label_dist:
                    category_available_image_indices.update(data_source_category['search'][label])

                for item_index in category_available_image_indices:
                    available_images.add(data_source_category['items'][item_index])

            available_images.difference_update(new_source_item_set)

            # Select images randomly from the available set.
            #
            item_count_to_add = min(len(available_images), category_population - len(new_source_item_set))
            new_source_item_set.update(random.sample(population=available_images, k=item_count_to_add))

        # Check if all the labels are represented if necessary.
        #
        if self.__strict_selection:
            available_labels_in_selection = set()
            for new_source_item in new_source_item_set:
                available_labels_in_selection.update(new_source_item.labels)

            if any(label not in available_labels_in_selection and 0.0 < self.__label_dist[label] for label in self.__label_dist):
                labels = list(self.__label_dist.keys())
                sources = [source_item.stat if source_item.stat else source_item.mask for source_item in new_source_item_set]

                raise dptdataerrors.FailedSourceSelectionError(labels, sources)

        return new_source_item_set

    def __patchdistribution(self, count):
        """
        Calculate distribution of patches over different images to sample.

        Args:
            count (int): Number of patches to sample.

        Returns:
            dict: per sampler patch count.

        Raises:
            LabelSourceConfigurationError: Label selected without source mask.
        """

        # Calculate the label extract ratios. If strict selection is enabled it is just the configured label distribution as it is guaranteed that each label can be
        # sampled from the current source set.
        #
        if self.__strict_selection:
            # With strict selection it is enforced that every label is represented in the image selection.
            #
            label_extract_ratios = {label: (self.__label_dist[label], 0, count) for label in self.__label_dist if 0.0 < self.__label_dist[label]}
        else:
            # Strict selection checking is not enabled. It is possible that a label cannot be sampled from the current source set.
            #
            available_labels = set()
            for source_item in self.__patch_samplers:
                available_labels.update(source_item.labels)

            label_extract_weights = {label: self.__label_dist[label] for label in self.__label_dist if label in available_labels and 0.0 < self.__label_dist[label]}

            # Check if there is at least one label to sample.
            #
            if label_extract_weights:
                label_extract_weights_sum = sum(extract_item for extract_item in label_extract_weights.values())
                label_extract_ratios = {label: (label_extract_weights[label] / label_extract_weights_sum, 0, count) for label in label_extract_weights}
            else:
                # There is not a single label that can be sampled from the current set.
                #
                labels = list(self.__label_dist.keys())
                sources = [source_item.stat if source_item.stat else source_item.mask for source_item in self.__patch_samplers]

                raise dptlabelerrors.LabelSourceConfigurationError(labels, sources)

        # Distribute the the label of the patches to extract between the available labels.
        #
        label_extract_counts = dptpopulation.distribute_population(population=count, ratios=label_extract_ratios)

        # Calculate the list of source images per label.
        #
        label_source_counts = {label: 0 for label in self.__label_dist}
        for source_item in self.__patch_samplers:
            for label in source_item.labels:
                if label in self.__label_dist:
                    label_source_counts[label] += 1

        # As a result of misconfiguration it can happen that a label is selected for extraction despite there is no mask image to source it. This can happen for example if there are
        # four labels: [1, 2, 3, 4], their sampling ratio is: {1: 0.25, 2: 0.25, 3: 0.25, 4: 0.25} while the currently opened images does not have any label 1. The population distribution
        # cannot take the opened images in consideration therefore it can select label 1 for extraction that would lead to error in trying to distribute the patches with label 1 between
        # the non-existent source images later. This is a fail-safe check. In theory this cannot happen any more.
        #
        if any(0 < label_extract_counts[label] and label_source_counts[label] <= 0 for label in label_extract_counts):
            labels = list(self.__label_dist.keys())
            sources = [source_item.stat if source_item.stat else source_item.mask for source_item in self.__patch_samplers]

            raise dptlabelerrors.LabelSourceConfigurationError(labels, sources)

        # Distribute each patches to extract from each label between the available samplers.
        #
        counts_per_sampler = dict()
        for label in label_extract_counts:
            # Only consider the labels that are sampled in this round. Sampling 0 number of a label causes exception in the population distribution.
            #
            label_count = label_extract_counts[label]
            if 0 < label_count:
                source_count = label_source_counts[label]
                # Calculate distribution.
                #
                label_ratios = {source_item: (1.0 / source_count, 0, label_count) if label in source_item.labels else (0.0, 0, 0) for source_item in self.__patch_samplers}
                label_extract_per_source = dptpopulation.distribute_population(population=label_count, ratios=label_ratios)

                # Rearrange data into per sampler layout. Only store the positive values to prevent unnecessary messages with 0 patches to extract.
                #
                for source_item in self.__patch_samplers:
                    if 0 < label_extract_per_source[source_item]:
                        if source_item in counts_per_sampler:
                            counts_per_sampler[source_item][label] = label_extract_per_source[source_item]
                        else:
                            counts_per_sampler[source_item] = {label: label_extract_per_source[source_item]}

        # Return the calculated patch distribution.
        #
        return counts_per_sampler

    def __stop(self):
        """Terminate and join all sampler processes."""

        # Processes are only spawned in multi-process mode.
        #
        if 0 < self.__process_count:
            process_id_list = list(self.__worker_processes.keys())
            for process_id in process_id_list:
                self.__terminateprocess(process_id=process_id)

    @property
    def workers(self):
        """
        Get the list of worker process IDs.

        Returns:
            list: List of worker process IDs.
        """

        return list(self.__worker_processes.keys())

    def stop(self):
        """Terminate and join all sampler processes."""

        # Processes are only spawned in multi-process mode.
        #
        if 0 < self.__process_count:
            self.__logger.debug('{tag}Stopping all sampler processes'.format(tag=self.__log_name_tag))

        self.__stop()

    def error(self, message):
        """
        Log an error that occurred outside the BatchSampler to save the reason of imminent shutdown.

        Args:
            message (str): Message to log.
        """

        self.__logger.error('{tag}{message}'.format(tag=self.__log_name_tag, message=message))

    def ping(self):
        """
        Check if all worker processes are alive and responding.

        Raises:
            ProcessTerminatedError: A patch sampler process is unexpectedly terminated.
        """

        if 0 < self.__process_count:
            self.__logger.debug('{tag}Checking sampler processes'.format(tag=self.__log_name_tag))

            # Check if all the worker processes are alive and responding.
            #
            for process_id in self.__worker_processes:
                if not self.__checkprocess(process_id=process_id):
                    # The worker process is not alive.
                    #
                    self.__logger.error('{tag}Process {pid} terminated before stepping'.format(tag=self.__log_name_tag, pid=process_id))

                    raise dptprocesserrors.ProcessTerminatedError(process_id)

    def step(self):
        """
        Clear the unnecessary patch samplers and create new ones with randomized sources.

        Raises:
            EmptyDataSetsError: The data sets are empty, the patch extractor is not initialized.
            FailedSourceSelectionError: A label cannot be represented in the source selection.
            ProcessTerminatedError: A patch sampler process is unexpectedly terminated.
            UnexpectedProcessResponseError: Unexpected sampler process response received.
            ProcessResponseTimeoutError: The processes did not responded in time.
            
            DigitalPathologyConfigError: Configuration errors.
            DigitalPathologyDataError: Data errors.
            DigitalPathologyImageError: Image errors.
            DigitalPathologyLabelError: Label errors.            
            DigitalPathologyStatError: Stats errors.
        """

        self.__logger.debug('{tag}Stepping patch sources'.format(tag=self.__log_name_tag))

        # Collect the source items to use.
        #
        new_source_items = self.__samplesources()

        # Check operation mode: spawn processes or calculate everything in this process sequentially.
        #
        if self.__process_count == 0:
            # Check the existing samplers and clear the unwanted ones.
            #
            del_source_item_keys = []
            for source_item in self.__patch_samplers:
                if source_item in new_source_items:
                    # The sampler exists: remove the item from the samplers-to-be-created list.
                    #
                    new_source_items.remove(source_item)
                else:
                    # Add the sampler to the list to be removed.
                    #
                    del_source_item_keys.append(source_item)

            # Pop the marked samplers from the actual storage.
            #
            for source_item in del_source_item_keys:
                self.__logger.debug('{tag}Removing patch sampler for \'{source}\''.format(tag=self.__log_name_tag, source=source_item.image))

                del self.__patch_samplers[source_item]
            del_source_item_keys.clear()

            # Open the selected images in a patch sampler. Sort the content of the set for reproducible results.
            #
            for source_item in sorted(new_source_items):
                self.__logger.debug('{tag}Adding patch sampler for \'{source}\''.format(tag=self.__log_name_tag, source=source_item.image))

                self.__patch_samplers[source_item] = dptpatchsampler.PatchSampler(patch_source=source_item,
                                                                                  create_stat=self.__create_stats,
                                                                                  mask_spacing=self.__mask_spacing,
                                                                                  spacing_tolerance=self.__spacing_tolerance,
                                                                                  input_channels=self.__input_channels,
                                                                                  label_mode=self.__label_mode)
        else:
            # Spawn all worker processes. It could happen that due to misconfiguration or to dynamic CPU scaling the number of processes to spawn is larger than the
            # patch sampler pool size. In that case a process without a patch sampler would just wait for a message after it is created and it would eventually time out.
            # In order to prevent this, the number of processes to create is also limited by the patch sampler pool size and by the current source item list.
            #
            spawned_process_ids = []
            target_process_count = min(self.__process_count, self.__pool_size, len(new_source_items))
            while len(self.__worker_processes) < target_process_count:
                new_process_id = self.__startprocess()
                spawned_process_ids.append(new_process_id)

            # Check if all the worker processes are alive and responding.
            #
            for process_id in self.__worker_processes:
                if not self.__checkprocess(process_id=process_id):
                    # The worker process is not alive.
                    #
                    self.__logger.error('{tag}Process {pid} terminated before stepping'.format(tag=self.__log_name_tag, pid=process_id))

                    raise dptprocesserrors.ProcessTerminatedError(process_id)

            # Serialize and send the patch augmenter pool to all the new processes.
            #
            for process_id in spawned_process_ids:
                process_random_seed = random.randint(a=0, b=np.iinfo(np.uint32).max)

                self.__logger.debug('{tag}Initializing sampler process: {pid}; with seed: {seed}; sending augmenters and label mappers'.format(tag=self.__log_name_tag,
                                                                                                                                               pid=process_id,
                                                                                                                                               seed=process_random_seed))

                # Set the random seed for the processes.
                #
                self.__command_pipes[process_id].send({'command': 'random', 'seed': process_random_seed})

                # Create the sampler in the process and store the host process identifier.
                #
                self.__command_pipes[process_id].send({'command': 'hooks', 'augmenter': self.__patch_augmenter, 'labeler': self.__label_mapper, 'weigher': self.__weight_mapper})

            # Wait for the fresh patch sampler objects to initialize and the old ones to die.
            #
            self.__waituntildone(count=len(spawned_process_ids), command='hooks')

            # Check the existing patch samplers and clear the unwanted ones.
            #
            del_source_item_keys = []
            for source_item in self.__patch_samplers:
                
                # Either att the item to the list to be removed or remove it from the list to be created.
                #
                if source_item in new_source_items:
                    new_source_items.remove(source_item)
                else:
                    del_source_item_keys.append(source_item)

            # Remove the unnecessary source items: remove it from the worker process and from the local bookkeeping.
            #
            for source_item in del_source_item_keys:
                self.__logger.debug('{tag}Removing source item \'{source}\' from sampler process {pid}'.format(tag=self.__log_name_tag,
                                                                                                               source=source_item.image,
                                                                                                               pid=self.__patch_samplers.get(source_item, None)))

                self.__command_pipes[self.__patch_samplers[source_item]].send({'command': 'remove', 'source': source_item})
                del self.__patch_samplers[source_item]

            # Collect the number of images per worker processes.
            #
            image_counts = {process_id: 0 for process_id in self.__worker_processes}
            for process_id in self.__patch_samplers.values():
                image_counts[process_id] += 1

            # Open the selected images in a worker process. Sort the content of the set for reproducible results.
            #
            for source_item in sorted(new_source_items):
                # Get the process with the minimal image count.
                #
                target_process_id = min(image_counts, key=image_counts.get)
                image_counts[target_process_id] += 1

                self.__logger.debug('{tag}Adding source item \'{source}\' to sampler process {pid}'.format(tag=self.__log_name_tag,
                                                                                                           source=source_item.image,
                                                                                                           pid=target_process_id))

                # Create the sampler in the process and store the host process identifier.
                #
                self.__command_pipes[target_process_id].send({'command': 'create',
                                                              'source': source_item,
                                                              'spacing': self.__mask_spacing,
                                                              'stat': self.__create_stats,
                                                              'tolerance': self.__spacing_tolerance,
                                                              'channels': self.__input_channels,
                                                              'mode': self.__label_mode})

                self.__patch_samplers[source_item] = target_process_id

            # Wait for the fresh patch sampler objects to initialize and the old ones to die.
            #
            self.__waituntildone(count=len(new_source_items) + len(del_source_item_keys), command='create|remove')

    def batch(self, batch_size):
        """
        Collect a batch of patches.

        Args:
            batch_size (int): Batch size.

        Returns:
            (dict): Dictionary {spacing: {'patches': patch array, 'labels': label array}} with as keys the pixel spacings where the patches were taken.

        Raises:
            InvalidBatchSizeError: The requested batch size is not valid.
            MissingPatchSamplersError: There are no configured patch samplers.
            LabelSourceConfigurationError: Label selected without source mask.
            ProcessTerminatedError: Patch sampler process is terminated.
            ProcessResponseTimeoutError: The processes did not responded in time.
            
            DigitalPathologyAugmentationError: Augmentation errors.
            DigitalPathologyConfigError: Configuration errors.
            DigitalPathologyImageError: Image errors.
            DigitalPathologyLabelError: Label errors.
        """

        self.__logger.debug('{tag}Sampling {count} patches'.format(tag=self.__log_name_tag, count=batch_size))

        # Check batch size it have to be at least 1 and if there are patch samplers.
        #
        if batch_size < 1:
            raise dptconfigerrors.InvalidBatchSizeError(batch_size)

        if not self.__patch_samplers:
            raise dptdataerrors.MissingPatchSamplersError()

        # Prepare result arrays.
        #
        sample_counts = self.__patchdistribution(count=batch_size)
        batch_dict = {spacing: {'patches': np.empty((batch_size, len(self.__input_channels)) + self.__patch_shapes[spacing], dtype=np.uint8),
                                'labels': np.empty((batch_size,), dtype=np.uint8) if self.__label_mode == 'central' else np.empty((batch_size, 1) + self.__patch_shapes[spacing], dtype=np.uint8)}
                      for spacing in self.__patch_shapes}

        # Add weight arrays if necessary.
        #
        if self.__weight_mapper is not None:
            for spacing in self.__patch_shapes:
                batch_dict[spacing]['weights'] = np.empty((batch_size, 1) + self.__patch_shapes[spacing], dtype=np.float32)

        # Check operation mode: single or multi-processing.
        #
        if self.__process_count == 0:
            # Construct batch.
            #
            first_patch_index = 0
            for source_item in sample_counts:
                # Extract patches.
                #
                patch_counts = sample_counts[source_item]
                total_patch_count = sum(count for count in patch_counts.values())
                extract_shapes = self.__patch_augmenter.shapes(target_shapes=self.__patch_shapes) if self.__patch_augmenter is not None else self.__patch_shapes

                self.__logger.debug('{tag}Extracting patches of shapes {shape} with {dist} distribution from \'{source}\''.format(tag=self.__log_name_tag,
                                                                                                                                  shape=self.__patch_shapes,
                                                                                                                                  dist=patch_counts,
                                                                                                                                  source=source_item.image))

                patch_dict = self.__patch_samplers[source_item].sample(counts=patch_counts, shapes=extract_shapes)

                # Apply augmentation.
                #
                if self.__patch_augmenter is not None:
                    patch_dict = self.__patch_augmenter.process(patches=patch_dict, shapes=self.__patch_shapes, randomize=True)

                # Apply label mapping. Generate valid label map if the weight mapper is also configured.
                #
                if self.__label_mapper is not None:
                    patch_dict = self.__label_mapper.process(patches=patch_dict, valid_map=self.__weight_mapper is not None)

                # Apply weight mapping.
                #
                if self.__weight_mapper is not None:
                    patch_dict = self.__weight_mapper.calculate(patches=patch_dict)

                # Push extracted patches, labels, and weights to the batch.
                #
                for spacing in patch_dict:
                    batch_dict[spacing]['patches'][first_patch_index: first_patch_index + total_patch_count] = patch_dict[spacing]['patches']
                    batch_dict[spacing]['labels'][first_patch_index: first_patch_index + total_patch_count] = patch_dict[spacing]['labels']

                    if self.__weight_mapper is not None:
                        batch_dict[spacing]['weights'][first_patch_index: first_patch_index + total_patch_count] = patch_dict[spacing]['weights']

                first_patch_index += total_patch_count
        else:
            # Send pings to the unused processes to keep them alive.
            #
            waiting_process_set = set(self.__command_pipes.keys()) - set(self.__patch_samplers[source_item] for source_item in self.__patch_samplers if source_item in sample_counts)
            for target_process_id in waiting_process_set:
                if not self.__checkprocess(process_id=target_process_id):
                    # The worker process is not alive.
                    #
                    self.__logger.error('{tag}Process {pid} terminated before patch extraction'.format(tag=self.__log_name_tag, pid=target_process_id))

                    raise dptprocesserrors.ProcessTerminatedError(target_process_id)

            # Issue sample collection commands.
            #
            for source_item in sample_counts:
                target_process_id = self.__patch_samplers[source_item]
                if self.__worker_processes[target_process_id].is_alive():
                    self.__logger.debug('{tag}Issuing extraction of patches of shapes {shape} with {dst} distribution to {pid} process from \'{src}\''.format(tag=self.__log_name_tag,
                                                                                                                                                              shape=self.__patch_shapes,
                                                                                                                                                              dst=sample_counts.get(source_item, None),
                                                                                                                                                              pid=target_process_id,
                                                                                                                                                              src=source_item.image))

                    self.__command_pipes[target_process_id].send({'command': 'sample',
                                                                  'source': source_item,
                                                                  'counts': sample_counts[source_item],
                                                                  'shapes': self.__patch_shapes})
                else:
                    self.__logger.error('{tag}Sampler process {pid} terminated before sampling request'.format(tag=self.__log_name_tag, pid=target_process_id))

                    raise dptprocesserrors.ProcessTerminatedError(target_process_id)

            # Calculate target indices for assembling the batch.
            #
            target_indices = {}
            total_patch_count = 0
            for source_item in sample_counts:
                target_indices[source_item] = total_patch_count
                patch_counts = sample_counts[source_item]
                total_patch_count += sum(count for count in patch_counts.values())

            # Collect responses.
            #
            for _ in range(len(sample_counts)):
                # Keep checking if all the child processes are alive. If a process disappears without sending and error response message it is very likely that the system
                # ran out of memory. In case of insufficient memory the python processes are shut down without any trace. Use 'docker stats' command on the host machine to
                # monitor memory usage since inside the docker it seems the all the physical memory is available to the running instance while it is not true.
                #
                for process_id in self.__worker_processes:
                    if not self.__worker_processes[process_id].is_alive():
                        self.__logger.error('{tag}Sampler process {pid} terminated before sampling response'.format(tag=self.__log_name_tag, pid=process_id))

                        raise dptprocesserrors.ProcessTerminatedError(process_id)

                # Pull one message from the inter-process queue.
                #
                try:
                    response_message = self.__response_queue.get(block=True, timeout=self.__response_timeout)

                except queue.Empty as empty_queue_error:
                    # No response in time.
                    #
                    self.__logger.error('{tag}Not all responses arrived in time for \'data\' command'.format(tag=self.__log_name_tag))

                    raise dptprocesserrors.ProcessResponseTimeoutError(self.__response_timeout, empty_queue_error)

                if response_message['response'] != 'data':
                    self.__logger.error('{tag}Unexpected response from {pid} sampler process for \'data\' command: {message}'.format(tag=self.__log_name_tag,
                                                                                                                                     pid=response_message.get('pid', None),
                                                                                                                                     message=response_message))

                    raise dptprocesserrors.UnexpectedProcessResponseError(response_message['pid'], 'sample', response_message)

                # Save the content of the message. The augmentation is already done in the child processes.
                #
                patch_source = response_message['source']
                patch_dict = response_message['patches']
                patch_count = next(iter(patch_dict.values()))['patches'].shape[0]
                target_index = target_indices[patch_source]

                for spacing in patch_dict:
                    batch_dict[spacing]['patches'][target_index: target_index + patch_count] = patch_dict[spacing]['patches']
                    batch_dict[spacing]['labels'][target_index: target_index + patch_count] = patch_dict[spacing]['labels']

                    if self.__weight_mapper is not None:
                        batch_dict[spacing]['weights'][target_index: target_index + patch_count] = patch_dict[spacing]['weights']

        # Return the assembled batch of patches and labels.
        #
        return batch_dict
