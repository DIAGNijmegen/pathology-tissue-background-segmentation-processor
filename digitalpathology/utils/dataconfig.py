"""
This module can build a data configuration file based on contents of a folder.
"""

from . import foldercontent as dptfoldercontent
from . import filesynchronizer as dptfilesynchronizer
from ..batch import batchgenerator as dptbatchgenerator
from ..batch import batchsource as dptbatchsource
from ..image import imagereader as dptimagereader

import numpy as np
import logging
import datetime
import time
import random
import sys
import os

#----------------------------------------------------------------------------------------------------

def _find_labels_in_mask(mask_path, read_level, read_spacing, spacing_tolerance, allowed_labels):
    """
    Find and return the available labels in a mask path by reading the mask file at a high level.

    Args:
        mask_path (string): Path to the mask file.
        read_level (int, None): The level from which patches should be extracted.
        read_spacing (float, None): Pixel spacing of the patches to be extracted (micrometer).
        spacing_tolerance (float, None): Pixel spacing tolerance (percentage).
        allowed_labels (list): List of labels that can be present.

    Returns:
        list: List of labels that are in the mask and are present in the allowed_labels argument.

    Raises:
        DigitalPathologyConfigError: Configuration errors.
        DigitalPathologyImageError: Image errors.
    """

    # Open the image.
    #
    mask_image = dptimagereader.ImageReader(image_path=mask_path, spacing_tolerance=spacing_tolerance, input_channels=None, cache_path=None)

    # Calculate processing level and add missing spacing information.
    #
    mask_level = mask_image.level(spacing=read_spacing) if read_level is None else read_level

    if any(mask_spacing is None for mask_spacing in mask_image.spacings):
        mask_image.correct(spacing=1.0, level=0)

    # Read the full image at this level.
    #
    mask_image_content = mask_image.content(spacing=mask_image.spacings[mask_level])
    labels_in_mask = np.unique(mask_image_content)

    # Intersect the set of collected labels with the set of allowed labels.
    #
    return np.intersect1d(labels_in_mask, allowed_labels).tolist()

#----------------------------------------------------------------------------------------------------

def _find_common_dir_seq(image_dir_seq, mask_dir_seq, stat_dir_seq):
    """
    Find the longest common directory path at the beginning of the image, mask and stats root directories.

    Args:
        image_dir_seq (list): Directory sequence of image root path. Image root path split by the path separator.
        mask_dir_seq (list): Directory sequence of mask root path. Mask root path split by the path separator.
        stat_dir_seq (list): Directory sequence of stat root path. Stat root path split by the path separator.

    Returns:
        int, int, int, int: Longest common root sequence length, image path sequence length, mask path sequence length, stat path sequence length after root.
    """

    # Identify the longest sequence in paths as root directory.
    #
    min_len = min(len(dir_sequence) for dir_sequence in [image_dir_seq, mask_dir_seq, stat_dir_seq] if dir_sequence)

    # Find the common root directory. Consider that in case of linux paths the path split by the path separator yields a list with an empty string as a first element.
    #
    root_include_len = 0
    while root_include_len < min_len and (os.path.isdir(os.path.sep.join(image_dir_seq[0:root_include_len + 1])) or image_dir_seq[0:root_include_len + 1] == ['']):

        if (not mask_dir_seq or image_dir_seq[0:root_include_len + 1] == mask_dir_seq[0:root_include_len + 1]) and \
                (not stat_dir_seq or image_dir_seq[0:root_include_len + 1] == stat_dir_seq[0:root_include_len + 1]):
            root_include_len += 1
        else:
            break

    # Find the whole-slide image directory.
    #
    image_include_len = root_include_len
    while image_include_len <= len(image_dir_seq) and (os.path.isdir(os.path.sep.join(image_dir_seq[0:image_include_len + 1])) or image_dir_seq[0:image_include_len + 1] == ['']):
        image_include_len += 1
    image_include_len -= root_include_len

    # Find the mask directory.
    #
    if mask_dir_seq:
        mask_include_len = root_include_len
        while mask_include_len <= len(mask_dir_seq) and (os.path.isdir(os.path.sep.join(mask_dir_seq[0:mask_include_len + 1])) or mask_dir_seq[0:mask_include_len + 1] == ['']):
            mask_include_len += 1
        mask_include_len -= root_include_len

    else:
        mask_include_len = 0

    # Find the stat directory.
    #
    if stat_dir_seq:
        stat_include_len = root_include_len
        while stat_include_len <= len(stat_dir_seq) and (os.path.isdir(os.path.sep.join(stat_dir_seq[0:stat_include_len + 1])) or stat_dir_seq[0:stat_include_len + 1] == ['']):
            stat_include_len += 1
        stat_include_len -= root_include_len

    else:
        stat_include_len = 0

    # Return the identified directory sequences.
    #
    return root_include_len, image_include_len, mask_include_len, stat_include_len

#----------------------------------------------------------------------------------------------------

def _replace_tags_in_path(directory_path, root_seq_len, root_tag, class_seq_len, class_tag):
    """
    Replace root and class tags in the image path.

    Args:
        directory_path (str): Path to process.
        root_seq_len (int): Sequence length of root part.
        root_tag (str): Tag to use for root part.
        class_seq_len (int): Sequence length of class part.
        class_tag (str): Tag to use for class part.

    Returns:
        str: Image path replaced tags.
    """

    # Decompose the directory path to list of directories.
    #
    directory_seq = directory_path.split(sep=os.path.sep)

    # Replace the root and class parts with tags.
    #
    if 0 < root_seq_len:
        result_directory_seq = ['{{{tag}}}'.format(tag=root_tag)] + directory_seq[root_seq_len:]
        if 0 < class_seq_len:
            result_directory_seq = [result_directory_seq[0], '{{{tag}}}'.format(tag=class_tag)] + result_directory_seq[1 + class_seq_len:]
    else:
        result_directory_seq = directory_seq

    # Assemble the new path with the tags.
    #
    return os.path.sep.join(result_directory_seq)

#----------------------------------------------------------------------------------------------------

def _repeat_batch_saving(output_path_list, batch_generator, file_sync, random_seed, compress, overwrite):
    """
    Repeat the batch saving process.

    Args:
        output_path_list (list): Output NPZ file path list.
        batch_generator (dptbatchgenerator.BatchGenerator): Configured batch generator.
        file_sync (dptfilesynchronizer.FileSynchronizer): File synchronizer object.
        random_seed (int, None): Random seed.
        compress (bool): Compress the output .NPZ file.
        overwrite (bool): Overwrite existing target.

    Raises:
        DigitalPathologyBufferError: Buffer errors.
        DigitalPathologyConfigError: Configuration errors.
        DigitalPathologyDataError: Data errors.
        DigitalPathologyImageError: Image errors.
        DigitalPathologyLabelError: Label errors.
        DigitalPathologyProcessError: Process errors.
        DigitalPathologyWeightMappingError: Weight mapping errors.
    """

    # Log parameters.
    #
    logger = logging.getLogger(name=__name__)

    # Initialize the batch generator.
    #
    batch_generator.start()
    batch_generator.step()

    # Repeat the extraction process.
    #
    for repetition in range(len(output_path_list)):
        # Start time measurement.
        #
        repetition_start_time = time.time()

        if 1 < len(output_path_list):
            logger.info('Repetition: {index} of {total} --------------------------------'.format(index=repetition + 1, total=len(output_path_list)))

        # Check if the target image already exits.
        #
        target_output_path = output_path_list[repetition]
        if not os.path.isfile(target_output_path) or overwrite:
            # Generate a batch and save it to file
            #
            logger.info('Extracting {total} patches...'.format(total=batch_generator.size))

            batch_generator.fill()
            batch, _ = batch_generator.batch(batch_size=batch_generator.size)

            # Lay out the batch in a savable format.
            #
            spacing_list = [spacing for spacing in batch]
            label_dist_list = [batch_generator.labeldistribution[label] if label in batch_generator.labeldistribution else 0.0 for label in range(np.iinfo(np.uint8).max)]

            batch['spacings'] = np.asarray(a=spacing_list, dtype=np.float64)
            batch['mask'] = np.asarray(a=[batch_generator.maskspacing], dtype=np.float64)
            batch['distribution'] = np.asarray(a=label_dist_list, dtype=np.float64)
            batch['channels'] = np.asarray(a=batch_generator.inputchannels, dtype=np.int64)
            batch['order'] = np.asarray(a=list(batch_generator.dimensionorder.encode(encoding='utf-8')), dtype=np.uint8)
            batch['seed'] = np.asarray(a=[random_seed], dtype=np.int64)
            batch['repetition'] = np.asarray(a=[repetition], dtype=np.int64)

            for spacing_index in range(len(spacing_list)):
                batch['patches_{index}'.format(index=spacing_index)] = batch[spacing_list[spacing_index]]['patches']
                batch['labels_{index}'.format(index=spacing_index)] = batch[spacing_list[spacing_index]]['labels']

                if 'weights' in batch[spacing_list[spacing_index]]:
                    batch['weights_{index}'.format(index=spacing_index)] = batch[spacing_list[spacing_index]]['weights']

                del batch[spacing_list[spacing_index]]

            # Save the formatted batch to file.
            #
            work_output_path = file_sync.work(target_path=target_output_path)
            logger.info('Saving patches to: {path}'.format(path=work_output_path))

            if compress:
                np.savez_compressed(file=work_output_path, **batch)
            else:
                np.savez(file=work_output_path, **batch)

            # Sync results to target path.
            #
            if work_output_path != target_output_path:
                logger.info('Synchronizing to: {path}'.format(path=target_output_path))

            file_sync.sync(target_path=target_output_path, move=True)

            # Log execution time.
            #
            repetition_execution_time = time.time() - repetition_start_time
            logger.debug('Done in {delta}'.format(delta=datetime.timedelta(seconds=repetition_execution_time)))

        else:
            logger.info('Skipping, target file already exits: {path}'.format(path=target_output_path))

#----------------------------------------------------------------------------------------------------

def build_batch_source(image_path, mask_path, stat_path, labels, read_level, read_spacing, purpose_distribution, mask_spacing_tolerance, random_item_order):
    """
    Build a BatchSource container that can be dumped to config file from matches with the configured image, mask and stat filters.

    Args:
        image_path (str): Slide image filter expression.
        mask_path (str, None): Mask image filter expression.
        stat_path (str, None): Stat file filter expression.
        labels (list): List of labels to add.
        read_level (int, None): Level to read the masks on for extracting label information. Setting both this and mask_spacing arguments to None disables this functionality.
        read_spacing (float, None): Pixel spacing to read the masks on for extracting label information (micrometer). Setting both this and mask_level arguments to None disables this functionality.
        purpose_distribution (dict, None): Purpose distribution.
        mask_spacing_tolerance (float, None): Pixel spacing tolerance (percentage).
        random_item_order (bool): Randomize item order in the output file.

    Returns:
        dptbatchsource.BatchSource: BatchSource object.

    Raises:
        DigitalPathologyConfigError: Configuration errors.
        DigitalPathologyImageError: Image errors.
    """

    # Start time measurement.
    #
    start_time = time.time()

    # Log parameters.
    #
    logger = logging.getLogger(name=__name__)

    logger.info('Constructing BatchSource object...')
    logger.info('Image path filter: {path}'.format(path=image_path))
    logger.info('Mask path filter: {path}'.format(path=mask_path))
    logger.info('Stat path filter: {path}'.format(path=stat_path))
    logger.info('Label list: {labels}'.format(labels=labels))

    if read_level is None:
        if read_spacing is None:
            logger.info('Label content checking is disabled. The masks are assumed to have the configured label list.')
        else:
            logger.info('Label content checking pixel spacing: {spacing} um'.format(spacing=read_spacing))
    else:
        logger.info('Label content checking  level: {level}'.format(level=read_level))

    logger.info('Purpose distribution: {purposes}'.format(purposes=purpose_distribution))
    logger.info('Randomize item order: {flag}'.format(flag=random_item_order))

    # Hardcoded values:
    #
    root_tag = 'root'
    images_tag = 'images'
    masks_tag = 'masks'
    stats_tag = 'stats'

    # Normalize paths.
    #
    norm_image_path = os.path.normpath(image_path)
    norm_mask_path = os.path.normpath(mask_path) if mask_path else ''
    norm_stat_path = os.path.normpath(stat_path) if stat_path else ''

    # Sequence path into directory elements.
    #
    image_path_seq = norm_image_path.split(sep=os.path.sep)
    mask_path_seq = norm_mask_path.split(sep=os.path.sep) if norm_mask_path else []
    stat_path_seq = norm_stat_path.split(sep=os.path.sep) if norm_stat_path else []

    # Collect image list from the root directory.
    #
    image_file_path_list = dptfoldercontent.folder_content(norm_image_path, recursive=False)
    image_file_path_map = {os.path.basename(image_path_item): image_path_item for image_path_item in image_file_path_list}

    # Replace the image matching string in the mask path to an asterisk to be able to collect all possible mask files.
    #
    mask_wildcard_path = norm_mask_path.format(image='*')
    mask_file_path_list = dptfoldercontent.folder_content(folder_path=mask_wildcard_path, recursive=False)
    mask_file_path_map = {os.path.basename(mask_path_item): mask_path_item for mask_path_item in mask_file_path_list}

    # Replace the image matching string in the mask path to an asterisk to be able to collect all possible stat files.
    #
    stat_wildcard_path = norm_stat_path.format(image='*')
    stat_file_path_list = dptfoldercontent.folder_content(folder_path=stat_wildcard_path, recursive=False)
    stat_file_path_map = {os.path.basename(stat_path_item): stat_path_item for stat_path_item in stat_file_path_list}

    # Find the longest common sequences in paths.
    #
    root_dir_len, image_dir_len, mask_dir_len, stat_dir_len = _find_common_dir_seq(image_dir_seq=image_path_seq, mask_dir_seq=mask_path_seq, stat_dir_seq=stat_path_seq)

    root_dir = os.path.sep.join(image_path_seq[:root_dir_len])
    image_dir = os.path.sep.join(image_path_seq[root_dir_len:root_dir_len + image_dir_len])
    mask_dir = os.path.sep.join(mask_path_seq[root_dir_len:root_dir_len + mask_dir_len])
    stat_dir = os.path.sep.join(stat_path_seq[root_dir_len:root_dir_len + stat_dir_len])

    # Log identified path information.
    #
    logger.debug('Root directory: {path}'.format(path=root_dir))
    logger.debug('Image directory: {{root}}{sep}{path}'.format(sep=os.path.sep, path=image_dir) if image_dir else '')
    logger.debug('Mask directory: {path}'.format(path='{{root}}{sep}{path}'.format(sep=os.path.sep, path=mask_dir) if mask_dir_len else ''))
    logger.debug('Stat directory: {path}'.format(path='{{root}}{sep}{path}'.format(sep=os.path.sep, path=stat_dir) if stat_dir_len else ''))

    # Order the keys.
    #
    image_file_path_map_ordered_key_list = list(image_file_path_map.keys())
    if random_item_order:
        random.shuffle(image_file_path_map_ordered_key_list)
    else:
        image_file_path_map_ordered_key_list.sort(reverse=False)

    # Process the list of image list.
    #
    mask_match = os.path.basename(norm_mask_path)
    stat_match = os.path.basename(norm_stat_path)
    data_items = []
    skipped_count = 0
    for image_key in image_file_path_map_ordered_key_list:
        image_base = os.path.splitext(image_key)[0]
        mask_key = mask_match.format(image=image_base)
        stat_key = stat_match.format(image=image_base)

        current_mask_file_path = mask_file_path_map.get(mask_key, None)
        current_stat_file_path = stat_file_path_map.get(stat_key, None)

        # Add item only of there is either a mask file or a stats file is defined for it.
        #
        if current_mask_file_path or current_stat_file_path:
            # Construct data item.
            #
            slide_item = {'image': _replace_tags_in_path(directory_path=image_file_path_map[image_key],
                                                         root_seq_len=root_dir_len,
                                                         root_tag=root_tag,
                                                         class_seq_len=image_dir_len,
                                                         class_tag=images_tag)}

            logger.debug('Adding: {path}'.format(path=slide_item['image'].format(**{root_tag: root_dir, images_tag: image_dir})))

            if current_mask_file_path:
                slide_item['mask'] = _replace_tags_in_path(directory_path=current_mask_file_path, root_seq_len=root_dir_len, root_tag=root_tag, class_seq_len=mask_dir_len, class_tag=masks_tag)

                # Load the mask file to determine which labels are present.
                #
                if read_level is not None or read_spacing is not None:
                    slide_item['labels'] = _find_labels_in_mask(mask_path=current_mask_file_path,
                                                                read_level=read_level,
                                                                read_spacing=read_spacing,
                                                                spacing_tolerance=mask_spacing_tolerance,
                                                                allowed_labels=labels)
                else:
                    slide_item['labels'] = list(labels)

            if current_stat_file_path:
                slide_item['stat'] = _replace_tags_in_path(directory_path=current_stat_file_path, root_seq_len=root_dir_len, root_tag=root_tag, class_seq_len=stat_dir_len, class_tag=stats_tag)

            # Add item to the collection.
            #
            data_items.append(slide_item)
        else:
            skipped_count += 1

    # Randomize the order of the data.
    #
    if random_item_order:
        random.shuffle(data_items)

    # Log match count for checking.
    #
    logger.info('Added image count: {match_count}, skipped: {skip_count}'.format(match_count=len(data_items), skip_count=skipped_count))

    # Construct the data structure.
    #
    data_container = {'default': data_items}
    path_replacements = {'root': root_dir, 'images': image_dir, 'masks': mask_dir, 'stats': stat_dir}

    # Construct a batch source object.
    #
    batch_source = dptbatchsource.BatchSource(source_items=data_container)
    batch_source.update(path_replacements=path_replacements)

    # Distribute the patch sources to purposes if configured.
    #
    batch_source.distribute(purpose_distribution=purpose_distribution)

    # Log execution time.
    #
    execution_time = time.time() - start_time
    logger.debug('Done in {delta}'.format(delta=datetime.timedelta(seconds=execution_time)))

    # Return the constructed BatchSource object.
    #
    return batch_source

#----------------------------------------------------------------------------------------------------

def extract_batch_of_patches(batch_source, patch_shapes, mask_spacing, patch_count, label_dist, label_mode, image_channels, purpose, category_dist, spacing_tolerance):
    """
    Extract a batch of patches based on a BatchSource object configuration. A BatchGenerator is instantiated with single threaded, single processing mode in BCHW configuration
    and maximal pool size.

    Args:
        batch_source (dptbatchsource.BatchSource): BatchSource object.
        patch_shapes (dict): Image spacing to patch shape mapping without the channels dimension.
        mask_spacing (float): Mask spacing to use for selecting patch center coordinates.
        patch_count (int): Number of patches to load.
        label_dist (dict): Mask label value to label ratio distribution.
        label_mode (str): Label generation mode. 'central', 'synthesize', or 'load'.
        image_channels (list): Desired channels that are extracted for each patch.
        purpose (str, None): List of purposes to use. If None, all purposes are used.
        category_dist (dict): Image category sampling distribution mapping from image category to ratio.
        spacing_tolerance (float): Pixel spacing tolerance (percentage).

    Returns:
        dict: A {'patches': patches, 'labels': labels, 'weights': weights} dictionary with extracted patches, corresponding labels or label patches and weight maps if extracted.

    Raises:
        DigitalPathologyBufferError: Buffer errors.
        DigitalPathologyConfigError: Configuration errors.
        DigitalPathologyDataError: Data errors.
        DigitalPathologyImageError: Image errors.
        DigitalPathologyLabelError: Label errors.
        DigitalPathologyProcessError: Process errors.
        DigitalPathologyWeightMappingError: Weight mapping errors.
    """

    # Start time measurement.
    #
    start_time = time.time()

    # Log parameters.
    #
    logger = logging.getLogger(name=__name__)

    logger.info('Extracting batch of patches from BatchSource object...')
    logger.info('Patch shapes: {shape}'.format(shape=patch_shapes))
    logger.info('Mask pixel spacing: {spacing} um'.format(spacing=mask_spacing))
    logger.info('Patch count: {count}'.format(count=patch_count))
    logger.info('Label distribution: {dist}'.format(dist=label_dist))
    logger.info('Label mode: {mode}'.format(mode=label_mode))
    logger.info('Image channels: {channels}'.format(channels=image_channels))
    logger.info('Purpose: {purpose}'.format(purpose=purpose))
    logger.info('Category distribution: {dist}'.format(dist=category_dist))

    # Generate values for the batch generator.
    #
    patch_sources = batch_source.collection(purpose_id=purpose, category_id=list(category_dist.keys()), replace=True)
    category_dist = category_dist if category_dist else {category_id: 1.0 for category_id in batch_source.categories(purpose_id=purpose)}

    # Configure the batch generator: channels first, single threaded, single processing configuration.
    #
    logger.debug('Instantiating the batch generator...')

    batch_generator = dptbatchgenerator.BatchGenerator(label_dist=label_dist,
                                                       patch_shapes=patch_shapes,
                                                       mask_spacing=mask_spacing,
                                                       spacing_tolerance=spacing_tolerance,
                                                       input_channels=image_channels,
                                                       dimension_order='BCHW',
                                                       label_mode=label_mode,
                                                       patch_sources=patch_sources,
                                                       category_dist=category_dist,
                                                       strict_selection=True,
                                                       create_stats=True,
                                                       main_buffer_size=patch_count,
                                                       buffer_chunk_size=sys.maxsize,
                                                       read_buffer_size=0,
                                                       labels_one_hot=False,
                                                       batch_normalizer=None,
                                                       patch_augmenter=None,
                                                       label_mapper=None,
                                                       free_label_range=True,
                                                       patch_weight_mapper=None,
                                                       batch_weight_mapper=None,
                                                       multi_threaded=False,
                                                       sampler_process_count=0,
                                                       sampler_pool_size=sys.maxsize,
                                                       sampler_chunk_size=sys.maxsize,
                                                       join_timeout=60,
                                                       response_timeout=60,
                                                       poll_timeout=60,
                                                       name_tag=None)

    # Initialize the batch generator.
    #
    logger.debug('Initializing the batch generator...')

    batch_generator.start()
    batch_generator.step()

    # Generate a batch and save it to file
    #
    logger.debug('Extracting patches...')

    batch_generator.fill()

    batch, _ = batch_generator.batch(batch_size=patch_count)

    # Shut down the batch generator.
    #
    logger.debug('Shutting down the batch generator...')

    batch_generator.stop()

    # Log execution time.
    #
    execution_time = time.time() - start_time
    logger.debug('Done in {delta}'.format(delta=datetime.timedelta(seconds=execution_time)))

    # Return the extracted batch.
    #
    return batch

#----------------------------------------------------------------------------------------------------

def save_batch_of_patches(output_path,
                          batch_source,
                          patch_shapes,
                          mask_spacing,
                          patch_count,
                          label_dist,
                          label_mode,
                          image_channels,
                          dimension_order,
                          purpose,
                          category_dist,
                          spacing_tolerance,
                          random_seed,
                          compress,
                          overwrite,
                          file_sync=None):
    """
    Save a batch of patches based on a BatchSource object configuration to NPZ file. A BatchGenerator is instantiated with single threaded, single processing mode in BCHW configuration
    and maximal pool size.

    Args:
        output_path (str, list): Output NPZ file path or paths.
        batch_source (dptbatchsource.BatchSource): BatchSource object.
        patch_shapes (dict): Image spacing to patch shape mapping without the channels dimension.
        mask_spacing (float): Mask spacing to use for selecting patch center coordinates.
        patch_count (int): Number of patches to extract and save.
        label_dist (dict): Mask label value to label ratio distribution.
        label_mode (str): Label generation mode. 'central', 'synthesize', or 'load'.
        image_channels (list): Desired channels that are extracted for each patch.
        dimension_order (str): Dimension order, 'BHWC' or 'BCHW'.
        purpose (str, None): Purpose to use. If None, all purposes are used.
        category_dist (dict): Image category sampling distribution mapping from image category to ratio.
        spacing_tolerance (float): Pixel spacing tolerance (percentage).
        random_seed (int, None): Random seed.
        compress (bool): Compress the output .NPZ file.
        overwrite (bool): Overwrite existing target.
        file_sync (dptfilesynchronizer.FileSynchronizer, None): File synchronizer object.

    Raises:
        DigitalPathologyBufferError: Buffer errors.
        DigitalPathologyConfigError: Configuration errors.
        DigitalPathologyDataError: Data errors.
        DigitalPathologyImageError: Image errors.
        DigitalPathologyLabelError: Label errors.
        DigitalPathologyProcessError: Process errors.
        DigitalPathologyWeightMappingError: Weight mapping errors.
    """

    # Start time measurement.
    #
    start_time = time.time()

    # Log parameters.
    #
    logger = logging.getLogger(name=__name__)

    logger.info('Extracting batch of patches from BatchSource object...')
    logger.info('Output paths: {count}'.format(count=1 if type(output_path) == str else len(output_path)))
    logger.info('Source item count: {count}'.format(count=batch_source.count(purpose_id=purpose, category_id=list(category_dist.keys()))))
    logger.info('Patch shapes: {shape}'.format(shape=patch_shapes))
    logger.info('Mask pixel spacing: {spacing} um'.format(spacing=mask_spacing))
    logger.info('Patch count: {count}'.format(count=patch_count))
    logger.info('Label distribution: {dist}'.format(dist=label_dist))
    logger.info('Label mode: {mode}'.format(mode=label_mode))
    logger.info('Image channels: {channels}'.format(channels=image_channels))
    logger.info('Purpose: {purpose}'.format(purpose=purpose))
    logger.info('Category distribution: {dist}'.format(dist=category_dist))
    logger.info('Pixel spacing tolerance: {tolerance}'.format(tolerance=spacing_tolerance))
    logger.info('Random seed: {seed}'.format(seed=random_seed))
    logger.info('Compress output: {flag}'.format(flag=compress))
    logger.info('Overwrite existing results: {flag}'.format(flag=overwrite))

    # Create file synchronizer.
    #
    actual_file_sync = file_sync if file_sync is not None else dptfilesynchronizer.FileSynchronizer(work_directory=None)

    # Initialize the random number generators. If the seed is None, the system time is used as seed.
    #
    actual_random_seed = random_seed if random_seed is not None else int(time.time() * 10000000)
    random.seed(a=actual_random_seed, version=2)
    np.random.seed(seed=random.randint(a=0, b=np.iinfo(np.uint32).max))

    # Generate values for the batch generator.
    #
    patch_sources = batch_source.collection(purpose_id=purpose, category_id=list(category_dist.keys()), replace=True)
    category_dist = category_dist if category_dist else {category_id: 1.0 for category_id in batch_source.categories(purpose_id=purpose)}

    # Configure the batch generator: channels first, single threaded, single processing configuration.
    #
    logger.info('Initializing the batch generator...')

    batch_generator = dptbatchgenerator.BatchGenerator(label_dist=label_dist,
                                                       patch_shapes=patch_shapes,
                                                       mask_spacing=mask_spacing,
                                                       spacing_tolerance=spacing_tolerance,
                                                       input_channels=image_channels,
                                                       dimension_order=dimension_order,
                                                       label_mode=label_mode,
                                                       patch_sources=patch_sources,
                                                       category_dist=category_dist,
                                                       strict_selection=True,
                                                       create_stats=True,
                                                       main_buffer_size=patch_count,
                                                       buffer_chunk_size=sys.maxsize,
                                                       read_buffer_size=0,
                                                       labels_one_hot=False,
                                                       batch_normalizer=None,
                                                       patch_augmenter=None,
                                                       label_mapper=None,
                                                       free_label_range=True,
                                                       patch_weight_mapper=None,
                                                       batch_weight_mapper=None,
                                                       multi_threaded=False,
                                                       sampler_process_count=0,
                                                       sampler_pool_size=sys.maxsize,
                                                       sampler_chunk_size=sys.maxsize,
                                                       join_timeout=60,
                                                       response_timeout=60,
                                                       poll_timeout=60,
                                                       name_tag=None)

    # Repeat the batch saving procedure.
    #
    output_path_list = [output_path] if type(output_path) == str else list(output_path)

    _repeat_batch_saving(output_path_list=output_path_list,
                         batch_generator=batch_generator,
                         file_sync=actual_file_sync,
                         random_seed=actual_random_seed,
                         compress=compress,
                         overwrite=overwrite)

    # Shut down the batch generator.
    #
    logger.info('Shutting down the batch generator...')

    batch_generator.stop()

    # Log total execution time.
    #
    execution_time = time.time() - start_time
    logger.debug('Complete batch saving done in {delta}'.format(delta=datetime.timedelta(seconds=execution_time)))

#----------------------------------------------------------------------------------------------------

def save_config_of_patches(output_path, batch_source, parameters, dimension_order, purpose, cpu_count_enforce, random_seed, compress, overwrite, file_sync=None):
    """
    Save a batch of patches based on a BatchSource object configuration to NPZ file. A BatchGenerator is instantiated with single threaded, single processing mode in BCHW configuration
    and maximal pool size.

    Args:
        output_path (str, list): Output NPZ file path or paths.
        batch_source (dptbatchsource.BatchSource): BatchSource object.
        parameters (dict): Loaded parameter configuration file.
        dimension_order (str): Dimension order, 'BHWC' or 'BCHW'.
        purpose (str) Purpose to use.
        cpu_count_enforce (int, None): Enforced available CPU count.
        random_seed (int, None): Random seed.
        compress (bool): Compress the output .NPZ file.
        overwrite (bool): Overwrite existing target.
        file_sync (dptfilesynchronizer.FileSynchronizer, None): File synchronizer object.

    Raises:
        ValueError: The configured purpose is not available in the parameter configuraions.
        ValueError: The enforced CPU count is not valid.

        DigitalPathologyBufferError: Buffer errors.
        DigitalPathologyConfigError: Configuration errors.
        DigitalPathologyDataError: Data errors.
        DigitalPathologyImageError: Image errors.
        DigitalPathologyLabelError: Label errors.
        DigitalPathologyProcessError: Process errors.
        DigitalPathologyWeightMappingError: Weight mapping errors.
    """

    # Start time measurement.
    #
    start_time = time.time()

    # Log parameters.
    #
    logger = logging.getLogger(name=__name__)

    logger.info('Output paths: {count}'.format(count=1 if type(output_path) == str else len(output_path)))
    logger.info('Source item count: {count}'.format(count=batch_source.count(purpose_id=None, category_id=None)))

    logger.info('Data/Images/Patch shapes: {shapes}'.format(shapes=parameters['data']['images']['patch shapes']))
    logger.info('Data/Spacing tolerance: {tolerance}'.format(tolerance=parameters['data']['spacing tolerance']))
    logger.info('Data/Images/Channels: {channels}'.format(channels=parameters['data']['images']['channels']))
    logger.info('Data/Labels/Mask pixel spacing: {spacing}'.format(spacing=parameters['data']['labels']['mask pixel spacing']))
    logger.info('Data/Labels/Label ratios: {distribution}'.format(distribution=parameters['data']['labels']['label ratios']))
    logger.info('Data/Labels/Label mode: \'{mode}\''.format(mode=parameters['data']['labels']['label mode']))
    logger.info('Data/Labels/Strict selection: {mode}'.format(mode=parameters['data']['labels']['strict selection']))
    logger.info('Data/Categories: {distribution}'.format(distribution=parameters['data']['categories']))

    logger.info('System/Process count: {count}'.format(count=parameters['system']['process count']))
    logger.info('System/Pool size: {size}'.format(size=parameters['system']['pool size']))
    logger.info('System/IPC chunk size: {size}'.format(size=parameters['system']['ipc chunk size']))
    logger.info('System/Join timeout: {secs} sec'.format(secs=parameters['system']['join timeout secs']))
    logger.info('System/Response timeout: {secs} sec'.format(secs=parameters['system']['response timeout secs']))
    logger.info('System/Poll timeout: {secs} sec'.format(secs=parameters['system']['poll timeout secs']))

    logger.info('Dimension order: \'{order}\''.format(order=dimension_order))
    logger.info('Purpose: \'{purpose}\''.format(purpose=purpose))
    logger.info('Enforced CPU count: {count}'.format(count=cpu_count_enforce))
    logger.info('Random seed: {seed}'.format(seed=random_seed))
    logger.info('Compress output: {flag}'.format(flag=compress))
    logger.info('Overwrite existing results: {flag}'.format(flag=overwrite))

    # Check if the purpose is available in both the data and the parameter configurations.
    #
    if purpose not in parameters['training']['iterations']:
        raise ValueError('Purpose not found in parameter configuration: {purpose}'.format(purpose=purpose))

    # Check if the CPU enforce value is valid.
    #
    if cpu_count_enforce is not None and cpu_count_enforce < 1:
        raise ValueError('Invalid CPU count setting: {count}'.format(count=cpu_count_enforce))

    # Create file synchronizer.
    #
    actual_file_sync = file_sync if file_sync is not None else dptfilesynchronizer.FileSynchronizer(work_directory=None)

    # Initialize the random number generators. If the seed is None, the system time is used as seed.
    #
    actual_random_seed = random_seed if random_seed is not None else int(time.time() * 10000000)
    random.seed(a=actual_random_seed, version=2)
    np.random.seed(seed=random.randint(a=0, b=np.iinfo(np.uint32).max))

    # Generate values for the batch generator.
    #
    patch_sources = batch_source.collection(purpose_id=purpose, category_id=list(parameters['data']['categories'].keys()), replace=True)

    # Configure the batch generator: channels first, single threaded, single processing configuration.
    #
    logger.info('Initializing the batch generator...')

    process_count_config = parameters['system']['process count']
    cpu_count = cpu_count_enforce if cpu_count_enforce is not None else os.cpu_count()
    process_count = round(cpu_count * process_count_config) if type(process_count_config) == float else process_count_config
    patch_count = parameters['training']['iterations'][purpose]['batch size'] * parameters['training']['iterations'][purpose]['iteration count']

    logger.info('Available CPUs: {count}'.format(count=cpu_count))
    logger.info('Process count: {processes}'.format(processes=process_count))
    logger.info('Patch count: {patches}'.format(patches=patch_count))

    if parameters['system']['pool size'] < process_count:
        logger.warning('CPU count: {cpu} is larger than the patch sampler pool size: {sampler}'.format(cpu=process_count, sampler=parameters['system']['pool size']))

    batch_generator = dptbatchgenerator.BatchGenerator(label_dist=parameters['data']['labels']['label ratios'],
                                                       patch_shapes=parameters['data']['images']['patch shapes'],
                                                       mask_spacing=parameters['data']['labels']['mask pixel spacing'],
                                                       spacing_tolerance=parameters['data']['spacing tolerance'],
                                                       input_channels=parameters['data']['images']['channels'],
                                                       dimension_order=dimension_order,
                                                       label_mode=parameters['data']['labels']['label mode'],
                                                       patch_sources=patch_sources,
                                                       category_dist=parameters['data']['categories'],
                                                       strict_selection=parameters['data']['labels']['strict selection'],
                                                       create_stats=True,
                                                       main_buffer_size=patch_count,
                                                       buffer_chunk_size=sys.maxsize,
                                                       read_buffer_size=0,
                                                       labels_one_hot=False,
                                                       batch_normalizer=None,
                                                       patch_augmenter=None,
                                                       label_mapper=None,
                                                       free_label_range=True,
                                                       patch_weight_mapper=None,
                                                       batch_weight_mapper=None,
                                                       multi_threaded=False,
                                                       sampler_process_count=process_count,
                                                       sampler_pool_size=parameters['system']['pool size'],
                                                       sampler_chunk_size=parameters['system']['ipc chunk size'],
                                                       join_timeout=parameters['system']['join timeout secs'],
                                                       response_timeout=parameters['system']['response timeout secs'],
                                                       poll_timeout=parameters['system']['poll timeout secs'],
                                                       name_tag=None)

    # Repeat the batch saving procedure.
    #
    output_path_list = [output_path] if type(output_path) == str else list(output_path)

    _repeat_batch_saving(output_path_list=output_path_list,
                         batch_generator=batch_generator,
                         file_sync=actual_file_sync,
                         random_seed=actual_random_seed,
                         compress=compress,
                         overwrite=overwrite)

    # Shut down the batch generator.
    #
    logger.info('Shutting down the batch generator...')

    batch_generator.stop()

    # Log total execution time.
    #
    execution_time = time.time() - start_time
    logger.debug('Complete batch saving done in {delta}'.format(delta=datetime.timedelta(seconds=execution_time)))
