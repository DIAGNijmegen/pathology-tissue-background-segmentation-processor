"""
This module can copy, move, anonymize and convert image files and contents of a JSON or YAML file.
"""

from ..patch import patchsource as dptpatchsource
from . import foldercontent as dptfoldercontent

import logging
import shutil
import os

#----------------------------------------------------------------------------------------------------

def check_image(image_path, allow_missing):
    """
    Check if source image exists.

    Args:
        image_path (str): Image path. If it is an MRXS file the corresponding directory will also be checked.
        allow_missing (bool): If true missing image will be logged as 'Absent' but reported as missing.

    Returns:
        bool: True if slide exists.
    """

    # Get a logger.
    #
    logger = logging.getLogger(name=__name__)

    # Check image path.
    #
    image_exists = os.path.exists(image_path)
    image_base_pair = os.path.splitext(image_path)
    if image_base_pair[1].lower() == '.mrxs':
        image_exists = image_exists and os.path.exists(image_base_pair[0])

    if image_exists:
        logger.info('Exists: {path}'.format(path=image_path))
        image_ok = True

    elif allow_missing:
        logger.info('Absent: {path}'.format(path=image_path))
        image_ok = True

    else:
        logger.info('Missing: {path}'.format(path=image_path))
        image_ok = False

    # Return result.
    #
    return image_ok

#----------------------------------------------------------------------------------------------------

def relocate_image(source_path, target_path, move, overwrite):
    """
    Move image to the target path.

    Args:
        source_path (str): Source image path. If it is an MRXS file the corresponding directory will also be copied/moved.
        target_path (str): Target image path. If it is file name the source file will be renamed.
        move (bool): If true the original files are removed.
        overwrite (bool): Overwrite existing target.

    Returns:
        bool, str: Image copied or moved and target image path.
    """

    # Get a logger.
    #
    logger = logging.getLogger(name=__name__)

    # Create operation string.
    #
    operation = 'Move' if move else 'Copy'

    # Construct target path.
    #
    target_file_path = os.path.join(target_path, os.path.basename(source_path)) if target_path.endswith(os.path.sep) else target_path

    # Check it target already exists.
    #
    if os.path.isfile(target_file_path) and not overwrite:
        logger.info('Exists: {path}'.format(path=target_file_path))

        relocated = False
    else:
        logger.info('{op}: {src} -> {dst}'.format(op=operation, src=source_path, dst=target_path))

        # Create directory structure if necessary.
        #
        target_dir_path = os.path.dirname(target_file_path)
        os.makedirs(target_dir_path, exist_ok=True)

        # Copy or move file.
        #
        if move:
            shutil.move(src=source_path, dst=target_file_path, copy_function=shutil.copyfile)
        else:
            shutil.copyfile(src=source_path, dst=target_file_path)

        relocated = True

    # Copy corresponding directory if the file was MRXS.
    #
    source_extension = os.path.splitext(source_path)[1].lower()
    if source_extension == '.mrxs':
        source_content_directory = os.path.splitext(source_path)[0]
        target_content_directory = os.path.splitext(target_file_path)[0]

        # Collect files from the MRXS content directory.
        #
        file_collection = dptfoldercontent.folder_content(folder_path=source_content_directory, filter_exp=None, recursive=False)
        file_mapping = {file_item: os.path.join(target_content_directory, os.path.basename(file_item)) for file_item in file_collection}

        # Check if target already exists.
        #
        if all(os.path.isfile(target_file_item) for target_file_item in file_mapping.values()) and not overwrite:
            logger.info('Exists: {path}'.format(op=operation, path=target_content_directory))

        else:
            logger.info('{op}: {src} -> {dst}'.format(op=operation, src=source_content_directory, dst=target_content_directory))

            # Create target directory.
            #
            os.makedirs(target_content_directory, exist_ok=True)

            # Copy content.
            #
            for source_file_path, target_file_path in file_mapping.items():
                if not os.path.isfile(target_file_path) or overwrite:
                    if move:
                        shutil.move(src=source_file_path, dst=target_file_path, copy_function=shutil.copyfile)
                    else:
                        shutil.copyfile(src=source_file_path, dst=target_file_path)

            # Remove the old content directory if necessary.
            #
            if move and not os.listdir(source_content_directory):
                shutil.rmtree(path=source_content_directory, ignore_errors=True)

            relocated = True

    # Return the relocated flag and the target file path.
    #
    return relocated, target_file_path

#----------------------------------------------------------------------------------------------------

def copy_image(source_path, target_path, overwrite):
    """
    Copy image file or image file with corresponding directory in case of MRXS.

    Args:
        source_path (str): Source image path. 
        target_path (str): Target image path.
        overwrite (bool): Overwrite existing target.

    Returns:
        bool: True if the image is copied, false if skipped.
    """

    # Copy image.
    #
    copied, _ = relocate_image(source_path=source_path, target_path=target_path, move=False, overwrite=overwrite)
    return copied

#----------------------------------------------------------------------------------------------------

def move_image(source_path, target_path, overwrite):
    """
    Move image file or image file with corresponding directory in case of MRXS.

    Args:
        source_path (str): Source image path.
        target_path (str): Target image path.
        overwrite (bool): Overwrite existing target.

    Returns:
        bool: True if the image is copied, false if skipped.
    """

    # Move image.
    #
    moved, _ = relocate_image(source_path=source_path, target_path=target_path, move=True, overwrite=overwrite)
    return moved

#----------------------------------------------------------------------------------------------------

def remove_image(image_path, ignore_errors=True):
    """
    Remove image.

    Args:
        image_path (str): Image path.
        ignore_errors (bool): Ignore file non-existence errors.
    """

    # Get a logger.
    #
    logger = logging.getLogger(name=__name__)

    logger.info('Remove: {slide}'.format(slide=image_path))

    # Delete file.
    #
    if not ignore_errors or os.path.isfile(path=image_path):
        os.remove(path=image_path)

    # If the file was MRXS delete the corresponding directory too.
    #
    folder, extension = os.path.splitext(image_path)
    if extension.lower() == '.mrxs':
        shutil.rmtree(path=folder, ignore_errors=ignore_errors)

#----------------------------------------------------------------------------------------------------

def remove_patch_source(patch_source, path_replacements=None, ignore_errors=True):
    """
    Removes the given image, mask pair.

    Args:
        patch_source (dptpatchsource.PatchSource): Patch source with paths.
        path_replacements (dict, None): Path replacements.
        ignore_errors (bool): Ignore file non-existence errors.
    """

    # Remove the image.
    #
    image_path = os.path.normpath(patch_source.image.format(**path_replacements) if path_replacements is not None else patch_source.image)
    remove_image(image_path=image_path, ignore_errors=ignore_errors)

    # Remove the mask.
    #
    if patch_source.mask:
        mask_path = os.path.normpath(patch_source.mask.format(**path_replacements) if path_replacements is not None else patch_source.mask)
        remove_image(image_path=mask_path, ignore_errors=ignore_errors)

    # Remove the stat.
    #
    if patch_source.stat:
        stat_path = os.path.normpath(patch_source.stat.format(**path_replacements) if path_replacements is not None else patch_source.stat)
        remove_image(image_path=stat_path, ignore_errors=ignore_errors)

#----------------------------------------------------------------------------------------------------

def copy_patch_source(patch_source, source_map, target_map, allow_missing_stat, overwrite):
    """
    Copies the given image, mask, stat tripled to the target path.

    Args:
        patch_source (dptpatchsource.PatchSource): Patch source with paths.
        source_map (dict): Source path replacements.
        target_map (dict): Target path replacements.
        allow_missing_stat (bool): Allow configured but missing stat file.
        overwrite (bool): Overwrite existing targets.

    Returns:
        int, int: Number of copied and skipped items.
    """

    # Get a logger.
    #
    logger = logging.getLogger(name=__name__)

    # Count the number of copied and skipped items.
    #
    copied_count = 0
    skipped_count = 0

    # Copy the image.
    #
    image_source_path = os.path.normpath(patch_source.image.format(**source_map))
    image_target_path = os.path.normpath(patch_source.image.format(**target_map))

    image_copied = copy_image(source_path=image_source_path, target_path=image_target_path, overwrite=overwrite)

    if image_copied:
        copied_count += 1
    else:
        skipped_count += 1

    # Copy the mask if present.
    #
    if patch_source.mask:
        mask_source_path = os.path.normpath(patch_source.mask.format(**source_map))
        mask_target_path = os.path.normpath(patch_source.mask.format(**target_map))

        mask_copied = copy_image(source_path=mask_source_path, target_path=mask_target_path, overwrite=overwrite)

        if mask_copied:
            copied_count += 1
        else:
            skipped_count += 1

    # Copy the stat file if present. If the file is configured but missing create the target directory for it.
    #
    if patch_source.stat:
        stat_source_path = os.path.normpath(patch_source.stat.format(**source_map))
        stat_target_path = os.path.normpath(patch_source.stat.format(**target_map))

        if os.path.isfile(stat_source_path) or not allow_missing_stat:
            stat_copied = copy_image(source_path=stat_source_path, target_path=stat_target_path, overwrite=overwrite)

            if stat_copied:
                copied_count += 1
            else:
                skipped_count += 1

        else:
            logger.info('Absent: {path}'.format(path=stat_source_path))

            os.makedirs(os.path.dirname(stat_target_path), exist_ok=True)

    # Return the number of copied and skipped items.
    #
    return copied_count, skipped_count

#----------------------------------------------------------------------------------------------------

def check_patch_source(patch_source, allow_missing_stat):
    """
    Check the existence of the given image, mask, stat triplet.

    Args:
        patch_source (dptpatchsource.PatchSource): Patch source with paths.
        allow_missing_stat (bool): Allow configured but missing stat file.

    Returns:
        int, int: Number of okay and missing items.
    """

    # Count the number of copied and skipped items.
    #
    okay_count = 0
    missing_count = 0

    # Check the image.
    #
    if check_image(image_path=patch_source.image, allow_missing=False):
        okay_count += 1
    else:
        missing_count += 1

    # Copy the mask if present.
    #
    if patch_source.mask:
        if check_image(image_path=patch_source.mask, allow_missing=False):
            okay_count += 1
        else:
            missing_count += 1

    # Copy the stat if present.
    #
    if patch_source.stat:
        if check_image(image_path=patch_source.stat, allow_missing=allow_missing_stat):
            okay_count += 1
        else:
            missing_count += 1

    # Return the number of okay and missing items.
    #
    return okay_count, missing_count

#----------------------------------------------------------------------------------------------------

def copy_batch_source(batch_source, source_replacements, target_replacements, purposes, categories, allow_missing_stat, overwrite):
    """
    Copy the batch source content.

    Args:
        batch_source (dptbatchsource.BatchSource): Batch source to copy.
        source_replacements (dict, None): Source path replacements.
        target_replacements (dict, None): Target path replacements.
        purposes (str, list, None): Purposes to copy. If None, all purposes are copied.
        categories (str, list, None): Categories to copy. If None, all categories are copied.
        allow_missing_stat (bool): Allow configured but missing stat file.
        overwrite (bool): Overwrite existing targets.

    Returns:
        int, int: Number of copied items and number of skipped items.
    """

    # Print the number of copied and skipped items.
    #
    copied_items = 0
    skipped_items = 0

    # Calculate the source and target replacements.
    #
    complete_source_replacements = batch_source.replacements()
    if source_replacements is not None:
        complete_source_replacements.update(source_replacements)

    complete_target_replacements = batch_source.replacements()
    if target_replacements is not None:
        complete_target_replacements.update(target_replacements)

    # Process each image-mask pair in the batch source.
    #
    for source_item in batch_source.items(purpose_id=purposes, category_id=categories, replace=False):
        # Copy items.
        #
        copied_from_source, skipped_from_source = copy_patch_source(patch_source=source_item,
                                                                    source_map=complete_source_replacements,
                                                                    target_map=complete_target_replacements,
                                                                    allow_missing_stat=allow_missing_stat,
                                                                    overwrite=overwrite)
        copied_items += copied_from_source
        skipped_items += skipped_from_source

    # Return the statistics.
    #
    return copied_items, skipped_items

#----------------------------------------------------------------------------------------------------

def check_batch_source(batch_source, purposes, categories, allow_missing_stat):
    """
    Check the existence of the given batch source content.

    Args:
        batch_source (dptbatchsource.BatchSource): Batch source to copy.
        purposes (str, list, None): Purposes to copy. If None, all purposes are copied.
        categories (str, list, None): Categories to copy. If None, all categories are copied.
        allow_missing_stat (bool): Allow configured but missing stat file.

    Returns:
        int, int: Number of okay items and number of missing items.
    """

    # Print the number of copied and skipped items.
    #
    okay_items = 0
    missing_items = 0

    # Process each image-mask pair in the batch source.
    #
    for source_item in batch_source.items(purpose_id=purposes, category_id=categories, replace=True):
        # Copy items.
        #
        okay_in_source, missing_in_source = check_patch_source(patch_source=source_item, allow_missing_stat=allow_missing_stat)
        okay_items += okay_in_source
        missing_items += missing_in_source

    # Return the statistics.
    #
    return okay_items, missing_items
