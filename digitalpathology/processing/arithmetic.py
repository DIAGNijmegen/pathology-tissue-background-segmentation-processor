"""
Functions in this module can execute arithmetic operations on images to combine them.
"""

from ..errors import processingerrors as dptprocessingerrors
from ..image import imagereader as dptimagereader
from ..image import imagewriter as dptimagewriter

import numpy as np
import logging
import datetime
import shutil
import time
import os

#----------------------------------------------------------------------------------------------------

def image_arithmetic(left, right, result_path, operand, accept_singles=False, overwrite=True):
    """
    Apply arithmetic operation on to images of the same size and save the result to a new image.

    Args:
        left (dptimagereader.ImageReader, str): Left image path of operation.
        right (dptimagereader.ImageReader, str): Right image path of operation.
        result_path (str): Result image path of operation.
        operand (str): Operation to execute. '+', '-' or '*'
        overwrite (bool): If true existing targets will be overwritten.
        accept_singles (bool): Flag controlling jobs with single participants (either left or right). If true the single participant will be copied as result. Non-existent participants must be None.

    Raises:
        UnknownImageArithmeticOperandError: The image arithmetic operand is unknown.
        ImageShapeMismatchError: The image arithmetic argument image shapes do not match.

        DigitalPathologyConfigError: Configuration errors.
        DigitalPathologyImageError: Image errors.
    """

    # Start time measurement.
    #
    start_time = time.time()

    # Log paths and operand.
    #
    logger = logging.getLogger(name=__name__)

    logger.info('Calculating image arithmetic...')
    logger.info('Left: {path}'.format(path=left.path if isinstance(left, dptimagereader.ImageReader) else left))
    logger.info('Right: {path}'.format(path=right.path if isinstance(right, dptimagereader.ImageReader) else right))
    logger.info('Result: {path}'.format(path=result_path))
    logger.info('Operand: {operand}'.format(operand=operand))

    # Check operand.
    #
    if operand not in ['+', '-', '*']:
        raise dptprocessingerrors.UnknownImageArithmeticOperandError(operand)

    # Check if the target image already exits.
    #
    if not os.path.isfile(result_path) or overwrite:
        # Check if both image is present.
        #
        if accept_singles and (left is None or right is None):
            # One image is missing and the function is configured to copy single participants to result.
            #
            source = left if left else right
            source_path = source.path if isinstance(source, dptimagereader.ImageReader) else source
            shutil.copy(src=source_path, dst=result_path)
        else:
            # Both images present. Execute the arithmetic operation.
            #
            left_image = left if isinstance(left, dptimagereader.ImageReader) else dptimagereader.ImageReader(image_path=left, spacing_tolerance=0.25, input_channels=None, cache_path=None)
            right_image = right if isinstance(right, dptimagereader.ImageReader) else dptimagereader.ImageReader(image_path=right, spacing_tolerance=0.25, input_channels=None, cache_path=None)

            if any(level_spacing is None for level_spacing in left_image.spacings):
                left_image.correct(spacing=1.0, level=0)

            if any(level_spacing is None for level_spacing in right_image.spacings):
                right_image.correct(spacing=1.0, level=0)

            # Check parameters.
            #
            if left_image.shapes[0] != right_image.shapes[0]:
                raise dptprocessingerrors.ImageShapeMismatchError(left_image.path, left_image.shapes[0], right_image.path, right_image.shapes[0])

            # Prepare result image.
            #
            tile_size = 512
            image_writer = dptimagewriter.ImageWriter(image_path=result_path,
                                                      shape=left_image.shapes[0],
                                                      spacing=left_image.spacings[0],
                                                      dtype=left_image.dtype,
                                                      coding=left_image.coding,
                                                      compression=None,
                                                      interpolation=None,
                                                      tile_size=tile_size,
                                                      jpeg_quality=None,
                                                      empty_value=0,
                                                      skip_empty=None,
                                                      cache_path=None)

            # Select operation.
            #
            if operand == '+':
                patch_operation = np.add
            elif operand == '-':
                patch_operation = np.subtract
            elif operand == '*':
                patch_operation = np.multiply
            else:
                patch_operation = None

            # Apply operation patch-by-patch and write out result.
            #
            logger.info('Calculating: {path}'.format(path=result_path))

            for row in range(0, left_image.shapes[0][0], tile_size):
                for col in range(0, left_image.shapes[0][1], tile_size):
                    left_patch = left_image.read(spacing=left_image.spacings[0], row=row, col=col, height=tile_size, width=tile_size)
                    right_patch = right_image.read(spacing=right_image.spacings[0], row=row, col=col, height=tile_size, width=tile_size)
                    result_patch = patch_operation(left_patch, right_patch)

                    image_writer.write(tile=result_patch, row=row, col=col)

            # Finalize the output image and close the input images.
            #
            image_writer.close()

            if not isinstance(right, dptimagereader.ImageReader):
                right_image.close()

            if not isinstance(left, dptimagereader.ImageReader):
                left_image.close()

        # Log execution time.
        #
        execution_time = time.time() - start_time
        logger.debug('Done in {delta}'.format(delta=datetime.timedelta(seconds=execution_time)))

    else:
        logger.info('Skipping, target file already exits: {path}'.format(path=result_path))

#----------------------------------------------------------------------------------------------------

def image_arithmetic_batch(job_list, operand, accept_singles, overwrite=True):
    """
    Apply arithmetic operation on a batch of images.

    Args:
        job_list (list): List of job (left path, right path, result path) tuples.
        operand (str): Operation to execute. '+', '-' or '*'.
        accept_singles (bool): Flag controlling jobs with single participants (either left or right). If true the single participant will be copied as result. Non-existent participants must be None.
        overwrite (bool): If true existing targets will be overwritten.

    Returns:
        (list, list): List of successfully processed items and list of failed items.
    """

    # Start time measurement.
    #
    start_time = time.time()

    # Log batch processing.
    #
    logger = logging.getLogger(name=__name__)

    logger.info('Calculating image arithmetics in batch mode...')
    logger.info('Job count: {count}'.format(count=len(job_list)))

    # Go through the job list.
    #
    failed_collection = []
    successful_collection = []
    for job_index in range(len(job_list)):
        # Get one job item.
        #
        left_path, right_path, result_path = job_list[job_index]

        try:
            # Print data paths if the mode is not a single file mode.
            #
            logger.info('Processing [{index}/{count}]: {path}'.format(index=job_index+1, count=len(job_list), path=result_path))

            # Execute the arithmetic operation.
            #
            image_arithmetic(left=left_path,
                             right=right_path,
                             result_path=result_path,
                             operand=operand,
                             accept_singles=accept_singles,
                             overwrite=overwrite)

        except Exception as exception:
            # Add case to the error collection.
            #
            failed_collection.append(result_path)
            logger.error('Error: {exception}'.format(exception=exception))

        else:
            # Save the mask to the list of successful conversions.
            #
            successful_collection.append(result_path)

    # Log execution time.
    #
    execution_time = time.time() - start_time
    logger.debug('Image arithmetic calculation batch done in {delta}'.format(delta=datetime.timedelta(seconds=execution_time)))

    # Return a list of successful normalizations.
    #
    return successful_collection, failed_collection
