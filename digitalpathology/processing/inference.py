"""
This module can load a network and apply classification on a whole slide image.
"""

from . import conversion as dptconversion
from . import dilate as dptdilate
from . import regions as dptregions
from . import threshold as dptthreshold
from . import zoom as dptzoom

from ..errors import processingerrors as dptprocessingerrors
from ..image import imagereader as dptimagereader
from ..normalizers import generalnormalizer as dtpgeneralnormalizer
from ..normalizers import passthroughnormalizer as dtppassthroughnormalizer
from ..normalizers import rgbnormalizer as dtprgbnormalizer
from ..normalizers import rgbtozeroonenormalizer as dtprgbtozeroonenormalizer
from ..utils import imagefile as dptimagefile
from ..utils import serialize as dptserialize

import diagmodels.models.modelbase as dmsmodelbase

import logging
import datetime
import time
import os
import numpy as np

#----------------------------------------------------------------------------------------------------

def _construct_intermediate_file_paths(output_path, applied_steps):
    """
     Construct intermediate file paths for each step.

    Args:
        output_path:
        applied_steps:

    Returns:
        list: List of individual file paths for every step.
    """

    # Construct file paths for intermediate files.
    #
    output_path_base = os.path.splitext(output_path)[0]
    result_paths = []

    for index in range(len(applied_steps)):
        if applied_steps[index][0]:
            step_suffix = applied_steps[index][1] if any(step_item[0] for step_item in applied_steps[index + 1:]) else ''
            step_path = '{base}{suffix}{ext}'.format(base=output_path_base, suffix=step_suffix, ext=applied_steps[index][2])
        else:
            step_path = result_paths[-1] if result_paths else None

        result_paths.append(step_path)

    return result_paths

#----------------------------------------------------------------------------------------------------

def _load_network_model(model_path, required_shape, unrestrict_network):
    """
    Load network model and instantiate it.

    Args:
        model_path (str): Path of the model file to load.
        required_shape (tuple): Processing patch shape.
        unrestrict_network (bool): Fix network for fixed in and output sizes.

    Returns:
        diagmodels.models.modelbase.ModelBase: Loaded network instance.

    Raises:
        MissingModelFileError: Missing model file.

        DigitalPathologyModelError: Model errors.
    """

    # Check if the model path is a valid file path.
    #
    if not os.path.isfile(path=model_path):
        raise dptprocessingerrors.MissingModelFileError(model_path)

    # Log progress.
    #
    logger = logging.getLogger(name=__name__)

    logger.info('Loading network ...')

    # Load the network data structure separately, in case the input shape needs to be unfixed.
    #
    network_data = dptserialize.load_object(path=model_path)

    # Return the prepared model.
    #
    network = dmsmodelbase.ModelBase.instantiate(file=network_data)

    if unrestrict_network:
        logger.info('Unfixing input shape ...')

        network.unfix()

    return network

#----------------------------------------------------------------------------------------------------

def _find_mask_level(input_image, input_spacing, mask_image):
    """
    Find the mask level to load based on image pixel spacing.

    Args:
        input_image (dptimagereader.ImageReader): Input image object.
        input_spacing (float): Pixel spacing of the input image to use.
        mask_image (dptimagereader.ImageReader, None): Mask image object.

    Returns:
        int: Mask image level.
    """

    if mask_image is not None:
        # If the spacing information is missing from the mask file, assume that the spacing of the mask is exactly the same as of the image file where the shape of the two files first match.
        #
        if any(mask_spacing is None for mask_spacing in mask_image.spacings):
            for image_level in range(len(input_image.shapes)):
                if input_image.shapes[image_level] == mask_image.shapes[0]:
                    mask_image.correct(spacing=input_image.spacings[image_level], level=0)
                    break

        # Find the corresponding level int the mask image.
        #
        if mask_image.test(spacing=input_spacing):
            return mask_image.level(spacing=input_spacing)

        else:
            # There is no corresponding level. Find the closest higher resolution level.
            #
            mask_level = 0
            for level in range(len(mask_image.spacings)):
                if input_spacing < mask_image.spacings[level]:
                    mask_level = max(level - 1, 0)
                    break

            return mask_level
    else:
        # There is no mask image.
        #
        return -1

#----------------------------------------------------------------------------------------------------

def _init_classifier(network_model, patch_size, padding_mode, padding_constant, output_class, soft_classification, quantize, class_count):
    """
    Initialize image classifier.

    Args:
        network_model (diadmodels.models.modelbase.ModelBase): Network instance.
        patch_size (int): Patch size to set for processing.
        padding_mode (str): Padding mode. Accepted modes are: 'constant' and 'reflect.
        padding_constant (int): Padding constant to use with 'constnat' padding mode.
        output_class (int): Output class for soft classification. If negative all class predictions will be written out in separate channels.
        soft_classification (bool): Soft classification flag. If true the output will be floating point from the [0.0, 1.0] interval, otherwise binary.
        quantize (bool): Quantization flag. If true and soft classification is selected the [0.0, 1.0] interval will be quantized to [0, 255].
        class_count (int): Number of output classes.

    Returns:
        WholeSlideImageClassifier: The configured image classifier object.
    """

    # Local import to prevent unnecessary GPU initialization.
    #
    import digitalpathology.classification.wholeslideimageclassifier as dptwholeslideimageclassifier

    # Configure classifier.
    #
    image_classifier = dptwholeslideimageclassifier.WholeSlideImageClassifier()
    image_classifier.model = network_model
    image_classifier.mode = 'fully_convolutional'
    image_classifier.tile_size = patch_size
    image_classifier.quantize = quantize
    image_classifier.quantize_min_and_max = (0.0, 1.0)
    image_classifier.soft = soft_classification
    image_classifier.verbose = True
    image_classifier.axis_description = 'cwh' if network_model.dimensionorder == 'bchw' else 'whc'
    image_classifier.channels_in_result = class_count
    image_classifier.padding_mode = padding_mode
    image_classifier.padding_constant = padding_constant

    if 0 <= output_class:
        image_classifier.output_class = output_class

    return image_classifier

#----------------------------------------------------------------------------------------------------

def _init_normalizer(normalizer_id, source_range, target_range):
    """
    Initializes a normalizer object.

    Args:
        normalizer_id (str): Normalizer identifier.
        source_range (tuple, None): Normalizer source range in (min, max) format.
        target_range (tuple, None): Normalizer target range in (min, max) format.

    Returns:
        function: A preprocessing normalizer function for the classifier.

    Raises:
        UnknownImageNormalizationModeError: Unknown image normalization mode.
    """

    # Configure normalizer_id
    #
    if normalizer_id == 'general':
        validation_normalizer = dtpgeneralnormalizer.GeneralNormalizer(target_range=target_range, source_range=source_range)

    elif normalizer_id == 'rgb':
        validation_normalizer = dtprgbnormalizer.RgbNormalizer(target_range=target_range)

    elif normalizer_id == 'rgb_to_0-1':
        validation_normalizer = dtprgbtozeroonenormalizer.RgbToZeroOneNormalizer()

    elif normalizer_id == 'pass-through':
        validation_normalizer = dtppassthroughnormalizer.PassThroughNormalizer()

    else:
        # The given normalization type is unknown.
        #
        raise dptprocessingerrors.UnknownImageNormalizationModeError(normalizer_id)

    # Define preprocessing function for the classifier.
    #
    def normalizer_function(image):
        batch_dict = {0: {'patches': np.array([image]), 'labels': np.empty(0), 'weight': np.empty(0)}}
        return validation_normalizer.normalize(batch_dict)[0]['patches'][0]

    return normalizer_function

#----------------------------------------------------------------------------------------------------

def apply_network_batch(job_list,
                        model_path,
                        patch_size,
                        output_class,
                        number_of_classes,
                        normalizer,
                        normalizer_source_range,
                        normalizer_target_range,
                        soft_mode,
                        input_spacing,
                        output_spacing=None,
                        spacing_tolerance=0.25,
                        unrestrict_network=False,
                        input_channels=(0, 1, 2),
                        padding_mode='constant',
                        padding_constant=255,
                        confidence=0.5,
                        minimum_region_diagonal=0.0,
                        dilation_distance=0.0,
                        minimum_hole_diagonal=0.0,
                        full_connectivity=False,
                        quantize=True,
                        interpolation_order=0,
                        copy_path=None,
                        work_path=None,
                        clear_cache=True,
                        keep_intermediates=False,
                        single_mode=False,
                        overwrite=True):
    """
    Apply network on batch of images.

    Args:
        job_list (list): List of job (image path, mask path, result path, interval path) tuples.
        model_path (str): Path of the model file to load.
        patch_size (int): Patch size to set for processing.
        output_class (int): Output class for soft classification. If negative all class predictions will be written out in separate channels.
        number_of_classes (int): Number of output classes. If negative will try to get number of channels of network.
        normalizer (str): Normalizer identifier.
        normalizer_source_range (tuple, None): Normalizer source range in (min, max) format.
        normalizer_target_range (tuple, None): Normalizer target range in (min, max) format.
        soft_mode (bool): Soft classification. If set too true the output will be the softmax.
        input_spacing (float): Pixel spacing level of WSI processed by the classifier.
        output_spacing (float, None): Required output pixel spacing: level 0 pixel spacing of the output image will be the given pixel spacing level of the input image.
        spacing_tolerance (float): Pixel spacing tolerance (percentage).
        unrestrict_network (bool): If true the network will be fixed (delete fixed input_size and output_size).
        input_channels (list): List of input channel indices.
        padding_mode (str): Padding mode. Accepted modes are: 'constant' and 'reflect'.
        padding_constant (int): Padding constant to use with 'constnat' padding mode.
        confidence (float): Network confidence for thresholding the predictions. Thresholding is omitted for threshold values smaller or equal to 0.0.
        minimum_region_diagonal (float): Region diagonal minimum (micrometer), for removing smaller than threshold sized regions from the output. Executed only if the diagonal is larger than 0.0.
        dilation_distance (float): Dilation distance (micrometer), for dilating images before hole filling.
        minimum_hole_diagonal (float): Hole diagonal minimum (micrometer), for filling smaller than threshold sized holes from the output. Executed only if the diagonal is larger than 0.0.
        full_connectivity (bool): Connectivity matrix for region and hole filtering. If true edge and point neighbors will be used otherwise edge neighbors only.
        quantize (bool): Quantization flag. If true the network output is quantized to [0, 255] uint8 interval.
        interpolation_order (int): Interpolation order from the [0, 5] interval.
        copy_path (str, None): Copy directory path where the images and masks are cached before processing.
        work_path (str, None): Work directory path where the output is cached first before writing it to the target path.
        clear_cache (bool): Remove the cached images.
        keep_intermediates (bool): Keep intermediate files.
        single_mode (bool): If true the batch execution acts as if it was a single call: exceptions raised, no processing count logged.
        overwrite (bool): If true existing targets will be overwritten.

    Returns:
        (list, list): List of successfully processed items and list of failed items.

    Raises:
        MissingModelFileError: Missing model file.
        UnknownImageNormalizationModeError: Unknown image normalization mode.
        InvalidConfigurationError: Steps enabled that only work on binary images while the network output is non-binary.
        InvalidPixelSpacingValueError: Both image level and pixel spacing are None.
        InvalidInterpolationOrderError: Interpolation order out of [0, 5] bounds.
        AsymmetricZoomFactorError: Asymmetric zoom values.
        InvalidZoomFactorError: Non-positive zoom factor.

        DigitalPathologyConfigError: Configuration errors.
        DigitalPathologyImageError: Image errors.
        DigitalPathologyModelError: Model errors.
        DigitalPathologyProcessingError: Processing errors.
    """

    # Start time measurement.
    #
    batch_start_time = time.time()

    # Log batch processing.
    #
    logger = logging.getLogger(name=__name__)

    if not single_mode:
        logger.info('Applying network in batch mode...')
        logger.info('Job count: {count}'.format(count=len(job_list)))

    try:
        # Calculate the required input shape for the network.
        #
        input_shape = (None, patch_size, patch_size, len(input_channels))

        # The model should be only loaded once:
        #
        network_model = _load_network_model(model_path=model_path, required_shape=input_shape, unrestrict_network=unrestrict_network)

    except Exception as exception:
        # Re-raise exception for the single file mode.
        #
        if single_mode:
            raise

        error_message = 'Error: {exception}'.format(exception=exception)
        logger.error(error_message)

        # Add all cases to the error collection and return: network inference is not possible without the loaded network.
        #
        return [], [(image_path, output_path, error_message) for image_path, _, output_path in job_list]

    # Go through the job list.
    #
    failed_collection = []
    successful_collection = []
    for job_index in range(len(job_list)):
        # Get one job item.
        #
        image_path, mask_path, output_path = job_list[job_index]

        # Construct direct work paths.
        #
        work_output_path = os.path.join(work_path, os.path.basename(output_path)) if work_path else output_path

        # Print data paths if the mode is not a single file mode.
        #
        if not single_mode:
            logger.info('Processing [{index}/{count}]: {path}'.format(index=job_index + 1, count=len(job_list), path=output_path))

        logger.info('Applying network...')
        logger.info('Model path: {path}'.format(path=model_path))
        logger.info('Input path: {path}'.format(path=image_path))
        logger.info('Mask path: {path}'.format(path=mask_path))
        logger.info('Output path: {path}'.format(path=output_path))
        logger.info('Input spacing: {spacing} um'.format(spacing=input_spacing))
        logger.info('Output class: {output} of {count} classes'.format(output=output_class, count=number_of_classes))
        logger.info('Normalizer: \'{normalizer}\''.format(normalizer=normalizer))
        logger.info('Soft mode: {mode}'.format(mode=soft_mode))

        try:
            # Check if the target image already exits.
            #
            if not os.path.isfile(output_path) or overwrite:
                # Progress reporting.
                #
                segmentation_start_time = time.time()

                # Open the input image.
                #
                input_image = dptimagereader.ImageReader(image_path=image_path, spacing_tolerance=spacing_tolerance, input_channels=input_channels, cache_path=copy_path)

                # Open the mask image.
                #
                mask_image = dptimagereader.ImageReader(image_path=mask_path, spacing_tolerance=spacing_tolerance, input_channels=None, cache_path=copy_path) if mask_path else None

                # Create work directory structure if necessary.
                #
                if work_path:
                    os.makedirs(work_path, exist_ok=True)

                # Initialize image classifier object: it is always executed in soft mode and the output will be thresholded manually later.
                #
                image_classifier = _init_classifier(network_model=network_model,
                                                    patch_size=patch_size,
                                                    padding_mode=padding_mode,
                                                    padding_constant=padding_constant,
                                                    output_class=output_class,
                                                    soft_classification=soft_mode,
                                                    quantize=quantize,
                                                    class_count=number_of_classes)

                # Determine processing steps.
                #
                apply_zoom = output_spacing is not None and input_image.level(spacing=output_spacing) != input_image.level(spacing=input_spacing)
                apply_threshold = soft_mode and 0.0 < confidence
                apply_diagonal_filter = 0.0 < minimum_region_diagonal
                apply_dilation = 0.0 < dilation_distance
                apply_hole_filling = 0.0 < minimum_hole_diagonal
                apply_outlining = os.path.splitext(work_output_path)[1].lower() == '.xml'

                # Check if the configuration is valid: diagonal filter, dilation, hole filling and outlining can only be applied to binary images.
                #
                if output_class < 0 and (apply_diagonal_filter or apply_dilation or apply_hole_filling or apply_outlining):
                    dptprocessingerrors.InvalidConfigurationError([apply_diagonal_filter, apply_dilation, apply_hole_filling, apply_outlining])

                # Construct file names.
                #
                applied_steps = [(True, '_prediction', '.tif'),
                                 (apply_zoom, '_zoomed', '.tif'),
                                 (apply_threshold, '_thresholded', '.tif'),
                                 (apply_diagonal_filter, '_filtered', '.tif'),
                                 (apply_dilation, '_dilated', '.tif'),
                                 (apply_hole_filling, '_filled', '.tif'),
                                 (apply_outlining, '', '.xml')]

                prediction_path, zoomed_path, thresholded_path, filtered_path, dilated_path, filled_path, outlined_path = _construct_intermediate_file_paths(output_path=work_output_path,
                                                                                                                                                             applied_steps=applied_steps)

                # Identify input and output levels.
                #
                input_level = input_image.level(spacing=input_spacing)
                output_level = input_image.level(spacing=output_spacing) if output_spacing is not None else input_level

                # Re-calculate confidence threshold based on quantization interval length.
                #
                low_threshold = int(255.0 * confidence) if quantize else confidence

                # Check if the target image already exits.
                #
                if not os.path.isfile(prediction_path) or overwrite:
                    # Configure inputs and outputs for the classifier.
                    #
                    mask_level = _find_mask_level(input_image=input_image, input_spacing=input_spacing, mask_image=mask_image)

                    image_classifier.set_input(image=input_image, level=input_level, channels=None, mask=mask_image,  mask_level=mask_level)
                    image_classifier.set_result_file_path(path=prediction_path)

                    # Configure normalizer.
                    #
                    image_classifier.preprocess_function = _init_normalizer(normalizer_id=normalizer, source_range=normalizer_source_range, target_range=normalizer_target_range)

                    # Classify image.
                    #
                    image_classifier.process()

                    # Measure inference time.
                    #
                    inference_exec_time = time.time() - segmentation_start_time
                    logger.debug('Done in {delta}'.format(delta=datetime.timedelta(seconds=inference_exec_time)))

                else:
                    logger.info('Skipping, target file already exits: {path}'.format(path=prediction_path))

                # Zoom image.
                #
                if apply_zoom:
                    # Check level difference: zoom in or zoom out = save at the given level.
                    #
                    if output_spacing < input_spacing:
                        dptzoom.zoom_image(image=prediction_path,
                                           output_path=zoomed_path,
                                           zoom=input_image.shapes[output_level],
                                           level=0,
                                           pixel_spacing=None,
                                           spacing_tolerance=spacing_tolerance,
                                           round_shape=True,
                                           interpolation_order=interpolation_order,
                                           jpeg_quality=None,
                                           overwrite=overwrite)
                    else:
                        dptzoom.save_image_at_level(image=prediction_path,
                                                    output_path=zoomed_path,
                                                    level=None,
                                                    pixel_spacing=output_spacing,
                                                    spacing_tolerance=spacing_tolerance,
                                                    jpeg_quality=None,
                                                    overwrite=overwrite)

                # Threshold the prediction to get the classification.
                #
                if apply_threshold:
                    dptthreshold.low_threshold_image(image=zoomed_path,
                                                     output_path=thresholded_path,
                                                     low_threshold=low_threshold,
                                                     overwrite=overwrite)

                # Filter the regions according to their size.
                #
                if apply_diagonal_filter:
                    dptregions.filter_regions_image(image=thresholded_path,
                                                    output_path=filtered_path,
                                                    diagonal_threshold=minimum_region_diagonal,
                                                    full_connectivity=full_connectivity,
                                                    pixel_spacing=None,
                                                    spacing_tolerance=spacing_tolerance,
                                                    foreground_labels=None,
                                                    background_label=0,
                                                    copy_path=None,
                                                    work_path=None,
                                                    clear_cache=True,
                                                    overwrite=overwrite)

                # Dilate regions.
                #
                if apply_dilation:
                    dptdilate.dilate_image(image=filtered_path,
                                           output_path=dilated_path,
                                           dilation_distance=dilation_distance,
                                           overwrite=overwrite)

                # Fill holes according to their size.
                #
                if apply_hole_filling:
                    dptregions.fill_holes_image(image=dilated_path,
                                                output_path=filled_path,
                                                diagonal_threshold=minimum_hole_diagonal,
                                                full_connectivity=full_connectivity,
                                                pixel_spacing=None,
                                                spacing_tolerance=spacing_tolerance,
                                                foreground_labels=None,
                                                fill_value=1,
                                                copy_path=None,
                                                work_path=None,
                                                clear_cache=True,
                                                overwrite=overwrite)

                if apply_outlining:
                    dptconversion.create_mask_annotation(mask=filled_path,
                                                         annotation=outlined_path,
                                                         label_map=None,
                                                         conversion_spacing=output_spacing,
                                                         target_spacing=input_image.spacings[0],
                                                         spacing_tolerance=spacing_tolerance,
                                                         keep_singles=False,
                                                         rdp_epsilon=1.0,
                                                         overwrite=overwrite)

                # Clean up.
                #
                if not keep_intermediates:
                    if prediction_path != work_output_path:
                        dptimagefile.remove_image(image_path=prediction_path, ignore_errors=True)

                    if apply_zoom and zoomed_path != work_output_path:
                        dptimagefile.remove_image(image_path=zoomed_path, ignore_errors=True)

                    if apply_threshold and thresholded_path != work_output_path:
                        dptimagefile.remove_image(image_path=thresholded_path, ignore_errors=True)

                    if apply_diagonal_filter and filtered_path != work_output_path:
                        dptimagefile.remove_image(image_path=filtered_path, ignore_errors=True)

                    if apply_dilation and dilated_path != work_output_path:
                        dptimagefile.remove_image(image_path=dilated_path, ignore_errors=True)

                    if apply_hole_filling and filled_path != work_output_path:
                        dptimagefile.remove_image(image_path=filled_path, ignore_errors=True)

                    if apply_outlining and outlined_path != work_output_path:
                        dptimagefile.remove_image(image_path=outlined_path, ignore_errors=True)

                # Processing finished. Close the images.
                #
                input_image.close(clear=clear_cache)
                if mask_image:
                    mask_image.close(clear=clear_cache)

                # Copy the result to the target location.
                #
                if work_path:
                    dptimagefile.relocate_image(source_path=work_output_path, target_path=output_path, move=clear_cache, overwrite=overwrite)

                # Measure execution time.
                #
                segmentation_exec_time = time.time() - segmentation_start_time
                logger.debug('Network application sequence done in {delta}'.format(delta=datetime.timedelta(seconds=segmentation_exec_time)))

            else:
                logger.info('Skipping, target file already exits: {path}'.format(path=output_path))

        except Exception as exception:
            # Re-raise exception for the single file mode.
            #
            if single_mode:
                raise

            error_message = 'Error: {exception}'.format(exception=exception)
            logger.error(error_message)

            # Add case to the error collection.
            #
            failed_collection.append((image_path, output_path, error_message))

        else:
            # Save the result to the list of successful inferences.
            #
            successful_collection.append((image_path, output_path, None))

    # Log execution time.
    #
    if not single_mode:
        batch_execution_time = time.time() - batch_start_time
        logger.debug('Network application batch done in {delta}'.format(delta=datetime.timedelta(seconds=batch_execution_time)))

    # Return a list of successful and failed inferences.
    #
    return successful_collection, failed_collection

#----------------------------------------------------------------------------------------------------

def apply_network(input_path,
                  mask_path,
                  output_path,
                  model_path,
                  patch_size,
                  output_class,
                  number_of_classes,
                  normalizer,
                  normalizer_source_range,
                  normalizer_target_range,
                  soft_mode,
                  input_spacing,
                  output_spacing=None,
                  spacing_tolerance=0.25,
                  input_channels=(0, 1, 2),
                  confidence=0.5,
                  minimum_region_diagonal=0.0,
                  dilation_distance=0.0,
                  minimum_hole_diagonal=0.0,
                  full_connectivity=False,
                  quantize=True,
                  interpolation_order=0,
                  copy_path=None,
                  work_path=None,
                  clear_cache=True,
                  keep_intermediates=False,
                  overwrite=True):
    """
    Apply network on image.

    Args:
        input_path (str): Path of the image to classify.
        mask_path (str): Path of the mask image to use.
        output_path (str): Path of the result image.
        model_path (str): Path of the model file to load.
        patch_size (int): Patch size to set for processing.
        output_class (int): Output class for soft classification. If negative all class predictions will be written out in separate channels.
        number_of_classes (int): Number of output classes. If negative will try to get number of channels of network.
        normalizer (str): Normalizer identifier.
        normalizer_source_range (tuple, None): Normalizer source range in (min, max) format.
        normalizer_target_range (tuple, None): Normalizer target range in (min, max) format.
        soft_mode (bool): Soft classification. If set too true the output will be the softmax.
        input_spacing (float): Pixel spacing level of WSI processed by the classifier.
        output_spacing (float, None): Required output pixel spacing: level 0 pixel spacing of the output image will be the given pixel spacing level of the input image.
        spacing_tolerance (float): Pixel spacing tolerance (percentage).
        input_channels (list): List of input channel indices.
        confidence (float): Network confidence for thresholding the predictions. Thresholding is omitted for threshold values smaller or equal to 0.0.
        minimum_region_diagonal (float): Region diagonal minimum (micrometer), for removing smaller than threshold sized regions from the output. Executed only if the diagonal is larger than 0.0.
        dilation_distance (float): Dilation distance (micrometer), for dilating images before hole filling.
        minimum_hole_diagonal (float): Hole diagonal minimum (micrometer), for filling smaller than threshold sized holes from the output. Executed only if the diagonal is larger than 0.0.
        full_connectivity (bool): Connectivity matrix for region and hole filtering. If true edge and point neighbors will be used otherwise edge neighbors only.
        quantize (bool): Quantization flag. If true the network output is quantized to [0, 255] uint8 interval.
        interpolation_order (int): Interpolation order from the [0, 5] interval.
        copy_path (str, None): Copy directory path where the images and masks are cached before processing.
        work_path (str, None): Work directory path where the output is cached first before writing it to the target path.
        clear_cache (bool): Remove the cached images.
        keep_intermediates (bool): Keep intermediate files.
        overwrite (bool): If true existing targets will be overwritten.

    Raises:
        MissingModelFileError: Missing model file.
        UnknownImageNormalizationModeError: Unknown image normalization mode.
        InvalidConfigurationError: Steps enabled that only work on binary images while the network output is non-binary.
        InvalidPixelSpacingValueError: Both image level and pixel spacing are None.
        InvalidInterpolationOrderError: Interpolation order out of [0, 5] bounds.
        AsymmetricZoomFactorError: Asymmetric zoom values.
        InvalidZoomFactorError: Non-positive zoom factor.

        DigitalPathologyConfigError: Configuration errors.
        DigitalPathologyImageError: Image errors.
        DigitalPathologyModelError: Model errors.
        DigitalPathologyProcessingError: Processing errors.
    """

    # Call the batch mode function with a single job.
    #
    apply_network_batch(job_list=[(input_path, mask_path, output_path)],
                        model_path=model_path,
                        patch_size=patch_size,
                        output_class=output_class,
                        number_of_classes=number_of_classes,
                        normalizer=normalizer,
                        normalizer_source_range=normalizer_source_range,
                        normalizer_target_range=normalizer_target_range,
                        soft_mode=soft_mode,
                        input_spacing=input_spacing,
                        output_spacing=output_spacing,
                        spacing_tolerance=spacing_tolerance,
                        input_channels=input_channels,
                        confidence=confidence,
                        minimum_region_diagonal=minimum_region_diagonal,
                        dilation_distance=dilation_distance,
                        minimum_hole_diagonal=minimum_hole_diagonal,
                        full_connectivity=full_connectivity,
                        quantize=quantize,
                        interpolation_order=interpolation_order,
                        copy_path=copy_path,
                        work_path=work_path,
                        clear_cache=clear_cache,
                        keep_intermediates=keep_intermediates,
                        single_mode=True,
                        overwrite=overwrite)
