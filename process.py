"""
This module can load a network model and apply classification on a whole slide image.
"""
import argparse
from pathlib import Path

import digitalpathology.processing.inference as dptinference
import digitalpathology.utils.foldercontent as dptfoldercontent

import logging
import json
import os
import sys


DEFAULT_CONFIG_FILE = str(Path(__file__).parent / "process.json")
DEFAULT_MODEL_FILE = str(Path(__file__).parent / "pathology-tissue-background-segmentation.net")


#----------------------------------------------------------------------------------------------------

def init_console_logger():
    """Initialize the logger to log all messages to the console."""

    # Configure logging.
    #
    logger = logging.getLogger(name=None)
    logger.setLevel(level=logging.NOTSET)

    # Clear current handlers.
    #
    logger.handlers.clear()

    # Add console stream log handler for errors only.
    #
    error_console_log_formatter = logging.Formatter(fmt='%(message)s')
    error_console_log_handler = logging.StreamHandler(stream=sys.stderr)
    error_console_log_handler.setFormatter(fmt=error_console_log_formatter)
    error_console_log_handler.setLevel(level=logging.ERROR)

    logger.addHandler(hdlr=error_console_log_handler)

    # Add console stream log handler for all messages.
    #
    normal_console_log_formatter = logging.Formatter(fmt='%(message)s')
    normal_console_log_handler = logging.StreamHandler(stream=sys.stdout)
    normal_console_log_handler.setFormatter(fmt=normal_console_log_formatter)
    normal_console_log_handler.setLevel(level=logging.DEBUG)

    logger.addHandler(hdlr=normal_console_log_handler)

#----------------------------------------------------------------------------------------------------

def load_config(input_config_path, default_config_path):
    """
    Load the settings from a JSON file.

    Args:
        input_config_path (str): Path of the configuration JSON file to load from the input folder.
        default_config_path (str): Path of the default configuration file.

    Returns:
        dict: Configuration.
    """

    # Load the default configuration file.
    #
    with open(file=default_config_path, mode='r') as default_config_file:
        config = json.load(fp=default_config_file)

    # Load configuration file from input.
    #
    if os.path.isfile(input_config_path):
        with open(file=input_config_path, mode='r') as input_config_file:
            input_config = json.load(fp=input_config_file)

            config.update(input_config)

    return config

#----------------------------------------------------------------------------------------------------

def assemble_jobs(image_path, mask_path, output_path):
    """
    Assemble (source image path, source mask path, target output path) job triplets for network application.

    Args:
        image_path (str): Path of the image to classify.
        mask_path (str, None): Path of the mask image to use.
        output_path (str): Path of the result image.

    Returns:
        list: List of job tuples.

    Raises:
        InvalidDataFileExtensionError: The format cannot be derived from the file extension.
        InvalidDataSourceTypeError: Invalid JSON or YAML file.
        PurposeListAndRatioMismatchError: The configured purpose distribution does not match the available purposes.
    """

    result_job_list = []

    # Replace the image matching string in the mask path to an asterisk to be able to collect all possible mask files.
    #
    mask_wildcard_path = mask_path.format(image='*') if mask_path else ''

    # Collect all source images and masks and match their file name for processing.
    #
    image_file_path_list = dptfoldercontent.folder_content(folder_path=image_path, filter_exp=None, recursive=False)
    mask_file_path_list = dptfoldercontent.folder_content(folder_path=mask_wildcard_path, filter_exp=None, recursive=False)

    # Build file name to path maps.
    #
    image_file_path_map = {os.path.splitext(os.path.basename(image_path_item))[0]: image_path_item for image_path_item in image_file_path_list}
    mask_file_path_map = {os.path.splitext(os.path.basename(mask_path_item))[0]: mask_path_item for mask_path_item in mask_file_path_list}

    # Construct image match expression.
    #
    if mask_path:
        mask_match_base = '{image}' if os.path.isdir(mask_path) else os.path.splitext(os.path.basename(mask_path))[0]
        mask_match = mask_match_base if 0 <= mask_match_base.find('{image}') else '{image}'
    else:
        mask_match = ''

    # Assemble list.
    #
    for image_key in image_file_path_map:
        if image_key not in mask_file_path_map:
            mask_key = mask_match.format(image=image_key)
            target_output_path = output_path.format(image=image_key)

            current_image_path = image_file_path_map[image_key]
            current_mask_path = mask_file_path_map.get(mask_key, None)

            # Add job item to the list.
            #
            job_item = (current_image_path, current_mask_path, target_output_path)
            result_job_list.append(job_item)

    # Print match count for checking.
    #
    print('Matching image count: {match_count}'.format(match_count=len(result_job_list)))

    # Return the result list.
    #
    return result_job_list

#----------------------------------------------------------------------------------------------------

def construct_results(successful_items, failed_items, results_path):
    """
    Construct and write the 'results.json' file.

    Args:
        successful_items (list): List of successfully processed (input, output) pairs.
        failed_items (list): List of failed (input, output, error) triplets.
        results_path (str): Result JSON file path.
    """

    # Prepare result list.
    #
    result_list = []

    # Add failed entities.
    #
    for item in failed_items:
        result_list.append({'entity': {'input': 'filepath:{path}'.format(path=item[0])},
                            'metrics': {},
                            'error_messages': [item[2]]})

    # Add successful entities.
    #
    for item in successful_items:
        result_list.append({'entity': {'input': 'filepath:{path}'.format(path=item[0])},
                            'metrics': {'output': 'filepath:{path}'.format(path=os.path.relpath(item[1], os.path.dirname(results_path)))},
                            'error_messages': []})

    # Write out result JSON file.
    #
    with open(file=results_path, mode='w') as results_file:
        json.dump(result_list, results_file, indent=4)

#----------------------------------------------------------------------------------------------------

def main(
    input_config_path = '',
    default_config_path = DEFAULT_CONFIG_FILE,
    input_images_file_path = '/input/*.tif',
    mask_images_file_path = None,
    output_images_file_path = '/output/{image}.tif',
    work_directory = '/home/user/work',
    model_path = DEFAULT_MODEL_FILE,
    input_spacing_um = 2.0,
    output_spacing_um = 2.0,
    spacing_tolerance = 0.25
):
    """
    Main function.

    Returns:
        int: Error code.
    """

    # Read the configuration files.
    #
    config = load_config(input_config_path=input_config_path, default_config_path=default_config_path)

    print('Identify tumor regions in lymph node tissue.')
    print('Image path: {path}'.format(path=input_images_file_path))
    print('Mask path: {path}'.format(path=mask_images_file_path))
    print('Output path: {path}'.format(path=output_images_file_path))
    print('Model path: {path}'.format(path=model_path))
    print('Patch size: {size}'.format(size=config['patch_size']))
    print('Output class: {index}'.format(index=config['output_class']))
    print('Normalizer: {name}'.format(name=config['normalizer']))
    print('Normalizer source range: {source}'.format(source=config['normalizer_source_range']))
    print('Normalizer target range: {target}'.format(target=config['normalizer_target_range']))
    print('Soft mode: {flag}'.format(flag=config['soft_mode']))
    print('Input pixel spacing: {spacing} um'.format(spacing=input_spacing_um))
    print('Output pixel spacing: {spacing} um'.format(spacing=output_spacing_um))
    print('Pixel spacing tolerance: {tolerance}'.format(tolerance=spacing_tolerance))
    print('Unrestrict network: {flag}'.format(flag=config['unrestrict_network']))
    print('Input channels: {channels}'.format(channels=config['input_channels']))
    print('Padding mode: \'{mode}\''.format(mode=config['padding_mode']))
    print('Padding constant: {value}'.format(value=config['padding_constant']))
    print('Network confidence: {confidence}'.format(confidence=config['confidence']))
    print('Region diagonal threshold: {threshold} um'.format(threshold=config['minimum_region_diagonal']))
    print('Dilation distance: {distance} um'.format(distance=config['dilation_distance']))
    print('Hole diagonal threshold: {threshold} um'.format(threshold=config['hole_diagonal_threshold']))
    print('Full connectivity: {flag}'.format(flag=config['full_connectivity']))
    print('Quantize: {flag}'.format(flag=config['quantize']))
    print('Interpolation order: {order}'.format(order=config['interpolation_order']))
    print('Overwrite existing results: {flag}'.format(flag=config['overwrite']))

    # Assemble job octets: (source image path, source mask path, copy image path, copy mask path, work output path, work interval path, target output path, target interval path).
    #
    job_list = assemble_jobs(image_path=input_images_file_path, mask_path=mask_images_file_path, output_path=output_images_file_path)

    # Check if there are any identified jobs.
    #
    if job_list:
        # Init the logger to print to the console.
        #
        init_console_logger()

        # Execute jobs.
        #
        successful_items, failed_items = dptinference.apply_network_batch(
            job_list=job_list,
            model_path=model_path,
            patch_size=config['patch_size'],
            output_class=config['output_class'],
            number_of_classes=-1,
            normalizer=config['normalizer'],
            normalizer_source_range=config['normalizer_source_range'],
            normalizer_target_range=config['normalizer_target_range'],
            soft_mode=config['soft_mode'],
            input_spacing=input_spacing_um,
            output_spacing=output_spacing_um,
            spacing_tolerance=spacing_tolerance,
            unrestrict_network=config['unrestrict_network'],
            input_channels=config['input_channels'],
            padding_mode=config['padding_mode'],
            padding_constant=config['padding_constant'],
            confidence=config['confidence'],
            minimum_region_diagonal=config['minimum_region_diagonal'],
            dilation_distance=config['dilation_distance'],
            minimum_hole_diagonal=config['hole_diagonal_threshold'],
            full_connectivity=config['full_connectivity'],
            quantize=config['quantize'],
            interpolation_order=config['interpolation_order'],
            copy_path=None,
            work_path=work_directory,
            clear_cache=True,
            keep_intermediates=False,
            single_mode=False,
            overwrite=config['overwrite']
        )

        # Print the collection of failed cases.
        #
        if failed_items:
            print('Failed on {count} items:'.format(count=len(failed_items)))
            for path in failed_items:
                print('{path}'.format(path=path), file=sys.stderr)

        error_code = len(failed_items)

    else:
        # Failed to identify any jobs.
        #
        print('No images matched the input filter.', file=sys.stderr)

        error_code = -1

    return error_code

#----------------------------------------------------------------------------------------------------


def cli():
    parser = argparse.ArgumentParser("CLI for WSI background segmentation algorithm")
    parser.add_argument("input_file_pattern", type=str, help="Input file pattern. Accepts wildcards '*'.")
    parser.add_argument("output_file_pattern", type=str, help="Output file pattern. Accepts '{image}' pattern for each image matching input file pattern.")
    parser.add_argument("--work-dir", type=str, default="/tmp/gc_wsi_bgseg", help="Directory to store intermediate working files")
    parser.add_argument("--input-spacing", type=float, default=2.0, help="Desired input spacing in um. Default 2.0")
    parser.add_argument("--output-spacing", type=float, default=2.0, help="Desired output spacing in um. Default 2.0")
    parser.add_argument("--spacing-tolerance", type=float, default=0.25, help="Tolerance for the input spacing. Default 0.25")
    args = parser.parse_args()
    return main(
        default_config_path=DEFAULT_CONFIG_FILE,
        input_images_file_path=args.input_file_pattern,
        output_images_file_path=args.output_file_pattern,
        work_directory=args.work_dir,
        model_path=DEFAULT_MODEL_FILE,
        input_spacing_um=args.input_spacing,
        output_spacing_um=args.output_spacing,
        spacing_tolerance=args.spacing_tolerance
    )


if __name__ == '__main__':
    # Return error code.
    sys.exit(cli())
