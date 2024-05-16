"""
Functions in this module can anonymize whole slide images. This script requires libtiff. Libtiff is a package in PyPI (https://pypi.python.org/pypi/libtiff).
built from https://github.com/pearu/pylibtiff source. Conda has a package (https://anaconda.org/anaconda/libtiff) with the same name but it is a binary
package made from the LibTIFF C library at http://www.libtiff.org origin. To install the libtiff package one must use pip: "pip install libtiff".
"""


import anonymize.anonymizeslide as ans

import logging
import xml.etree.ElementTree as xmlet
import mmap
import struct
import libtiff
import os

#----------------------------------------------------------------------------------------------------

def _directory_offsets(file_object):
    """
    Get the list of directory offsets in the TIFF file.

    Args:
        file_object (file): File object of the TIFF file.

    Returns:
        list: List of directory offsets.
    """

    file_map = mmap.mmap(file_object.fileno(), 0)
    offsets = []

    # Get TIFF version.
    #
    ver = int(struct.unpack('H', file_map[2:4])[0])
    if ver == 43:
        # Get first IDF.
        #
        offset = struct.unpack('Q', file_map[8:16])[0]
        while offset != 0:
            offsets.append(offset)
            # Get number of tags in this IDF.
            #
            tags = struct.unpack('Q', file_map[offset:offset + 8])[0]
            offset = struct.unpack('Q', file_map[offset + 8 + 20 * tags:offset + 8 + 8 + 20 * tags])[0]

    return offsets

#----------------------------------------------------------------------------------------------------

def _image_description_offsets(file_object, dir_offsets):
    """
    Get list list of image description offsets in the directories.

    Args:
        file_object (file): File object of the TIFF file.
        dir_offsets (list): List of directory offsets.

    Returns:
        (list, list, list): List of image description offsets, tag value offsets and tag lengths.
    """

    image_descriptions = []
    tag_value_offsets = []
    tag_lengths_in_bytes = []
    file_map = mmap.mmap(file_object.fileno(), 0)
    for dirOffset in dir_offsets:
        tag_offset = dirOffset + 8
        tags = struct.unpack('Q', file_map[dirOffset:tag_offset])[0]
        tag_found = False
        for tag in range(tags):
            tag_id = struct.unpack('H', file_map[tag_offset:tag_offset + 2])[0]
            if tag_id == 270:
                tag_length = struct.unpack('Q', file_map[tag_offset + 4:tag_offset + 12])[0]
                tag_lengths_in_bytes.append(tag_length)
                if tag_length <= 8:
                    tag_value = file_map[tag_offset + 12:tag_offset + 20][:tag_length - 1]
                    tag_value_offsets.append(tag_offset + 12)
                    image_descriptions.append(tag_value)
                    tag_found = True
                else:
                    tag_value_offset = struct.unpack('Q', file_map[tag_offset + 12:tag_offset + 20])[0]
                    tag_value_offsets.append(tag_value_offset)
                    tag_value = file_map[tag_value_offset:tag_value_offset + tag_length - 1]
                    image_descriptions.append(tag_value)
                    tag_found = True
            tag_offset += 20
        if not tag_found:
            image_descriptions.append("")

    return image_descriptions, tag_value_offsets, tag_lengths_in_bytes

#----------------------------------------------------------------------------------------------------

def _zero_out_leftover_XML(file_path, xml_length_to_zero_out, img_desc_tag_offset, img_desc_tag_length):
    """
    Fill the unused part of the XML tag with zeroes.

    Args:
        file_path (str): Path of the anonymised Philips TIFF to process.
        xml_length_to_zero_out (int): Length of part to be zeroed out.
        img_desc_tag_offset (int): Tag offset.
        img_desc_tag_length (int): Tag length.
    """

    with open(file=file_path, mode='r+b') as file_object:
        file_map = mmap.mmap(file_object.fileno(), 0)
        file_map[img_desc_tag_offset + img_desc_tag_length - xml_length_to_zero_out:
                 img_desc_tag_offset + img_desc_tag_length] = b'\x00' * xml_length_to_zero_out

#----------------------------------------------------------------------------------------------------

def _zero_next_dir_offset(file_object, dir_offset):
    """
    Zero out the next tag after this one.

    Args:
        file_object (file): File object of the TIFF file.
        dir_offset (int): Offset. The offset of the next tag will be zeroed out in this one.
    """

    file_map = mmap.mmap(file_object.fileno(), 0)
    tags = struct.unpack('Q', file_map[dir_offset:dir_offset + 8])[0]
    file_map[dir_offset + 8 + 20 * tags:dir_offset + 8 + 8 + 20 * tags] = b'\x00\x00\x00\x00\x00\x00\x00\x00'
    file_map.flush()

#----------------------------------------------------------------------------------------------------

def _zero_out_pixel_data(file_object, strip_offset, strip_size):
    """
    Zeros out the pixel data belonging to the label.

    Args:
        file_object (file): File object of the TIFF file.
        strip_offset (int): Offset of the strip of pixel data which will be zeroed out.
        strip_size (int): Size of the strip in bytes.
    """

    file_map = mmap.mmap(file_object.fileno(), 0)
    file_map[strip_offset:strip_offset + strip_size] = b'\x00' * strip_size
    file_map.flush()

#----------------------------------------------------------------------------------------------------

def _zero_out_label_dir(file_path, dir_offsets, img_descs, strip_offset, strip_size):
    """
    This step zeros out the label image if there is a label directory. It also removes the reference to the directory.

    Args:
        file_path (str): Path of the anonymised Philips TIFF to process.
        dir_offsets (list): List of directory offsets.
        img_descs (list): ist of image description offsets
        strip_offset (int): Image strip offset.
        strip_size (int): Image strip size.
    """

    with open(file=file_path, mode='r+b') as file_object:
        for index, desc in enumerate(img_descs[1:]):
            if 'macro' in desc.decode('utf-8').lower():
                if strip_offset > 0 and strip_size > 0:
                    _zero_out_pixel_data(file_object, strip_offset, strip_size)
                else:
                    raise ValueError("Could not zero out the label image pixel data, the strip offset or size were not found.")
                _zero_next_dir_offset(file_object, dir_offsets[index + 1])
                break

#----------------------------------------------------------------------------------------------------

def anonymize_philips_tiff(file_path):
    """
    Anonymize Philips TIFF files.

    Args:
        file_path (str): Path of TIFF file to anonymize.

    Raises:
        IOError: The file is not a Philips TIFF.
    """

    # Get some information on offsets from the raw file
    #
    with open(file=file_path, mode='r+b') as file_object:
        dir_offsets = _directory_offsets(file_object)
        img_descs, img_desc_val_offsets, img_desc_val_lengths = _image_description_offsets(file_object, dir_offsets)

    # Open TIFF.
    #
    tiff_file = libtiff.TIFF.open(file_path, mode='a')
    tiff_file.SetDirectory(0)

    # Check if TIFF is a philips TIFF:
    # 1. The TIFF Software tag starts with Philips.
    # 2. The ImageDescription tag contains valid XML.
    # 3. The root element of the XML is 'DataObject' and has an 'ObjectType' attribute with a value of 'DPUfsImport'.
    #
    vendor = tiff_file.GetField('Software')
    if not vendor.startswith(b'Philips'):
        raise IOError('The file is not a Philips TIFF (invalid vendor)', file_path, vendor)

    # Read data and check XML content.
    #
    try:
        xml_string = tiff_file.GetField('ImageDescription')
        xml_string_length = len(xml_string)
        xml_tree = xmlet.fromstring(xml_string)
    except xmlet.ParseError as exception:
        raise IOError('The file is not a Philips TIFF (XML parse error)', file_path, exception)
    else:
        if xml_tree.tag != 'DataObject' or xml_tree.get(key='ObjectType', default='') != 'DPUfsImport':
            raise IOError('The file is not a Philips TIFF (XML content error)', file_path)

    # Remove label from XML tags.
    #
    for el in xml_tree.findall('.//*[@Name=\'DICOM_DERIVATION_DESCRIPTION\']'):
        el.clear()

    for el in xml_tree.findall('.//*[@Name=\'PIM_DP_IMAGE_DATA\']'):
        el.clear()

    for el in xml_tree.findall('.//*[@Name=\'PIM_DP_UFS_BARCODE\']'):
        el.clear()

    xml_string = xmlet.tostring(xml_tree)
    xml_length_to_zero_out = xml_string_length - len(xml_string)
    tiff_file.SetField('ImageDescription', xml_string)
    tiff_file.WriteDirectory()
    tiff_file.close()

    # Zero-out left over XML text parts.
    #
    if xml_length_to_zero_out > 0:
        _zero_out_leftover_XML(file_path, xml_length_to_zero_out, img_desc_val_offsets[0], img_desc_val_lengths[0])

    # Remove label from directories.
    #
    i = 1
    tiff_file = libtiff.TIFF.open(file_path, mode='r')
    tiff_file.SetDirectory(i)
    while not tiff_file.GetField('ImageDescription').lower().decode("utf-8") == 'label' and not tiff_file.LastDirectory():
        tiff_file.SetDirectory(i)
        i += 1

    if tiff_file.GetField('ImageDescription').decode("utf-8") .lower() == 'label':
        strip_offset = tiff_file.GetField("StripOffsets").value
        strip_size = tiff_file.GetField("StripByteCounts").value
        _zero_out_label_dir(file_path, dir_offsets, img_descs, strip_offset, strip_size)

    tiff_file.close()

#----------------------------------------------------------------------------------------------------

def anonymize_image(image_path):
    """
    Anonymize image. It can handle TIFF files, Philips TIFF files, MRXS files, NDPI files and SVS files. The standard TIFF files will be passed through
    without modification.

    Args:
        image_path (str): Slide path.

    Raises:
        IOError: Unrecognized file type.
        IOError: The file is not a Philips TIFF.
    """

    # Log parameters.
    #
    logger = logging.getLogger(name=__name__)

    logger.info('Anonymize: {path}'.format(path=image_path))

    # Select anonymizer based on file extension.
    #
    if os.path.splitext(image_path)[1].lower() in ['.tif', '.tiff']:
        # Only philips TIFF files can be anonymized.
        #
        anonymize_philips_tiff(file_path=image_path)
    else:
        # MRXS, NDPI and SVS files are always anonymized with the standard tool.
        #
        ans.anonymize_slide(filename=image_path)
