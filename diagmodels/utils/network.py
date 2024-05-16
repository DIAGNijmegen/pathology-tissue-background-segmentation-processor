"""
This file contains utility functions that update a saved network to the newest format.
"""

import h5py

#----------------------------------------------------------------------------------------------------

def hf5_memory_file(name, mode):
    """
    Create a purely in-memory HF5 file that does not synchronize to any file on disk.

    Args:
        name (str): Name of the file. It still has to be set despite it is not synchronized to disk.
        mode (str): Read/write mode.

    Returns:
        h5py.File: The constructed HF5 file.
    """

    return h5py.File(name=name, mode=mode, driver='core', backing_store=False)

#----------------------------------------------------------------------------------------------------

def hf5_memory_file_image(hf5_file):
    """
    Get the byte content of the HF5 file.

    Args:
        hf5_file (h5py.File): HF5 file.

    Returns:
        bytes: The byte content of the HF5 file.
    """

    h5py.h5f.flush(obj=hf5_file.id, scope=h5py.h5f.SCOPE_GLOBAL)

    return h5py.h5f.FileID.get_file_image(hf5_file.id)

#----------------------------------------------------------------------------------------------------

def load_bytes_to_hf5_memory_file(byte_stream, name):
    """
    Loads a HF5 file from a byte stream.

    Args:
        byte_stream (bytes): Byte content of the HF5 file.
        name (str): Name of the file.

    Returns:
        h5py.File: The HF5 file loaded from the byte stream.
    """

    # Create a file access property ist for the HF5 file with in-memory representation and set the stored binary data to it.
    #
    file_access_property_list = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
    file_access_property_list.set_fapl_core(backing_store=False)
    file_access_property_list.set_file_image(byte_stream)

    # Construct the H5 file with the settings. The filename is mandatory even though the content will not be read or written to the file at any point.
    #
    file_id = h5py.h5f.open(name=name.encode(), flags=h5py.h5f.ACC_RDONLY, fapl=file_access_property_list)
    hf5_file = h5py.File(name=file_id)

    # Return the constructed HF5 file.
    #
    return hf5_file

#----------------------------------------------------------------------------------------------------

def export_model_to_hf5(network, hf5_path):
    """
    Export the Keras model data to a separate HF5 file.

    Args:
        network (dict): Network data structure.
        hf5_path (str): Target HF5 file path.

    Returns:
        bool: True if the HF5 file is successfully exported, False otherwise.
    """

    # The function only works with the newest format of Keras networks.
    #
    if type(network['parameters']) is dict and 'keras_model_file_name' in network['parameters']:
        # Write out the binary content of the HF5 file to the target file.
        #
        with open(file=hf5_path, mode='wb') as hf5_file:
            hf5_file.write(network['parameters']['hf5_image'])

            # File successfully saved.
            #
            return True

    # The data is missing or the network is not a Keras network.
    #
    return False
