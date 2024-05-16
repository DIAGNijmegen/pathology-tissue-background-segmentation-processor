"""
This file contains functions for serializing and reconstructing numpy ndarrays.
"""

import numpy as np
import io
import pickle
import zlib

#----------------------------------------------------------------------------------------------------

def serialize_ndarray(array):
    """
    Serialize a numpy ndarray into an array of bytes.

    Args:
        array (numpy.ndarray): Array to serialize.

    Returns:
        bytes: Serialized numpy ndarray object.
    """

    # Stream the content of the numpy ndarray to a byte buffer.
    #
    content_buffer = io.BytesIO()
    np.save(file=content_buffer, arr=array, allow_pickle=False)

    # Return the byte array representation of the numpy ndarray.
    #
    return content_buffer.getvalue()

#----------------------------------------------------------------------------------------------------

def reconstruct_ndarray(content):
    """
    Reconstruct a serialized numpy ndarray.

    Args:
        content (bytes): Serialized numpy ndarray.

    Returns:
        numpy.ndarray: The reconstructed ndarray.
    """

    # Reconstruct the numpy ndarray from the byte content.
    #
    return np.load(file=io.BytesIO(content), allow_pickle=False)

#----------------------------------------------------------------------------------------------------

def save_object(content, path):
    """
    Save an object to a file with compressing the content.

    Args:
        content (object): Object to save.
        path (str): Target file path.
    """

    # Pickle the object, compress it and write the compressed version to file.
    #
    with open(file=path, mode='wb') as file:
        file.write(zlib.compress(pickle.dumps(content), level=9))

#----------------------------------------------------------------------------------------------------

def load_object(path):
    """
    Load an object from file.

    Args:
        path (str): Target file path.

    Returns:
        Loaded object.
    """

    # Load the data from file, uncompress it and reconstruct the original object from it.
    #
    with open(file=path, mode='rb') as file:
        return pickle.loads(zlib.decompress(file.read()))
