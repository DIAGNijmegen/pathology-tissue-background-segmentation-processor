"""
Functions in this module can do operations related to stain normalization applied to input tiles.
"""

import numpy as np

#----------------------------------------------------------------------------------------------------

def apply_lut(tile, lut):
    """ 
    Apply look-up-table to tile to normalize H&E staining.
    
    Args:
        tile (np.ndarray): RGB tile to normalize. The shape is (channels, rows, cols)
        lut (np.ndarray): Look-up table that maps RGB input values into stain-normalized values.

    Returns:
        np.ndarray: Normalized tile with values in range [0.0, 1.0].
    """

    # Transform the tile back to the [0.0, 255.0] value range.
    #
    tile = tile.transpose((1, 2, 0)) * 255.0

    # Reshape tile for normalization.
    #
    tile_shape = tile.shape
    reshaped_tile = tile.reshape((tile_shape[0] * tile_shape[1], 3))
    normalized_tile = np.zeros(reshaped_tile.shape, dtype=np.float32)

    # Calculate indices.
    #
    pixel_indices = range(reshaped_tile.shape[0])
    all_indices = reshaped_tile[pixel_indices, 0] * 256.0 * 256.0 + reshaped_tile[pixel_indices, 1] * 256.0 + reshaped_tile[pixel_indices, 2]

    # Normalize image.
    #
    normalized_tile[pixel_indices] = lut[all_indices.astype(int)]
    normalized_tile = normalized_tile.reshape(tile_shape)
    normalized_tile = normalized_tile.transpose(2, 0, 1)
    normalized_tile /= 255.0

    # Return the normalized tile as float from the original [0.0, 1.0] range.
    #
    return normalized_tile
