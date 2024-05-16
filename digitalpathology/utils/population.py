"""
This file contains functions for distributing a population into sum of numbers.
"""

import numpy as np
import math

#----------------------------------------------------------------------------------------------------

def distribute_population(population, ratios):
    """
    Distribute a number X into sum of numbers: Y0, Y1, ..., Yn in a way that the sum of the numbers equals to the distributed number: X = sum(Y0, Y1, ..., Ym)
    The correction between the ratio and the actual distribution is done randomly.

    Each element in the ratio list must be from [0.0, 1.0] and the list must sum to 1.0.

    Args:
        population (int): Number to distribute.
        ratios (list, dict): List or dictionary of (ratio, minimum, maximum) triplets, where:
            ratio (float): The ratio of population for the given group.
            minimum (int): Minimal number of item for the given group that the algorithm will keep if possible.
            maximum (int): Maximal number of item for the given group.

    Returns:
        list, dict: List of numbers that sums to population or dictionary with the same keys as the ratios.

    Raises:
        ValueError: Invalid value in ratio list.
        ValueError: Ratio list does not sum to 1.0.
        ValueError: Sum of minimums is more than the population.
        ValueError: Sum of maximums is less than the population.
    """

    # Convert the ratios to list if necessary.
    #
    ratio_list = list(ratios.values()) if type(ratios) is dict else ratios

    # Check ratio list.
    #
    if any(0.0 > ratio or 1.0 < ratio for ratio, _, _ in ratio_list):
        raise ValueError('Invalid ratio value in ratios', ratios)

    if not math.isclose(sum(ratio for ratio, _, _ in ratio_list), 1.0):
        raise ValueError('Ratios does not sum to 1.0', ratios)

    # Check the minimums and maximums.
    #
    if population < sum(minimum for _, minimum, _ in ratio_list):
        raise ValueError('Sum of minimums is more than the population', ratios, population)

    if sum(maximum for _, _, maximum in ratio_list) < population:
        raise ValueError('Sum of maximums is less than the population', ratios, population)

    # Calculate initial distribution.
    #
    distribution_list = [min(max(minimum, int(round(ratio * population))), maximum) for ratio, minimum, maximum in ratio_list]
    count_diff = population - sum(distribution_list)

    # Check if the population have been distributed.
    #
    if 0 < count_diff:
        # Collect indices that are correctable items in the list: the items that are below maximum.
        #
        correctable_indices = [index for index in range(len(distribution_list)) if distribution_list[index] < ratio_list[index][2]]

        # The boundary that needs to be checked is the maximum.
        #
        boundary_index = 2
        item_increment = 1
    elif count_diff < 0:
        # Collect indices that are correctable items in the list: : the items that are above minimum.
        #
        correctable_indices = [index for index in range(len(distribution_list)) if ratio_list[index][1] < distribution_list[index]]

        # The boundary that needs to be checked is the minimum.
        #
        boundary_index = 1
        item_increment = -1
    else:
        # Just to avoid warnings.
        #
        correctable_indices = []
        boundary_index = 0
        item_increment = 0

    # Correct values to the right sum.
    #
    while count_diff and correctable_indices:
        # Select an index from the table of correctable indices randomly and correct the number in the final list.
        #
        correction_table_index = np.random.randint(low=0, high=len(correctable_indices))
        correction_index = correctable_indices[correction_table_index]
        distribution_list[correction_index] += item_increment

        # Update the table: if the number in the final list reaches the minimum remove it from the table of correctable indices.
        #
        if distribution_list[correction_index] == ratio_list[correction_index][boundary_index]:
            correctable_indices.pop(correction_table_index)

        # Decrease the difference.
        #
        count_diff -= item_increment

    # Convert the result back to dictionary if necessary.
    #
    return dict(zip(ratios.keys(), distribution_list)) if type(ratios) is dict else distribution_list
