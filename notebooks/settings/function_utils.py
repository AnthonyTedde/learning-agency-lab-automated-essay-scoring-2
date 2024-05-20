import numpy as np


def category_to_ordinal(category):
    """
    Converts a list of categorical labels to an ordinal matrix.

    Parameters:
    category (list or array-like): A list or array of categorical labels.

    Returns:
    np.ndarray: A 2D numpy array where each row corresponds to the ordinal encoding of the input category.
    """
    y = np.array(category, dtype=int)
    num_class = np.max(y)
    range_values = np.arange(num_class)
    # Convert to columns vector and broadcast
    ordinal = (range_values < y[:, np.newaxis]).astype(int)
    return ordinal
