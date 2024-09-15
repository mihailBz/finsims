import numpy as np
import pywt


def wavelet_transform(data, wavelet_name):
    """
    Applies wavelet decomposition to each time series in the data array.
    Args:
        data: A 2D numpy array of shape (n, M) where n is the number
         of timesteps and M is the number of time series.
        wavelet_name: The name of the wavelet to use for decomposition.

    Returns:
        concatenated_coeffs_list: A list of concatenated wavelet coefficients for each time series.
        coeff_shapes_list: A list of coefficient shapes for each time series.
        coeffs_list: A list of original coefficients for each time series.
    """
    concatenated_coeffs_list = []
    coeff_shapes_list = []
    coeffs_list = []

    # Loop through each time series (column in the data)
    for i in range(data.shape[1]):
        # Perform wavelet decomposition on the i-th time series
        coeffs = pywt.wavedec(data[:, i], wavelet_name)

        # Store the shape of each coefficient array for restoration later
        coeff_shapes = [c.shape for c in coeffs]

        # Concatenate coefficients into a single array
        concatenated_coeffs = np.concatenate([c.ravel() for c in coeffs])

        # Append results to the lists
        concatenated_coeffs_list.append(concatenated_coeffs)
        coeff_shapes_list.append(coeff_shapes)
        coeffs_list.append(coeffs)

    return np.array(concatenated_coeffs_list).T, np.array(coeff_shapes_list)


def restore_coeffs(concatenated_coeffs_list, coeff_shapes_list):
    """
    Restores wavelet coefficients from concatenated arrays for each time series.

    Args:
        concatenated_coeffs_list: A list of concatenated wavelet coefficients for each time series.
        coeff_shapes_list: A list of coefficient shapes for each time series.

    Returns:
        restored_coeffs_list: A list of restored wavelet coefficients for each time series.
    """
    restored_coeffs_list = []

    # Loop through each time series
    for concatenated_coeffs, coeff_shapes in zip(
        concatenated_coeffs_list.T, coeff_shapes_list
    ):
        restored_coeffs = []
        start = 0

        # Reshape the concatenated array back to the original coefficient shapes
        for shape in coeff_shapes:
            size = np.prod(shape)  # Total number of elements in the coefficient
            restored_coeffs.append(
                concatenated_coeffs[start : start + size].reshape(shape)
            )
            start += size

        restored_coeffs_list.append(restored_coeffs)

    return restored_coeffs_list


def wavelet_inverse_transform(restored_coeffs_list, wavelet_name):
    """
    Reconstructs each time series from its wavelet coefficients.

    Args:
        restored_coeffs_list: A list of restored wavelet coefficients for each time series.
        wavelet_name: The name of the wavelet used for reconstruction.

    Returns:
        restored_data: A 2D numpy array of shape (n, M) where n is
         the number of timesteps and M is the number of time series.
    """
    restored_data_list = []

    # Loop through each time series' restored coefficients
    for restored_coeffs in restored_coeffs_list:
        # Perform inverse wavelet transform to reconstruct the time series
        restored_data = pywt.waverec(restored_coeffs, wavelet_name)
        restored_data_list.append(restored_data)

    # Stack the restored data columns into a 2D array (n, M)
    restored_data_array = np.column_stack(restored_data_list)

    return np.array(restored_data_array)
