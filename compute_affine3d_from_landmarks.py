import numpy as np

def compute_affine_transform_3d(em_points, lm_points):
    """
    Compute a best-fit affine transformation that maps
    em_points -> lm_points in 3D via least squares.

    Parameters
    ----------
    em_points : (N, 3) ndarray
        Array of source landmarks (EM).
    lm_points : (N, 3) ndarray
        Array of target landmarks (LM).

    Returns
    -------
    T : (4, 4) ndarray
        The homogeneous 3D affine transformation matrix.
    """

    # Convert to float numpy arrays
    em_points = np.asarray(em_points, dtype=float)
    lm_points = np.asarray(lm_points, dtype=float)

    # Number of points
    n_points = em_points.shape[0]

    # 1) Build the homogeneous coordinate array for EM (source)
    #    Shape will be (N, 4)
    ones_col = np.ones((n_points, 1), dtype=float)
    X = np.hstack([em_points, ones_col])  # (N x 4)

    # 2) LM (target) is (N x 3)
    Y = lm_points  # (N x 3)

    # 3) Solve the linear system X * M = Y in a least-squares sense
    #    M will be of shape (4, 3).
    #    So for each row i, X[i] * M = Y[i], in 3D.
    M, residuals, rank, s = np.linalg.lstsq(X, Y, rcond=None)

    # 4) Build the full 4x4 homogeneous transform T
    T = np.eye(4)
    # M is 4x3, so its transpose is 3x4.
    # We want the top-left 3x4 part of T to be M^T, i.e. T[:3, :4] = M.T
    T[:3, :4] = M.T

    return T

from scipy.ndimage import affine_transform


def transform_image_stack(image_stack, affine_matrix, **kwargs):
    """
    Transforms a 3D image stack using an affine 3D transformation matrix.

    Parameters:
        image_stack (np.ndarray): The input 3D image stack (shape: Z, Y, X).
        affine_matrix (np.ndarray): 4x4 affine transformation matrix.
        interpolation_order (int): Interpolation order (1=linear, 3=cubic).
        fill_mode (str): How to fill values outside the boundaries ('constant', 'nearest', etc.).
        fill_value (float): Fill value used when fill_mode='constant'.

    Returns:
        np.ndarray: The transformed 3D image stack.
    """
    # Validate inputs
    assert affine_matrix.shape == (4, 4), "Affine matrix must be 4x4."
    assert image_stack.ndim == 3, "Image stack must be 3-dimensional (Z, Y, X)."

    # Separate the linear and translation components
    linear_part = affine_matrix[:3, :3]
    translation_part = affine_matrix[:3, 3]

    # Apply affine transform
    transformed_stack = affine_transform(
        image_stack,
        matrix=linear_part,
        offset=translation_part,
        **kwargs
    )
    return transformed_stack


# Compute the inverse of the affine transformation
def invert_affine_matrix(affine_matrix):
    # Separate the linear and translation components
    linear_part = affine_matrix[:3, :3]
    translation_part = affine_matrix[:3, 3]

    # Invert the linear part
    inv_linear_part = np.linalg.inv(linear_part)
    inv_translation_part = -np.dot(inv_linear_part, translation_part)

    # Reconstruct the inverse affine matrix
    inv_affine_matrix = np.zeros((4, 4))
    inv_affine_matrix[:3, :3] = inv_linear_part
    inv_affine_matrix[:3, 3] = inv_translation_part
    inv_affine_matrix[3, 3] = 1
    return inv_affine_matrix


def transform_points(points, affine_matrix):
    """
    Transforms a set of 3D points using an affine 3D transformation matrix.

    Parameters:
        points (np.ndarray): The input 3D points (shape: N, 3).
        affine_matrix (np.ndarray): 4x4 affine transformation matrix.

    Returns:
        np.ndarray: The transformed 3D points.
    """
    # Validate inputs
    assert affine_matrix.shape == (4, 4), "Affine matrix must be 4x4."
    assert points.shape[1] == 3, "Points must be 3D (shape: N, 3)."

    # Add homogeneous coordinates
    n_points = points.shape[0]
    homog_points = np.hstack([points, np.ones((n_points, 1))])

    # Apply the affine transformation
    transformed_points = np.dot(homog_points, affine_matrix.T)
    transformed_points = transformed_points[:, :3]
    return transformed_points