import numpy as np
from numpy import linalg as la


def calc_metric(
    ni: np.int32,
    nj: np.int32,
    x: np.ndarray,
    y: np.ndarray,
    cell_center: np.ndarray,
    cell_volume: np.ndarray,
    i_face_center: np.ndarray,
    i_face_vector: np.ndarray,
    j_face_center: np.ndarray,
    j_face_vector: np.ndarray,
):

    for i in range(ni):
        for j in range(nj - 1):
            i_face_vector[i, j, 0] = y[i, j + 1] - y[i, j]
            i_face_vector[i, j, 1] = x[i, j] - x[i, j + 1]
            i_face_center[i, j, 0] = 0.5 * (x[i, j] + x[i, j + 1])
            i_face_center[i, j, 1] = 0.5 * (y[i, j] + y[i, j + 1])

    for i in range(ni - 1):
        for j in range(nj):
            j_face_vector[i, j, 0] = y[i, j] - y[i + 1, j]
            j_face_vector[i, j, 1] = x[i + 1, j] - x[i, j]
            j_face_center[i, j, 0] = 0.5 * (x[i, j] + x[i + 1, j])
            j_face_center[i, j, 1] = 0.5 * (y[i, j] + y[i + 1, j])

    r = np.zeros(2, dtype=float)
    for i in range(1, ni):
        for j in range(1, nj):
            r[0] = x[i, j] - x[i - 1, j - 1]
            r[1] = y[i, j] - y[i - 1, j - 1]
            cell_volume[i, j] = 0.5 * (
                np.dot(i_face_vector[i - 1, j - 1, :], r)
                + np.dot(j_face_vector[i - 1, j - 1, :], r)
            )

    for i in range(1, ni):
        for j in range(1, nj):
            cell_center[i, j] = 0.0
            total_square = 0.0

            for k in (i - 1, i):
                square = la.norm(i_face_vector[k, j - 1, :])
                total_square += square
                cell_center[i, j] += i_face_center[k, j - 1, :] * square

            for k in (j - 1, j):
                square = la.norm(j_face_vector[i - 1, k, :])
                total_square += square
                cell_center[i, j] += j_face_center[i - 1, k, :] * square

            cell_center[i, j] /= total_square

    for j in range(1, nj):
        cell_center[0, j, :] = i_face_center[0, j - 1, :]
        cell_center[-1, j, :] = i_face_center[-1, j - 1, :]

    for i in range(1, ni):
        cell_center[i, 0, :] = j_face_center[i - 1, 0, :]
        cell_center[i, -1, :] = j_face_center[i - 1, -1, :]

    cell_center[0, 0, 0] = x[0, 0]
    cell_center[0, 0, 1] = y[0, 0]
    cell_center[0, -1, 0] = x[0, -1]
    cell_center[0, -1, 1] = y[0, -1]
    cell_center[-1, 0, 0] = x[-1, 0]
    cell_center[-1, 0, 1] = y[-1, 0]
    cell_center[-1, -1, 0] = x[-1, -1]
    cell_center[-1, -1, 1] = y[-1, -1]
