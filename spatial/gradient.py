import numba
import numpy as np
from numpy import linalg as la

from spatial.interp import linear_interp

@numba.njit
def green_gauss(
    ni: int,
    nj: int,
    p: np.ndarray,
    gradp: np.ndarray,
    cell_volume: np.ndarray,
    cell_center: np.ndarray,
    i_face_center: np.ndarray,
    i_face_vector: np.ndarray,
    j_face_center: np.ndarray,
    j_face_vector: np.ndarray,
):
    rm = np.zeros(2, dtype=np.float64)
    rf = np.zeros(2, dtype=np.float64)
    sf = np.zeros(2, dtype=np.float64)
    r_right = np.zeros(2, dtype=np.float64)
    r_left = np.zeros(2, dtype=np.float64)
    grad = np.copy(gradp)
    gradp[:, :, :] = 0.0
    gradm = np.zeros(2, dtype=np.float64)
    d_right = np.float64(0.0)

    for i in range(ni):
        for j in range(nj - 1):
            rf = i_face_center[i, j, :]
            sf = i_face_vector[i, j, :]

            r_right = cell_center[i + 1, j + 1, :]
            vol_right = cell_volume[i + 1, j + 1]
            d_right = la.norm(r_right[:] - rf[:])
            p_right = p[i + 1, j + 1]
            gradr = grad[i + 1, j + 1]

            r_left = cell_center[i, j + 1, :]
            vol_left = cell_volume[i, j + 1]
            d_left = la.norm(r_left[:] - rf[:])
            p_left = p[i, j + 1]
            gradl = grad[i, j + 1]

            pm = linear_interp(d_right, d_left, p_right, p_left)
            rm[:] = linear_interp(d_right, d_left, r_right[:], r_left[:])
            gradm[:] = linear_interp(d_right, d_left, gradr[:], gradl[:])
            pf = pm + np.dot(rf[:] - rm[:], gradm[:])

            if vol_left >= 1e-14:
                gradp[i, j + 1, :] += sf[:] * pf / vol_left

            if vol_right >= 1e-14:
                gradp[i + 1, j + 1, :] -= sf[:] * pf / vol_right

    for i in range(ni - 1):
        for j in range(nj):
            rf = j_face_center[i, j, :]
            sf = j_face_vector[i, j, :]

            r_right = cell_center[i + 1, j + 1, :]
            vol_right = cell_volume[i + 1, j + 1]
            d_right = la.norm(r_right[:] - rf[:])
            p_right = p[i + 1, j + 1]
            gradr = grad[i + 1, j + 1]

            r_left = cell_center[i + 1, j, :]
            vol_left = cell_volume[i + 1, j]
            d_left = la.norm(r_left[:] - rf[:])
            p_left = p[i + 1, j]
            gradl = grad[i + 1, j]

            pm = linear_interp(d_right, d_left, p_right, p_left)
            rm[:] = linear_interp(d_right, d_left, r_right[:], r_left[:])
            gradm[:] = linear_interp(d_right, d_left, gradr[:], gradl[:])
            pf = pm + np.dot(rf[:] - rm[:], gradm[:])

            if vol_left >= 1e-14:
                gradp[i + 1, j, :] += sf[:] * pf / vol_left

            if vol_right >= 1e-14:
                gradp[i + 1, j + 1, :] -= sf[:] * pf / vol_right

@numba.njit
def least_squares(
    ni: np.int32,
    nj: np.int32,
    p: np.ndarray,
    gradp: np.ndarray,
    cell_volume: np.ndarray,
    cell_center: np.ndarray,
    i_face_center: np.ndarray,
    i_face_vector: np.ndarray,
    j_face_center: np.ndarray,
    j_face_vector: np.ndarray,
):
    gradp[:, :, :] = 0.0
    rc = np.zeros(2, dtype=np.float64)
    rn = np.zeros(2, dtype=np.float64)

    for i in range(1, ni):
        for j in range(1, nj):
            ncell = np.array(
                [[i - 1, j], [i + 1, j], [i, j - 1], [i, j + 1]], dtype=np.int32
            )
            dx_dx = 0.0
            dx_dy = 0.0
            dy_dy = 0.0
            rc[:] = cell_center[i, j, :]
            for neighbour in ncell:
                i_n, j_n = neighbour
                rn[:] = cell_center[i_n, j_n, :]
                dx = rn[0] - rc[0]
                dy = rn[1] - rc[1]
                weight_sqr = 1.0 / (dx * dx + dy * dy)
                dx_dx += dx * dx * weight_sqr
                dx_dy += dx * dy * weight_sqr
                dy_dy += dy * dy * weight_sqr

            r11 = np.sqrt(dx_dx)
            r12 = dx_dy / r11
            r22 = np.sqrt(dy_dy - r12 * r12)

            for neighbour in ncell:
                i_n, j_n = neighbour
                rn[:] = cell_center[i_n, j_n, :]
                dx = rn[0] - rc[0]
                dy = rn[1] - rc[1]
                weight_sqr = 1 / (dx * dx + dy * dy)
                a1 = dx / (r11 * r11)
                a2 = (dy - r12 * dx / r11) / (r22 * r22)
                theta = np.array([a1 - r12 * a2 / r11, a2], dtype=np.float64)
                gradp[i, j, :] += theta[:] * (p[i_n, j_n] - p[i, j]) * weight_sqr
