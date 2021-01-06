import numba
import numpy as np
from numpy import linalg as la

from spatial.interp import linear_interp


@numba.njit
def calc_loctime(
    ni: np.int32,
    nj: np.int32,
    CFL: np.float64,
    ACP: np.float64,
    dt: np.ndarray,
    V: np.ndarray,
    cell_volume: np.ndarray,
    cell_center: np.ndarray,
    i_face_center: np.ndarray,
    i_face_vector: np.ndarray,
    j_face_center: np.ndarray,
    j_face_vector: np.ndarray,
):

    rf = np.zeros((4, 2), dtype=np.float64)
    sf = np.copy(rf)
    vf = np.zeros(2, dtype=np.float64)
    rc = np.zeros(2, dtype=np.float64)
    rn = np.zeros(2, dtype=np.float64)

    for i in range(1, ni):
        for j in range(1, nj):
            ncell = np.array(
                [[i - 1, j], [i + 1, j], [i, j - 1], [i, j + 1]], dtype=np.int32
            )
            rf[0, :] = i_face_center[i - 1, j - 1, :]
            rf[1, :] = i_face_center[i, j - 1, :]
            rf[2, :] = j_face_center[i - 1, j - 1, :]
            rf[3, :] = j_face_center[i - 1, j, :]

            sf[0, :] = -i_face_vector[i - 1, j - 1, :]
            sf[1, :] = i_face_vector[i, j - 1, :]
            sf[2, :] = -j_face_vector[i - 1, j - 1, :]
            sf[3, :] = j_face_vector[i - 1, j, :]

            vol = np.absolute(cell_volume[i, j])
            rc[:] = cell_center[i, j, :]
            first = True
            gf_max = 0.0
            sf_max = 0.0
            for iface, neighbour in enumerate(ncell):
                i_n, j_n = neighbour
                rn[:] = cell_center[i_n, j_n, :]
                dc = la.norm(rf[iface, :] - rc[:])
                dn = la.norm(rf[iface, :] - rn[:])
                vf[:] = linear_interp(dc, dn, V[i, j, :], V[i_n, j_n, :])
                if first:
                    first = False
                    gf_max = np.absolute(np.dot(vf[:] + np.sqrt(ACP), sf[iface, :]))
                    sf_max = np.absolute(la.norm(sf[iface, :]))
                else:
                    gf = np.absolute(np.dot(vf[:] + np.sqrt(ACP), sf[iface, :]))
                    sf_norm = np.absolute(la.norm(sf[iface, :]))
                    if gf > gf_max:
                        gf_max = gf
                    if sf_norm > sf_max:
                        sf_max = sf_norm
            dt_conv_rev = gf_max / (CFL * vol)
            dt_diff_rev = (2.0 * sf_max * la.norm(V[i, j, :]) + np.sqrt(ACP)) / (
                CFL * vol
            )
            if (dt_conv_rev + dt_diff_rev) > 1.0e-10:
                dt[i, j] = 1.0 / (dt_conv_rev + dt_diff_rev)
            else:
                dt[i, j] = 0.0
