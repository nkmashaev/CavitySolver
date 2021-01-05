import os

from typing import Tuple
from numpy import linalg as la
import numpy as np
import numba

import local_time
from spatial.interp import linear_interp
from meshtools import metric
from meshtools import output as out
from meshtools import taskinit as inp
from spatial import gradient

@numba.njit
def explicit_step(dt : np.ndarray,
                  p: np.ndarray,
                  pn: np.ndarray,
                  gradp: np.ndarray,
                  T: np.ndarray,
                  Tn: np.ndarray,
                  gradT: np.ndarray,
                  V: np.ndarray,
                  Vn: np.ndarray,
                  gradu: np.ndarray,
                  gradv: np.ndarray,
                  resc: np.ndarray,
                  resx: np.ndarray,
                  resy: np.ndarray,
                  resT: np.ndarray,
                  cell_volume: np.ndarray,
                  cell_center: np.ndarray,
                  i_face_center: np.ndarray,
                  i_face_vector: np.ndarray,
                  j_face_center: np.ndarray,
                  j_face_vector: np.ndarray,
                  Re: np.float64,
                  Pr: np.float64,
                  CFL: np.float64,
                  ACP: np.float64,
                  RK_coeff: np.ndarray,
                  gradmode: np.int32,
                  ggiter: np.int32,
                  calc_local: np.int32,
                  do_smoothing: np.int32
                  )->Tuple[np.float64, np.float64, np.float64,np.float64]:
    #grad calculation
    gradp[:, :, :] = 0.0
    gradT[:, :, :] = 0.0
    gradu[:, :, :] = 0.0
    gradv[:, :, :] = 0.0

    if gradmode == 0:
        pass
    else:
        rc = np.zeros(2, dtype=np.float64)
        rn = np.zeros(2, dtype=np.float64)
        for i in range(1, ni):
            for j in range(1, nj):
                ncell = np.array(
                        [[i - 1, j], [i + 1, j], [i, j -1], [i ,j + 1]], dtype=np.int32
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
                    wsqr = 1.0 / (dx * dx + dy * dy)
                    dx_dx += dx * dx * wsqr
                    dx_dy += dx * dy * wsqr
                    dy_dy += dy * dy * wsqr
                r11 = np.sqrt(dx_dx)
                r12 = dx_dy / r11
                r22 = np.sqrt(dy_dy - r12*r12)

                for neighbour in ncell:
                    i_n, j_n = neighbour
                    rn[:] = cell_center[i_n, j_n, :]
                    dx = rn[0] - rc[0]
                    dy = rn[1] - rc[1]
                    wsqr = 1.0 / (dx * dx + dy * dy)
                    a1 = dx / (r11 * r11)
                    a2 = (dy - r12 * dx / r11) / (r22 * r22)
                    theta = np.array([a1 - r12 * a2 / r11, a2], dtype = np.float64)
                    theta[:] *= wsqr
                    gradp[i, j, :] += theta[:] * (p[i_n, j_n] - p[i, j])
                    gradu[i, j, :] += theta[:] * (V[i_n, j_n, 0] - V[i, j, 0])
                    gradv[i, j, :] += theta[:] * (V[i_n, j_n, 1] - V[i, j, 1])
                    gradT[i, j, :] += theta[:] * (T[i_n, j_n] - T[i, j])
    
    # calc fluxes on faces
    rnc = np.zeros(2, dtype=np.float64)
    v_left = np.zeros(2, dtype=np.float64)
    v_right = np.zeros(2, dtype=np.float64)
    vf = np.zeros(2, dtype=np.float64)
    uf = np.zeros(2, dtype=np.float64)
    dVdn = np.zeros(2, dtype=np.float64)
    dVdn_c = np.zeros(2, dtype=np.float64)
    gradu_l = np.zeros(2, dtype=np.float64)
    gradu_r = np.zeros(2, dtype=np.float64)
    graduf = np.zeros(2, dtype=np.float64)
    gradvf = np.zeros(2, dtype=np.float64)
    gradTf = np.zeros(2, dtype=np.float64)
    gradv_l = np.zeros(2, dtype=np.float64)
    gradv_r = np.zeros(2, dtype=np.float64)
    gradT_l = np.zeros(2, dtype=np.float64)
    gradT_r = np.zeros(2, dtype=np.float64)
    resc_max = 0.0
    resx_max = 0.0
    resy_max = 0.0
    resT_max = 0.0
    for alpha in RK_coeff:
        resc[:, :] = 0.0
        resx[:, :] = 0.0
        resy[:, :] = 0.0
        resT[:, :] = 0.0

        for i in range(ni):
            for j in range(nj - 1):
                rf = i_face_center[i, j, :]
                sf = i_face_vector[i, j, :]
                nf = i_face_vector[i, j, :] / la.norm(sf[:])

                dt_right = dt[i + 1, j + 1]
                p_right = p[i + 1, j + 1]
                T_right = T[i + 1, j + 1]
                r_right = cell_center[i + 1, j + 1, :]
                vol_right = cell_volume[i + 1, j + 1]
                d_right = la.norm(r_right[:] - rf[:])
                v_right[:] = V[i + 1, j + 1, :]
                gradu_r[:] = gradu[i + 1, j + 1, :]
                gradv_r[:] = gradv[i + 1, j + 1, :]
                gradT_r[:] = gradT[i + 1, j + 1, :]

                p_left = p[i, j + 1]
                T_left = T[i, j + 1]
                r_left = cell_center[i, j + 1, :]
                dt_left= dt[i, j + 1]
                vol_left = cell_volume[i, j + 1]
                d_left = la.norm(r_left[:] - rf[:])
                v_left[:] = V[i, j + 1, :]
                gradu_l[:] = gradu[i, j + 1, :]
                gradv_l[:] = gradv[i, j + 1, :]
                gradT_l[:] = gradT[i, j + 1, :]
            
                dnc = la.norm(r_right[:] - r_left[:])
                rnc[:] = (r_right[:] - r_left[:]) / dnc
            
                graduf[:] = linear_interp(d_right, d_left, gradu_r[:], gradu_l[:])
                gradvf[:] = linear_interp(d_right, d_left, gradv_r[:], gradv_l[:])
                gradTf[:] = linear_interp(d_right, d_left, gradT_r[:], gradT_l[:])
                vf[:] = linear_interp(d_right, d_left, v_right[:], v_left[:])
                pf = linear_interp(d_right, d_left, p_right, p_left)
                dVdn[:] = (v_right[:] - v_left[:]) /dnc
                dTdn = (T_right - T_left) /dnc
                if np.absolute(vol_left) < 1e-14:
                    dVdn_c[0] = np.dot(gradu_r[:], nf[:])
                    dVdn_c[1] = np.dot(gradv_r[:], nf[:])
                    dVdn[:] = (5.0 * dVdn[:] - 2.0 * dVdn_c[:]) / 3.0
                    graduf[:] = gradu_r[:]
                    gradvf[:] = gradv_r[:]

                    dTdn_c = np.dot(gradT_r[:], nf[:])
                    dTdn = (5.0 * dTdn - 2.0 * dTdn_c) / 3.0
                    gradTf[:] = gradT_r[:]
                    #dTdn = 0.0
                    #gradTf[:] = 0.0
                if np.absolute(vol_right) < 1e-14:
                    dVdn_c[0] = np.dot(gradu_l[:], nf[:])
                    dVdn_c[1] = np.dot(gradv_l[:], nf[:])
                    dVdn[:] = (5.0 * dVdn[:] - 2.0 * dVdn_c[:]) / 3.0
                    graduf[:] = gradu_l[:]
                    gradvf[:] = gradv_l[:]

                    dTdn_c = np.dot(gradT_l[:], nf[:])
                    dTdn = (5.0 * dTdn - 2.0 * dTdn_c) / 3.0
                    gradTf[:] = gradT_l[:]
                    #dTdn = 0.0
                    #gradTf[:] = 0.0
                dVdn[0] += np.dot(nf[:] - rnc[:], graduf[:])
                dVdn[1] += np.dot(nf[:] - rnc[:], gradvf[:])
                dTdn += np.dot(nf[:] - rnc[:], gradTf[:]) 

                if np.dot(sf[:], vf[:]) > 0:
                    if np.absolute(vol_left) >= 1e-14:
                        uf[:] = v_left[:]
                        Tf = T_left
                    else:
                        uf[:] = 2.0 * v_left[:] - v_right[:]
                        Tf = 2.0 * T_left - T_right
                else:
                    if np.absolute(vol_right) >= 1e-14:
                        uf[:] = v_right[:]
                        Tf = T_right
                    else:
                        uf[:] = 2.0 * v_right[:] - v_left[:]
                        Tf = 2.0 * T_right - T_left
            
                sf_norm = la.norm(sf)
                sf_dot_vf = np.dot(sf[:], vf[:])
                rescf = sf_dot_vf
            
                resxf = uf[0] * sf_dot_vf + pf * sf[0] - dVdn[0] * sf_norm / Re

                resyf = uf[1] * sf_dot_vf + pf * sf[1] - dVdn[1] * sf_norm / Re

                resTf = Tf * sf_dot_vf - dTdn * sf_norm / (Re * Pr)

                if np.absolute(vol_left) >= 1e-14:
                    resc[i, j + 1] += rescf
                    resx[i, j + 1] += resxf
                    resy[i, j + 1] += resyf
                    resT[i, j + 1] += resTf
                if np.absolute(vol_right) >= 1e-14:
                    resc[i + 1, j + 1] -= rescf
                    resx[i + 1, j + 1] -= resxf
                    resy[i + 1, j + 1] -= resyf
                    resT[i + 1, j + 1] -= resTf
    
        for i in range(ni - 1):
            for j in range(nj):
                rf = j_face_center[i, j, :]
                sf = j_face_vector[i, j, :]
                nf = j_face_vector[i, j, :] / la.norm(sf[:])

                dt_right = dt[i + 1, j + 1]
                p_right = p[i + 1, j + 1]
                T_right = T[i + 1, j + 1]
                r_right = cell_center[i + 1, j + 1, :]
                vol_right = cell_volume[i + 1, j + 1]
                d_right = la.norm(r_right[:] - rf[:])
                v_right[:] = V[i + 1, j + 1, :]
                gradu_r[:] = gradu[i + 1, j + 1, :]
                gradv_r[:] = gradv[i + 1, j + 1, :]
                gradT_r[:] = gradT[i + 1, j + 1, :]

                p_left = p[i + 1, j]
                T_left = T[i + 1, j]
                r_left = cell_center[i + 1, j, :]
                dt_left= dt[i + 1, j]
                vol_left = cell_volume[i + 1, j]
                d_left = la.norm(r_left[:] - rf[:])
                v_left[:] = V[i + 1, j, :]
                gradu_l[:] = gradu[i + 1, j, :]
                gradv_l[:] = gradv[i + 1, j, :]
                gradT_l[:] = gradT[i + 1, j, :]

                dnc = la.norm(r_right[:] - r_left[:])
                rnc[:] = (r_right[:] - r_left[:]) / dnc

                graduf[:] = linear_interp(d_right, d_left, gradu_r[:], gradu_l[:])
                gradvf[:] = linear_interp(d_right, d_left, gradv_r[:], gradv_l[:])
                gradTf[:] = linear_interp(d_right, d_left, gradT_r[:], gradT_l[:])
                vf[:] = linear_interp(d_right, d_left, v_right[:], v_left[:])
                pf = linear_interp(d_right, d_left, p_right, p_left)
                dVdn[:] = (v_right[:] - v_left[:]) /dnc
                dTdn = (T_right - T_left) /dnc
                if np.absolute(vol_left) < 1e-14:
                    dVdn_c[0] = np.dot(gradu_r[:], nf[:])
                    dVdn_c[1] = np.dot(gradv_r[:], nf[:])
                    dVdn[:] = (5.0 * dVdn[:] - 2.0 * dVdn_c[:]) / 3.0
                    graduf[:] = gradu_r[:]
                    gradvf[:] = gradv_r[:]

                    #dTdn_c = np.dot(gradT_r[:], nf[:])
                    #dTdn = (5.0 * dTdn - 2.0 * dTdn_c) / 3.0
                    #gradTf[:] = gradT_r[:]
                    dTdn = 0.0
                    gradTf[:] = 0.0
                if np.absolute(vol_right) < 1e-14:
                    dVdn_c[0] = np.dot(gradu_l[:], nf[:])
                    dVdn_c[1] = np.dot(gradv_l[:], nf[:])
                    dVdn[:] = (5.0 * dVdn[:] - 2.0 * dVdn_c[:]) / 3.0
                    graduf[:] = gradu_l[:]
                    gradvf[:] = gradv_l[:]

                    #dTdn_c = np.dot(gradT_l[:], nf[:])
                    #dTdn = (5.0 * dTdn - 2.0 * dTdn_c) / 3.0
                    #gradTf[:] = gradT_l[:]
                    dTdn = 0.0
                    gradTf[:] = 0.0
                dVdn[0] += np.dot(nf[:] - rnc[:], graduf[:])
                dVdn[1] += np.dot(nf[:] - rnc[:], gradvf[:])
                dTdn += np.dot(nf[:] - rnc[:], gradTf[:])

                if np.dot(sf[:], vf[:]) > 0:
                    if np.absolute(vol_left) >= 1e-14:
                        uf[:] = v_left[:]
                        Tf = T_left
                    else:
                        uf[:] = 2.0 * v_left[:] - v_right[:]
                        Tf = 2.0 * T_left - T_right
                else:
                    if np.absolute(vol_right) >= 1e-14:
                        uf[:] = v_right[:]
                        Tf = T_right
                    else:
                        uf[:] = 2.0 * v_right[:] - v_left[:]
                        Tf = 2.0 * T_right - T_left

                sf_norm = la.norm(sf)
                sf_dot_vf = np.dot(sf[:], vf[:])
                rescf = sf_dot_vf

                resxf = uf[0] * sf_dot_vf + pf * sf[0] - dVdn[0] * sf_norm / (Re * Pr)

                resyf = uf[1] * sf_dot_vf + pf * sf[1] - dVdn[1] * sf_norm / (Re * Pr)

                resTf = Tf * sf_dot_vf - dTdn * sf_norm / (Re * Pr)

                if np.absolute(vol_left) >= 1e-14:
                    resc[i + 1, j] += rescf
                    resx[i + 1, j] += resxf
                    resy[i + 1, j] += resyf
                    resT[i + 1, j] += resTf
                if np.absolute(vol_right) >= 1e-14:
                    resc[i + 1, j + 1] -= rescf
                    resx[i + 1, j + 1] -= resxf
                    resy[i + 1, j + 1] -= resyf
                    resT[i + 1, j + 1] -= resTf
        if do_smoothing:
            for i in range(1, ni):
                for j in range(1, nj):
                    ncell = np.array(
                        [[i - 1, j], [i + 1, j], [i, j -1], [i ,j + 1]], dtype=np.int32
                    )
                    for i_n, j_n in ncell:
                        resc[i, j] += 0.5 * resc[i_n, j_n]
                        resx[i, j] += 0.5 * resx[i_n, j_n]
                        resy[i, j] += 0.5 * resy[i_n, j_n]
                        resT[i, j] += 0.5 * resT[i_n, j_n]
                    resc[i, j] /= 3.0
                    resx[i, j] /= 3.0
                    resy[i, j] /= 3.0
                    resT[i, j] /= 3.0
            for i in range(ni - 1, 0, -1):
                for j in range(nj - 1, 0, -1):
                    ncell = np.array(
                        [[i, j + 1], [i, j - 1], [i + 1, j], [i - 1, j]], dtype=np.int32
                    )
                    for i_n, j_n in ncell:
                        resc[i, j] += 0.5 * resc[i_n, j_n]
                        resx[i, j] += 0.5 * resx[i_n, j_n]
                        resy[i, j] += 0.5 * resy[i_n, j_n]
                        resT[i, j] += 0.5 * resT[i_n, j_n]
                    resc[i, j] /= 3.0
                    resx[i, j] /= 3.0
                    resy[i, j] /= 3.0
                    resT[i, j] /= 3.0

        if calc_local:
            local_time.calc_loctime(
                                ni,
                                nj,
                                CFL,
                                ACP,
                                dt,
                                V,
                                cell_volume,
                                cell_center,
                                i_face_center,
                                i_face_vector,
                                j_face_center,
                                j_face_vector)

        p[1:ni, 1:nj] = pn[1:ni, 1:nj] - alpha * resc[1:ni, 1:nj] * dt[1:ni, 1:nj] * ACP / cell_volume[1:ni, 1:nj]
        V[1:ni, 1:nj, 0] = Vn[1:ni, 1:nj, 0] - alpha * resx[1:ni, 1:nj] * dt[1:ni, 1:nj] / cell_volume[1:ni, 1:nj]
        V[1:ni, 1:nj, 1] = Vn[1:ni, 1:nj, 1] - alpha * resy[1:ni, 1:nj] * dt[1:ni, 1:nj] / cell_volume[1:ni, 1:nj]
        T[1:ni, 1:nj] = Tn[1:ni, 1:nj] - alpha * resT[1:ni, 1:nj] * dt[1:ni, 1:ni] / cell_volume[1:ni, 1:nj]
    
    resc_max = np.amax(np.absolute(resc))
    resx_max = np.amax(np.absolute(resx))
    resy_max = np.amax(np.absolute(resy))
    resT_max = np.amax(np.absolute(resT))

    return resc_max, resx_max, resy_max, resT_max

if __name__ == "__main__":
    # read initialization parameters
    input_name = os.path.join(os.path.dirname(__file__), "input.txt")
    print(f'Read initialization parameters from file: "{input_name}"')
    taskinit = inp.InputManager(input_name)
    print(f'Mesh file name is "{taskinit.msh}"')
    print(f"Gradient calculation approach: {taskinit.grad[1]}")
    grad_calc = gradient.least_squares
    gg_iter = 1
    if taskinit.grad[0] == 0:
        print(f"Number of green gauss iterations equals {taskinit.gauss_iter}")
        gg_iter = taskinit.gauss_iter
        grad_calc = gradient.green_gauss
    
    print(f"Re={taskinit.Re}")
    Re = taskinit.Re
    
    print(f"CFL={taskinit.CFL}")
    CFL = taskinit.CFL
    
    print(f"Artificial compressibility parameter={taskinit.ACP}")
    ACP = taskinit.ACP

    calc_local = taskinit.loctime
    if calc_local == 0:
        print("Uniform time step")
    else:
        print("Local time step acceleration")

    do_smoothing = taskinit.smoothing
    if do_smoothing == 0:
        print("Without smoothing")
    else:
        print("Central smoothing implicit scheme")

    RK = taskinit.RK
    print(f"RK level {RK}")
    if RK == 1:
        RK_coeff = np.array([1.0], dtype=np.float64)
    elif RK == 2:
        RK_coeff = np.array([0.4242, 1.000], dtype=np.float64)
    elif RK == 3:
        RK_coeff = np.array([0.1918, 0.4929, 1.0000], dtype=np.float64)
    elif RK == 4:
        RK_coeff = np.array([1.084, 0.2602, 0.5052, 1.000], dtype=np.float64)
    elif RK == 5:
        RK_coeff = np.array([0.0695, 0.1602, 0.2898, 0.5060, 1.000], dtype =np.float64)

    # read mesh file
    with open(taskinit.msh, "r") as in_file:
        size_list = in_file.readline().strip().split()
        ni = np.int32(size_list[0])
        nj = np.int32(size_list[1])
        x = np.zeros((ni, nj), dtype=np.float64)
        y = np.zeros((ni, nj), dtype=np.float64)
        cell_center = np.zeros((ni + 1, nj + 1, 2), dtype=np.float64)
        cell_volume = np.zeros((ni + 1, nj + 1), dtype=np.float64)
        i_face_vector = np.zeros((ni, nj - 1, 2), dtype=np.float64)
        i_face_center = np.zeros((ni, nj - 1, 2), dtype=np.float64)
        j_face_vector = np.zeros((ni - 1, nj, 2), dtype=np.float64)
        j_face_center = np.zeros((ni - 1, nj, 2), dtype=np.float64)
        for j in range(nj):
            for i in range(ni):
                coord_list = in_file.readline().strip().split()
                x[i, j] = np.float64(coord_list[0])
                y[i, j] = np.float64(coord_list[1])

    metric.calc_metric(
        ni,
        nj,
        x,
        y,
        cell_center,
        cell_volume,
        i_face_center,
        i_face_vector,
        j_face_center,
        j_face_vector,
    )

    p = np.zeros((ni + 1, nj + 1), dtype=np.float64)
    pn = np.copy(p)
    gradp = np.zeros((ni + 1, nj + 1, 2), dtype=np.float64)
    V = np.zeros((ni + 1, nj + 1, 2) ,dtype=np.float64)
    Vn = np.copy(V)
    gradu = np.zeros((ni + 1, nj + 1, 2), dtype=np.float64)
    gradv = np.zeros((ni + 1, nj + 1, 2), dtype=np.float64)
    T = np.zeros((ni + 1, nj + 1), dtype=np.float64)
    Tn = np.copy(T)
    gradT = np.zeros((ni + 1, nj + 1, 2), dtype=np.float64)
    dt = np.zeros((ni + 1, nj + 1), dtype=np.float64)
    V[1:ni, -1, 0] = 1.0
    T[0, 1:nj] = 0.0
    T[-1, 1:nj ] = 0.1
    resc = np.zeros((ni + 1, nj + 1), dtype=np.float64)
    resx = np.zeros((ni + 1, nj + 1), dtype=np.float64)
    resy = np.zeros((ni + 1, nj + 1), dtype=np.float64)
    resT = np.zeros((ni + 1, nj + 1), dtype=np.float64)

    vol_max = np.amax(np.absolute(cell_volume))
    if not calc_local:
        dt[1:ni, 1:nj] = CFL * vol_max / np.amax(np.absolute(V))
    
    Pr = 1.0

    outp = out.OutputManager(x, y)
    outp.save_scalar("Pressure", p)
    outp.save_scalar("X-Velocity", V[:, :, 0])
    outp.save_scalar("Y-Velocity", V[:, :, 1])
    outp.save_scalar("Temperature", T)

    residual_file = open("residual.dat", "w")
    for n in range(1, 100000):
        if (n % 1000 == 0):
            outp.output("cavity.plt")

        pn[:, :] = p[:, :]
        Vn[:, :, :] = V[:, :, :]
        Tn[:, :] = T[:, :]
        resc_max, resx_max, resy_max, resT_max = explicit_step(dt,
                                            p,
                                            pn,
                                            gradp,
                                            T,
                                            Tn,
                                            gradT,
                                            V,
                                            Vn,
                                            gradu,
                                            gradv,
                                            resc,
                                            resx,
                                            resy,
                                            resT,
                                            cell_volume,
                                            cell_center,
                                            i_face_center,
                                            i_face_vector,
                                            j_face_center,
                                            j_face_vector,
                                            Re,
                                            Pr,
                                            CFL,
                                            ACP,
                                            RK_coeff,
                                            taskinit.grad[0],
                                            gg_iter,
                                            calc_local,
                                            do_smoothing)
        p[0, 1:nj] = p[1, 1:nj]
        p[-1, 1:nj] = p[-2, 1:nj]
        p[1:ni, 0] = p[1:ni, 1]
        p[1:ni, -1] = p[1:ni, -2]
        #T[0, 1:nj] = -T[1, 1:nj]
        #T[-1, 1:nj] = 0.2 - T[-2, 1:nj]
        T[1:ni, 0] = T[1:ni, 1]
        T[1:ni, -1] = T[1:ni, -2]
        #V[1:ni, -1, 0] = 2.0 - V[1:ni, -2, 0]
        #V[1:ni, -1, 1] = -V[1:ni, -2, 1]
        #V[1:ni, 0, :] = -V[1:ni, 1, :]
        #V[0, 1:nj, :] = -V[1, 1:nj , :]
        #V[-1, 1:nj, :] = -V[-2, 1:nj, :]
        print(f"{ n} {resc_max:.11e} {resx_max:.11e} {resy_max:.11e} {resT_max:.11e}")
        print(f"{ n} {resc_max:.11e} {resx_max:.11e} {resy_max:.11e} {resT_max:.11e}", file=residual_file)
    residual_file.close()
