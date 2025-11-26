#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import math
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from scipy.ndimage import gaussian_filter, distance_transform_edt, map_coordinates
from scipy.spatial import cKDTree
import nibabel as nib

# 如果可用，开启 cuDNN benchmark
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# =========================
# 基础构型与工具
# =========================
def centerline_vox(t, N, a1_vox=12.0, b1=1.0, a2_vox=8.0, b2=1.5, margin_vox=8.0):
    x0 = (N - 1) * 0.5
    z0 = (N - 1) * 0.5
    y = margin_vox + (N - 1 - 2.0 * margin_vox) * t
    x = x0 + a1_vox * np.sin(np.pi * b1 * t)
    z = z0 + a2_vox * np.sin(np.pi * b2 * t)
    return np.stack([x, y, z], axis=1).astype(np.float32)

def exponential_radius_vox(t, r0_vox=4.0, r_max_vox=10.0):
    k = np.log(max(1e-6, r_max_vox / max(1e-6, r0_vox)))
    return (r0_vox * np.exp(k * t)).astype(np.float32)

def safe_margin_vox(r0_vox, r_max_vox, base_delta_r_vox, alpha, N, safety_factor=1.25, pad_vox=2, cap_frac=0.45):
    outer_max = max(r0_vox, r_max_vox) + base_delta_r_vox * (1.0 + alpha)
    margin = safety_factor * outer_max + pad_vox
    cap_vox = cap_frac * (N - 1)
    return float(min(margin, cap_vox, (N - 1) * 0.5 - 1.0))

def clamp_centerline_bend_vox(a1_vox, b1, a2_vox, b2, margin_vox, N, s_max=0.8):
    Ly = max(1e-6, (N - 1) - 2.0 * margin_vox)
    a1_lim = s_max * Ly / (np.pi * max(1e-6, b1))
    a2_lim = s_max * Ly / (np.pi * max(1e-6, b2))
    return min(a1_vox, a1_lim), b1, min(a2_vox, a2_lim), b2

def _to_t(a, device):
    return torch.as_tensor(a, dtype=torch.float32, device=device)

def _grid_centers(x_axis, y_axis, z_axis):
    X, Y, Z = np.meshgrid(x_axis, y_axis, z_axis, indexing="ij")
    P = np.stack([X, Y, Z], axis=-1).reshape(-1, 3).astype(np.float32)
    return P, X.shape

def _sample_trilinear_torch(vol, x_axis, y_axis, z_axis, pts):
    x0, x1 = float(x_axis[0]), float(x_axis[-1])
    y0, y1 = float(y_axis[0]), float(y_axis[-1])
    z0, z1 = float(z_axis[0]), float(z_axis[-1])
    gx = 2.0*(pts[:,0]-x0)/(x1-x0) - 1.0
    gy = 2.0*(pts[:,1]-y0)/(y1-y0) - 1.0
    gz = 2.0*(pts[:,2]-z0)/(z1-z0) - 1.0
    grid = torch.stack([gz, gy, gx], dim=-1).view(1, -1, 1, 1, 3)
    vol5 = vol.view(1,1,*vol.shape)
    vals = F.grid_sample(vol5, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return vals.view(-1)

# =========================
# 连续 SDF 构建
# =========================
@torch.no_grad()
def build_sdfs_continuous_gpu(
    N=96, T=400,
    a1_vox=12.0, b1=1.0, a2_vox=8.0, b2=1.5,
    r0_vox=4.0, r_max_vox=10.0,
    base_delta_r_vox=3.0, alpha=0.6, sigma_vox=3.0,
    seed=2024, device=None, chunk_pts=120000, margin_override_vox=None
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    x_axis = np.arange(N, dtype=np.float32)
    y_axis = np.arange(N, dtype=np.float32)
    z_axis = np.arange(N, dtype=np.float32)

    margin_vox = safe_margin_vox(r0_vox, r_max_vox, base_delta_r_vox, alpha, N) if margin_override_vox is None else float(margin_override_vox)
    t_vals = np.linspace(0.0, 1.0, T, dtype=np.float32)
    a1_c, b1_c, a2_c, b2_c = clamp_centerline_bend_vox(a1_vox, b1, a2_vox, b2, margin_vox, N, s_max=0.8)
    C = centerline_vox(t_vals, N, a1_c, b1_c, a2_c, b2_c, margin_vox=margin_vox)
    R = exponential_radius_vox(t_vals, r0_vox, r_max_vox)

    rng = np.random.default_rng(seed)
    noise = rng.uniform(-1, 1, size=(N,N,N)).astype(np.float32)
    smooth = gaussian_filter(noise, sigma=sigma_vox)
    smooth = smooth / max(1e-6, np.max(np.abs(smooth)))
    delta_r = base_delta_r_vox * (1 + alpha * smooth)
    delta_r = np.maximum(delta_r, 0.25).astype(np.float32)

    C0 = _to_t(C[:-1], dev)
    V  = _to_t(C[1:] - C[:-1], dev)
    L2 = (V*V).sum(dim=1) + 1e-12
    R0 = _to_t(R[:-1], dev)
    R1 = _to_t(R[1:],  dev)
    C0_dot_V = (C0*V).sum(dim=1)

    P_np, shape3 = _grid_centers(x_axis, y_axis, z_axis)
    P = _to_t(P_np, dev)
    inner_vals = torch.empty((P.shape[0],), dtype=torch.float32, device=dev)

    for s in range(0, P.shape[0], chunk_pts):
        e = min(P.shape[0], s + chunk_pts)
        X = P[s:e]
        Xv = X @ V.t()
        num = Xv - C0_dot_V.unsqueeze(0)
        lam = (num / L2.unsqueeze(0)).clamp(0.0, 1.0)
        Q = C0.unsqueeze(0) + lam.unsqueeze(-1) * V.unsqueeze(0)
        D = X.unsqueeze(1) - Q
        dist = torch.sqrt((D*D).sum(dim=2) + 1e-12)
        r_lin = (1.0 - lam) * R0.unsqueeze(0) + lam * R1.unsqueeze(0)
        sdf_seg = dist - r_lin
        inner_vals[s:e] = sdf_seg.min(dim=1).values

    delta_r_t = _to_t(delta_r, dev)
    x_t = _to_t(x_axis, dev); y_t = _to_t(y_axis, dev); z_t = _to_t(z_axis, dev)
    dr_vals = _sample_trilinear_torch(delta_r_t, x_t, y_t, z_t, P)
    outer_vals = inner_vals - dr_vals

    F_inner = inner_vals.view(*shape3).detach().cpu().numpy().astype(np.float32)
    F_outer = outer_vals.view(*shape3).detach().cpu().numpy().astype(np.float32)
    return F_inner, F_outer, x_axis, y_axis, z_axis, C.astype(np.float32), R.astype(np.float32), delta_r.astype(np.float32), margin_vox, (a1_c, b1_c, a2_c, b2_c)

# =========================
# Frenet 与投影
# =========================
def frenet_frame_discrete(C):
    dC = np.gradient(C, axis=0)
    T = dC / (np.linalg.norm(dC, axis=1, keepdims=True) + 1e-12)
    # 与第二段保持一致的方向稳定
    if np.mean(T[:, 1]) < 0:
        T = -T
    dT = np.gradient(T, axis=0)
    kappa_mag = np.linalg.norm(dT, axis=1, keepdims=True)
    N = np.zeros_like(T)
    valid = (kappa_mag[:, 0] > 1e-6)
    N[valid] = dT[valid] / (kappa_mag[valid] + 1e-12)
    for i in range(1, len(N)):
        if np.linalg.norm(N[i]) < 1e-9:
            N[i] = N[i-1]
        elif np.dot(N[i], N[i-1]) < 0:
            N[i] = -N[i]
    B = np.cross(T, N)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return T.astype(np.float32), N.astype(np.float32), B.astype(np.float32)

def project_points_to_centerline_continuous(pts, C, R, T, N, batch=1000000):
    seg_vec = C[1:] - C[:-1]
    seg_len2 = np.sum(seg_vec*seg_vec, axis=1)
    mid = 0.5*(C[1:] + C[:-1])
    tree = cKDTree(mid)
    n = pts.shape[0]
    tau = np.empty((n,), dtype=np.float32)
    r_itp = np.empty((n,), dtype=np.float32)
    side  = np.empty((n,), dtype=np.float32)
    for s in range(0, n, batch):
        e = min(n, s+batch)
        pb = pts[s:e]
        _, k = tree.query(pb, k=1, workers=-1)
        Ck = C[k]; Vk = seg_vec[k]; L2 = seg_len2[k] + 1e-12
        lam = np.clip(np.sum((pb - Ck)*Vk, axis=1)/L2, 0.0, 1.0).astype(np.float32)
        q = Ck + lam[:,None]*Vk
        tau[s:e] = (k.astype(np.float32) + lam) / float(len(C)-1)
        r0 = R[k]; r1 = R[k+1]
        r_itp[s:e] = (1.0 - lam)*r0 + lam*r1
        n0 = N[k]; n1 = N[k+1]
        n_itp = (1.0 - lam)[:,None]*n0 + lam[:,None]*n1
        n_itp = n_itp / (np.linalg.norm(n_itp, axis=1, keepdims=True) + 1e-12)
        side[s:e] = np.sum((pb - q)*n_itp, axis=1)
    return tau, r_itp, side

def precompute_projection_volumes(F_inner, C, R, x_axis, y_axis, z_axis, upsample=2, batch=1000000):
    """
    令 N 对齐到凹侧，随后计算每个点的 tau、局部半径 r 与 side
    """
    Ngrid = len(x_axis)
    xu = np.linspace(x_axis[0], x_axis[-1], upsample*Ngrid, dtype=np.float32)
    yu = np.linspace(y_axis[0], y_axis[-1], upsample*Ngrid, dtype=np.float32)
    zu = np.linspace(z_axis[0], z_axis[-1], upsample*Ngrid, dtype=np.float32)
    X, Y, Z = np.meshgrid(xu, yu, zu, indexing="ij")
    pts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3).astype(np.float32)

    Tvec, Nvec, _ = frenet_frame_discrete(C)

    dT = np.gradient(Tvec, axis=0)
    kappa = np.linalg.norm(dT, axis=1)
    kappaN = (kappa[:, None]) * Nvec
    sum_kappaN = kappaN.sum(axis=0)
    if np.linalg.norm(sum_kappaN) < 1e-8:
        sum_kappaN = Nvec.sum(axis=0)
    N_mean = Nvec.mean(axis=0)
    # 若 mean(N) 与 −Σ(κN) 同向，则翻转，使 N 指向凹侧
    if np.dot(N_mean, -sum_kappaN) > 0.0:
        Nvec = -Nvec

    tau, r_itp, side = project_points_to_centerline_continuous(pts, C, R, Tvec, Nvec, batch=batch)

    D = upsample * Ngrid
    tau_vol = tau.reshape(D, D, D).astype(np.float32)
    r_vol   = r_itp.reshape(D, D, D).astype(np.float32)
    s_vol   = side.reshape(D, D, D).astype(np.float32)
    return {
        "x_hr": xu, "y_hr": yu, "z_hr": zu,
        "tau_hr": tau_vol, "r_hr": r_vol, "side_hr": s_vol,
        "concave_positive": True,
        "concave_vote_mean_diff": float(np.linalg.norm(sum_kappaN))
    }

# =========================
# SDF 与体素工具
# =========================
def sdf_from_mask(mask, x_axis, y_axis, z_axis):
    dist_out = distance_transform_edt(~mask, sampling=(1.0,1.0,1.0)).astype(np.float32)
    dist_in  = distance_transform_edt( mask, sampling=(1.0,1.0,1.0)).astype(np.float32)
    return dist_out - dist_in

@torch.no_grad()
def geometric_volume_and_change_gpu(
    F_before, F_after, x_axis, y_axis, z_axis,
    nx=24, ny=24, nz=24, chunk_voxels=128, device=None
):
    if device is None: device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    Fb = torch.from_numpy(F_before).to(device=device, dtype=torch.float32)
    Fa = torch.from_numpy(F_after ).to(device=device, dtype=torch.float32)
    D, H, W = Fb.shape; vox_vol = 1.0

    ox = ((np.arange(nx)+0.5)/nx - 0.5); oy = ((np.arange(ny)+0.5)/ny - 0.5); oz = ((np.arange(nz)+0.5)/nz - 0.5)
    OX, OY, OZ = np.meshgrid(ox, oy, oz, indexing="ij")
    offsets = np.stack([OX.ravel(), OY.ravel(), OZ.ravel()], axis=1).astype(np.float32)
    offsets_t = torch.from_numpy(offsets).to(device=device, dtype=torch.float32); S = offsets.shape[0]

    Xc, Yc, Zc = np.meshgrid(x_axis, y_axis, z_axis, indexing='ij')
    centers = np.stack([Xc.ravel(), Yc.ravel(), Zc.ravel()], axis=1).astype(np.float32)
    centers_t = torch.from_numpy(centers).to(device=device, dtype=torch.float32); V = centers.shape[0]

    def norm_coords(pts, lo, hi): return 2.0 * (pts - lo) / (hi - lo) - 1.0
    def sample_trilinear(vol, pts_world):
        x0, x1 = float(x_axis[0]), float(x_axis[-1])
        y0, y1 = float(y_axis[0]), float(y_axis[-1])
        z0, z1 = float(z_axis[0]), float(z_axis[-1])
        gx = norm_coords(pts_world[:,0], x0, x1); gy = norm_coords(pts_world[:,1], y0, y1); gz = norm_coords(pts_world[:,2], z0, z1)
        grid = torch.stack([gz, gy, gx], dim=-1).view(1, -1, 1, 1, 3)
        return F.grid_sample(vol.view(1,1,D,H,W), grid, mode="bilinear", padding_mode="border", align_corners=True).view(-1)

    base_frac  = torch.zeros((V,), dtype=torch.float32, device=device)
    after_frac = torch.zeros((V,), dtype=torch.float32, device=device)
    add_frac   = torch.zeros((V,), dtype=torch.float32, device=device)
    rem_frac   = torch.zeros((V,), dtype=torch.float32, device=device)

    for s in range(0, V, chunk_voxels):
        e = min(V, s + chunk_voxels); cen = centers_t[s:e]; M = cen.shape[0]
        pts = (cen[:, None, :] + offsets_t[None, :, :]).reshape(M*S, 3)
        fb = sample_trilinear(Fb, pts); fa = sample_trilinear(Fa, pts)
        in_b = (fb <= 0.0); in_a = (fa <= 0.0)
        add = (~in_b) & in_a
        rem = in_b & (~in_a)
        base_frac[s:e]  += in_b.view(M, S).float().mean(dim=1)
        after_frac[s:e] += in_a.view(M, S).float().mean(dim=1)
        add_frac[s:e]   += add.view(M, S).float().mean(dim=1)
        rem_frac[s:e]   += rem.view(M, S).float().mean(dim=1)

    base_map  = base_frac.view(D, H, W).detach().cpu().numpy().astype(np.float32)
    after_map = after_frac.view(D, H, W).detach().cpu().numpy().astype(np.float32)
    add_map   = add_frac.view(D, H, W).detach().cpu().numpy().astype(np.float32)
    rem_map   = rem_frac.view(D, H, W).detach().cpu().numpy().astype(np.float32)
    signed_map = add_map - rem_map  # 增加为正，减少为负；未变化为 0

    base_vol  = float(base_frac.sum().item()  * vox_vol)
    after_vol = float(after_frac.sum().item() * vox_vol)
    add_vol   = float(add_frac.sum().item()   * vox_vol)
    rem_vol   = float(rem_frac.sum().item()   * vox_vol)
    net_vol   = after_vol - base_vol

    stats = dict(
        base_volume=base_vol, after_volume=after_vol,
        add_volume=add_vol, rem_volume=rem_vol, net_volume=net_vol,
        pct_add=100.0 * add_vol / max(base_vol, 1e-12),
        pct_rem=100.0 * rem_vol / max(base_vol, 1e-12),
        pct_net=100.0 * net_vol / max(base_vol, 1e-12)
    )
    return base_map, after_map, add_map, rem_map, signed_map, stats

# =========================
# 方法一 局部球体变化
# =========================
def _interp_sdf_at_point(Fvol, x_axis, y_axis, z_axis, p):
    ix = (p[0]-x_axis[0])/(x_axis[1]-x_axis[0])
    iy = (p[1]-y_axis[0])/(y_axis[1]-y_axis[0])
    iz = (p[2]-z_axis[0])/(z_axis[1]-z_axis[0])
    v = map_coordinates(Fvol, [[ix],[iy],[iz]], order=1, mode='nearest')
    return float(v[0])

def _ray_to_zero(F, x_axis, y_axis, z_axis, origin, direction, s_max, step=0.5, refine=4):
    d = direction / (np.linalg.norm(direction) + 1e-12)
    f0 = _interp_sdf_at_point(F, x_axis, y_axis, z_axis, origin)
    if f0 > 0:
        return np.nan
    s = step
    while s <= s_max:
        p = origin + d * s
        fs = _interp_sdf_at_point(F, x_axis, y_axis, z_axis, p)
        if fs > 0:
            lo, hi = s - step, s
            for _ in range(refine):
                mid = 0.5 * (lo + hi)
                fm = _interp_sdf_at_point(F, x_axis, y_axis, z_axis, origin + d * mid)
                if fm > 0: hi = mid
                else: lo = mid
            return 0.5 * (lo + hi)
        s += step
    return s_max

def estimate_concave_sign_by_thickness(F_inner, C, N, R, x_axis, y_axis, z_axis, top_q=0.30, max_samples=50):
    dC = np.gradient(C, axis=0)
    T = dC / (np.linalg.norm(dC, axis=1, keepdims=True) + 1e-12)
    dT = np.gradient(T, axis=0)
    kappa = np.linalg.norm(dT, axis=1)
    idx = np.where(kappa >= np.quantile(kappa, 1.0 - top_q))[0]
    if idx.size == 0:
        idx = np.arange(0, len(C), max(1, len(C)//10))
    step = max(1, len(idx)//max_samples)
    idx = idx[::step]

    diffs = []
    for i in idx:
        p = C[i]
        n = N[i]
        rloc = float(R[i])
        smax = max(2.0, 1.5 * rloc)
        d_plus  = _ray_to_zero(F_inner, x_axis, y_axis, z_axis, p,  n, smax)
        d_minus = _ray_to_zero(F_inner, x_axis, y_axis, z_axis, p, -n, smax)
        if np.isfinite(d_plus) and np.isfinite(d_minus):
            diffs.append(d_plus - d_minus)
    if len(diffs) == 0:
        return 1, 0.0
    mean_diff = float(np.mean(diffs))
    return (1 if mean_diff < 0.0 else -1), mean_diff

@torch.no_grad()
def apply_local_spherical_changes_to_inner_geometric(
    F_inner, F_outer, C, R, x_axis, y_axis, z_axis,
    target_frac_range=(0.03, 0.05),
    nx_target=24, ny_target=24, nz_target=24,
    seed=2028,
    max_trials_per_sphere=400,
    max_total_spheres=5,
    change_mode='expand',
    expand_center_dist_vox_range=(0.5, 2.5),
    shrink_center_dist_vox_range=(2.0, 8.0),
    centers_chunk=256,
    offsets_chunk=1024,
    step_frac_range=(0.30, 0.55),
    tol_frac=0.0015
):
    assert change_mode in ('expand','shrink')
    rng = np.random.default_rng(seed)
    N = F_inner.shape[0]
    hmin = 1.0
    vox_vol = 1.0
    sub_vol = vox_vol/(nx_target*ny_target*nz_target)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ox = ((np.arange(nx_target)+0.5)/nx_target - 0.5)
    oy = ((np.arange(ny_target)+0.5)/ny_target - 0.5)
    oz = ((np.arange(nz_target)+0.5)/nz_target - 0.5)
    OX, OY, OZ = np.meshgrid(ox, oy, oz, indexing="ij")
    offsets = np.stack([OX.ravel(), OY.ravel(), OZ.ravel()], axis=1).astype(np.float32)
    offsets_t_all = torch.from_numpy(offsets).to(device=device, dtype=torch.float32)
    S_total = offsets.shape[0]

    Xg, Yg, Zg = np.meshgrid(x_axis, y_axis, z_axis, indexing='ij')
    tree = cKDTree(C)

    def _geom_base(Fin):
        D,H,W = Fin.shape
        Fin_t = torch.from_numpy(Fin).to(device=device, dtype=torch.float32)
        Xc, Yc, Zc = np.meshgrid(x_axis, y_axis, z_axis, indexing="ij")
        centers = np.stack([Xc.ravel(), Yc.ravel(), Zc.ravel()], axis=1).astype(np.float32)
        centers_t = torch.from_numpy(centers).to(device=device, dtype=torch.float32)
        V = centers_t.shape[0]
        def norm_coords(pts, lo, hi): return 2.0*(pts-lo)/(hi-lo) - 1.0
        def sample(vol, pts):
            x0, x1 = float(x_axis[0]), float(x_axis[-1])
            y0, y1 = float(y_axis[0]), float(y_axis[-1])
            z0, z1 = float(z_axis[0]), float(z_axis[-1])
            gx = norm_coords(pts[:,0], x0, x1); gy = norm_coords(pts[:,1], y0, y1); gz = norm_coords(pts[:,2], z0, z1)
            grid = torch.stack([gz, gy, gx], dim=-1).view(1, -1, 1, 1, 3)
            return F.grid_sample(vol.view(1,1,D,H,W), grid, mode="bilinear", padding_mode="border", align_corners=True).view(-1)
        inside_cnt = 0
        for s in range(0, V, 256):
            e = min(V, s+256); cen = centers_t[s:e]; M = cen.shape[0]
            for ss in range(0, S_total, 1024):
                se = min(S_total, ss + 1024)
                offs = offsets_t_all[ss:se]
                pts = (cen[:,None,:] + offs[None,:,:]).reshape(M*(se-ss), 3)
                fin = sample(Fin_t, pts)
                inside_cnt += (fin <= 0).sum().item()
        return inside_cnt * sub_vol

    base_vol = _geom_base(F_inner)
    target_frac = float(rng.uniform(*target_frac_range))
    target_total = target_frac * base_vol
    hi_bound = 0.05 * base_vol
    tol_abs = tol_frac * base_vol

    F_in_cur = F_inner.copy()
    F_out_cur = F_outer.copy()
    shell_inside_mask = (F_out_cur <= 0)

    progressed = 0.0
    spheres = []
    placed = 0
    guard = 0

    def norm_coords(pts, lo, hi): return 2.0 * (pts - lo) / (hi - lo) - 1.0

    while (progressed + 1e-12) < (target_total - tol_abs) and placed < max_total_spheres and guard < 10*max_total_spheres:
        guard += 1

        if change_mode == 'expand':
            lo, hi = expand_center_dist_vox_range
            band = (F_in_cur <= -lo*hmin) & (F_in_cur >= -hi*hmin)
            if not np.any(band): band = (F_in_cur <= 0)
        else:
            lo, hi = shrink_center_dist_vox_range
            band = (F_in_cur >= lo*hmin) & (F_in_cur <= hi*hmin)
            if not np.any(band): band = (F_in_cur > 0)

        ii, jj, kk = np.nonzero(band)
        if ii.size == 0: break
        sel = np.random.randint(0, ii.size)
        i, j, k = int(ii[sel]), int(jj[sel]), int(kk[sel])
        p = np.array([x_axis[i], y_axis[j], z_axis[k]], dtype=np.float32)
        d0 = _interp_sdf_at_point(F_in_cur, x_axis, y_axis, z_axis, p)

        idc = int(tree.query(p[None,:], k=1, workers=-1)[1][0])
        R_local = float(R[idc])

        if change_mode == 'expand':
            if d0 >= 0: continue
            margin = -d0
            r_hi = max(margin + 4.0, min(3.0*margin, 1.25*R_local))
            if r_hi <= margin + 1e-6: r_hi = margin + 2.0
            if r_hi <= 0: continue
        else:
            if d0 <= 0: continue
            r_hi = min(2.5*d0, 1.25*R_local) - 1e-6
            if r_hi <= d0: continue

        pad = r_hi + 1.5
        xr0, xr1 = p[0]-pad, p[0]+pad
        yr0, yr1 = p[1]-pad, p[1]+pad
        zr0, zr1 = p[2]-pad, p[2]+pad

        Ntot = N
        idx = lambda axis, r0, r1: (max(0, int(np.searchsorted(axis, r0, side='left'))),
                                    min(Ntot, int(np.searchsorted(axis, r1, side='right'))))
        i0, i1 = idx(x_axis, xr0, xr1); j0, j1 = idx(y_axis, yr0, yr1); k0, k1 = idx(z_axis, zr0, zr1)
        if i0 >= i1 or j0 >= j1 or k0 >= k1: continue

        Xc, Yc, Zc = np.meshgrid(x_axis[i0:i1], y_axis[j0:j1], z_axis[k0:k1], indexing='ij')

        centers = np.stack([Xc.ravel(), Yc.ravel(), Zc.ravel()], axis=1).astype(np.float32)
        centers_t_all = torch.from_numpy(centers).to(device=device, dtype=torch.float32)
        M_total = centers_t_all.shape[0]

        Fin_t = torch.from_numpy(F_in_cur).to(device=device, dtype=torch.float32)
        Fou_t = torch.from_numpy(F_out_cur).to(device=device, dtype=torch.float32)
        p_t = torch.from_numpy(p).to(device=device, dtype=torch.float32)

        d_all_chunks = []
        r_limit = r_hi + 1e-9
        S_total = 24*24*24  # 仅用于预估候选球半径采样密度，不影响最终准确性

        # 构建 ROI 中落入球内且满足内外壳条件的距离分布
        for cs in range(0, M_total, 256):
            ce = min(M_total, cs + 256)
            cen = centers_t_all[cs:ce]
            # 采用单点中心采样近似统计，效率优先
            pts = cen
            dist = torch.sqrt(((pts - p_t)**2).sum(dim=1))
            def samp(vol):
                return F.grid_sample(vol.view(1,1,*F_in_cur.shape),
                                     torch.stack([((pts[:,2]-z_axis[0])/(z_axis[-1]-z_axis[0])*2-1),
                                                  ((pts[:,1]-y_axis[0])/(y_axis[-1]-y_axis[0])*2-1),
                                                  ((pts[:,0]-x_axis[0])/(x_axis[-1]-x_axis[0])*2-1)], dim=-1
                                                 ).view(1,-1,1,1,3),
                                     mode="bilinear", padding_mode="border", align_corners=True).view(-1)
            fin_vals = samp(Fin_t)
            if change_mode == 'expand':
                fou_vals = samp(Fou_t)
                elig = (dist <= r_limit) & (fin_vals > 0.0) & (fou_vals <= 0.0)
            else:
                elig = (dist <= r_limit) & (fin_vals <= 0.0)
            if elig.any():
                d_sel = dist[elig].detach().cpu().numpy()
                d_all_chunks.append(d_sel)

        if len(d_all_chunks) == 0:
            continue
        d_all = np.concatenate(d_all_chunks, axis=0)
        d_all.sort()

        remain = max(0.0, target_total - progressed)
        s_left = max_total_spheres - placed
        if s_left <= 1:
            step_goal = remain
        else:
            frac = float(rng.uniform(*step_frac_range))
            step_goal = np.clip(frac * remain, 0.15*remain, 0.60*remain)

        need_cnt = int(max(1, np.ceil(step_goal / (1.0/(24*24*24)))))
        need_cnt = min(need_cnt, d_all.shape[0])

        # 应用结果到 SDF
        r_star = float(d_all[need_cnt-1])

        if change_mode == 'expand':
            r_star = min(r_star, 1.25 * R_local)
            r_star = min(r_star, 3.0 * (-d0))
            r_star = max(r_star, -d0 + 1e-6)
            region_backup = F_in_cur[i0:i1, j0:j1, k0:k1].copy()
            sphere_sdf_roi = np.sqrt((Xc - p[0])**2 + (Yc - p[1])**2 + (Zc - p[2])**2).astype(np.float32) - r_star
            shell_roi = (F_out_cur[i0:i1, j0:j1, k0:k1] <= 0)
            F_in_cur[i0:i1, j0:j1, k0:k1] = np.where(
                shell_roi,
                np.minimum(region_backup, sphere_sdf_roi),
                region_backup
            )
        else:
            r_star = min(r_star, 1.25 * R_local - 1e-6)
            r_star = min(r_star, 2.5 * d0 - 1e-6)
            r_star = max(r_star, d0 + 1e-6)
            region_backup = F_in_cur[i0:i1, j0:j1, k0:k1].copy()
            dist_roi = np.sqrt((Xc - p[0])**2 + (Yc - p[1])**2 + (Zc - p[2])**2).astype(np.float32)
            carve = (r_star - dist_roi).astype(np.float32)
            F_in_cur[i0:i1, j0:j1, k0:k1] = np.maximum(region_backup, carve)

        placed += 1

    return F_in_cur.astype(np.float32), F_out_cur.astype(np.float32)

# =========================
# 方法二 段放缩
# =========================
@torch.no_grad()
def _build_inner_from_C_R(C, R, x_axis, y_axis, z_axis, chunk_pts=120000, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    C0 = _to_t(C[:-1], dev)
    V  = _to_t(C[1:] - C[:-1], dev)
    L2 = (V*V).sum(dim=1) + 1e-12
    R0 = _to_t(R[:-1], dev)
    R1 = _to_t(R[1:],  dev)
    C0_dot_V = (C0*V).sum(dim=1)
    P_np, shape3 = _grid_centers(x_axis, y_axis, z_axis)
    P = _to_t(P_np, dev)
    inner_vals = torch.empty((P.shape[0],), dtype=torch.float32, device=dev)
    for s in range(0, P.shape[0], chunk_pts):
        e = min(P.shape[0], s + chunk_pts)
        X = P[s:e]
        Xv = X @ V.t()
        num = Xv - C0_dot_V.unsqueeze(0)
        lam = (num / L2.unsqueeze(0)).clamp(0.0, 1.0)
        Q = C0.unsqueeze(0) + lam.unsqueeze(-1) * V.unsqueeze(0)
        D = X.unsqueeze(1) - Q
        dist = torch.sqrt((D*D).sum(dim=2) + 1e-12)
        r_lin = (1.0 - lam) * R0.unsqueeze(0) + lam * R1.unsqueeze(0)
        sdf_seg = dist - r_lin
        inner_vals[s:e] = sdf_seg.min(dim=1).values
    F_inner = inner_vals.view(*shape3).detach().cpu().numpy().astype(np.float32)
    return F_inner

def _outer_from_inner_and_delta(F_inner, delta_r_field, x_axis, y_axis, z_axis, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    P_np, shape3 = _grid_centers(x_axis, y_axis, z_axis)
    P = _to_t(P_np, dev)
    dr_t = _to_t(delta_r_field.astype(np.float32), dev)
    x_t = _to_t(x_axis, dev); y_t = _to_t(y_axis, dev); z_t = _to_t(z_axis, dev)
    dr_vals = _sample_trilinear_torch(dr_t, x_t, y_t, z_t, P).view(*shape3).detach().cpu().numpy().astype(np.float32)
    return (F_inner - dr_vals).astype(np.float32)

def _make_scale_profile(T, t0, t1, w, f):
    tau = np.linspace(0.0, 1.0, T, dtype=np.float32)
    s = np.ones((T,), dtype=np.float32)
    t0w = t0 + w
    t1w = t1 - w
    t1w = max(t1w, t0w + 1e-6)
    m1 = (tau >= t0) & (tau < t0w)
    s[m1] = 1.0 + (f - 1.0) * ((tau[m1] - t0) / max(1e-12, (t0w - t0)))
    m2 = (tau >= t0w) & (tau <= t1w)
    s[m2] = f
    m3 = (tau > t1w) & (tau <= t1)
    s[m3] = f + (1.0 - f) * ((tau[m3] - t1w) / max(1e-12, (t1 - t1w)))
    return s

@torch.no_grad()
def apply_segment_radius_scaling_to_sdfs(
    F_inner_base, delta_r_field, C, R, x_axis, y_axis, z_axis,
    mode='expand',
    target_frac_range=(0.03, 0.05),
    seg_tau_len_range=(0.18, 0.32),
    blend_tau_len_range=(0.05, 0.10),
    factor_bounds_expand=(1.02, 1.35),
    factor_bounds_shrink=(0.65, 0.98),
    seed=26000,
    nx_eval=24, ny_eval=24, nz_eval=24,
    tol_frac=0.0015,
    max_iter=14,
    device=None,
    chunk_pts=120000
):
    assert mode in ('expand','shrink')
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    rng = np.random.default_rng(seed)
    T = len(R)
    tau_len = float(rng.uniform(*seg_tau_len_range))
    blend_len = float(rng.uniform(*blend_tau_len_range))
    margin = max(0.06, blend_len + 0.02)
    t0 = float(rng.uniform(margin, 1.0 - margin - tau_len))
    t1 = t0 + tau_len
    w = min(blend_len, 0.49 * (t1 - t0))

    target_pct = float(rng.uniform(*target_frac_range)) * 100.0
    if mode == 'shrink':
        lo, hi = factor_bounds_shrink
    else:
        lo, hi = factor_bounds_expand

    def try_factor(f, return_fields=False):
        s = _make_scale_profile(T, t0, t1, w, f).astype(np.float32)
        R_mod = (R * s).astype(np.float32)
        F_inner_mod = _build_inner_from_C_R(C, R_mod, x_axis, y_axis, z_axis, chunk_pts=chunk_pts, device=device)
        F_outer_mod = _outer_from_inner_and_delta(F_inner_mod, delta_r_field, x_axis, y_axis, z_axis, device=device)
        _, _, _, _, _, stats = geometric_volume_and_change_gpu(
            F_inner_base, F_inner_mod, x_axis, y_axis, z_axis,
            nx=nx_eval, ny=ny_eval, nz=nz_eval, chunk_voxels=128, device=device
        )
        pct_net = stats["pct_net"]
        if return_fields:
            return pct_net, F_inner_mod, F_outer_mod, R_mod
        return pct_net

    f_lo, f_hi = lo, hi
    best = {"gap": 1e9, "fields": None, "f": None, "pct": None}
    for _ in range(max_iter):
        f_mid = 0.5 * (f_lo + f_hi)
        pct = try_factor(f_mid)
        # 以绝对误差匹配目标幅度
        gap = abs(abs(pct) - target_pct)
        if gap < best["gap"]:
            pct_b, Fi_b, Fo_b, R_b = try_factor(f_mid, return_fields=True)
            best.update({"gap": gap, "fields": (Fi_b, Fo_b, R_b), "f": f_mid, "pct": pct_b})
        # 二分区间收缩
        if mode == 'expand':
            if pct < target_pct: f_lo = f_mid
            else: f_hi = f_mid
        else:
            if -pct < target_pct: f_hi = f_mid
            else: f_lo = f_mid
        if best["gap"] <= tol_frac * 100.0:
            break

    if best["fields"] is None:
        edge_f = hi if mode == 'expand' else lo
        pct_b, Fi_b, Fo_b, R_b = try_factor(edge_f, return_fields=True)
        best.update({"fields": (Fi_b, Fo_b, R_b), "f": edge_f, "pct": pct_b})

    Fi_mod, Fo_mod, R_mod = best["fields"]
    info = {
        "tau_start": t0, "tau_end": t1, "blend": w,
        "factor": best["f"], "achieved_pct_net": best["pct"]
    }
    return Fi_mod.astype(np.float32), Fo_mod.astype(np.float32), R_mod.astype(np.float32), info

# =========================
# 强度体生成 采用方案B 统一实现
# =========================
def to_device_np(a, device): return torch.from_numpy(a).to(device=device, dtype=torch.float32)
def norm_coords(pts, lo, hi): return 2.0 * (pts - lo) / (hi - lo) - 1.0

def sample_trilinear_torch(vol, x_axis_t, y_axis_t, z_axis_t, pts_world, device):
    D, H, W = vol.shape
    x0, x1 = float(x_axis_t[0].item()), float(x_axis_t[-1].item())
    y0, y1 = float(y_axis_t[0].item()), float(y_axis_t[-1].item())
    z0, z1 = float(z_axis_t[0].item()), float(z_axis_t[-1].item())
    gx = norm_coords(pts_world[:,0], x0, x1)
    gy = norm_coords(pts_world[:,1], y0, y1)
    gz = norm_coords(pts_world[:,2], z0, z1)
    grid = torch.stack([gz, gy, gx], dim=-1).view(1, -1, 1, 1, 3)
    vol5d = vol.view(1,1,D,H,W)
    vals = F.grid_sample(vol5d, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return vals.view(-1)

def partial_volume_gray_schemeB_gpu(
    F_inner, F_outer, C, R, x_axis, y_axis, z_axis,
    proj_vols_hr, head_quantile_range=(0.85, 0.93), w_wm_vox_range=(20.0, 60.0),
    boundary_seed=2027, nx=24, ny=24, nz=24,
    core_low=150.0, core_high=150.0 + 255.0*(0.55*0.01),
    const_intensity={1:0.8,2:0.56,3:0.2,4:0.558,5:0.203,6:0.5535},
    chunk_voxels=96, device=None
):
    if device is None: device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    N = F_inner.shape[0]
    rng = np.random.default_rng(boundary_seed)
    q = float(rng.uniform(*head_quantile_range))
    r_thr = float(np.quantile(R, q))
    w_wm = max(2.0, float(rng.uniform(*w_wm_vox_range)))

    x_t = to_device_np(x_axis, device); y_t = to_device_np(y_axis, device); z_t = to_device_np(z_axis, device)
    fin_t = to_device_np(F_inner, device); fou_t = to_device_np(F_outer, device)
    shell_outside = (F_outer > 0)

    xg, yg, zg = np.meshgrid(x_axis, y_axis, z_axis, indexing="ij")
    xu = proj_vols_hr["x_hr"]; yu = proj_vols_hr["y_hr"]; zu = proj_vols_hr["z_hr"]
    ix = (xg - xu[0]) / (xu[-1] - xu[0]) * (len(xu)-1)
    iy = (yg - yu[0]) / (yu[-1] - yu[0]) * (len(yu)-1)
    iz = (zg - zu[0]) / (zu[-1] - zu[0]) * (len(zu)-1)
    r_itp_on_base = map_coordinates(proj_vols_hr["r_hr"], [ix, iy, iz], order=1, mode='nearest')
    head_out_mask = shell_outside & (r_itp_on_base >= r_thr)

    D_extGM1 = sdf_from_mask(head_out_mask, x_axis, y_axis, z_axis).astype(np.float32)
    Dext_t = to_device_np(D_extGM1, device)

    xh_t = to_device_np(proj_vols_hr["x_hr"], device)
    yh_t = to_device_np(proj_vols_hr["y_hr"], device)
    zh_t = to_device_np(proj_vols_hr["z_hr"], device)
    tau_hr_t  = to_device_np(proj_vols_hr["tau_hr"], device)
    r_hr_t    = to_device_np(proj_vols_hr["r_hr"], device)
    side_hr_t = to_device_np(proj_vols_hr["side_hr"], device)

    ox = ((np.arange(nx)+0.5)/nx - 0.5); oy = ((np.arange(ny)+0.5)/ny - 0.5); oz = ((np.arange(nz)+0.5)/nz - 0.5)
    OX, OY, OZ = np.meshgrid(ox, oy, oz, indexing='ij')
    offsets = np.stack([OX.ravel(), OY.ravel(), OZ.ravel()], axis=1).astype(np.float32)
    offsets_t = torch.from_numpy(offsets).to(device=device, dtype=torch.float32); S = offsets.shape[0]

    Xc, Yc, Zc = np.meshgrid(x_axis, y_axis, z_axis, indexing='ij')
    centers = np.stack([Xc.ravel(), Yc.ravel(), Zc.ravel()], axis=1).astype(np.float32)
    total = centers.shape[0]
    out = torch.zeros((total,), dtype=torch.float32, device=device)

    reg_ids = torch.tensor(sorted(const_intensity.keys()), device=device, dtype=torch.int64)
    reg_vals_norm = torch.tensor([const_intensity[int(k)] for k in reg_ids.tolist()], device=device, dtype=torch.float32)

    core_low_norm  = 0.55
    core_width_norm = float(core_high - core_low) / 255.0
    core_high_norm = core_low_norm + core_width_norm

    for s in range(0, total, chunk_voxels):
        e = min(total, s + chunk_voxels); cen = torch.from_numpy(centers[s:e]).to(device=device, dtype=torch.float32)
        M = cen.shape[0]
        pts = (cen[:, None, :] + offsets_t[None, :, :]).reshape(M*S, 3)

        fin_vals  = sample_trilinear_torch(fin_t,  x_t, y_t, z_t, pts, device)
        fou_vals  = sample_trilinear_torch(fou_t,  x_t, y_t, z_t, pts, device)
        tau_vals  = sample_trilinear_torch(tau_hr_t,  xh_t, yh_t, zh_t, pts, device)
        r_vals    = sample_trilinear_torch(r_hr_t,    xh_t, yh_t, zh_t, pts, device)
        side_vals = sample_trilinear_torch(side_hr_t, xh_t, yh_t, zh_t, pts, device)
        Dext_vals = sample_trilinear_torch(Dext_t,    x_t,  y_t,  z_t,  pts, device)

        core_mask  = (fin_vals <= 0)
        shell_in = (~core_mask) & (fou_vals <= 0)
        shell_out = (fou_vals > 0)

        head_in = shell_in & (r_vals >= r_thr)
        rest_in = shell_in & (~head_in)

        reg = torch.zeros_like(fin_vals, dtype=torch.int64, device=device)
        reg[head_in] = 2

        concave_positive = bool(proj_vols_hr.get("concave_positive", True))
        if concave_positive:
            reg[rest_in & (side_vals >= 0)] = 1
            reg[rest_in & (side_vals <  0)] = 3
        else:
            reg[rest_in & (side_vals >= 0)] = 3
            reg[rest_in & (side_vals <  0)] = 1

        head_out = shell_out & (r_vals >= r_thr); reg[head_out] = 4

        x0, x1 = float(x_axis[0]), float(x_axis[-1]); y0, y1 = float(y_axis[0]), float(y_axis[-1]); z0, z1 = float(z_axis[0]), float(z_axis[-1])
        d2b = torch.minimum(torch.minimum(pts[:,0]-x0, x1-pts[:,0]), torch.minimum(pts[:,1]-y0, y1-pts[:,1]))
        d2b = torch.minimum(d2b, torch.minimum(pts[:,2]-z0, z1-pts[:,2]))
        w_wm_t = torch.tensor(w_wm, device=device)
        wm_limit_local = torch.clamp(torch.minimum(w_wm_t, d2b - 1.5), min=0.0)

        band = shell_out & (~head_out) & (Dext_vals > 0) & (Dext_vals <= wm_limit_local + 1e-6)
        reg[band] = 5
        reg[shell_out & (~head_out) & (~band)] = 6
        reg[core_mask] = 0

        reg_blocks = reg.view(M, S)
        gray_vox_norm = torch.zeros((M,), dtype=torch.float32, device=device)

        mask_core = (reg_blocks == 0)
        if mask_core.any():
            tau_blocks = tau_vals.view(M, S)
            core_vals_norm = core_low_norm + core_width_norm * tau_blocks
            core_vals_norm = core_vals_norm * mask_core.float()
            denom = torch.clamp(mask_core.sum(dim=1).float(), min=1.0)
            core_mean_norm = core_vals_norm.sum(dim=1) / denom
            core_weight = mask_core.float().mean(dim=1)
            gray_vox_norm = gray_vox_norm + core_weight * core_mean_norm

        for k_id, k_val_norm in zip(reg_ids, reg_vals_norm):
            w = (reg_blocks == k_id).float().mean(dim=1)
            gray_vox_norm = gray_vox_norm + w * k_val_norm

        out[s:e] = gray_vox_norm * 255.0

    gray = out.view(N, N, N).detach().cpu().numpy().astype(np.float32)
    return gray

# =========================
# NIfTI 保存 与 加噪工具
# =========================
def save_nifti(volume_3d: np.ndarray, x_axis: np.ndarray, y_axis: np.ndarray, z_axis: np.ndarray, out_path: str):
    dx = float(x_axis[1] - x_axis[0]) if len(x_axis) > 1 else 1.0
    dy = float(y_axis[1] - y_axis[0]) if len(y_axis) > 1 else 1.0
    dz = float(z_axis[1] - z_axis[0]) if len(z_axis) > 1 else 1.0
    ox = float(x_axis[0]); oy = float(y_axis[0]); oz = float(z_axis[0])
    affine = np.array([[dx, 0,  0,  ox],
                       [0,  dy, 0,  oy],
                       [0,  0,  dz, oz],
                       [0,  0,  0,  1 ]], dtype=np.float32)
    img = nib.Nifti1Image(volume_3d.astype(np.float32), affine)
    nib.save(img, out_path)

def add_gaussian_noise_by_snr(vol: np.ndarray, target_snr: float, rng: np.random.Generator) -> np.ndarray:
    """
    采用整体强度标准差定义 SNR，噪声标准差 = std(vol) / target_snr
    """
    sig = float(np.std(vol))
    if not np.isfinite(sig) or sig < 1e-6:
        sig = 1.0
    noise_std = sig / max(target_snr, 1e-6)
    noise = rng.normal(0.0, noise_std, size=vol.shape).astype(np.float32)
    out = vol.astype(np.float32) + noise
    out = np.clip(out, 0.0, 255.0)
    return out

# =========================
# 形状参数采样
# =========================
def sample_shape_params_to_vox(N, base_delta_r_frac, alpha, seed, T,
                               a1_frac_range=(0.22, 0.40),
                               a2_frac_range=(0.10, 0.28),
                               b1_range=(0.85, 1.20),
                               b2_range=(1.00, 1.50),
                               r0_frac_range=(0.04, 0.07),
                               rmax_frac_range=(0.12, 0.20),
                               max_tries=200):
    rng = np.random.default_rng(seed)
    S = (N - 1) * 0.5
    for _ in range(max_tries):
        a1f = float(rng.uniform(*a1_frac_range))
        a2f = float(rng.uniform(*a2_frac_range))
        b1  = float(rng.uniform(*b1_range))
        b2  = float(rng.uniform(*b2_range))
        r0f = float(rng.uniform(*r0_frac_range))
        rmax_min = max(r0f + 0.02, rmax_frac_range[0])
        rmaxf = float(rng.uniform(rmax_min, rmax_frac_range[1]))
        a1_vox = a1f * S; a2_vox = a2f * S
        r0_vox = r0f * S; r_max_vox = rmaxf * S
        base_delta_r_vox = base_delta_r_frac * S
        margin_vox = safe_margin_vox(r0_vox, r_max_vox, base_delta_r_vox, alpha, N)
        t_vals = np.linspace(0.0, 1.0, T, dtype=np.float32)
        a1_c, b1_c, a2_c, b2_c = clamp_centerline_bend_vox(a1_vox, b1, a2_vox, b2, margin_vox, N, s_max=0.8)
        C = centerline_vox(t_vals, N, a1_c, b1_c, a2_c, b2_c, margin_vox=margin_vox)
        ok_x = (C[:,0].min() >= 0.5) and (C[:,0].max() <= (N-1) - 0.5)
        ok_z = (C[:,2].min() >= 0.5) and (C[:,2].max() <= (N-1) - 0.5)
        if ok_x and ok_z:
            return dict(a1_vox=a1_c, b1=b1_c, a2_vox=a2_c, b2=b2_c,
                        r0_vox=r0_vox, r_max_vox=r_max_vox,
                        base_delta_r_vox=base_delta_r_vox,
                        margin_vox=margin_vox)
    raise RuntimeError("形状参数采样失败，建议调整范围")

# =========================
# 主流程
# =========================
def generate_sample(
    method_name: str, mode: str, idx: int,
    base_dir: str,
    # 通用形状与噪声参数
    N=96, T=400, base_delta_r_frac=0.05, alpha=0.6, sigma_vox=3.0,
    nx_vol=24, ny_vol=24, nz_vol=24,
    nx_pv=24, ny_pv=24, nz_pv=24,
    # 方法一种子
    shape_seed_base_m1=12000, shell_seed_base_m1=13000, region_seed_base_m1=14000, sphere_seed_base_m1=16000,
    # 方法二种子
    shape_seed_base_m2=22000, shell_seed_base_m2=23000, region_seed_base_m2=24000, seg_seed_base_m2=26000,
    device=None
):
    """
    生成单个样本并保存到 NIfTI
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    # 目录
    sub_dir = os.path.join(base_dir, f"{method_name}", f"{mode}")
    os.makedirs(sub_dir, exist_ok=True)

    # 统一命名
    prefix = f"{'M1' if method_name=='method1_sphere' else 'M2'}_{'sphere' if method_name=='method1_sphere' else 'seg'}_{mode}_{idx:03d}"
    path_before = os.path.join(sub_dir, f"{prefix}_before.nii.gz")
    path_after  = os.path.join(sub_dir, f"{prefix}_after.nii.gz")
    path_label  = os.path.join(sub_dir, f"{prefix}_label.nii.gz")
    path_before_snr100 = os.path.join(sub_dir, f"{prefix}_before_snr100.nii.gz")
    path_after_snr100  = os.path.join(sub_dir, f"{prefix}_after_snr100.nii.gz")
    path_before_snr50  = os.path.join(sub_dir, f"{prefix}_before_snr50.nii.gz")
    path_after_snr50   = os.path.join(sub_dir, f"{prefix}_after_snr50.nii.gz")

    # 轴
    x_axis = np.arange(N, dtype=np.float32)
    y_axis = np.arange(N, dtype=np.float32)
    z_axis = np.arange(N, dtype=np.float32)

    # 采样形状参数
    if method_name == "method1_sphere":
        shape_seed  = shape_seed_base_m1  + idx
        shell_seed  = shell_seed_base_m1  + idx
        region_seed = region_seed_base_m1 + idx
        local_seed  = sphere_seed_base_m1 + idx
    else:
        shape_seed  = shape_seed_base_m2  + idx
        shell_seed  = shell_seed_base_m2  + idx
        region_seed = region_seed_base_m2 + idx
        local_seed  = seg_seed_base_m2    + idx

    shp = sample_shape_params_to_vox(
        N=N, base_delta_r_frac=base_delta_r_frac, alpha=alpha, seed=shape_seed, T=T,
        a1_frac_range=(0.22, 0.40), a2_frac_range=(0.10, 0.28),
        b1_range=(0.85, 1.20), b2_range=(1.00, 1.50),
        r0_frac_range=(0.04, 0.07), rmax_frac_range=(0.12, 0.20)
    )
    # 按两段代码一致的缩放
    s_curve = 0.85
    a1_vox = shp['a1_vox'] * s_curve
    a2_vox = shp['a2_vox'] * s_curve
    b1 = shp['b1']; b2 = shp['b2']
    r0_vox = shp['r0_vox']
    r_max_vox = shp['r_max_vox']
    base_delta_r_vox = shp['base_delta_r_vox']
    margin_use = shp['margin_vox']

    t_vals = np.linspace(0.0, 1.0, T, dtype=np.float32)
    a1_c, b1_c, a2_c, b2_c = clamp_centerline_bend_vox(a1_vox, b1, a2_vox, b2, margin_use, N, s_max=0.8)
    C = centerline_vox(t_vals, N, a1_c, b1_c, a2_c, b2_c, margin_vox=margin_use)
    R = exponential_radius_vox(t_vals, r0_vox, r_max_vox)

    # 基础 SDF
    F_inner, F_outer, xa, ya, za, C_used, R_used, delta_r_field, margin_used, ab_mod = build_sdfs_continuous_gpu(
        N=N, T=T, a1_vox=a1_c, b1=b1_c, a2_vox=a2_c, b2=b2_c,
        r0_vox=r0_vox, r_max_vox=r_max_vox,
        base_delta_r_vox=base_delta_r_vox, alpha=alpha, sigma_vox=sigma_vox,
        seed=shell_seed, device=dev, chunk_pts=120000, margin_override_vox=margin_use
    )
    Fi0, Fo0 = F_inner.copy(), F_outer.copy()

    # 变化后 SDF
    if method_name == "method1_sphere":
        Fi_after, Fo_after = apply_local_spherical_changes_to_inner_geometric(
            F_inner=F_inner, F_outer=F_outer, C=C_used, R=R_used,
            x_axis=x_axis, y_axis=y_axis, z_axis=z_axis,
            target_frac_range=(0.03, 0.05),
            nx_target=nx_vol, ny_target=ny_vol, nz_target=nz_vol,
            seed=local_seed, max_total_spheres=5, change_mode=mode,
            expand_center_dist_vox_range=(0.5, 2.5), shrink_center_dist_vox_range=(2.0, 8.0),
            centers_chunk=256, offsets_chunk=1024,
            step_frac_range=(0.30, 0.55), tol_frac=0.0015
        )
    else:
        Fi_after, Fo_after, R_mod, info = apply_segment_radius_scaling_to_sdfs(
            F_inner_base=F_inner, delta_r_field=delta_r_field, C=C_used, R=R_used,
            x_axis=x_axis, y_axis=y_axis, z_axis=z_axis,
            mode=mode, target_frac_range=(0.03, 0.05),
            seg_tau_len_range=(0.18, 0.32), blend_tau_len_range=(0.05, 0.10),
            factor_bounds_expand=(1.02, 1.35), factor_bounds_shrink=(0.65, 0.98),
            seed=local_seed, nx_eval=nx_vol, ny_eval=ny_vol, nz_eval=nz_vol,
            tol_frac=0.0015, max_iter=14, device=dev, chunk_pts=120000
        )

    # 计算标签，统一采用方法二的量化方式
    _, _, _, _, signed_map, stats = geometric_volume_and_change_gpu(
        Fi0, Fi_after, x_axis, y_axis, z_axis,
        nx=nx_vol, ny=ny_vol, nz=nz_vol, chunk_voxels=128, device=dev
    )
    subdiv = int(nx_vol * ny_vol * nz_vol)  # 24*24*24
    signed_map_q = np.round(signed_map * subdiv) / float(subdiv)

    # 合成强度体，使用相同的方案B
    proj_before = precompute_projection_volumes(Fi0,      C_used, R_used, x_axis, y_axis, z_axis, upsample=2, batch=1000000)
    if method_name == "method1_sphere":
        proj_after  = precompute_projection_volumes(Fi_after, C_used, R_used, x_axis, y_axis, z_axis, upsample=2, batch=1000000)
    else:
        proj_after  = precompute_projection_volumes(Fi_after, C_used, R_mod, x_axis, y_axis, z_axis, upsample=2, batch=1000000)

    const_intensity = {1:0.8, 2:0.56, 3:0.2, 4:0.558, 5:0.203, 6:0.5535}
    gray_before = partial_volume_gray_schemeB_gpu(
        Fi0, Fo0, C_used, R_used, x_axis, y_axis, z_axis,
        proj_vols_hr=proj_before,
        head_quantile_range=(0.85, 0.93),
        w_wm_vox_range=(20.0, 60.0),
        boundary_seed=region_seed,
        nx=nx_pv, ny=ny_pv, nz=nz_pv,
        core_low=150.0, core_high=150.0 + 255.0*(0.55*0.01),
        const_intensity=const_intensity,
        chunk_voxels=96, device=dev
    )
    gray_after = partial_volume_gray_schemeB_gpu(
        Fi_after, Fo_after, C_used, R_used if method_name=="method1_sphere" else R_mod,
        x_axis, y_axis, z_axis,
        proj_vols_hr=proj_after,
        head_quantile_range=(0.85, 0.93),
        w_wm_vox_range=(20.0, 60.0),
        boundary_seed=region_seed + 1000,
        nx=nx_pv, ny=ny_pv, nz=nz_pv,
        core_low=150.0, core_high=150.0 + 255.0*(0.55*0.01),
        const_intensity=const_intensity,
        chunk_voxels=96, device=dev
    )

    # 保存 before after label
    save_nifti(gray_before.astype(np.float32), x_axis, y_axis, z_axis, path_before)
    save_nifti(gray_after.astype(np.float32),  x_axis, y_axis, z_axis, path_after)
    save_nifti(signed_map_q.astype(np.float32), x_axis, y_axis, z_axis, path_label)

    # 加噪并保存
    rng_noise = np.random.default_rng(10_000_000 + idx*37 + (0 if mode=='expand' else 1) + (0 if method_name=='method1_sphere' else 2))
    before_snr100 = add_gaussian_noise_by_snr(gray_before, 100.0, rng_noise)
    after_snr100  = add_gaussian_noise_by_snr(gray_after,  100.0, rng_noise)
    before_snr50  = add_gaussian_noise_by_snr(gray_before, 50.0,  rng_noise)
    after_snr50   = add_gaussian_noise_by_snr(gray_after,  50.0,  rng_noise)

    save_nifti(before_snr100, x_axis, y_axis, z_axis, path_before_snr100)
    save_nifti(after_snr100,  x_axis, y_axis, z_axis, path_after_snr100)
    save_nifti(before_snr50,  x_axis, y_axis, z_axis, path_before_snr50)
    save_nifti(after_snr50,   x_axis, y_axis, z_axis, path_after_snr50)

    return True

def main():
    # 总体设置
    base_dir = "all_data"
    os.makedirs(base_dir, exist_ok=True)

    # 每种方法 400 个样本，其中 expand 200 个 shrink 200 个
    per_method_total = 400
    per_mode_each = per_method_total // 2  # 200

    device = "cuda" if torch.cuda.is_available() else "cpu"

    methods = [("method1_sphere", "M1 sphere"), ("method2_seg", "M2 seg")]
    total_tasks = len(methods) * per_method_total
    pbar = tqdm(total=total_tasks, ncols=100, desc="生成数据集")

    # 方法一
    for method_name, _ in methods:
        # expand
        for i in range(1, per_mode_each + 1):
            t0 = time.perf_counter()
            generate_sample(method_name=method_name, mode="expand", idx=i,
                            base_dir=base_dir, device=device)
            dt = time.perf_counter() - t0
            pbar.set_postfix_str(f"{method_name} expand #{i} 用时 {dt:.1f}s")
            pbar.update(1)
        # shrink
        for i in range(1, per_mode_each + 1):
            t0 = time.perf_counter()
            # 为避免与 expand 序号混淆，这里同样从 1 到 200 命名，目录区分
            generate_sample(method_name=method_name, mode="shrink", idx=i,
                            base_dir=base_dir, device=device)
            dt = time.perf_counter() - t0
            pbar.set_postfix_str(f"{method_name} shrink #{i} 用时 {dt:.1f}s")
            pbar.update(1)

    pbar.close()
    print("全部完成，数据保存在:", os.path.abspath(base_dir))

if __name__ == "__main__":
    main()
