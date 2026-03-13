import os
import numpy as np
from numpy.linalg import pinv
from joblib import Parallel, delayed
from skglm import WeightedLasso

from .sub_prob import compute_Zu, compute_Zv
from .utils import construct_active_set, merge_intervals


def make_kfold_splits(n, M, seed=None):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    folds = np.array_split(idx, M)
    splits = []
    for m in range(M):
        val_idx = folds[m]
        train_idx = np.concatenate([folds[k] for k in range(M) if k != m])
        splits.append((train_idx, val_idx))
    return splits


def solve_quadratic_ineq(A, B, C, z_lo, z_hi):
    if z_lo >= z_hi:
        return []

    if A == 0:
        if B == 0:
            return [(z_lo, z_hi)] if C <= 0 else []
        root = -C / B
        if B > 0:
            hi = min(root, z_hi)
            return [(z_lo, hi)] if hi > z_lo else []
        else:
            lo = max(root, z_lo)
            return [(lo, z_hi)] if lo < z_hi else []

    disc = B * B - 4.0 * A * C
    if disc < 0:
        z_mid = (z_lo + z_hi) / 2.0
        val = A * z_mid * z_mid + B * z_mid + C
        return [(z_lo, z_hi)] if val <= 0 else []

    sqrt_disc = np.sqrt(disc)
    r1 = (-B - sqrt_disc) / (2.0 * A)
    r2 = (-B + sqrt_disc) / (2.0 * A)
    root_lo, root_hi = min(r1, r2), max(r1, r2)

    if A > 0:
        lo = max(root_lo, z_lo)
        hi = min(root_hi, z_hi)
        return [(lo, hi)] if hi > lo else []
    else:
        result = []
        hi1 = min(root_lo, z_hi)
        if hi1 > z_lo:
            result.append((z_lo, hi1))
        lo2 = max(root_hi, z_lo)
        if lo2 < z_hi:
            result.append((lo2, z_hi))
        return result


def intersect_interval_lists(list1, list2):
    result = []
    i = j = 0
    while i < len(list1) and j < len(list2):
        lo = max(list1[i][0], list2[j][0])
        hi = min(list1[i][1], list2[j][1])
        if lo < hi:
            result.append((lo, hi))
        if list1[i][1] < list2[j][1]:
            i += 1
        else:
            j += 1
    return result


def calculate_gh(info, a, b, w_tilde):
    p = len(w_tilde)
    g = np.zeros(p)
    h = np.zeros(p)
    Ou = info["active_set"]
    if len(Ou) == 0:
        return g, h
    XOu = info["X_active"]
    SOu = info["sign_active"].ravel()
    w_tilde_Ou = w_tilde[Ou]
    inv = pinv(XOu.T @ XOu)
    g[Ou] = inv @ (XOu.T @ a - w_tilde_Ou * SOu)
    h[Ou] = inv @ (XOu.T @ b)
    return g, h


def calculate_ABC(g, h, a_val, b_val, X_val):
    alpha_vec = a_val - X_val @ g
    beta_vec = b_val - X_val @ h
    A = 0.5 * (beta_vec @ beta_vec)
    B = alpha_vec @ beta_vec
    C = 0.5 * (alpha_vec @ alpha_vec)
    return A, B, C


def compute_sh_path(X_train, a_train, b_train, w_tilde, z_start, z_end):
    n_train = X_train.shape[0]
    path = []
    z = z_start
    seg_left = z_start
    while z < z_end:
        Yz = a_train + b_train * z
        model = WeightedLasso(alpha=1.0 / n_train, fit_intercept=False, tol=1e-13,
                              weights=w_tilde)
        model.fit(X_train, Yz)
        info = construct_active_set(model.coef_, X_train)
        left, right = compute_Zu(info, a_train.reshape(-1, 1), b_train.reshape(-1, 1),
                                 w_tilde.reshape(-1, 1), n_train)
        g, h = calculate_gh(info, a_train, b_train, w_tilde)
        z_hi = min(right, z_end)
        if z_hi > seg_left:
            path.append((seg_left, z_hi, g, h, info["active_set"]))
        seg_left = z_hi
        z = right + 1e-5
    return path


def compute_indiv_path(XK_train, a_K_eff, b_K_eff, a_tilde, z_start, z_end):
    nK_train = XK_train.shape[0]
    p = XK_train.shape[1]
    path = []
    z = z_start
    seg_left = z_start
    while z < z_end:
        rKz = a_K_eff + b_K_eff * z
        model = WeightedLasso(alpha=1.0 / nK_train, fit_intercept=False, tol=1e-13,
                              weights=a_tilde)
        model.fit(XK_train, rKz)
        info = construct_active_set(model.coef_, XK_train)
        left, right = compute_Zv(info, b_K_eff.reshape(-1, 1), a_K_eff.reshape(-1, 1),
                                 nK_train, a_tilde.reshape(-1, 1))
        g_indiv = np.zeros(p)
        h_indiv = np.zeros(p)
        L = info["active_set"]
        if len(L) > 0:
            XK_L = info["X_active"]
            SL = info["sign_active"].ravel()
            a_tilde_L = a_tilde[L]
            inv = pinv(XK_L.T @ XK_L)
            g_indiv[L] = inv @ (XK_L.T @ a_K_eff - a_tilde_L * SL)
            h_indiv[L] = inv @ (XK_L.T @ b_K_eff)
        z_hi = min(right, z_end)
        if z_hi > seg_left:
            path.append((seg_left, z_hi, g_indiv, h_indiv))
        seg_left = z_hi
        z = right + 1e-5
    return path


def merge_cv_paths(fold_quad_paths):
    M = len(fold_quad_paths)
    if M == 0:
        return []
    all_bp = set()
    for path in fold_quad_paths:
        for (z_lo, z_hi, A, B, C) in path:
            all_bp.add(z_lo)
            all_bp.add(z_hi)
    breakpoints = sorted(all_bp)
    if len(breakpoints) < 2:
        return []
    avg_path = []
    for i in range(len(breakpoints) - 1):
        z_lo = breakpoints[i]
        z_hi = breakpoints[i + 1]
        z_mid = (z_lo + z_hi) / 2.0
        sum_A = sum_B = sum_C = 0.0
        for path in fold_quad_paths:
            for (lo, hi, A, B, C) in path:
                if lo <= z_mid <= hi:
                    sum_A += A
                    sum_B += B
                    sum_C += C
                    break
        avg_path.append((z_lo, z_hi, sum_A / M, sum_B / M, sum_C / M))
    return avg_path


def compute_cv_region(obs_avg_path, comp_avg_path):
    winner_region = []
    for (z_lo, z_hi, A_obs, B_obs, C_obs) in obs_avg_path:
        z_mid = (z_lo + z_hi) / 2.0
        A_comp = B_comp = C_comp = 0.0
        for (lo2, hi2, A2, B2, C2) in comp_avg_path:
            if lo2 <= z_mid <= hi2:
                A_comp, B_comp, C_comp = A2, B2, C2
                break
        dA = A_obs - A_comp
        dB = B_obs - B_comp
        dC = C_obs - C_comp
        winner_region.extend(solve_quadratic_ineq(dA, dB, dC, z_lo, z_hi))
    return winner_region


# def compute_cv_region(obs_avg_path, comp_avg_path):
#     all_bp = set()
#     for (lo, hi, *_) in obs_avg_path:
#         all_bp.add(lo)
#         all_bp.add(hi)
#     for (lo, hi, *_) in comp_avg_path:
#         all_bp.add(lo)
#         all_bp.add(hi)
#     breakpoints = sorted(all_bp)

#     winner_region = []
#     for i in range(len(breakpoints) - 1):
#         z_lo = breakpoints[i]
#         z_hi = breakpoints[i + 1]
#         z_mid = (z_lo + z_hi) / 2.0
#         A_obs = B_obs = C_obs = 0.0
#         for (lo1, hi1, A1, B1, C1) in obs_avg_path:
#             if lo1 <= z_mid <= hi1:
#                 A_obs, B_obs, C_obs = A1, B1, C1
#                 break
#         A_comp = B_comp = C_comp = 0.0
#         for (lo2, hi2, A2, B2, C2) in comp_avg_path:
#             if lo2 <= z_mid <= hi2:
#                 A_comp, B_comp, C_comp = A2, B2, C2
#                 break
#         dA = A_obs - A_comp
#         dB = B_obs - B_comp
#         dC = C_obs - C_comp
#         winner_region.extend(solve_quadratic_ineq(dA, dB, dC, z_lo, z_hi))
#     return winner_region

def cv_select_lambda_sh(X_all, Y_all, Lambda, fold_splits):
    p = X_all.shape[1]
    best_lam = None
    best_loss = np.inf
    for lam in Lambda:
        w_tilde = np.zeros(p)
        w_tilde[1:] = lam
        total_loss = 0.0
        for (train_idx, val_idx) in fold_splits:
            n_train = len(train_idx)
            model = WeightedLasso(alpha=1.0 / n_train, fit_intercept=False, tol=1e-13,
                                  weights=w_tilde)
            model.fit(X_all[train_idx], Y_all[train_idx])
            resid = Y_all[val_idx] - X_all[val_idx] @ model.coef_
            total_loss += 0.5 * (resid @ resid)
        avg_loss = total_loss / len(fold_splits)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_lam = lam
    return best_lam


def cv_select_Phi(X_all, XK, Y_all, YK, lambda_sh_obs, Lambda_tilde, fold_splits_K):
    n = X_all.shape[0]
    p = X_all.shape[1]
    w_tilde = np.zeros(p)
    w_tilde[1:] = lambda_sh_obs
    model_sh = WeightedLasso(alpha=1.0 / n, fit_intercept=False, tol=1e-13,
                             weights=w_tilde)
    model_sh.fit(X_all, Y_all)
    beta_sh = model_sh.coef_

    best_Phi = None
    best_loss = np.inf
    for Phi in Lambda_tilde:
        rho, lambda_K = Phi
        w_inactive = (lambda_K / rho) if rho != 0.0 else 1e15
        a_tilde = np.where(beta_sh == 0.0, w_inactive, lambda_K).astype(float)
        a_tilde[0] = 0.0
        total_loss = 0.0
        for (train_K_idx, val_K_idx) in fold_splits_K:
            nK_train = len(train_K_idx)
            XK_tr = XK[train_K_idx]
            YK_tr = YK[train_K_idx]
            residual = YK_tr - (1 - rho) * (XK_tr @ beta_sh)
            model_indiv = WeightedLasso(alpha=1.0 / nK_train, fit_intercept=False, tol=1e-13,
                                        weights=a_tilde)
            model_indiv.fit(XK_tr, residual)
            beta_K = (1 - rho) * beta_sh + model_indiv.coef_
            resid_val = YK[val_K_idx] - XK[val_K_idx] @ beta_K
            total_loss += 0.5 * (resid_val @ resid_val)
        avg_loss = total_loss / len(fold_splits_K)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_Phi = Phi
    return best_Phi


# def compute_Z2(X_all, a, b, Lambda, lambda_sh_obs, fold_splits, z_min, z_max):
#     p = X_all.shape[1]
#     cv_error_paths = {}
#     for lam in Lambda:
#         w_tilde = np.zeros(p)
#         w_tilde[1:] = lam
#         fold_quad_paths = []
#         for (train_idx, val_idx) in fold_splits:
#             lasso_path = compute_sh_path(
#                 X_all[train_idx], a[train_idx], b[train_idx], w_tilde, z_min, z_max)
#             quad_path = []
#             for (z_lo, z_hi, g, h, _) in lasso_path:
#                 A, B, C = calculate_ABC(g, h, a[val_idx], b[val_idx], X_all[val_idx])
#                 quad_path.append((z_lo, z_hi, A, B, C))
#             fold_quad_paths.append(quad_path)
#         cv_error_paths[lam] = merge_cv_paths(fold_quad_paths)
#
#     Z2 = [(z_min, z_max)]
#     obs_path = cv_error_paths[lambda_sh_obs]
#     for lam in Lambda:
#         if lam == lambda_sh_obs:
#             continue
#         winner_region = compute_cv_region(obs_path, cv_error_paths[lam])
#         Z2 = intersect_interval_lists(Z2, winner_region)
#         if not Z2:
#             break
#     return merge_intervals(Z2, tol=1e-4)


def compute_Z2(X_all, a, b, Lambda, lambda_sh_obs, fold_splits, z_min, z_max, num_segments=1):
    p = X_all.shape[1]
    n_jobs = min(num_segments, os.cpu_count())
    seg_w = (z_max - z_min) / num_segments
    segments = [(z_min + i * seg_w, z_min + (i + 1) * seg_w) for i in range(num_segments)]

    cv_error_paths = {}
    for lam in Lambda:
        w_tilde = np.zeros(p)
        w_tilde[1:] = lam
        fold_quad_paths = []
        for (train_idx, val_idx) in fold_splits:
            results = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(compute_sh_path)(
                    X_all[train_idx], a[train_idx], b[train_idx], w_tilde, seg[0], seg[1])
                for seg in segments
            )
            lasso_path = []
            for seg_path in results:
                lasso_path.extend(seg_path)
            quad_path = []
            for (z_lo, z_hi, g, h, _) in lasso_path:
                A, B, C = calculate_ABC(g, h, a[val_idx], b[val_idx], X_all[val_idx])
                quad_path.append((z_lo, z_hi, A, B, C))
            fold_quad_paths.append(quad_path)
        cv_error_paths[lam] = merge_cv_paths(fold_quad_paths)

    Z2 = [(z_min, z_max)]
    obs_path = cv_error_paths[lambda_sh_obs]
    for lam in Lambda:
        if lam == lambda_sh_obs:
            continue
        winner_region = compute_cv_region(obs_path, cv_error_paths[lam])
        Z2 = intersect_interval_lists(Z2, winner_region)
        if not Z2:
            break
    return merge_intervals(Z2, tol=1e-4)


# def compute_Z3(X_all, XK, a, b, lambda_sh_obs, Lambda_tilde, Phi_obs, fold_splits_K, z_min, z_max):
#     n = X_all.shape[0]
#     nK = XK.shape[0]
#     n_source = n - nK
#     p = X_all.shape[1]
#
#     a_K = a[n_source:]
#     b_K = b[n_source:]
#
#     w_tilde = np.zeros(p)
#     w_tilde[1:] = lambda_sh_obs
#     sh_path = compute_sh_path(X_all, a, b, w_tilde, z_min, z_max)
#
#     cv_error_paths = {}
#     for Phi in Lambda_tilde:
#         rho, lambda_K = Phi
#         w_inactive = (lambda_K / rho) if rho != 0.0 else 1e15
#         fold_quad_paths = []
#         for (train_K_idx, val_K_idx) in fold_splits_K:
#             XK_train = XK[train_K_idx]
#             XK_val = XK[val_K_idx]
#             a_K_tr = a_K[train_K_idx]
#             b_K_tr = b_K[train_K_idx]
#             a_K_va = a_K[val_K_idx]
#             b_K_va = b_K[val_K_idx]
#             nK_train = len(train_K_idx)
#
#             quad_path = []
#             for (z_lo_sh, z_hi_sh, g_sh, h_sh, Ou) in sh_path:
#                 a_tilde = np.full(p, w_inactive)
#                 if len(Ou) > 0:
#                     a_tilde[Ou] = lambda_K
#                 a_tilde[0] = 0.0
#
#                 a_K_eff = a_K_tr - (1 - rho) * (XK_train @ g_sh)
#                 b_K_eff = b_K_tr - (1 - rho) * (XK_train @ h_sh)
#
#                 indiv_path = compute_indiv_path(
#                     XK_train, a_K_eff, b_K_eff, a_tilde, z_lo_sh, z_hi_sh)
#
#                 for (z_lo_indiv, z_hi_indiv, g_indiv, h_indiv) in indiv_path:
#                     g_tilde = (1 - rho) * g_sh + g_indiv
#                     h_tilde = (1 - rho) * h_sh + h_indiv
#                     A, B, C = calculate_ABC(g_tilde, h_tilde, a_K_va, b_K_va, XK_val)
#                     quad_path.append((z_lo_indiv, z_hi_indiv, A, B, C))
#
#             fold_quad_paths.append(quad_path)
#         cv_error_paths[Phi] = merge_cv_paths(fold_quad_paths)
#
#     Z3 = [(z_min, z_max)]
#     obs_path = cv_error_paths[Phi_obs]
#     for Phi in Lambda_tilde:
#         if Phi == Phi_obs:
#             continue
#         winner_region = compute_cv_region(obs_path, cv_error_paths[Phi])
#         Z3 = intersect_interval_lists(Z3, winner_region)
#         if not Z3:
#             break
#     return merge_intervals(Z3, tol=1e-4)


def compute_Z3(X_all, XK, a, b, lambda_sh_obs, Lambda_tilde, Phi_obs, fold_splits_K, z_min, z_max, num_segments=1):
    n = X_all.shape[0]
    nK = XK.shape[0]
    n_source = n - nK
    p = X_all.shape[1]

    a_K = a[n_source:]
    b_K = b[n_source:]

    n_jobs = min(num_segments, os.cpu_count())
    seg_w = (z_max - z_min) / num_segments
    segments = [(z_min + i * seg_w, z_min + (i + 1) * seg_w) for i in range(num_segments)]

    w_tilde = np.zeros(p)
    w_tilde[1:] = lambda_sh_obs
    sh_results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(compute_sh_path)(X_all, a, b, w_tilde, seg[0], seg[1])
        for seg in segments
    )
    sh_path = []
    for seg_path in sh_results:
        sh_path.extend(seg_path)

    cv_error_paths = {}
    for Phi in Lambda_tilde:
        rho, lambda_K = Phi
        w_inactive = (lambda_K / rho) if rho != 0.0 else 1e15
        fold_quad_paths = []
        for (train_K_idx, val_K_idx) in fold_splits_K:
            XK_train = XK[train_K_idx]
            XK_val = XK[val_K_idx]
            a_K_tr = a_K[train_K_idx]
            b_K_tr = b_K[train_K_idx]
            a_K_va = a_K[val_K_idx]
            b_K_va = b_K[val_K_idx]
            nK_train = len(train_K_idx)

            quad_path = []
            for (z_lo_sh, z_hi_sh, g_sh, h_sh, Ou) in sh_path:
                a_tilde = np.full(p, w_inactive)
                if len(Ou) > 0:
                    a_tilde[Ou] = lambda_K
                a_tilde[0] = 0.0

                a_K_eff = a_K_tr - (1 - rho) * (XK_train @ g_sh)
                b_K_eff = b_K_tr - (1 - rho) * (XK_train @ h_sh)

                indiv_path = compute_indiv_path(
                    XK_train, a_K_eff, b_K_eff, a_tilde, z_lo_sh, z_hi_sh)

                for (z_lo_indiv, z_hi_indiv, g_indiv, h_indiv) in indiv_path:
                    g_tilde = (1 - rho) * g_sh + g_indiv
                    h_tilde = (1 - rho) * h_sh + h_indiv
                    A, B, C = calculate_ABC(g_tilde, h_tilde, a_K_va, b_K_va, XK_val)
                    quad_path.append((z_lo_indiv, z_hi_indiv, A, B, C))

            fold_quad_paths.append(quad_path)
        cv_error_paths[Phi] = merge_cv_paths(fold_quad_paths)

    Z3 = [(z_min, z_max)]
    obs_path = cv_error_paths[Phi_obs]
    for Phi in Lambda_tilde:
        if Phi == Phi_obs:
            continue
        winner_region = compute_cv_region(obs_path, cv_error_paths[Phi])
        Z3 = intersect_interval_lists(Z3, winner_region)
        if not Z3:
            break
    return merge_intervals(Z3, tol=1e-4)
