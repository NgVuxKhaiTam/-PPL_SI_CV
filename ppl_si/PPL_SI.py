import numpy as np
import os
from threadpoolctl import threadpool_limits
from joblib import Parallel, delayed
from scipy.linalg import block_diag

from .algorithms import PretrainedLasso, DTransFusion, source_estimator, Finetune_Lasso
from .utils import (
    add_intercept_column, construct_active_set, construct_Q, construct_P,
    construct_XY_tilde, construct_test_statistic_pretrained,
    calculate_a_b_pretrained, merge_intervals, calculate_TN_p_value,
    construct_test_statistic, calculate_a_b
)
from .sub_prob import (
    compute_Zu, compute_Zv, compute_Zt, calculate_xi_zeta_star, calculate_phi_iota_xi_zeta,
    compute_Zu_dtf, compute_Zv_dtf, calculate_phi_iota_xi_zeta_dtf, calculate_c_d
)
from .cv_utils import (
    make_kfold_splits, cv_select_lambda_sh, cv_select_Phi,
    compute_Z2, compute_Z3, intersect_interval_lists
)

# Pretrained Lasso
def identify_intervals_in_segment(X, XK, a, b, Mobs, n, nK, Q, w_tilde, lambda_K, rho, weight_inactive, z_start, z_end):
    with threadpool_limits(limits=1, user_api='blas'):
        intervals = []
        z = z_start
        while z < z_end:
            Yz = (a + b * z).ravel()
            YKz = Q @ Yz

            beta_sh, beta_indiv, betaK = PretrainedLasso(X, Yz, XK, YKz, w_tilde, lambda_K, rho, n, nK)
            beta_sh_info = construct_active_set(beta_sh, X)
            beta_indiv_info = construct_active_set(beta_indiv, XK)
            betaK_info = construct_active_set(betaK, XK)

            Ou = beta_sh_info["active_set"]
            a_tilde = np.full((X.shape[1], 1), weight_inactive)

            if len(Ou) > 0:
                a_tilde[Ou, 0] = lambda_K
            a_tilde[0, 0] = 0.0

            phi_u, iota_u, xi_uv, zeta_uv = calculate_phi_iota_xi_zeta(beta_sh_info, beta_indiv_info, XK, a, b, Q, rho, w_tilde, a_tilde, n, nK)

            lu, ru = compute_Zu(beta_sh_info, a, b, w_tilde, n)
            lv, rv = compute_Zv(beta_indiv_info, phi_u, iota_u, nK, a_tilde)
            lt, rt = compute_Zt(betaK_info, xi_uv, zeta_uv)
        
            right = min(ru, rv, rt)
            left = max(lu, lv, lt)
            if right < left or right < z: 
                print('Error')
                return ([], [])

            Mt = betaK_info["active_set"]
            if np.array_equal(Mobs, Mt):
                intervals.append((left, right))

            z = right + 1e-5

    return intervals



def divide_and_conquer(X, XK, a, b, Mobs, n, nK, w_tilde, lambda_K, rho, z_min, z_max, num_segments=1):
    seg_w = (z_max - z_min) / num_segments
    segments = [(z_min + i * seg_w, z_min + (i + 1) * seg_w) for i in range(num_segments)]
    n_jobs = min(num_segments, os.cpu_count())
    
    w_inactive = lambda_K / rho if rho != 0.0 else 1e15
    Q = construct_Q(n, nK)


    results = Parallel(n_jobs=n_jobs, backend="loky")(
        (delayed(identify_intervals_in_segment)(X, XK, a, b, Mobs, n, nK, Q, w_tilde, lambda_K, rho, w_inactive, seg[0], seg[1]) for seg in segments)
    )

    intervals = []

    for seg_intervals in results:
        intervals.extend(seg_intervals)
    
    intervals = merge_intervals(intervals, tol=1e-4)

    return intervals


def PPL_SI(X_list, Y_list, lambda_sh, lambda_K, rho, Sigma_list, z_min=-20, z_max=20, num_segments=1):
    X_ = np.concatenate(X_list)
    Y = np.concatenate(Y_list)
    XK_ = X_list[-1]
    YK = Y_list[-1]
    n_list = [Xk.shape[0] for Xk in X_list]
    n = sum(n_list)
    nK = n_list[-1]

    X = add_intercept_column(X_)
    XK = add_intercept_column(XK_)


    w_tilde = np.concatenate(([0.0], np.full(X.shape[1] - 1, lambda_sh))).reshape(-1, 1)
    beta_sh_hat, beta_indiv_hat, betaK_hat = PretrainedLasso(X, Y, XK, YK, w_tilde, lambda_K, rho, n, nK)

    Mobs = [i for i in range(len(betaK_hat)) if betaK_hat[i] != 0.0]
    
    if len(Mobs) == 0:
        return None

    XK_M = XK[:, Mobs]
    Sigma = block_diag(*Sigma_list)


    
    p_sel_list = []
    
    for j in Mobs:
        etaj, etajTY = construct_test_statistic(j, XK_M, Y, Mobs, n, nK)
        a, b = calculate_a_b(etaj, Y, Sigma, n)
        intervals = divide_and_conquer(X, XK, a, b, Mobs, n, nK, w_tilde, lambda_K, rho, z_min, z_max, num_segments)
        p_value = calculate_TN_p_value(intervals, etaj, etajTY, Sigma, 0)
        p_sel_list.append((j, p_value))
    
    return p_sel_list


def PPL_SI_randj(X_list, Y_list, lambda_sh, lambda_K, rho, Sigma_list, z_min=-20, z_max=20, num_segments=1):
    X_ = np.concatenate(X_list)
    Y = np.concatenate(Y_list)
    XK_ = X_list[-1]
    YK = Y_list[-1]
    n_list = [Xk.shape[0] for Xk in X_list]
    n = sum(n_list)
    nK = n_list[-1]

    X = add_intercept_column(X_)
    XK = add_intercept_column(XK_)


    w_tilde = np.concatenate(([0.0], np.full(X.shape[1] - 1, lambda_sh))).reshape(-1, 1)
    beta_sh_hat, beta_indiv_hat, betaK_hat = PretrainedLasso(X, Y, XK, YK, w_tilde, lambda_K, rho, n, nK)

    Mobs = [i for i in range(len(betaK_hat)) if betaK_hat[i] != 0.0]
    
    if len(Mobs) == 0:
        return None

    XK_M = XK[:, Mobs]
    Sigma = block_diag(*Sigma_list)

    
    j = np.random.choice(Mobs)
    etaj, etajTY = construct_test_statistic(j, XK_M, Y, Mobs, n, nK)
    a, b = calculate_a_b(etaj, Y, Sigma, n)
    intervals = divide_and_conquer(X, XK, a, b, Mobs, n, nK, w_tilde, lambda_K, rho, z_min, z_max, num_segments)
    p_value = calculate_TN_p_value(intervals, etaj, etajTY, Sigma, 0)

    return j, p_value



# Pretrained Lasso Only Parameters
def identify_intervals_in_segment_param_only(XK, beta_sh, a, b, Mobs, nK, a_tilde, rho, z_start, z_end):
    with threadpool_limits(limits=1, user_api='blas'):
        intervals = []
        z = z_start

        phi = b.copy()
        iota = a - (1 - rho) * (XK @ beta_sh).reshape(-1, 1)
        while z < z_end:
            Yz = (a + b * z).ravel()

            beta_indiv, betaK = Finetune_Lasso(XK, Yz, beta_sh, a_tilde, rho, nK)

            beta_indiv_info = construct_active_set(beta_indiv, XK)
            betaK_info = construct_active_set(betaK, XK)

            xi_uv, zeta_uv = calculate_xi_zeta_star(beta_sh, beta_indiv_info, XK, phi, iota, rho, a_tilde)

            lv, rv = compute_Zv(beta_indiv_info, phi, iota, nK, a_tilde)
            lt, rt = compute_Zt(betaK_info, xi_uv, zeta_uv)
        

            right = min(rv, rt)
            left = max(lv, lt)
            if right < left or right < z: 
                print ('Error')
                return ([], [])

            Mt = betaK_info["active_set"]
            if np.array_equal(Mobs, Mt):
                intervals.append((left, right))


            z = right + 1e-5  

    return intervals



def divide_and_conquer_param_only(beta_sh, XK, a, b, Mobs, nK, a_tilde, rho, z_min, z_max, num_segments=24):
    seg_w = (z_max - z_min) / num_segments
    segments = [(z_min + i * seg_w, z_min + (i + 1) * seg_w) for i in range(num_segments)]

    n_jobs = min(num_segments, os.cpu_count())
    
    
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        (delayed(identify_intervals_in_segment_param_only)(XK, beta_sh, a, b, Mobs, nK, a_tilde, rho, seg[0], seg[1]) for seg in segments)
    )

    intervals = []

    for seg_intervals in results:
        intervals.extend(seg_intervals)
    
    intervals = merge_intervals(intervals, tol=1e-4)

    return intervals


def PPL_SI_param_only(beta_sh, XK, YK, lambda_K, rho, Sigma_K, z_min=-20, z_max=20):

    nK = XK.shape[0]

    XK = add_intercept_column(XK)

    a_tilde = np.where(beta_sh == 0.0, lambda_K / rho if rho != 0.0 else 1e15, lambda_K).astype(float).reshape(-1, 1)

    beta_indiv_hat, betaK_hat = Finetune_Lasso(XK, YK, beta_sh, a_tilde, rho, nK)

    Mobs = [i for i in range(len(betaK_hat)) if betaK_hat[i] != 0.0]
    
    if len(Mobs) == 0:
        return None

    XK_M = XK[:, Mobs]
    Sigma = Sigma_K


    p_sel_list = []
    
    for j in Mobs:
        etaj, etajTY = construct_test_statistic_pretrained(j, XK_M, YK, Mobs)
        a, b = calculate_a_b(etaj, YK, Sigma, nK) 
        intervals = divide_and_conquer_param_only(beta_sh, XK, a, b, Mobs, nK, a_tilde, rho, z_min, z_max)
        p_value = calculate_TN_p_value(intervals, etaj, etajTY, Sigma, 0)
        p_sel_list.append((j, p_value))
    
    return p_sel_list


def PPL_SI_param_only_randj(beta_sh, XK, YK, lambda_K, rho, Sigma_K, z_min=-20, z_max=20, num_segments=1):

    nK = XK.shape[0]

    XK = add_intercept_column(XK)

    a_tilde = np.where(beta_sh == 0.0, lambda_K / rho if rho != 0.0 else 1e15, lambda_K).astype(float).reshape(-1, 1)

    beta_indiv_hat, betaK_hat = Finetune_Lasso(XK, YK, beta_sh, a_tilde, rho, nK)

    Mobs = [i for i in range(len(betaK_hat)) if betaK_hat[i] != 0.0]
    
    if len(Mobs) == 0:
        return None

    XK_M = XK[:, Mobs]
    Sigma = Sigma_K

    j = np.random.choice(Mobs)
    etaj, etajTY = construct_test_statistic_pretrained(j, XK_M, YK, Mobs)
    a, b = calculate_a_b(etaj, YK, Sigma, nK) 
    intervals = divide_and_conquer_param_only(beta_sh, XK, a, b, Mobs, nK, a_tilde, rho, z_min, z_max, num_segments)
    p_value = calculate_TN_p_value(intervals, etaj, etajTY, Sigma, 0)

    return j, p_value





# DTransFusion

def identify_intervals_in_segment_dtf(X_tilde, XK, a, b, c, d, Mobs, q_tilde, lambda_tilde, P, n, nK, K, z_start, z_end):
    with threadpool_limits(limits=1, user_api='blas'):

        intervals = []
        z = z_start
        while z < z_end:
            Yz = (a + b * z).ravel()
            Y_tilde_z = (c + d * z).ravel()

            theta, delta, betaK = DTransFusion(X_tilde, Y_tilde_z, XK, Yz, q_tilde, lambda_tilde, P, n)
            theta_info = construct_active_set(theta, X_tilde)
            delta_info = construct_active_set(delta, XK)
            betaK_info = construct_active_set(betaK, XK)

            phi_u, iota_u, xi_uv, zeta_uv = calculate_phi_iota_xi_zeta_dtf(theta_info, delta_info, XK, a, b, c, d, P, q_tilde, lambda_tilde, n, nK, K)
        
            lu, ru = compute_Zu_dtf(theta_info, c, d, q_tilde, n)
            lv, rv = compute_Zv_dtf(delta_info, phi_u, iota_u, lambda_tilde, nK)
            lt, rt = compute_Zt(betaK_info, xi_uv, zeta_uv)
        
            right = min(ru, rv, rt)
            left = max(lu, lv, lt)
            if right < left or right < z: 
                print('Error')
                return ([], [])

            Mt = betaK_info["active_set"]
            
            if np.array_equal(Mobs, Mt):
                intervals.append((left, right))

            z = right + 1e-5

    return intervals


def divide_and_conquer_dtf(X_tilde, XK, a, b, c, d, Mobs, q_tilde, lambda_tilde, P, n, nK, K, z_min, z_max, num_segments=24):
    seg_w = (z_max - z_min) / num_segments
    segments = [(z_min + i * seg_w, z_min + (i + 1) * seg_w) for i in range(num_segments)]
    n_jobs = min(num_segments, os.cpu_count())

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        (delayed(identify_intervals_in_segment_dtf)(X_tilde, XK, a, b, c, d, Mobs, q_tilde, lambda_tilde, P, n, nK, K, seg[0], seg[1]) for seg in segments)
    )

    intervals = []

    for seg_intervals in results:
        intervals.extend(seg_intervals)
    
    intervals = merge_intervals(intervals, tol=1e-4)

    return intervals


def PPL_SI_DTF(XK, YK, beta_tilde_list, n_list, lambda_0, lambda_tilde, qk_weights, Sigma_K, z_min=-20, z_max=20, num_segments=1):
    K = len(n_list)
    nK = XK.shape[0]
    p = XK.shape[1]
    n = sum(n_list)


    P = construct_P(K, p, n, n_list)
    X_tilde, Y_tilde = construct_XY_tilde(beta_tilde_list, n_list, XK, YK, K, p)

    q_tilde = np.concatenate(
        [lambda_0 * qk_weights[k] * np.ones(p) for k in range(K - 1)] +
        [lambda_0 * np.ones(p)]
    ).reshape(-1, 1)
    
    theta_hat, delta_hat, betaK_hat = DTransFusion(X_tilde, Y_tilde, XK, YK, q_tilde, lambda_tilde, P, n)

    Mobs = [i for i in range(p) if betaK_hat[i] != 0.0]
    
    if len(Mobs) == 0:
        return None
    
    XK_M = XK[:, Mobs]
    Sigma = Sigma_K

    p_sel_list = []

    for j in Mobs:
        etaj, etajTY = construct_test_statistic_pretrained(j, XK_M, YK, Mobs)
        a, b = calculate_a_b_pretrained(etaj, YK, Sigma, nK)
        c, d = calculate_c_d(a, b, beta_tilde_list, n_list, K, p)
        intervals = divide_and_conquer_dtf(X_tilde, XK, a, b, c, d, Mobs, q_tilde, lambda_tilde, P, n, nK, K, z_min, z_max, num_segments)
        p_value = calculate_TN_p_value(intervals, etaj, etajTY, Sigma, 0)
        p_sel_list.append((j, p_value))

    return p_sel_list


def PPL_SI_DTF_randj(XK, YK, beta_tilde_list, n_list, lambda_0, lambda_tilde, qk_weights, Sigma_K, z_min=-20, z_max=20, num_segments=1):
    K = len(n_list)
    nK = XK.shape[0]
    p = XK.shape[1]
    n = sum(n_list)


    P = construct_P(K, p, n, n_list)
    X_tilde, Y_tilde = construct_XY_tilde(beta_tilde_list, n_list, XK, YK, K, p)

    q_tilde = np.concatenate(
        [lambda_0 * qk_weights[k] * np.ones(p) for k in range(K - 1)] +
        [lambda_0 * np.ones(p)]
    ).reshape(-1, 1)
    
    theta_hat, delta_hat, betaK_hat = DTransFusion(X_tilde, Y_tilde, XK, YK, q_tilde, lambda_tilde, P, n)

    Mobs = [i for i in range(p) if betaK_hat[i] != 0.0]
    
    if len(Mobs) == 0:
        return None
    
    XK_M = XK[:, Mobs]
    Sigma = Sigma_K

    j = np.random.choice(Mobs)
    etaj, etajTY = construct_test_statistic_pretrained(j, XK_M, YK, Mobs)
    a, b = calculate_a_b_pretrained(etaj, YK, Sigma, nK)
    c, d = calculate_c_d(a, b, beta_tilde_list, n_list, K, p)
    intervals = divide_and_conquer_dtf(X_tilde, XK, a, b, c, d, Mobs, q_tilde, lambda_tilde, P, n, nK, K, z_min, z_max, num_segments)
    p_value = calculate_TN_p_value(intervals, etaj, etajTY, Sigma, 0)

    return j, p_value



# Pretrained Lasso with CV
def PPL_SI_CV(X_list, Y_list, Lambda, Lambda_tilde, Sigma_list, n_folds=5, z_min=-20, z_max=20, num_segments=1, seed=None):
    X_ = np.concatenate(X_list)
    Y = np.concatenate(Y_list)
    XK_ = X_list[-1]
    YK = Y_list[-1]
    n_list = [Xk.shape[0] for Xk in X_list]
    n = sum(n_list)
    nK = n_list[-1]

    X = add_intercept_column(X_)
    XK = add_intercept_column(XK_)
    Sigma = block_diag(*Sigma_list)

    fold_splits_all = make_kfold_splits(n, n_folds, seed)
    fold_splits_K = make_kfold_splits(nK, n_folds, seed)

    lambda_sh_obs = cv_select_lambda_sh(X, Y, Lambda, fold_splits_all)
    Phi_obs = cv_select_Phi(X, XK, Y, YK, lambda_sh_obs, Lambda_tilde, fold_splits_K)
    rho_obs, lambda_K_obs = Phi_obs

    w_tilde = np.concatenate(([0.0], np.full(X.shape[1] - 1, lambda_sh_obs))).reshape(-1, 1)
    beta_sh_hat, beta_indiv_hat, betaK_hat = PretrainedLasso(X, Y, XK, YK, w_tilde, lambda_K_obs, rho_obs, n, nK)

    Mobs = [i for i in range(len(betaK_hat)) if betaK_hat[i] != 0.0]

    if len(Mobs) == 0:
        return None

    XK_M = XK[:, Mobs]

    p_sel_list = []

    for j in Mobs:
        etaj, etajTY = construct_test_statistic(j, XK_M, Y, Mobs, n, nK)
        a, b = calculate_a_b(etaj, Y, Sigma, n)

        intervals_Z1 = divide_and_conquer(X, XK, a, b, Mobs, n, nK, w_tilde, lambda_K_obs, rho_obs, z_min, z_max, num_segments)

        intervals_Z2 = compute_Z2(X, a.ravel(), b.ravel(), Lambda, lambda_sh_obs, fold_splits_all, z_min, z_max, num_segments)
        intervals_Z3 = compute_Z3(X, XK, a.ravel(), b.ravel(), lambda_sh_obs, Lambda_tilde, Phi_obs, fold_splits_K, z_min, z_max, num_segments)

        intervals_CV = intersect_interval_lists(
            intersect_interval_lists(intervals_Z1, intervals_Z2),
            intervals_Z3
        )

        p_value = calculate_TN_p_value(intervals_CV, etaj, etajTY, Sigma, 0)
        p_sel_list.append((j, p_value))

    return p_sel_list


def PPL_SI_CV_randj(X_list, Y_list, Lambda, Lambda_tilde, Sigma_list, n_folds=5, z_min=-20, z_max=20, num_segments=1, seed=None):
    X_ = np.concatenate(X_list)
    Y = np.concatenate(Y_list)
    XK_ = X_list[-1]
    YK = Y_list[-1]
    n_list = [Xk.shape[0] for Xk in X_list]
    n = sum(n_list)
    nK = n_list[-1]

    X = add_intercept_column(X_)
    XK = add_intercept_column(XK_)
    Sigma = block_diag(*Sigma_list)

    fold_splits_all = make_kfold_splits(n, n_folds, seed)
    fold_splits_K = make_kfold_splits(nK, n_folds, seed)

    lambda_sh_obs = cv_select_lambda_sh(X, Y, Lambda, fold_splits_all)
    Phi_obs = cv_select_Phi(X, XK, Y, YK, lambda_sh_obs, Lambda_tilde, fold_splits_K)
    rho_obs, lambda_K_obs = Phi_obs

    w_tilde = np.concatenate(([0.0], np.full(X.shape[1] - 1, lambda_sh_obs))).reshape(-1, 1)
    beta_sh_hat, beta_indiv_hat, betaK_hat = PretrainedLasso(X, Y, XK, YK, w_tilde, lambda_K_obs, rho_obs, n, nK)

    Mobs = [i for i in range(len(betaK_hat)) if betaK_hat[i] != 0.0]

    if len(Mobs) == 0:
        return None

    XK_M = XK[:, Mobs]

    j = np.random.choice(Mobs)
    etaj, etajTY = construct_test_statistic(j, XK_M, Y, Mobs, n, nK)
    a, b = calculate_a_b(etaj, Y, Sigma, n)

    intervals_Z1 = divide_and_conquer(X, XK, a, b, Mobs, n, nK, w_tilde, lambda_K_obs, rho_obs, z_min, z_max, num_segments)

    intervals_Z2 = compute_Z2(X, a.ravel(), b.ravel(), Lambda, lambda_sh_obs, fold_splits_all, z_min, z_max, num_segments)
    intervals_Z3 = compute_Z3(X, XK, a.ravel(), b.ravel(), lambda_sh_obs, Lambda_tilde, Phi_obs, fold_splits_K, z_min, z_max, num_segments)

    intervals_CV = intersect_interval_lists(
        intersect_interval_lists(intervals_Z1, intervals_Z2),
        intervals_Z3
    )

    p_value = calculate_TN_p_value(intervals_CV, etaj, etajTY, Sigma, 0)

    return j, p_value
