"""
Successive Multivariate Variational Mode Decomposition (SMVMD)
==============================================================
Exact formulation from:
Chandela, Faisal, & Sharma (IEEE Transactions on Cognitive and Developmental Systems, 2025)
Equations (3), (4), (5), (6).
"""

from typing import Tuple, List, Optional, Dict, Any
import numpy as np


class SuccessiveMVMD:
    """
    Successive Multivariate Variational Mode Decomposition (SMVMD)
    Equations (4)-(6) in Chandela et al. (IEEE TCDS 2025).
    """

    def __init__(
        self,
        alpha: float = 2000.0,
        tau: float = 0.0,
        tol: float = 1e-10,
        max_modes: int = 8,
        max_iter_mode: int = 200,
        tol_res: float = 0.015,
    ):
        self.alpha = alpha
        self.tau = tau
        self.tol = tol
        self.max_modes = max_modes
        self.max_iter_mode = max_iter_mode
        self.tol_res = tol_res

    def fit_transform(
        self,
        X: np.ndarray,
        fs: float = 128.0,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        return smvmd_decompose(
            X=X,
            fs=fs,
            alpha=self.alpha,
            tau=self.tau,
            tol=self.tol,
            max_modes=self.max_modes,
            max_iter_mode=self.max_iter_mode,
            tol_res=self.tol_res,
        )


def smvmd_decompose(
    X: np.ndarray,
    fs: float = 128.0,
    alpha: float = 2000.0,
    tau: float = 0.0,
    tol: float = 1e-10,
    max_modes: int = 8,
    max_iter_mode: int = 200,
    tol_res: float = 0.015,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Decompose multichannel signal X into successive Multivariate Intrinsic Mode Functions (MIMFs).

    Implements exact Equations (4)-(6) of Chandela et al. (2025):
    - Mode update (Eq. 4):
      u_hat_{k,c}^{n+1}(w) = [x_hat_c(w) + a^2(w - w_k^n)^4 u_hat_{k,c}^n(w) + lambda_hat_c / 2] /
                             [(1 + a^2(w - w_k^n)^4) * (1 + 2a(w - w_k^n)^2 + sum_{i=1}^{k-1} 1/(a^2(w - w_i^n)^4))]
    - Center frequency update (Eq. 5):
      w_k^{n+1} = sum_c integral(w * |u_hat_{k,c}|^2) / sum_c integral(|u_hat_{k,c}|^2)
    - Lagrange multiplier update (Eq. 6):
      lambda_hat_c^{n+1} = lambda_hat_c^n + tau * [(x_hat_c - u_hat_{k,c}^{n+1} + lambda_hat_c/2)/(1 + a^2(w - w_k^{n+1})^4) - lambda_hat_c/2]
    """
    if X.ndim == 1:
        X = X[np.newaxis, :]
        was_1d = True
    else:
        was_1d = False

    C, T = X.shape
    T_half = T // 2
    N = 2 * T

    # Mirroring to eliminate boundary artifacts
    f_mirrored = np.zeros((C, N))
    for c in range(C):
        f_mirrored[c, :T_half] = X[c, T_half:0:-1]
        f_mirrored[c, T_half : T_half + T] = X[c, :]
        f_mirrored[c, T_half + T :] = X[c, -1 : -T_half - 1 : -1]

    # One-sided FFT on positive frequencies
    f_hat = np.fft.rfft(f_mirrored, axis=1)  # shape (C, N_freqs), N_freqs = N//2 + 1
    N_freqs = f_hat.shape[1]
    freqs_norm = np.fft.rfftfreq(N, 1.0)  # [0, 0.5] normalized frequency

    total_energy = float(np.sum(np.abs(f_hat) ** 2)) + 1e-12

    u_hat_list: List[np.ndarray] = []
    omega_list: List[float] = []
    residual_energy_ratios: List[float] = []
    mode_iterations: List[int] = []

    current_res_hat = np.copy(f_hat)

    # Successive mode extraction
    for k in range(max_modes):
        # 1. Initialize center frequency from peak of residual power spectrum
        res_power = np.sum(np.abs(current_res_hat) ** 2, axis=0)
        if len(res_power) > 2:
            peak_idx = np.argmax(res_power[1:-1]) + 1
            omega_k = float(freqs_norm[peak_idx])
        else:
            omega_k = 0.25

        u_k_hat = np.zeros((C, N_freqs), dtype=complex)
        lambda_hat = np.zeros((C, N_freqs), dtype=complex)

        # Precompute sum_{i=1}^{k-1} 1 / (alpha^2 * (w - w_i)^4) penalty term (Eq. 4)
        overlap_penalty = np.zeros(N_freqs)
        for i in range(k):
            omega_i = omega_list[i]
            diff_4 = (alpha ** 2) * ((freqs_norm - omega_i) ** 4) + 1e-8
            overlap_penalty += 1.0 / diff_4

        # ADMM inner loop
        n = 0
        u_diff = 1.0
        while (u_diff > tol) and (n < max_iter_mode):
            u_prev_hat = np.copy(u_k_hat)

            # Eq. (4) mode update
            diff_w4 = (alpha ** 2) * ((freqs_norm - omega_k) ** 4)
            diff_w2 = 2.0 * alpha * ((freqs_norm - omega_k) ** 2)

            denom1 = 1.0 + diff_w4
            denom2 = 1.0 + diff_w2 + overlap_penalty
            denom_total = denom1 * denom2

            for c in range(C):
                num = current_res_hat[c, :] + diff_w4 * u_prev_hat[c, :] + (lambda_hat[c, :] / 2.0)
                u_k_hat[c, :] = num / denom_total

            # Eq. (5) joint center frequency update across all channels
            power_sum = np.sum(np.abs(u_k_hat) ** 2, axis=0)
            total_p = np.sum(power_sum) + 1e-12
            omega_k = float(np.sum(freqs_norm * power_sum) / total_p)

            # Eq. (6) Lagrangian multiplier update
            if tau > 0.0:
                diff_w4_new = (alpha ** 2) * ((freqs_norm - omega_k) ** 4)
                for c in range(C):
                    term = (current_res_hat[c, :] - u_k_hat[c, :] + lambda_hat[c, :] / 2.0) / (1.0 + diff_w4_new)
                    lambda_hat[c, :] += tau * (term - lambda_hat[c, :] / 2.0)

            # Convergence check
            u_diff = np.sum(np.abs(u_k_hat - u_prev_hat) ** 2) / (
                np.sum(np.abs(u_prev_hat) ** 2) + 1e-12
            )
            n += 1

        u_hat_list.append(u_k_hat)
        omega_list.append(omega_k)
        mode_iterations.append(n)

        # Update remaining residual spectrum
        current_res_hat = current_res_hat - u_k_hat
        res_energy = np.sum(np.abs(current_res_hat) ** 2)
        res_ratio = float(res_energy / total_energy)
        residual_energy_ratios.append(res_ratio)

        # Stopping condition
        if res_ratio < tol_res:
            break
        if k > 0 and abs(residual_energy_ratios[k - 1] - residual_energy_ratios[k]) < 1e-4:
            break

    # Reconstruct time domain modes
    K = len(u_hat_list)
    modes = np.zeros((K, C, T))
    for k in range(K):
        for c in range(C):
            u_mirrored = np.fft.irfft(u_hat_list[k][c, :], n=N)
            modes[k, c, :] = u_mirrored[T_half : T_half + T]

    omega_hz = np.array(omega_list) * fs

    if was_1d:
        modes = modes[:, 0, :]

    info = {
        "num_modes": K,
        "omega_hz": omega_hz,
        "residual_energy_ratios": residual_energy_ratios,
        "mode_iterations": mode_iterations,
        "total_energy": total_energy,
    }

    return modes, omega_hz, info


def mvmd_decompose(
    X: np.ndarray,
    K: int = 5,
    alpha: float = 2000.0,
    tau: float = 0.0,
    tol: float = 1e-10,
    max_iter: int = 200,
    fs: float = 128.0,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Standard fixed-K MVMD baseline."""
    if X.ndim == 1:
        X = X[np.newaxis, :]
        was_1d = True
    else:
        was_1d = False

    C, T = X.shape
    T_half = T // 2
    N = 2 * T
    f_mirrored = np.zeros((C, N))
    for c in range(C):
        f_mirrored[c, :T_half] = X[c, T_half:0:-1]
        f_mirrored[c, T_half : T_half + T] = X[c, :]
        f_mirrored[c, T_half + T :] = X[c, -1 : -T_half - 1 : -1]

    f_hat = np.fft.rfft(f_mirrored, axis=1)
    N_freqs = f_hat.shape[1]
    freqs_norm = np.fft.rfftfreq(N, 1.0)

    u_hat = np.zeros((K, C, N_freqs), dtype=complex)
    omega = np.linspace(0.05, 0.45, K)
    lambda_hat = np.zeros((C, N_freqs), dtype=complex)

    u_diff = 1.0
    n = 0

    while (u_diff > tol) and (n < max_iter):
        u_hat_old = np.copy(u_hat)

        for k in range(K):
            sum_other = np.sum(u_hat, axis=0) - u_hat[k]
            denom = 1.0 + 2.0 * alpha * ((freqs_norm - omega[k]) ** 2)
            for c in range(C):
                num = f_hat[c, :] - sum_other[c, :] + (lambda_hat[c, :] / 2.0)
                u_hat[k, c, :] = num / denom

            power_sum = np.sum(np.abs(u_hat[k, :, :]) ** 2, axis=0)
            total_power = np.sum(power_sum) + 1e-12
            omega[k] = float(np.sum(freqs_norm * power_sum) / total_power)

        if tau > 0.0:
            res_sum = f_hat - np.sum(u_hat, axis=0)
            lambda_hat += tau * res_sum

        u_diff = np.sum(np.abs(u_hat - u_hat_old) ** 2) / (
            np.sum(np.abs(u_hat_old) ** 2) + 1e-12
        )
        n += 1

    modes = np.zeros((K, C, T))
    for k in range(K):
        for c in range(C):
            u_mirrored = np.fft.irfft(u_hat[k, c, :], n=N)
            modes[k, c, :] = u_mirrored[T_half : T_half + T]

    omega_hz = omega * fs
    if was_1d:
        modes = modes[:, 0, :]

    info = {"num_modes": K, "omega_hz": omega_hz, "iterations": n}
    return modes, omega_hz, info


def svmd_decompose(
    x: np.ndarray,
    fs: float = 128.0,
    alpha: float = 2000.0,
    max_modes: int = 8,
    tol: float = 1e-10,
    tol_res: float = 0.015,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    return smvmd_decompose(X=x, fs=fs, alpha=alpha, max_modes=max_modes, tol=tol, tol_res=tol_res)


def vmd_decompose(
    x: np.ndarray,
    K: int = 5,
    alpha: float = 2000.0,
    fs: float = 128.0,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    return mvmd_decompose(X=x, K=K, alpha=alpha, fs=fs)
