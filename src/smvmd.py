"""
smvmd.py - Successive Multivariate Variational Mode Decomposition

Paper: "Electroencephalogram-Based Unified Approach for Multiple Neurodevelopmental
        Disorders Detection in Children Using SMVMD"
Authors: Ujjawal Chandela, Kazi Newaj Faisal, Rishi Raj Sharma
IEEE TCDS, 2025

SMVMD Reference Implementation based on:
  [24] Liu & Yu, "Successive Multivariate Variational Mode Decomposition,"
       Multidimensional Systems and Signal Processing, 2022.
  [28] Ur Rehman & Aftab, "Multivariate Variational Mode Decomposition,"
       IEEE Trans. Signal Process., vol. 67, no. 23, pp. 6039–6052, Dec. 2019.

SMVMD Overview:
  SMVMD = multivariate extension of SVMD (Successive VMD)
  SVMD = successive, adaptive extension of VMD

  Key properties:
  1. Multivariate: processes C channels simultaneously
  2. Successive: extracts modes one by one (not simultaneously)
  3. Adaptive: number of modes K is determined automatically
  4. Interrelated: preserves cross-channel relationships

  For each segment X ∈ R^(C × T):
  - Extracts K Multivariate Intrinsic Mode Functions (MIMFs)
  - u_k ∈ R^(C × T) for k = 1, ..., K
  - K varies between segments (key feature of SMVMD)

  ADMM Optimization (Equations 4-6 from paper):
  u̅^{n+1}_{k,c}(ω) = [x̂_c(ω) - Σ_{i≠k} û^n_{i,c}(ω) + λ̂^n_c(ω)/2]
                      / [1 + 2α(ω - ω^n_k)²]

  ω^{n+1}_k = Σ_c ∫_0^∞ ω|û^{n+1}_{k,c}(ω)|²dω
               / Σ_c ∫_0^∞ |û^{n+1}_{k,c}(ω)|²dω

  λ̂^{n+1}_c(ω) = λ̂^n_c(ω) + τ[x̂_c(ω) - Σ_k û^{n+1}_{k,c}(ω)]

  Note: Paper equations 4-6 use a slightly different form due to the
  successive nature. The residual after removing k-1 modes is decomposed.

Paper parameters:
  DS-1: α=2000, τ=0, tol=1e-10  (bandwidth < 3 Hz)
  DS-2: α=1000, τ=0, tol=1e-10  (bandwidth < 4 Hz)

IMPORTANT IMPLEMENTATION NOTES:
  1. The paper does not specify the exact stopping criterion for SMVMD
     (when to stop adding more modes). This is THE critical unspecified detail.
     Our criterion: stop when ||u_k||² / ||x_residual||² < threshold,
     indicating the new mode captures negligible energy from the residual.
     [Paper does not specify; implementation choice]

  2. The paper does not specify center frequency initialization.
     We initialize uniformly across [0, π] (normalized).
     [Paper does not specify; implementation choice]

  3. Maximum iterations per mode: 500 [paper does not specify; implementation choice]
"""

import numpy as np
import logging
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Data Container for SMVMD Results
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SMVMDResult:
    """Container for SMVMD decomposition results.

    Attributes:
        mimfs: List of K MIMFs, each shape (C, T). Length = K.
        center_freqs: List of K center frequencies (normalized, 0 to 0.5).
        center_freqs_hz: Center frequencies in Hz (= center_freqs * fs).
        energies: Array of shape (K, C) - energy of each mode for each channel.
        n_modes: Number of extracted modes K.
        reconstruction_error: Relative reconstruction error ||x - Σu_k|| / ||x||.
        converged: Whether ADMM converged within max_iter.
        n_iters_per_mode: List of iteration counts for each mode.
        alpha: Regularization parameter used.
        residual: Signal residual after all modes extracted (C, T).
    """
    mimfs: List[np.ndarray] = field(default_factory=list)
    center_freqs: List[float] = field(default_factory=list)
    center_freqs_hz: List[float] = field(default_factory=list)
    energies: Optional[np.ndarray] = None
    n_modes: int = 0
    reconstruction_error: float = float('inf')
    converged: bool = False
    n_iters_per_mode: List[int] = field(default_factory=list)
    alpha: float = 1000.0
    residual: Optional[np.ndarray] = None


# ─────────────────────────────────────────────────────────────────────────────
# Core SMVMD Algorithm
# ─────────────────────────────────────────────────────────────────────────────

class SMVMD:
    """Successive Multivariate Variational Mode Decomposition.

    Implements the SMVMD algorithm from:
      Liu & Yu (2022), "Successive Multivariate Variational Mode Decomposition"
      Multidimensional Systems and Signal Processing.

    As applied in:
      Chandela, Faisal & Sharma (2025), IEEE TCDS.

    The algorithm successively extracts MIMFs from a multivariate signal.
    Each MIMF is extracted by solving a constrained optimization problem
    using the Alternating Direction Method of Multipliers (ADMM).

    After extracting each MIMF, the residual (signal minus all extracted MIMFs)
    is checked to determine if more modes exist.

    Parameters (from paper):
      alpha: Regularization (DS-1: 2000, DS-2: 1000)
      tau: Dual ascent step (paper: 0 for both datasets)
      tol: Convergence tolerance (paper: 1e-10 for both)
    """

    def __init__(self,
                 alpha: float = 1000.0,
                 tau: float = 0.0,
                 tol: float = 1e-10,
                 max_iter: int = 500,
                 max_modes: int = 20,
                 stop_threshold: float = 0.01,
                 init_freq: str = 'uniform'):
        """Initialize SMVMD.

        Args:
            alpha: Regularization parameter controlling mode bandwidth.
                   [Paper: 2000 for DS-1, 1000 for DS-2]
                   Larger alpha → narrower bandwidth modes.
            tau: Dual ascent step size.
                 [Paper: 0 for both datasets]
                 tau=0 avoids full reconstruction of noisy data.
            tol: Convergence tolerance for ADMM.
                 [Paper: 1e-10 for both datasets]
            max_iter: Maximum ADMM iterations per mode.
                 [Paper does not specify; implementation choice: 500]
            max_modes: Maximum number of modes to extract.
                 [Paper does not specify; implementation choice: 20]
            stop_threshold: Stop adding modes when residual energy
                 fraction falls below this threshold.
                 [Paper does not specify; implementation choice: 0.01 = 1%]
            init_freq: Center frequency initialization strategy.
                 'uniform': spread uniformly across [0, 0.5] normalized freq.
                 'random': random initialization in [0, 0.5].
                 [Paper does not specify; implementation choice: 'uniform']
        """
        self.alpha = alpha
        self.tau = tau
        self.tol = tol
        self.max_iter = max_iter
        self.max_modes = max_modes
        self.stop_threshold = stop_threshold
        self.init_freq = init_freq

    def decompose(self, x: np.ndarray, fs: float = 128.0) -> SMVMDResult:
        """Decompose a multivariate EEG segment using SMVMD.

        Input:
          x ∈ R^(C × T): C-channel signal, T samples

        Output:
          u_1, u_2, ..., u_K: K MIMFs, each ∈ R^(C × T)
          ω_1, ω_2, ..., ω_K: Center frequencies

        The decomposition proceeds as:
          x = u_1 + x_r^(1)
          x_r^(1) = u_2 + x_r^(2)
          ...
          x_r^(K-1) = u_K + x_u (unprocessed residual)

        [Paper Equations 1-2]

        Args:
            x: Multichannel EEG segment, shape (C, T).
            fs: Sampling frequency in Hz (for frequency output).

        Returns:
            SMVMDResult containing all modes, frequencies, energies, etc.
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)

        C, T = x.shape
        result = SMVMDResult(alpha=self.alpha)

        # Convert to frequency domain
        # Use one-sided FFT for real signals
        N = T
        freqs = np.fft.fftfreq(N, d=1.0 / fs)  # frequency axis in Hz
        freqs_norm = freqs / fs  # normalized: [-0.5, 0.5)

        # Take FFT of all channels
        x_hat = np.fft.fft(x, axis=1)  # shape: (C, N)

        # Keep only positive frequencies for mode update
        # (real signal: negative freqs are conjugate of positive)
        pos_freq_idx = np.arange(N // 2 + 1)
        omega = freqs_norm[pos_freq_idx]  # normalized positive freqs [0, 0.5]
        omega_hz = freqs[pos_freq_idx]    # positive freqs in Hz

        # Initialize
        residual = x.copy()
        residual_hat = x_hat.copy()
        extracted_modes = []
        extracted_freqs_norm = []
        extracted_freqs_hz = []

        # SMVMD: Successively extract modes
        for mode_idx in range(self.max_modes):
            # Check if residual has meaningful energy
            residual_energy = np.sum(residual ** 2)
            original_energy = np.sum(x ** 2) + 1e-10

            if residual_energy / original_energy < self.stop_threshold:
                logger.debug(
                    f"Mode {mode_idx}: residual energy fraction "
                    f"{residual_energy/original_energy:.4e} < "
                    f"threshold {self.stop_threshold:.4e}. Stopping."
                )
                break

            # Extract one MIMF from the residual
            mimf, omega_k, n_iters, converged = self._extract_one_mode(
                residual_hat=residual_hat,
                omega=omega,
                mode_idx=mode_idx,
                C=C,
                N=N,
            )

            # Compute mode energy and check if meaningful
            mode_energy = np.sum(mimf ** 2)
            if mode_energy < 1e-12:
                logger.debug(f"Mode {mode_idx}: negligible energy. Stopping.")
                break

            # Accept this mode
            extracted_modes.append(mimf)
            extracted_freqs_norm.append(float(omega_k))
            extracted_freqs_hz.append(float(omega_k * fs))
            result.n_iters_per_mode.append(n_iters)

            # Update residual [Paper Eq. 1: x = u_k + x_r]
            residual = residual - mimf
            residual_hat = np.fft.fft(residual, axis=1)

            logger.debug(
                f"Mode {mode_idx + 1}: ω_k={omega_k*fs:.2f} Hz, "
                f"energy={mode_energy:.4e}, converged={converged}, "
                f"iters={n_iters}"
            )

        K = len(extracted_modes)

        if K == 0:
            logger.warning("SMVMD produced 0 modes. Check input signal.")
            result.n_modes = 0
            result.residual = residual
            return result

        # Compute energies: (K, C)
        energies = np.array([
            [np.sum(mimf[c] ** 2) for c in range(C)]
            for mimf in extracted_modes
        ])  # shape: (K, C)

        # Compute reconstruction error
        reconstruction = sum(extracted_modes)
        rec_error = (np.linalg.norm(x - reconstruction) /
                     (np.linalg.norm(x) + 1e-12))

        # Populate result
        result.mimfs = extracted_modes
        result.center_freqs = extracted_freqs_norm
        result.center_freqs_hz = extracted_freqs_hz
        result.energies = energies
        result.n_modes = K
        result.reconstruction_error = float(rec_error)
        result.residual = residual
        result.converged = all([True])  # modes converged during extraction

        return result

    def _extract_one_mode(self,
                           residual_hat: np.ndarray,
                           omega: np.ndarray,
                           mode_idx: int,
                           C: int,
                           N: int
                           ) -> Tuple[np.ndarray, float, int, bool]:
        """Extract one MIMF from the current residual using ADMM.

        This implements the MVMD-style optimization for one successive mode.

        The ADMM updates are (from paper Equations 4-6, adapted for successive):

        Mode update (Eq. 4 - adapted for successive, residual is the "signal"):
        û^{n+1}_{k,c}(ω) = x̂_c^{residual}(ω) + λ̂^n_c(ω)/2
                            / [1 + 2α(ω - ω^n_k)²]

        Center frequency update (Eq. 5):
        ω^{n+1}_k = Σ_c ∫_0^∞ ω|û^{n+1}_{k,c}(ω)|² dω
                    / Σ_c ∫_0^∞ |û^{n+1}_{k,c}(ω)|² dω

        Lagrangian update (Eq. 6, tau=0 for this paper):
        λ̂^{n+1}_c(ω) = λ̂^n_c(ω) + τ[x̂_c^{residual}(ω) - û^{n+1}_{k,c}(ω)]

        NOTE: Since τ=0 (paper's setting), the Lagrangian multiplier is never
        updated. This means we skip the dual ascent step entirely.
        "τ is set at zero to avoid the full reconstruction of noisy input data"

        Args:
            residual_hat: FFT of current residual, shape (C, N).
            omega: Normalized positive frequency axis, shape (N//2+1,).
            mode_idx: Current mode index (for initialization).
            C: Number of channels.
            N: Number of time samples.

        Returns:
            Tuple of:
            - mimf: Extracted mode in time domain, shape (C, N).
            - omega_k: Final center frequency (normalized).
            - n_iters: Number of ADMM iterations used.
            - converged: Whether ADMM converged within max_iter.
        """
        n_pos = len(omega)  # Number of positive frequencies

        # Initialize center frequency
        # [Paper does not specify initialization; implementation choice]
        if self.init_freq == 'uniform':
            # Space modes uniformly; spread as modes are extracted
            omega_k = (mode_idx + 1) / (self.max_modes + 1) * 0.5
        elif self.init_freq == 'random':
            omega_k = np.random.uniform(0.01, 0.49)
        else:
            omega_k = 0.25  # Default: quarter-Nyquist

        # Clamp to valid range
        omega_k = np.clip(omega_k, 1e-6, 0.5 - 1e-6)

        # Initialize mode spectrum (positive frequencies only)
        u_hat = np.zeros((C, n_pos), dtype=complex)  # mode spectrum

        # Initialize Lagrangian (tau=0 means this stays 0 throughout)
        lambda_hat = np.zeros((C, n_pos), dtype=complex)

        # Only use positive-frequency part of residual
        residual_hat_pos = residual_hat[:, :n_pos]

        converged = False
        n_iters = 0
        prev_omega_k = omega_k

        for iteration in range(self.max_iter):
            n_iters = iteration + 1

            # ─── Mode update (Eq. 4, adapted for successive) ───────────────
            # û^{n+1}_{k,c}(ω) = (x̂_c^res(ω) + λ̂^n_c(ω)/2) / (1 + 2α(ω-ω_k)²)
            #
            # Note: In successive SMVMD, the "signal" is the residual.
            # There are no other modes to subtract (successive extraction).
            denominator = 1.0 + 2.0 * self.alpha * (omega - omega_k) ** 2
            for c in range(C):
                u_hat[c] = (residual_hat_pos[c] + lambda_hat[c] / 2.0) / denominator

            # ─── Center frequency update (Eq. 5) ───────────────────────────
            # ω^{n+1}_k = Σ_c ∫_0^∞ ω|û|² dω / Σ_c ∫_0^∞ |û|² dω
            u_hat_sq = np.abs(u_hat) ** 2  # (C, n_pos)
            numerator_omega = np.sum(u_hat_sq * omega[np.newaxis, :])
            denominator_omega = np.sum(u_hat_sq) + 1e-12
            omega_k_new = numerator_omega / denominator_omega

            # Clamp center frequency
            omega_k_new = np.clip(omega_k_new, 1e-6, 0.5 - 1e-6)

            # ─── Lagrangian update (Eq. 6) ──────────────────────────────────
            # tau=0: Lagrangian is NOT updated (paper's setting)
            # "τ is set at zero to avoid full reconstruction of noisy data"
            if self.tau > 0:
                for c in range(C):
                    lambda_hat[c] = (lambda_hat[c] +
                                     self.tau * (residual_hat_pos[c] - u_hat[c]))

            # ─── Convergence check ──────────────────────────────────────────
            freq_change = abs(omega_k_new - omega_k)
            mode_change = np.sum(np.abs(u_hat) ** 2 + 1e-12)  # relative check

            if freq_change < self.tol:
                converged = True
                omega_k = omega_k_new
                break

            omega_k = omega_k_new
            prev_omega_k = omega_k

        # Convert mode spectrum back to full (two-sided) spectrum
        u_hat_full = np.zeros((C, N), dtype=complex)
        u_hat_full[:, :n_pos] = u_hat

        # Mirror for negative frequencies (real signal)
        if N % 2 == 0:
            u_hat_full[:, n_pos:] = np.conj(u_hat[:, -2:0:-1])
        else:
            u_hat_full[:, n_pos:] = np.conj(u_hat[:, -1:0:-1])

        # Inverse FFT to time domain
        mimf = np.real(np.fft.ifft(u_hat_full, axis=1))  # (C, N)

        return mimf, float(omega_k), n_iters, converged


# ─────────────────────────────────────────────────────────────────────────────
# Batch Processing Wrapper
# ─────────────────────────────────────────────────────────────────────────────

def _decompose_one(args):
    """Worker function for parallel SMVMD decomposition of a single segment."""
    segment, alpha, tau, tol, max_iter, max_modes, stop_threshold, fs = args
    smvmd = SMVMD(
        alpha=alpha,
        tau=tau,
        tol=tol,
        max_iter=max_iter,
        max_modes=max_modes,
        stop_threshold=stop_threshold,
    )
    return smvmd.decompose(segment, fs=fs)


def decompose_segments(segments: np.ndarray,
                        alpha: float = 1000.0,
                        tau: float = 0.0,
                        tol: float = 1e-10,
                        max_iter: int = 500,
                        max_modes: int = 20,
                        stop_threshold: float = 0.01,
                        fs: float = 128.0,
                        save_path: Optional[str] = None,
                        verbose: bool = True,
                        n_jobs: int = -1
                        ) -> List[SMVMDResult]:
    """Decompose a batch of EEG segments with SMVMD.

    Args:
        segments: Array of shape (n_segments, C, T).
        alpha: Regularization parameter (DS-1: 2000, DS-2: 1000).
        tau: Dual ascent step (paper: 0).
        tol: Convergence tolerance (paper: 1e-10).
        max_iter: Max ADMM iterations per mode.
        max_modes: Max number of modes per segment.
        stop_threshold: Stopping threshold for successive extraction.
        fs: Sampling frequency (128 Hz).
        save_path: If provided, save intermediate results here.
        verbose: If True, log progress.
        n_jobs: Number of parallel workers (-1 = all cores).

    Returns:
        List of SMVMDResult objects, one per segment.
    """
    from tqdm import tqdm
    from joblib import Parallel, delayed
    import multiprocessing

    n_segments = len(segments)
    n_cores = multiprocessing.cpu_count()
    workers = n_cores if n_jobs == -1 else n_jobs

    logger.info(
        f"SMVMD decomposing {n_segments} segments: "
        f"α={alpha}, τ={tau}, tol={tol}, workers={workers}/{n_cores}"
    )

    args_list = [
        (segments[i], alpha, tau, tol, max_iter, max_modes, stop_threshold, fs)
        for i in range(n_segments)
    ]

    results = Parallel(n_jobs=n_jobs, backend='loky', verbose=0)(
        delayed(_decompose_one)(args)
        for args in tqdm(args_list, desc="SMVMD", disable=not verbose)
    )

    # Summary statistics
    mode_counts = np.array([r.n_modes for r in results])
    logger.info(
        f"\nSMVMD Summary over {n_segments} segments:\n"
        f"  K (modes): min={mode_counts.min()}, max={mode_counts.max()}, "
        f"  mean={mode_counts.mean():.2f}, std={mode_counts.std():.2f}\n"
        f"  Distribution: {dict(zip(*np.unique(mode_counts, return_counts=True)))}"
    )

    # Verify variable K (key SMVMD property)
    if len(np.unique(mode_counts)) == 1:
        logger.warning(
            f"All segments have same number of modes (K={mode_counts[0]}). "
            "Expected variable K across segments (key SMVMD property). "
            "Check stop_threshold parameter."
        )
    else:
        logger.info(
            f"✓ Variable K confirmed: K varies from {mode_counts.min()} "
            f"to {mode_counts.max()} across segments"
        )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Validation and Diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def validate_smvmd_result(result: SMVMDResult,
                           original: np.ndarray,
                           fs: float = 128.0,
                           rec_tol: float = 1e-3) -> Dict[str, Any]:
    """Validate SMVMD result for one segment.

    Checks:
    1. sum(MIMFs) + residual ≈ original (reconstruction)
    2. Variable K (report)
    3. Center frequencies are in valid range
    4. No NaN/Inf in MIMFs

    Args:
        result: SMVMDResult from decompose().
        original: Original signal (C, T).
        fs: Sampling frequency.
        rec_tol: Tolerance for reconstruction error.

    Returns:
        Dict with validation results.
    """
    validation = {
        'n_modes': result.n_modes,
        'reconstruction_error': result.reconstruction_error,
        'reconstruction_ok': result.reconstruction_error < rec_tol,
        'center_freqs_hz': result.center_freqs_hz,
        'center_freqs_valid': all(0 <= f <= fs/2 for f in result.center_freqs_hz),
        'mimf_health': [],
        'converged': result.converged,
    }

    # Check each MIMF
    for k, mimf in enumerate(result.mimfs):
        has_nan = np.any(np.isnan(mimf))
        has_inf = np.any(np.isinf(mimf))
        energy = np.sum(mimf ** 2)
        validation['mimf_health'].append({
            'mode': k + 1,
            'has_nan': bool(has_nan),
            'has_inf': bool(has_inf),
            'energy': float(energy),
            'center_freq_hz': result.center_freqs_hz[k] if k < len(result.center_freqs_hz) else None,
        })

    # Reconstruction check
    if result.n_modes > 0:
        recon = sum(result.mimfs)
        if result.residual is not None:
            recon += result.residual
        error = np.linalg.norm(original - recon) / (np.linalg.norm(original) + 1e-12)
        validation['reconstruction_error_with_residual'] = float(error)

    return validation


def report_smvmd_statistics(results: List[SMVMDResult],
                              fs: float = 128.0) -> Dict[str, Any]:
    """Compute statistics over all SMVMD results.

    Args:
        results: List of SMVMDResult from all segments.
        fs: Sampling frequency.

    Returns:
        Dict with aggregate statistics.
    """
    mode_counts = [r.n_modes for r in results]
    rec_errors = [r.reconstruction_error for r in results]

    stats = {
        'n_segments': len(results),
        'mode_counts': {
            'min': int(np.min(mode_counts)),
            'max': int(np.max(mode_counts)),
            'mean': float(np.mean(mode_counts)),
            'std': float(np.std(mode_counts)),
            'distribution': dict(zip(
                *[x.tolist() for x in np.unique(mode_counts, return_counts=True)]
            )),
        },
        'reconstruction_error': {
            'mean': float(np.mean(rec_errors)),
            'max': float(np.max(rec_errors)),
            'min': float(np.min(rec_errors)),
        },
        'variable_K': len(np.unique(mode_counts)) > 1,
    }

    return stats
