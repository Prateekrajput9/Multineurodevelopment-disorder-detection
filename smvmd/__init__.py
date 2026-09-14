"""
SMVMD: Successive Multivariate Variational Mode Decomposition
==============================================================
Implementation of the unified approach for pediatric neurodevelopmental
disorders detection (ADHD, IDD) based on:

Ujjawal Chandela, Kazi Newaj Faisal, and Rishi Raj Sharma,
"Electroencephalogram-Based Unified Approach for Multiple Neurodevelopmental
Disorders Detection in Children Using Successive Multivariate Variational Mode
Decomposition", IEEE Transactions on Cognitive and Developmental Systems, 2025.
"""

from .smvmd import (
    SuccessiveMVMD,
    smvmd_decompose,
    mvmd_decompose,
    svmd_decompose,
    vmd_decompose,
)
from .preprocessing import (
    preprocess_eeg,
    bandpass_filter,
    notch_filter,
    create_epochs,
    normalize_eeg,
)
from .features import (
    FEATURE_NAMES_TABLE_1,
    extract_9_features_1d,
    extract_mode_features,
    extract_all_features,
    compute_std,
    compute_var,
    compute_rms,
    compute_iqr,
    compute_assr,
    compute_ap,
    compute_mfl,
    compute_ip,
    compute_pcc,
)
from .feature_integration import (
    EnergyBasedFeatureIntegrator,
    integrate_mode_features,
)
from .models import (
    NeuroDisorderClassifier,
    evaluate_cross_validation,
    compute_metrics,
    print_classification_report,
)
from .dataset import (
    PediatricEEGDataset,
    generate_synthetic_eeg_cohort,
    generate_synthetic_eeg_subject,
    load_eeg_file,
    DS1_14_CHANNELS,
    DS2_19_CHANNELS,
)
from .visualization import (
    plot_raw_vs_modes,
    plot_mode_spectra,
    plot_energy_distribution,
    plot_confusion_matrix,
    plot_roc_curves,
    plot_feature_importance,
    plot_eeg_channels,
)

__version__ = "1.0.0"
