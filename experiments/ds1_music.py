"""
ds1_music.py - DS-1 IDD vs TDC (Music Stimulus) Experiment

Paper target: Accuracy = 100%, Precision = 100%, Recall = 100%, F1 = 100%
KNN params (Table II): n=2, Cityblock, Squared Inverse weight, standardize=False
mRMR: 47 features selected

Run:
    python -m experiments.ds1_music
    python -m experiments.ds1_music --no-cache   # force full recomputation
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import run_ds1_experiment
from src.results_io import ensure_dirs
from src.utils import load_config, setup_logger, set_random_seed

logger = setup_logger('DS1_Music', 'results/logs/ds1_music.log')


def main(use_cache: bool = True):
    ensure_dirs()
    config = load_config('configs/config.yaml')
    set_random_seed(config.get('random_seed', 42))

    return run_ds1_experiment(
        config=config,
        conditions=['music'],
        exp_key='DS1_Music',
        knn_key='music',
        n_features_key='n_features_music',
        title='DS-1 Music (IDD vs TDC)',
        logger=logger,
        use_cache=use_cache,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-cache', action='store_true',
                        help='Ignore any cached feature matrix and recompute '
                             'preprocessing, SMVMD and feature extraction.')
    args = parser.parse_args()
    main(use_cache=not args.no_cache)
