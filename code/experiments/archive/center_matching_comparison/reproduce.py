"""Re-run the original center-matching observation without altering core code.

This tool resumes from the FIVE SAVED original rounds, rather than starting
again from nearly one-hot centers. It recomputes new Monte Carlo observations
under the same Q-optimization/walk configuration and writes a separate replay.
The full 49 rounds can take several minutes on an ordinary laptop.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
ORIGINAL = REPO / 'code' / 'experiments' / 'center_matching_observation'
GENERATOR = REPO / 'code' / 'data_generator' / 'generate.py'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rounds', type=int, default=49,
                        help='Total number of rounds including five original rounds')
    parser.add_argument('--data-dir', type=Path, default=REPO / 'generated_data')
    parser.add_argument('--output-dir', type=Path,
                        default=REPO / 'generated_data' / 'center_matching_replay')
    parser.add_argument('--force', action='store_true',
                        help='Replace an existing replay checkpoint, never source results')
    args = parser.parse_args()
    if args.rounds < 5:
        parser.error('--rounds must be >=5 to resume from original five rounds')
    output_dir = args.output_dir.resolve()
    checkpoint = output_dir / 'replay.json'
    source = ORIGINAL / 'results' / 'center_matching_round1.json'
    if not source.is_file():
        parser.error(f'Original five-round checkpoint is missing: {source}')
    if checkpoint.exists() and not args.force:
        parser.error(f'Replay already exists: {checkpoint}. Use --force to replace it.')
    output_dir.mkdir(parents=True, exist_ok=True)
    train = args.data_dir / 'train.npz'
    validation = args.data_dir / 'validation.npz'
    if not (train.exists() and validation.exists()):
        subprocess.run([sys.executable, str(GENERATOR),
                        '--config', str(REPO / 'code' / 'data_generator' / 'config.json'),
                        '--out', str(args.data_dir)], cwd=REPO, check=True)
    shutil.copy2(source, checkpoint)
    cmd = [
        sys.executable, str(ORIGINAL / 'observe.py'),
        '--train', str(train), '--validation', str(validation),
        '--policy', str(REPO / 'code' / 'policy_evaluator' / 'policy_uniform.json'),
        '--rounds', str(args.rounds), '--eta', '0.5', '--epsilon', '0.08',
        '--walks', '100', '--optimizer-steps', '45', '--seed', '20261104',
        '--resume', '--out', str(checkpoint),
    ]
    print('Replaying with identical model parameters from original five-round checkpoint.', flush=True)
    subprocess.run(cmd, cwd=REPO, check=True)
    subprocess.run([sys.executable, str(HERE / 'analyze.py'),
                    '--history', str(checkpoint),
                    '--outdir', str(output_dir / 'results')], cwd=REPO, check=True)
    print('Replay and comparison saved under:', output_dir)


if __name__ == '__main__':
    main()
