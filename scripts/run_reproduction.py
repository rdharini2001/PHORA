#!/usr/bin/env python3
"""Execute the reproducibility notebook in fast or paper mode."""
from __future__ import annotations
import argparse, os, subprocess, sys
from pathlib import Path


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True, type=Path, help='Folder with batch_*.zip, train_metadata.csv and val_metadata.csv')
    ap.add_argument('--mode', choices=['fast','paper'], default='fast')
    ap.add_argument('--output-dir', type=Path, default=None)
    ap.add_argument('--n-jobs', type=int, default=4)
    args=ap.parse_args()
    repo=Path(__file__).resolve().parents[1]
    out=(args.output_dir or (repo/'runs'/args.mode)).resolve(); out.mkdir(parents=True,exist_ok=True)
    executed=out/'PHORA_reproduction_executed.ipynb'
    env=os.environ.copy()
    env['PHORA_REPO_ROOT']=str(repo)
    env['PHORA_DATA_ROOT']=str(args.data_root.resolve())
    env['PHORA_OUTPUT_DIR']=str(out)
    env['PHORA_MODE']=args.mode
    env['PHORA_N_JOBS']=str(args.n_jobs)
    cmd=[sys.executable,'-m','jupyter','nbconvert','--to','notebook','--execute',
         str(repo/'notebooks/01_reproduce_paper.ipynb'),'--output',str(executed),
         '--ExecutePreprocessor.timeout=-1']
    print('Running:', ' '.join(map(str,cmd)))
    subprocess.run(cmd,check=True,env=env,cwd=repo)
    print('Executed notebook:', executed)

if __name__=='__main__': main()
