#!/usr/bin/env python3
"""Rebuild the ready-to-upload Codabench ZIP from challenge/codabench_source."""
from pathlib import Path
import zipfile
repo=Path(__file__).resolve().parents[1]
src=repo/'challenge/codabench_source'
out=repo/'challenge/PHORA_AMPLIFAI_Codabench.zip'
with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(src.rglob('*')):
        if p.is_file(): z.write(p,p.relative_to(src))
print(out)
