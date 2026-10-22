# Publishing PHORA — phase-to-decision research site

This package is an overlay for the existing `rdharini2001/PHORA` repository. It preserves the scientific source tree and original validation CSVs.

1. Clone `https://github.com/rdharini2001/PHORA`.
2. Extract this ZIP, then run `bash scripts/install_in_repo.sh /path/to/PHORA` from the extracted `PHORA_Research_Companion` folder.
3. Review the resulting changes with `git diff`, then commit and push.
4. In GitHub Settings → Pages, choose GitHub Actions. The supplied workflow deploys `docs/`.

Static site entry: `docs/index.html`. Signed image maps are pre-rendered; no backend or CT data download is needed to browse them. Source NIfTI series are excluded from the public website archive.

Only the website and viewer assets are updated; published article and PHORA code remain linked. The frozen inference examples are distinct from the two CT visualization cases.
