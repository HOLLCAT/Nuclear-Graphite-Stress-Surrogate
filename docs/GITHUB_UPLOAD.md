# Private GitHub Upload

Target: `https://github.com/HOLLCAT/Nuclear-Graphite-Stress-Surrogate.git`.

These commands run on the local Mac, not CSF3. Run each block separately and stop if a command reports an error. Preparation does not require another model fit or final-test run. Authentication, installation and the final push are performed by the owner.

## Permissions and Storage

The owner has confirmed permission to upload all 199 FEM files and completed artifacts to a private repository. Verify that the target really is private before pushing. Do not change it to public without additional permission. Historical paths and account names remain visible to collaborators who can read the repository.

The archive is approximately 8.70 GB. `.gitattributes` already assigns the raw FEM TXT files, NPY/NPZ arrays, PKL checkpoints and GZ exports to Git LFS. Do not run blanket `git lfs track "*.txt"` or place all outputs in LFS: small formulas and tables should remain readable in Git. Existing newline-preservation rules must remain intact.

Check available Git LFS storage and bandwidth in the repository owner's GitHub billing settings before uploading. Included allowances are account-wide, not reserved for this repository. Uploads, new versions and subsequent downloads must be considered; this guide does not purchase storage or enable paid overages.

## 1. Install and Authenticate

The Mac already has Homebrew. Install the two missing command-line tools:

```bash
brew install git-lfs gh
git lfs version
gh --version
```

Log in through the browser with the account that owns or can write to the repository:

```bash
gh auth login --hostname github.com --git-protocol https --web
gh auth setup-git
gh auth status
```

Complete the browser approval yourself. Never paste a password, device code or access token into a chat or the repository. Do not run `gh auth token` for troubleshooting. If the CLI warns that it cannot use a credential store, resolve that issue rather than choosing insecure storage.

Check repository access and visibility:

```bash
gh repo view HOLLCAT/Nuclear-Graphite-Stress-Surrogate --json nameWithOwner,url,visibility,isEmpty,viewerPermission
git ls-remote https://github.com/HOLLCAT/Nuclear-Graphite-Stress-Surrogate.git
```

Continue only if `visibility` is `PRIVATE`, write permission is available and `isEmpty` is `true`. An empty `ls-remote` output counts as an empty repository only when the command succeeded. Authentication failure is not proof of emptiness.

If the repository is not empty, stop before initialization and inspect its contents. Do not delete remote commits, force-push, or use `--allow-unrelated-histories` just to bypass the mismatch. Existing remote work needs a reviewed import plan.

## 2. Enter and Validate the Local Project

Replace the path below with the actual local project directory, retaining quotation marks:

```bash
cd "/absolute/path/Nuclear Graphite Project"
pwd
ls README.md .gitattributes scripts/validate_delivery.py
python3 scripts/test_delivery_curation.py
python3 scripts/validate_delivery.py --full-hashes --smoke
```

Use the Python environment with the scientific dependencies already installed. This validator reproduces one full development-case prediction; it does not reopen the final test. Keep the original experiment directories and another local backup outside GitHub.

## 3. Initialize Locally and Enable LFS

The following block is for the checked, empty remote and a project directory with no existing `.git`. Run it once:

```bash
git init -b main
git lfs install --local
git remote add origin https://github.com/HOLLCAT/Nuclear-Graphite-Stress-Surrogate.git
git remote -v
git check-attr filter text -- FE_Results_Cases_All/FE_Results_Case_0.txt
```

The last command must report `filter: lfs` and `text: unset` for the raw file. Do not stage anything if LFS installation failed.

Check the commit identity. This is not a GitHub login credential:

```bash
git var GIT_AUTHOR_IDENT
```

If no identity is configured, set a name and the GitHub-provided private commit email from your account settings, using the real values rather than the placeholders:

```bash
git config user.name "YOUR_COMMIT_NAME"
git config user.email "YOUR_GITHUB_NOREPLY_EMAIL"
```

## 4. Stage and Inspect

```bash
git add .
git status --short
git diff --cached --stat
git lfs ls-files
git show :FE_Results_Cases_All/FE_Results_Case_0.txt
```

Staging several gigabytes can take time and needs additional local disk space for the LFS object cache. It is still a local operation. The `git show` command should display a short LFS pointer containing `version`, `oid sha256` and `size`, not 400,360 FEM rows. The working-tree TXT file remains the full original file.

Check that `.venv`, `.cache`, `.DS_Store`, credentials and unrelated personal files are not staged. `provenance`, `shared` and the full research outputs are intentionally included in this complete private archive.

## 5. Commit, Check LFS Objects and Push

Only proceed after reviewing the staged files and confirming storage capacity:

```bash
git commit -m "Add verified research code, FEM data and frozen results"
git lfs fsck
git lfs push --dry-run origin main
```

Stop if the object check fails. The dry run lists intended LFS uploads; it does not establish sufficient quota. The next command is the actual upload:

```bash
git push -u origin main
```

This transfers both Git history and LFS objects to the private repository. Keep the Mac awake and connected. If the network fails, inspect the message and retry `git push -u origin main` after fixing connectivity; do not recreate the repository or commit duplicate copies. If quota is exhausted, decide on storage before retrying. Do not force-push to resolve errors.

## 6. Verify the Result

```bash
git status --short
git rev-parse HEAD
git ls-remote origin refs/heads/main
gh repo view HOLLCAT/Nuclear-Graphite-Stress-Surrogate --json url,visibility
```

The local commit and remote `main` hash should match. Matching commits alone does not prove that every large object can be downloaded. The strongest transfer check is a fresh clone and full delivery validation, which require additional disk space and LFS download bandwidth:

```bash
gh repo clone HOLLCAT/Nuclear-Graphite-Stress-Surrogate ../Nuclear-Graphite-Stress-Surrogate-verify
cd ../Nuclear-Graphite-Stress-Surrogate-verify
git lfs pull
git lfs fsck
python3 scripts/validate_delivery.py --full-hashes --smoke
```

Do not remove your original local project after uploading. GitHub is one copy of the archive, not the only backup.

## Sources

- [GitHub CLI browser authentication](https://cli.github.com/manual/gh_auth_login)
- [GitHub LFS configuration](https://docs.github.com/en/repositories/working-with-files/managing-large-files/configuring-git-large-file-storage)
- [GitHub repository limits](https://docs.github.com/en/repositories/creating-and-managing-repositories/repository-limits)
- [GitHub LFS storage and bandwidth](https://docs.github.com/en/billing/concepts/product-billing/git-lfs)
