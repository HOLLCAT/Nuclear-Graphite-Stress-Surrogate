"""Read-only publication checks; no training, unpickling, Git commit or upload."""
from __future__ import annotations

import argparse
import codecs
from collections import Counter
import csv
import gzip
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SKIP_NAMES = {".git", ".venv", ".cache", "__pycache__", ".ipynb_checkpoints", ".DS_Store"}
TEXT_SUFFIXES = {".py", ".ipynb", ".json", ".csv", ".txt", ".log", ".out", ".err",
                 ".md", ".tex", ".sh", ".slurm", ".toml", ".yml", ".yaml", ".ini"}
HAN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\U00020000-\U0002ffff]")
PRIVATE_PATH = re.compile(r"/Users/[^/\s]+/|/(?:net/)?scratch/[^/\s]+/|/mnt/iusers[0-9]+/")
SECRET_PATTERNS = {
    "private_key_header": re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |ENCRYPTED )?PRIVATE KEY-----"),
    "github_token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,})\b"),
    "openai_token": re.compile(r"\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{40,}\b"),
    "aws_access_key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "literal_password": re.compile(r"(?i)\b(?:password|passwd|api_key|access_token)\s*[:=]\s*['\"][^'\"\r\n]{8,}['\"]"),
}


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)
    elif isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from strings(item)


def scan_text(path):
    chinese = False
    private_path = False
    credentials = set()
    crlf = False
    scanned_bytes = 0

    def inspect(text):
        nonlocal chinese, private_path
        if not chinese and not text.isascii():
            chinese = bool(HAN.search(text))
        if not private_path and "/" in text:
            private_path = bool(PRIVATE_PATH.search(text))
        lower = text.lower()
        possible = {
            "private_key_header": "PRIVATE KEY" in text,
            "github_token": "gh" in text,
            "openai_token": "sk-" in text,
            "aws_access_key": "AKIA" in text or "ASIA" in text,
            "literal_password": any(key in lower for key in ("password", "passwd", "api_key", "access_token")),
        }
        for label, pattern in SECRET_PATTERNS.items():
            if possible[label] and pattern.search(text):
                credentials.add(label)

    opener = gzip.open if path.suffix == ".gz" else open
    decoder = codecs.getincrementaldecoder("utf-8")("strict")
    carry = ""
    with opener(path, "rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            scanned_bytes += len(block)
            crlf = crlf or b"\r\n" in block
            text = carry + decoder.decode(block)
            inspect(text)
            carry = text[-256:]
        inspect(carry + decoder.decode(b"", final=True))
    if path.suffix in {".json", ".ipynb"}:
        payload = json.loads(path.read_text())
        for item in strings(payload):
            inspect(item)
    return {"chinese": chinese, "private_path": private_path,
            "credential_patterns": sorted(credentials), "crlf": crlf,
            "scanned_bytes": scanned_bytes}


def check_git_byte_preservation():
    samples = [
        "FE_Results_Cases_All/FE_Results_Case_0.txt",
        "provenance/source_file_manifest.csv",
        "outputs/19_one_time_final_50case_locked_149/formal_final_50case/frozen_artifact_hash_audit.csv",
        "provenance/original_code/notebooks/Multi-Case_FEM_Parametric_and_Position_Sensitivity_Stability_Analysis_199Cases.ipynb",
    ]
    results = []
    with tempfile.TemporaryDirectory(prefix="graphite-git-byte-check-") as temp:
        workspace = Path(temp)
        subprocess.run(["git", "init", "--quiet", str(workspace)], check=True, capture_output=True)
        shutil.copyfile(ROOT / ".gitattributes", workspace / ".gitattributes")
        for name in samples:
            target = workspace / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
            raw = subprocess.check_output(["git", "-C", temp, "hash-object", "--no-filters", str(target)], text=True).strip()
            modes = {}
            for setting in ["false", "true", "input"]:
                actual = subprocess.check_output(
                    ["git", "-C", temp, "-c", f"core.autocrlf={setting}",
                     "-c", "filter.lfs.process=", "-c", "filter.lfs.clean=",
                     "-c", "filter.lfs.required=false", "hash-object",
                     f"--path={name}", str(target)], text=True).strip()
                modes[setting] = actual == raw
            if not all(modes.values()):
                raise RuntimeError(f"Git newline conversion changes archived bytes: {name}")
            results.append({"file": name, "unchanged_by_autocrlf_mode": modes,
                            "scope": "newline_conversion_only_lfs_filters_disabled"})
    return results


def audit():
    files = [p for p in sorted(ROOT.rglob("*")) if p.is_file()
             and not any(part in SKIP_NAMES for part in p.relative_to(ROOT).parts)]
    # Exclude this generated report from its own text findings and size accounting.
    files = [p for p in files if p != ROOT / "provenance/github_upload_audit.json"]
    with (ROOT / "provenance/delivery_file_inventory.csv").open(newline="") as handle:
        inventory = list(csv.DictReader(handle))
    missing = [r["path"] for r in inventory if not (ROOT / r["path"]).is_file()]
    han_files, active_han, privacy_files, secret_hits, crlf_files = [], [], [], [], []
    scan_errors, binary_skipped = [], []
    scanned_files, scanned_bytes = 0, 0
    by_folder = Counter()
    oversize_browser, oversize_git = [], []
    for index, path in enumerate(files, 1):
        if index % 100 == 1:
            print(f"Scanning {index}/{len(files)} files", flush=True)
        relative = str(path.relative_to(ROOT))
        size = path.stat().st_size
        by_folder[relative.split("/")[0]] += size
        if size > 25 * 1024 ** 2:
            oversize_browser.append({"file": relative, "bytes": size})
        if size > 100 * 1024 ** 2:
            oversize_git.append({"file": relative, "bytes": size})
        gz_text = path.suffix == ".gz" and Path(path.stem).suffix in TEXT_SUFFIXES
        if path.suffix not in TEXT_SUFFIXES and not gz_text and path.name not in {".gitignore", ".gitattributes"}:
            binary_skipped.append(relative)
            continue
        try:
            result = scan_text(path)
        except (ValueError, OSError, UnicodeError) as exc:
            scan_errors.append({"file": relative, "error": type(exc).__name__})
            continue
        scanned_files += 1
        scanned_bytes += result["scanned_bytes"]
        if result["chinese"]:
            han_files.append(relative)
            if relative.split("/")[0] in {"src", "scripts", "cluster", "notebooks", "docs", "formula_appendix"} or relative == "README.md":
                active_han.append(relative)
        if result["private_path"]:
            privacy_files.append(relative)
        if result["credential_patterns"]:
            secret_hits.append({"file": relative, "patterns": result["credential_patterns"]})
        if result["crlf"]:
            crlf_files.append(relative)

    lfs = subprocess.run(["git", "lfs", "version"], capture_output=True, text=True)
    symlinks = [str(p.relative_to(ROOT)) for p in ROOT.rglob("*") if p.is_symlink()]
    attrs = (ROOT / ".gitattributes").read_text()
    return {
        "audit_kind": "publication_readiness_not_new_research",
        "status": "publication_decisions_required",
        "missing_inventory_files": missing, "symlinks": symlinks,
        "file_count_excluding_audit_and_ignored_runtime_files": len(files),
        "total_bytes_excluding_this_audit": sum(by_folder.values()), "bytes_by_folder": dict(by_folder),
        "text_files_scanned_including_decompressed_text": scanned_files, "decoded_bytes_scanned": scanned_bytes,
        "unscanned_binary_files": len(binary_skipped), "text_scan_errors": scan_errors,
        "active_chinese_files": active_han, "all_chinese_files_including_archives": han_files,
        "historical_or_literal_private_path_files": privacy_files,
        "potential_credential_pattern_matches": secret_hits,
        "credential_scan_limit": "Pattern screening of UTF-8 text and decoded JSON, not a comprehensive secret or binary audit; matching values are not recorded.",
        "files_over_browser_25_MiB": oversize_browser, "files_over_git_100_MiB": oversize_git,
        "files_with_CRLF": crlf_files,
        "git_byte_preservation_checks": check_git_byte_preservation(),
        "git_lfs_available_on_this_machine": lfs.returncode == 0,
        "git_lfs_configured_in_attributes": "filter=lfs" in attrs,
        "git_lfs_upload_or_download_tested": False,
        "licence_files": [p.name for p in ROOT.iterdir() if p.name.lower().startswith(("license", "licence", "copying"))],
        "publication_permission_context": "Owner confirmed permission for this complete private upload; public release and downstream licensing are not authorised by that confirmation.",
        "requires_owner_confirmation": ["Authenticated check of target private visibility and write access", "Available Git LFS storage and bandwidth", "Separate permissions before any public release or reuse licence"],
        "new_training_or_final_evaluation": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, help="Optional curation audit, never inside frozen outputs")
    args = parser.parse_args()
    report = audit()
    if args.report:
        output = args.report.resolve()
        if output.is_relative_to(ROOT / "outputs"):
            raise ValueError("Do not write publication audits into research outputs")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=True, indent=2) + "\n")
    summary = {k: report[k] for k in ["status", "missing_inventory_files", "active_chinese_files", "text_scan_errors",
                                     "potential_credential_pattern_matches", "git_lfs_available_on_this_machine",
                                     "git_lfs_configured_in_attributes", "licence_files"]}
    summary.update({"chinese_file_count_including_archives": len(report["all_chinese_files_including_archives"]),
                    "files_over_25_MiB": len(report["files_over_browser_25_MiB"]),
                    "files_over_100_MiB": len(report["files_over_git_100_MiB"]),
                    "CRLF_file_count": len(report["files_with_CRLF"]),
                    "git_byte_preservation_samples": len(report["git_byte_preservation_checks"])})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
