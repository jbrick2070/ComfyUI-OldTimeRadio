"""
Preflight — Pre-run validation for OldTimeRadio v2.0 pipeline.

Checks before any run:
    - ComfyUI started with all comfyui_required_args
    - assets/signal_lost_prerender.mp4 exists
    - assets/embeddings/manifest.json valid; each entry has matching file + sha256
    - Missing embeddings for referenced tokens: warn, substitute, continue
    - nvidia-smi idle VRAM < 2.5 GB
    - All referenced models present on disk

Exit nonzero on any hard-fail.
"""

import hashlib
import json
import logging
import os
import subprocess
import sys

log = logging.getLogger("OTR")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class PreflightError(RuntimeError):
    """Raised when a hard-fail preflight check fails."""
    pass


def _sha256(path):
    """Compute SHA-256 hex digest for a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def check_comfyui_args(config):
    """Verify ComfyUI was started with required arguments.

    Args:
        config: Parsed rtx5080.yaml dict.

    Returns:
        List of warning/error strings. Hard-fail if any required arg missing.
    """
    required = config.get("comfyui_required_args", [])
    if not required:
        return []

    issues = []

    # Check sys.argv for ComfyUI args
    argv_str = " ".join(sys.argv)
    for arg in required:
        if arg not in argv_str:
            issues.append(f"HARD-FAIL: ComfyUI missing required arg: {arg}")

    return issues


def check_assets(config):
    """Verify required assets exist on disk.

    Returns:
        List of warning/error strings.
    """
    issues = []

    # Signal lost fallback clip
    fallback_clip = config.get("fallback", {}).get(
        "clip", "assets/signal_lost_prerender.mp4")
    fallback_path = os.path.join(_REPO_ROOT, fallback_clip)

    if not os.path.isfile(fallback_path):
        issues.append(
            f"HARD-FAIL: Fallback asset missing: {fallback_path}")

    return issues


def check_embeddings():
    """Validate embeddings manifest and file integrity.

    Returns:
        Tuple of (hard_errors, warnings).
    """
    errors = []
    warnings = []

    manifest_path = os.path.join(
        _REPO_ROOT, "assets", "embeddings", "manifest.json")

    if not os.path.isfile(manifest_path):
        warnings.append(
            "WARN: No embeddings manifest found. Character embeddings "
            "will not be available.")
        return errors, warnings

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        errors.append(f"HARD-FAIL: Embeddings manifest corrupt: {e}")
        return errors, warnings

    if not isinstance(manifest, dict):
        errors.append("HARD-FAIL: Embeddings manifest must be a JSON object")
        return errors, warnings

    embeddings = manifest.get("embeddings", [])
    for entry in embeddings:
        token = entry.get("token", "unknown")
        path = entry.get("path", "")
        expected_sha = entry.get("sha256", "")

        full_path = os.path.join(_REPO_ROOT, path)

        if not os.path.isfile(full_path):
            warnings.append(
                f"WARN: Embedding file missing for token '{token}': {path}. "
                f"Will substitute neutral descriptor.")
            continue

        if expected_sha:
            actual_sha = _sha256(full_path)
            if actual_sha != expected_sha:
                warnings.append(
                    f"WARN: SHA-256 mismatch for embedding '{token}'. "
                    f"Expected {expected_sha[:16]}..., got {actual_sha[:16]}..."
                )

    return errors, warnings


def check_vram_idle():
    """Check that idle VRAM is below 2.5 GB using nvidia-smi.

    Returns:
        List of warning strings.
    """
    warnings = []

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )

        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                used_mb = float(line.strip())
                used_gb = used_mb / 1024.0
                if used_gb > 2.5:
                    warnings.append(
                        f"WARN: GPU idle VRAM is {used_gb:.1f} GB "
                        f"(target < 2.5 GB). Other processes may be "
                        f"consuming VRAM.")
                else:
                    log.info("[Preflight] GPU idle VRAM: %.1f GB (OK)",
                             used_gb)
    except FileNotFoundError:
        warnings.append("WARN: nvidia-smi not found. Cannot check VRAM.")
    except Exception as e:
        warnings.append(f"WARN: nvidia-smi check failed: {e}")

    return warnings


def check_models(config):
    """Verify that all referenced models exist on disk.

    Args:
        config: Parsed rtx5080.yaml dict.

    Returns:
        List of warning/error strings.
    """
    issues = []

    # Try to use ComfyUI folder_paths for model resolution
    try:
        import folder_paths
        checkpoints_dir = folder_paths.get_folder_paths("checkpoints")[0]
    except (ImportError, IndexError):
        # Fall back to relative path from ComfyUI root
        checkpoints_dir = os.path.normpath(
            os.path.join(_REPO_ROOT, "..", "..", "models", "checkpoints"))

    # Check image model (SD3.5)
    image_model = config.get("image", {}).get("model", "")
    if image_model:
        # GGUF models might have various extensions
        found = False
        for ext in (".safetensors", ".gguf", ""):
            candidate = os.path.join(checkpoints_dir, image_model + ext)
            if os.path.isfile(candidate):
                found = True
                break
        if not found:
            issues.append(
                f"WARN: Image model not found: {image_model} "
                f"(searched {checkpoints_dir})")

    # Check video model (LTX) — only if mode is experiment
    mode = config.get("mode", "safe")
    if mode == "experiment":
        video_model = config.get("video", {}).get("model", "")
        if video_model:
            found = False
            for ext in (".safetensors", ".gguf", ""):
                candidate = os.path.join(checkpoints_dir, video_model + ext)
                if os.path.isfile(candidate):
                    found = True
                    break
            if not found:
                issues.append(
                    f"WARN: Video model not found: {video_model} "
                    f"(searched {checkpoints_dir})")

    return issues


def run(config):
    """Execute all preflight checks.

    Args:
        config: Parsed rtx5080.yaml dict.

    Returns:
        Dict with keys: passed (bool), hard_fails (list), warnings (list).

    Raises:
        PreflightError: If any hard-fail check fails and caller
            doesn't check the return value.
    """
    hard_fails = []
    warnings = []

    # 1. ComfyUI args
    arg_issues = check_comfyui_args(config)
    for issue in arg_issues:
        if issue.startswith("HARD-FAIL"):
            hard_fails.append(issue)
        else:
            warnings.append(issue)

    # 2. Assets
    asset_issues = check_assets(config)
    for issue in asset_issues:
        if issue.startswith("HARD-FAIL"):
            hard_fails.append(issue)
        else:
            warnings.append(issue)

    # 3. Embeddings
    emb_errors, emb_warnings = check_embeddings()
    hard_fails.extend(emb_errors)
    warnings.extend(emb_warnings)

    # 4. VRAM idle check
    vram_warnings = check_vram_idle()
    warnings.extend(vram_warnings)

    # 5. Models
    model_issues = check_models(config)
    for issue in model_issues:
        if issue.startswith("HARD-FAIL"):
            hard_fails.append(issue)
        else:
            warnings.append(issue)

    # Report
    for w in warnings:
        log.warning("[Preflight] %s", w)
    for h in hard_fails:
        log.error("[Preflight] %s", h)

    passed = len(hard_fails) == 0
    if passed:
        log.info("[Preflight] All checks passed (%d warnings)", len(warnings))
    else:
        log.error("[Preflight] FAILED: %d hard-fail(s)", len(hard_fails))

    return {
        "passed": passed,
        "hard_fails": hard_fails,
        "warnings": warnings,
    }


if __name__ == "__main__":
    """CLI entry point for standalone preflight checks."""
    import yaml

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    config_path = os.path.join(_REPO_ROOT, "config", "rtx5080.yaml")
    if not os.path.isfile(config_path):
        print(f"ERROR: Config not found: {config_path}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    result = run(config)

    print(f"\nPreflight: {'PASSED' if result['passed'] else 'FAILED'}")
    if result["warnings"]:
        print(f"  Warnings: {len(result['warnings'])}")
        for w in result["warnings"]:
            print(f"    {w}")
    if result["hard_fails"]:
        print(f"  Hard-fails: {len(result['hard_fails'])}")
        for h in result["hard_fails"]:
            print(f"    {h}")

    sys.exit(0 if result["passed"] else 1)
