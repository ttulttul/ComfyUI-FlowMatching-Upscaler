#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMFY_ROOT="${HOME}/git/ComfyUI"
VENV_PATH="${REPO_ROOT}/.venv"

export COMFY_ROOT
export REPO_ROOT

if [ ! -d "${VENV_PATH}" ]; then
  python -m venv "${VENV_PATH}"
fi

source "${VENV_PATH}/bin/activate"

pip uninstall -y comfy >/dev/null 2>&1 || true

SITE_PACKAGES="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
mkdir -p "${SITE_PACKAGES}"
rm -rf "${SITE_PACKAGES}/comfy.py" \
       "${SITE_PACKAGES}/comfy" \
       "${SITE_PACKAGES}/comfy-"*.dist-info 2>/dev/null || true
# Remove any existing comfy namespace placeholders.
find "${SITE_PACKAGES}" -maxdepth 1 -name "comfy*.pth" -print0 | while IFS= read -r -d '' file; do
  rm -f "$file"
done
echo "${COMFY_ROOT}" > "${SITE_PACKAGES}/comfyui-src.pth"

export PYTHONPATH="${COMFY_ROOT}:${REPO_ROOT}/src:${PYTHONPATH:-}"

python <<'PY'
import importlib.util
import os
import sys

comfy_root = os.environ.get("COMFY_ROOT")
spec = importlib.util.find_spec("comfy")
if spec is None:
    sys.stderr.write("Unable to locate 'comfy' after adding ComfyUI source to PYTHONPATH.\n")
    sys.exit(1)

origin = spec.origin or ""
paths = spec.submodule_search_locations or []
if comfy_root and not any(str(path).startswith(comfy_root) for path in paths or [os.path.dirname(origin)]):
    sys.stderr.write(f"'comfy' resolves to {origin}, expected paths under {comfy_root}.\n")
    sys.exit(1)
PY

python -m pip install --upgrade pip >/dev/null 2>&1
python -m pip install -r requirements.txt >/dev/null 2>&1

if ! python -c "import pytest" >/dev/null 2>&1; then
  python -m pip install pytest >/dev/null 2>&1
fi

python -m pytest "$@"
