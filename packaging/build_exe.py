"""
Build the MonkeyGrab desktop app (.exe) with PyInstaller.

Orchestrates the full freeze: (1) ensure the React UI is built, (2) generate the
window/app icon from the logo, (3) run PyInstaller against ``MonkeyGrab.spec``.
Output: ``dist/MonkeyGrab/MonkeyGrab.exe`` (one-folder bundle; multi-GB).

Usage (from the repository root):
    pip install pyinstaller pywebview          # build-time deps
    python packaging/build_exe.py              # full build
    python packaging/build_exe.py --skip-frontend   # reuse existing dist/

Prerequisites on the *target* machine (not bundled):
    - Ollama installed and the desired models pulled.
    - Microsoft Edge WebView2 runtime (preinstalled on Windows 11).

Dependencies: PyInstaller, pywebview; Node.js only when (re)building the frontend.
"""

# ─────────────────────────────────────────────
# MODULE MAP -- Section index
# ─────────────────────────────────────────────
#
#  +-- 1. Paths and CLI
#  +-- 2. Frontend build              npm run build (skippable)
#  +-- 3. Icon generation            logo.png -> MonkeyGrab.ico (optional)
#  +-- 4. PyInstaller invocation
#  +-- 5. main()
#
# ─────────────────────────────────────────────

import argparse
import os
import shutil
import subprocess
import sys


# ─────────────────────────────────────────────
# SECTION 1: PATHS AND CLI
# ─────────────────────────────────────────────

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
SPEC = os.path.join(HERE, "MonkeyGrab.spec")
FRONTEND = os.path.join(PROJ, "rag", "web", "frontend")
DIST_UI = os.path.join(FRONTEND, "dist")
ICON = os.path.join(HERE, "MonkeyGrab.ico")
LOGO = os.path.join(FRONTEND, "public", "logo.png")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the MonkeyGrab desktop app.")
    p.add_argument("--skip-frontend", action="store_true", help="Reuse the existing React build in dist/.")
    p.add_argument("--dist", default=os.path.join(PROJ, "dist"), help="Output directory for the bundle.")
    return p.parse_args()


# ─────────────────────────────────────────────
# SECTION 2: FRONTEND BUILD
# ─────────────────────────────────────────────

def build_frontend() -> None:
    """Build the React UI into rag/web/frontend/dist via npm."""
    npm = shutil.which("npm")
    if npm is None:
        print("! npm not found; skipping frontend build (using existing dist/ if present).")
        return
    if not os.path.isdir(os.path.join(FRONTEND, "node_modules")):
        print(">> npm install ...")
        subprocess.run([npm, "install"], cwd=FRONTEND, check=True, shell=(os.name == "nt"))
    print(">> npm run build ...")
    subprocess.run([npm, "run", "build"], cwd=FRONTEND, check=True, shell=(os.name == "nt"))


# ─────────────────────────────────────────────
# SECTION 3: ICON GENERATION
# ─────────────────────────────────────────────

def build_icon() -> None:
    """Generate MonkeyGrab.ico from the logo if Pillow is available."""
    if os.path.isfile(ICON):
        return
    try:
        from PIL import Image
    except ImportError:
        print("! Pillow not installed; building without a custom icon.")
        return
    if not os.path.isfile(LOGO):
        print("! logo.png not found; building without a custom icon.")
        return
    Image.open(LOGO).convert("RGBA").save(
        ICON, format="ICO", sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
    )
    print(f">> icon generated: {ICON}")


# ─────────────────────────────────────────────
# SECTION 4: PYINSTALLER INVOCATION
# ─────────────────────────────────────────────

def run_pyinstaller(dist_dir: str) -> None:
    """Freeze the app with PyInstaller using the versioned spec."""
    work_dir = os.path.join(HERE, "build")
    cmd = [
        sys.executable, "-m", "PyInstaller", SPEC,
        "--noconfirm", "--clean",
        "--distpath", dist_dir,
        "--workpath", work_dir,
    ]
    print(">> " + " ".join(cmd))
    subprocess.run(cmd, cwd=PROJ, check=True)
    exe = os.path.join(dist_dir, "MonkeyGrab", "MonkeyGrab.exe")
    print(f"\nDone. Bundle: {exe}" if os.path.isfile(exe) else f"\nBuild finished; check {dist_dir}")


# ─────────────────────────────────────────────
# SECTION 5: ENTRY POINT
# ─────────────────────────────────────────────

def main() -> None:
    args = _parse_args()
    if not args.skip_frontend:
        build_frontend()
    if not os.path.isdir(DIST_UI):
        sys.exit(f"React build missing at {DIST_UI}. Run without --skip-frontend or build it first.")
    build_icon()
    run_pyinstaller(args.dist)


if __name__ == "__main__":
    main()
