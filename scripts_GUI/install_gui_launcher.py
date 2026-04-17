#!/usr/bin/env python3
"""Install a desktop launcher for the PyHaemo compact GUI on Linux.

Usage
-----
python3 scripts/install_gui_launcher.py

This creates:
  ~/.local/share/applications/pyhaemo-mesh-generator.desktop

The launcher points to:
  <repo>/partery/gui.py
and uses icon:
  <repo>/assets/logo/pictogram.png (fallbacks included)
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
  # scripts/install_gui_launcher.py -> repo root is parent of scripts/
  return Path(__file__).resolve().parent.parent


def _resolve_icon(repo_root: Path) -> Path | None:
  candidates = (
    repo_root / "assets" / "logo" / "pictogram_2x.png",
    repo_root / "assets" / "pictogram_2x.png",
    repo_root / "assets" / "logo" / "pictogram.png",
    repo_root / "assets" / "logo" / "logotipo.png",
    repo_root / "assets" / "pictogram.png",
    repo_root / "assets" / "logotipo.png",
  )
  for p in candidates:
    if p.is_file():
      return p
  return None


def _desktop_text(
  desktop_id: str,
  app_name: str,
  exec_cmd: str,
  workdir: Path,
  icon_path: Path | None,
) -> str:
  icon_line = str(icon_path) if icon_path is not None else "applications-science"
  return (
    "[Desktop Entry]\n"
    "Type=Application\n"
    "Version=1.0\n"
    f"Name={app_name}\n"
    "Comment=Interactive PyHaemo mesh GUI\n"
    f"Exec={exec_cmd}\n"
    f"Path={workdir}\n"
    f"Icon={icon_line}\n"
    "Terminal=false\n"
    "Categories=Science;Engineering;\n"
    "StartupNotify=true\n"
    f"StartupWMClass={desktop_id}\n"
  )


def _python_has_pyqt6(python_exe: Path) -> bool:
  try:
    proc = subprocess.run(
      [str(python_exe), "-c", "import PyQt6"],
      stdout=subprocess.DEVNULL,
      stderr=subprocess.DEVNULL,
      check=False,
    )
    return proc.returncode == 0
  except Exception:
    return False


def _pick_python(user_python: str | None) -> Path:
  candidates: list[Path] = []
  if user_python:
    candidates.append(Path(user_python))
  candidates.append(Path(sys.executable))

  env_python = os.environ.get("PYHAEMO_GUI_PYTHON")
  if env_python:
    candidates.append(Path(env_python))

  conda_prefix = os.environ.get("CONDA_PREFIX")
  if conda_prefix:
    candidates.append(Path(conda_prefix) / "bin" / "python")

  home = Path.home()
  # Common env names for this project.
  candidates.extend(
    [
      home / "miniconda3" / "envs" / "pyhaemo" / "bin" / "python",
      home / "anaconda3" / "envs" / "pyhaemo" / "bin" / "python",
    ]
  )

  for exe in candidates:
    if exe.is_file() and os.access(exe, os.X_OK) and _python_has_pyqt6(exe):
      return exe

  # Fallback: first existing executable candidate.
  for exe in candidates:
    if exe.is_file() and os.access(exe, os.X_OK):
      return exe
  return Path(sys.executable)


def main() -> int:
  parser = argparse.ArgumentParser(description="Install PyHaemo GUI desktop launcher.")
  parser.add_argument(
    "--desktop-id",
    default="pyhaemo-mesh-generator",
    help="Desktop ID and WM class hint (default: pyhaemo-mesh-generator).",
  )
  parser.add_argument(
    "--app-name",
    default="PyHaemo Mesh Generator",
    help="Display name in app launcher.",
  )
  parser.add_argument(
    "--python",
    default=None,
    help="Python interpreter to use in Exec line. If omitted, auto-detect one with PyQt6.",
  )
  args = parser.parse_args()

  repo_root = _repo_root()
  gui_script = repo_root / "partery" / "gui.py"
  if not gui_script.is_file():
    raise FileNotFoundError(f"Cannot find GUI script: {gui_script}")

  desktop_dir = Path.home() / ".local" / "share" / "applications"
  desktop_dir.mkdir(parents=True, exist_ok=True)
  desktop_file = desktop_dir / f"{args.desktop_id}.desktop"

  icon_path = _resolve_icon(repo_root)
  pyexe = _pick_python(args.python)
  exec_cmd = f"{shlex.quote(str(pyexe))} {shlex.quote(str(gui_script))}"
  desktop_content = _desktop_text(
    desktop_id=args.desktop_id,
    app_name=args.app_name,
    exec_cmd=exec_cmd,
    workdir=repo_root,
    icon_path=icon_path,
  )

  desktop_file.write_text(desktop_content, encoding="utf-8")
  os.chmod(desktop_file, 0o644)

  update_db = shutil.which("update-desktop-database")
  if update_db:
    subprocess.run([update_db, str(desktop_dir)], check=False)

  print(f"Installed launcher: {desktop_file}")
  print(f"Desktop ID / WM class: {args.desktop_id}")
  if icon_path is not None:
    print(f"Icon: {icon_path}")
  else:
    print("Icon: fallback system icon (no project PNG found)")
  if _python_has_pyqt6(pyexe):
    print(f"Python: {pyexe} (PyQt6 OK)")
  else:
    print(f"Python: {pyexe} (WARNING: PyQt6 not found; install it in this environment)")
  print("Open app menu and launch 'PyHaemo Mesh Generator'.")
  print("Then pin it to the dock to keep the correct icon.")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
