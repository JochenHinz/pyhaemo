#!/usr/bin/env python3
"""Install a macOS app launcher for the PyHaemo compact GUI.

Creates:
  ~/Applications/PyHaemo Mesh Generator.app

This launcher runs:
  <repo>/partery/gui.py
using the selected Python interpreter.
"""

from __future__ import annotations

import argparse
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
  return Path(__file__).resolve().parent.parent


def _resolve_icon_png(repo_root: Path) -> Path | None:
  candidates = (
    repo_root / "assets" / "pictogram.png",
  )
  for p in candidates:
    if p.is_file():
      return p
  return None


def _make_icns_from_png(png_path: Path, out_icns: Path) -> bool:
  sips = shutil.which("sips")
  iconutil = shutil.which("iconutil")
  if not sips or not iconutil:
    return False
  iconset = out_icns.with_suffix(".iconset")
  if iconset.exists():
    shutil.rmtree(iconset)
  iconset.mkdir(parents=True, exist_ok=True)

  sizes = (16, 32, 128, 256, 512)
  for size in sizes:
    png1 = iconset / f"icon_{size}x{size}.png"
    png2 = iconset / f"icon_{size}x{size}@2x.png"
    subprocess.run([sips, "-z", str(size), str(size), str(png_path), "--out", str(png1)], check=False)
    subprocess.run([sips, "-z", str(size * 2), str(size * 2), str(png_path), "--out", str(png2)], check=False)

  res = subprocess.run([iconutil, "-c", "icns", str(iconset), "-o", str(out_icns)], check=False)
  shutil.rmtree(iconset, ignore_errors=True)
  return res.returncode == 0 and out_icns.is_file()


def _plist_text(app_name: str, bundle_id: str, icon_basename: str | None) -> str:
  icon_key = f"\n  <key>CFBundleIconFile</key>\n  <string>{icon_basename}</string>" if icon_basename else ""
  return (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" '
    '"http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
    "<plist version=\"1.0\">\n"
    "<dict>\n"
    "  <key>CFBundleDevelopmentRegion</key>\n"
    "  <string>en</string>\n"
    "  <key>CFBundleExecutable</key>\n"
    "  <string>run</string>\n"
    f"  <key>CFBundleIdentifier</key>\n  <string>{bundle_id}</string>\n"
    f"  <key>CFBundleName</key>\n  <string>{app_name}</string>\n"
    f"  <key>CFBundleDisplayName</key>\n  <string>{app_name}</string>\n"
    "  <key>CFBundlePackageType</key>\n"
    "  <string>APPL</string>\n"
    "  <key>CFBundleVersion</key>\n"
    "  <string>1.0</string>\n"
    "  <key>CFBundleShortVersionString</key>\n"
    "  <string>1.0</string>"
    f"{icon_key}\n"
    "  <key>LSBackgroundOnly</key>\n"
    "  <false/>\n"
    "</dict>\n"
    "</plist>\n"
  )


def main() -> int:
  parser = argparse.ArgumentParser(description="Install macOS launcher app for PyHaemo GUI.")
  parser.add_argument("--app-name", default="PyHaemo")
  parser.add_argument("--bundle-id", default="org.pyhaemo.meshgenerator")
  parser.add_argument("--python", default=sys.executable, help="Python interpreter for launcher")
  args = parser.parse_args()

  if sys.platform != "darwin":
    print("This installer is for macOS only.")
    return 1

  repo_root = _repo_root()
  gui_script = repo_root / "partery" / "gui.py"
  if not gui_script.is_file():
    raise FileNotFoundError(f"Cannot find GUI script: {gui_script}")

  app_dir = Path.home() / "Applications" / f"{args.app_name}.app"
  contents = app_dir / "Contents"
  macos_dir = contents / "MacOS"
  resources_dir = contents / "Resources"
  macos_dir.mkdir(parents=True, exist_ok=True)
  resources_dir.mkdir(parents=True, exist_ok=True)

  run_script = macos_dir / "run"
  run_script.write_text(
    "#!/usr/bin/env bash\n"
    "set -euo pipefail\n"
    f"cd '{repo_root}'\n"
    f"exec '{Path(args.python)}' '{gui_script}'\n",
    encoding="utf-8",
  )
  mode = run_script.stat().st_mode
  run_script.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

  icon_png = _resolve_icon_png(repo_root)
  icon_basename = None
  if icon_png is not None:
    icns_path = resources_dir / "appicon.icns"
    if _make_icns_from_png(icon_png, icns_path):
      icon_basename = "appicon.icns"

  plist = contents / "Info.plist"
  plist.write_text(_plist_text(args.app_name, args.bundle_id, icon_basename), encoding="utf-8")

  # Refresh LaunchServices registration.
  lsreg = "/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister"
  if Path(lsreg).is_file():
    subprocess.run([lsreg, "-f", str(app_dir)], check=False)

  print(f"Installed macOS app: {app_dir}")
  if icon_basename:
    print("Icon: installed via .icns")
  else:
    print("Icon: .icns conversion unavailable (install Xcode command line tools for iconutil/sips).")
  print("Launch from Applications and pin to Dock.")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
