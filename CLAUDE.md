# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A desktop application for annotating retinal fundus images, designed for medical image researchers. Supports two annotation tasks: Artery/Vein classification (distinguishing blood vessels as arteries, veins, or overlap) and Optic Disc segmentation. Built with PySide6 for cross-platform deployment.

## Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3 |
| GUI | PySide6 (Qt for Python) |
| Image Processing | OpenCV, scikit-image, NumPy, PIL |
| Configuration | PyYAML |
| Packaging | PyInstaller |

## Directory Structure

```
app/                # Main application package (modular components)
datasets/           # Sample data for testing (images/, masks_artery_vein/)
docs/               # Architecture and changelog documentation
```

## Commands

| Task | Command |
|------|---------|
| Run application | `python App.py` |
| Package (macOS) | `pyinstaller --onefile --windowed --name "MaskEditor" --icon=icon.icns App.py` |
| Package (Windows) | `pyinstaller --onefile --windowed --name "MaskEditor" --icon=icon.ico App.py` |

## Key Files

| File | Purpose |
|------|---------|
| `App.py` | Entry point (imports from app/) |
| `AppConfig.py` | All configuration constants (colors, shortcuts, settings) |
| `MaskEditorUI.ui` | Qt Designer UI definition |
| `app/main.py` | MaskEditor main window class |
| `app/canvas.py` | ImageCanvas widget (painting, zoom, pan) |
| `app/paint_ops.py` | Brush operations and mask modification |
| `app/rendering.py` | Mask-to-RGBA conversion functions |

## Data Format

- **Masks**: PNG with bit-encoded values (bit0=artery, bit1=vein, both=overlap)
- **Records**: JSON files tracking finalized images per task
- **Settings**: `app_settings.yaml` for user preferences

## Additional Documentation

| Document | When to Consult |
|----------|-----------------|
| `docs/architecture.md` | When modifying core modules, adding features, or understanding data flow |
| `docs/changelog.md` | When preparing releases or reviewing recent changes |
| `AppConfig.py` | When adding shortcuts, colors, or any configurable values |
