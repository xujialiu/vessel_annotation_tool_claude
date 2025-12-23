# Vessel Annotation Tool

## Project Overview

A medical image annotation desktop application for retinal fundus images. Supports two annotation tasks:
- **Artery/Vein Classification** - Distinguish blood vessels as arteries (red), veins (blue), or overlapping (yellow)
- **Optic Disc Segmentation** - Annotate optic disc regions (green)

## Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3 |
| GUI Framework | PySide6 (Qt for Python) |
| Image Processing | OpenCV, scikit-image, NumPy, PIL |
| Configuration | PyYAML |
| Compression | zlib (for undo/redo history) |
| Packaging | PyInstaller |

## Project Structure

```
├── App.py              # Main application (entry point, UI logic, painting operations)
├── AppConfig.py        # Centralized configuration (colors, shortcuts, settings)
├── MaskEditorUI.py     # Generated Qt UI class (from .ui file)
├── MaskEditorUI.ui     # Qt Designer UI definition
├── enhance_contrast.py # Standalone image enhancement utilities
├── debugger.py         # Development helper utilities
├── packing_script.sh   # PyInstaller packaging script
└── datasets/           # Sample data (images/, masks_artery_vein/)
```

## Key Components

### App.py
- `MaskEditor` - Main window class (`App.py:921`)
- `ImageCanvas` - Custom QLabel for image rendering and mouse events (`App.py:394`)
- `PaintOperations` - Static methods for brush operations (`App.py:261`)
- `CompressedHistory` - Undo/redo with zlib compression (`App.py:857`)

### AppConfig.py
- Nested static classes for all configuration values
- `AppConfig.Task.CONFIGS` - Task-specific settings (`AppConfig.py:73`)
- `AppConfig.Color` - Color definitions (`AppConfig.py:120`)
- `AppConfig.Shortcut` - Keyboard bindings (`AppConfig.py:183`)

## Build & Run Commands

```bash
# Run application
python App.py

# Package for distribution (macOS example)
pyinstaller --onefile --windowed --name "MaskEditor" --icon=icon.icns App.py
```

## Data Format

- **Masks:** PNG images with bit-encoded values (artery=bit0, vein=bit1)
- **Records:** JSON files tracking finalized images per task
- **Settings:** `app_settings.yaml` stores user preferences

## Key Workflows

1. **Image Loading:** File matching by basename (`App.py:1523-1553`)
2. **Painting:** Mouse events → coordinate transform → mask modification (`App.py:572-704`)
3. **Saving:** Encode masks → write PNG → update finalization JSON (`App.py:1799-1847`)

## Additional Documentation

When working on specific topics, check these files:

| Topic | File |
|-------|------|
| Architecture & Patterns | `.claude/docs/architectural_patterns.md` |

## Quick Reference

- Entry point: `App.py:main()` at line 1886
- Configuration hub: `AppConfig.py` (all magic numbers defined here)
- UI layout: `MaskEditorUI.ui` (Qt Designer file)
- Image enhancement: `enhance_fundus_image()` at `App.py:122`
