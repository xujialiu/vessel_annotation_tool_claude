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
- `MaskEditor` - Main window class (`App.py:983`)
- `ImageCanvas` - Custom QLabel for image rendering and mouse events (`App.py:421`)
- `PaintOperations` - Static methods for brush operations (`App.py:289`)
- `CompressedHistory` - Undo/redo with zlib compression (`App.py:918`)

### AppConfig.py
- Nested static classes for all configuration values
- `AppConfig.Task.CONFIGS` - Task-specific settings (`AppConfig.py:83`)
- `AppConfig.Color` - Color definitions (`AppConfig.py:130`)
- `AppConfig.Shortcut` - Keyboard bindings (`AppConfig.py:219`)

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

1. **Image Loading:** File selection and loading (`App.py:1649-1728`)
2. **Painting:** Mouse events → coordinate transform → mask modification (`App.py:652-680`)
3. **Saving:** Encode masks → write PNG → update finalization JSON (`App.py:1953`)

## Dedicated Annotation Mode (A/V Task)

For the Artery/Vein task, dedicated annotation modes allow focused editing of a single vessel type:

| Shortcut | Action |
|----------|--------|
| `Shift+C` | Enter Artery Mode - shows only artery mask (red), allows draw/delete artery |
| `Shift+V` | Enter Vein Mode - shows only vein mask (blue), allows draw/delete vein |
| `Shift+X` | Exit to Normal Mode - shows combined A/V mask with overlap |

**Key implementation locations:**
- `AppConfig.AnnotationMode` - Mode constants (`AppConfig.py:67-72`)
- `MaskEditor.set_annotation_mode()` - Mode switching logic (`App.py:1848-1877`)
- `mask_to_rgba_artery_only()` / `mask_to_rgba_vein_only()` - Single-mask visualization (`App.py:201-218`)

## Additional Documentation

When working on specific topics, check these files:

| Topic | File |
|-------|------|
| Architecture & Patterns | `.claude/docs/architectural_patterns.md` |
| Annotation Mode Pattern | `.claude/docs/architectural_patterns.md` (Section 16) |

## Quick Reference

- Entry point: `App.py:main()` at line 2018
- Configuration hub: `AppConfig.py` (all magic numbers defined here)
- UI layout: `MaskEditorUI.ui` (Qt Designer file)
- Image enhancement: `enhance_fundus_image()` at `App.py:122`
