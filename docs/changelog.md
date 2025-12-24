# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Refactor: Split monolithic `App.py` into modular `app/` package with separate modules for canvas, painting, rendering, history, and utilities

## [1.0.0] - 2025-12-24

### Added

- Initial release with two annotation tasks: Artery/Vein classification and Optic Disc segmentation
- Dedicated annotation modes for focused A/V editing (`Shift+C` artery, `Shift+V` vein, `Shift+X` exit)
- Logging system with file output to `app.log`
- Stroke preview optimization during painting
- Undo/redo with zlib-compressed history
- Enhanced image view with Gaussian contrast enhancement
- Configurable brush size, opacity, and zoom
- Finalization tracking with JSON records
- User settings persistence via `app_settings.yaml`

### Fixed

- Stroke preview step threshold logic (check >1000 before >500)
- Bug where modify mode incorrectly switched to artery mode after editing

### Changed

- REDO shortcut changed to `Shift+Ctrl+Z`
