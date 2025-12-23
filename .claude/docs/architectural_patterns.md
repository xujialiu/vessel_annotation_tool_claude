# Architectural Patterns

This document describes recurring patterns and design decisions in the codebase.

## 1. Centralized Configuration

All configuration values are centralized in `AppConfig.py` using nested static classes.

**Usage locations:**
- Application metadata: `AppConfig.py:17-62`
- Task configs: `AppConfig.py:73-101`
- Color schemes: `AppConfig.py:120-178`
- Keyboard shortcuts: `AppConfig.py:183-336`

**Pattern:** Reference `AppConfig.X.Y` instead of hardcoding values.

## 2. Task-Type Polymorphism

Two annotation tasks (ARTERY_VEIN, OPTIC_DISC) share the same UI with branching logic.

**Definition:** `AppConfig.py:70-101`

**Branching locations:**
- Canvas mask loading: `App.py:495-512`
- Paint operations: `App.py:1667-1699`
- History tracking: `App.py:1706-1710`
- Mode shortcuts: `App.py:1309-1323`

**Pattern:** Check `self.current_task` and branch using `AppConfig.Task.CONFIGS[task]`.

## 3. Callback Pattern for Canvas Events

`ImageCanvas` exposes callback hooks instead of direct method calls.

**Definition:** `App.py:437-440`
```python
self.on_paint = None
self.on_stroke_start = None
self.on_stroke_end = None
```

**Connection:** `App.py:1251-1256`

**Pattern:** Decouples canvas rendering from business logic.

## 4. Dual State Management

Separate reference and working copies of mask data.

**MaskEditor state:** `App.py:960-968`
- `self.artery_mask`, `self.vein_mask`, `self.disc_mask` (originals)
- `self.working_artery`, `self.working_vein`, `self.working_disc` (editable)

**Pattern:** Working copies modified during editing; originals preserved for comparison.

## 5. Compressed History (Undo/Redo)

`CompressedHistory` class manages state with zlib compression.

**Implementation:** `App.py:857-918`
- Combines masks: `np.stack(masks, axis=0)` at line 875
- Compresses: `zlib.compress()` at line 876
- Max 50 states: `AppConfig.History.MAX_UNDO_STEPS`

**Pattern:** Memory-efficient history for large image masks.

## 6. Viewport Transformation

Coordinate conversion between screen and image space.

**Scale calculation:** `App.py:542-545`
```python
self.display_scale_x = scaled.width() / self.base_pixmap.width()
self.display_scale_y = scaled.height() / self.base_pixmap.height()
```

**Screen→Image:** `App.py:572-581`
```python
x_img = int(pos.x() / self.display_scale_x)
y_img = int(pos.y() / self.display_scale_y)
```

**Pattern:** All mask operations use image coordinates; display uses scaled coordinates.

## 7. Bit-Packed Mask Encoding

Two masks encoded into single byte for storage efficiency.

**Encoding:** `App.py:162-166`
- Artery in bit 0, vein in bit 1
- Values 0-3 map to: background, artery, vein, overlap

**Decoding:** `App.py:169-170`

**Pattern:** Reduces storage and simplifies history tracking.

## 8. LRU Cache for Brush Masks

Expensive brush shape computations are memoized.

**Implementation:** `App.py:201-217`
```python
@lru_cache(maxsize=256)
def get_brush_mask(size):
```

**Pattern:** Avoids redundant numpy operations during continuous painting.

## 9. Static Method Grouping

Related operations grouped as static methods in `PaintOperations` class.

**Location:** `App.py:261-389`
- `apply_brush()` at line 263
- `modify_artery()` at line 306
- `modify_vein()` at line 349

**Pattern:** Groups domain operations without instance state.

## 10. Settings Persistence (YAML)

User preferences stored in `app_settings.yaml`.

**Default template:** `App.py:1005-1032`
**Load/Save:** `App.py:1036-1065`
**Apply to UI:** `App.py:1067-1101`

**Pattern:** Human-readable configuration separate from `AppConfig` (code constants).

## 11. Resource Path Abstraction

Works in both development and PyInstaller contexts.

**Implementation:** `App.py:54-68`
```python
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS  # PyInstaller
    except AttributeError:
        base_path = os.path.abspath(".")  # Development
```

**Pattern:** Use `resource_path()` for bundled assets (icons, etc.).

## 12. File Matching by Basename

Pairs mask and image files using dictionary intersection.

**Implementation:** `App.py:1523-1553`
```python
masks = {os.path.splitext(f)[0]: f for f in mask_files}
imgs = {os.path.splitext(f)[0]: f for f in image_files}
common = sorted(set(masks) & set(imgs))
```

**Pattern:** O(1) lookup for file pairs with different extensions.

## 13. NumPy Dtype Validation

Explicit type checking before image I/O.

**In imwrite:** `App.py:79-87`
- Handles float32/64, bool, and uint8
- Normalizes to uint8 for saving

**In ensure_mask:** `App.py:156-159`

**Pattern:** Normalize all internal masks to uint8.

## 14. Stroke Preview Optimization

Long strokes use step-skipping for performance.

**Implementation:** `App.py:742-774`
- Thresholds at 500/1000 points: `AppConfig.py:420-421`
- Step increases to 2 or 3 for long strokes

**Pattern:** Adaptive rendering maintains responsiveness during continuous drawing.

## 15. Qt Signal-Slot Connections

Standard Qt pattern for UI event handling.

**Connection hub:** `App.py:1251-1271`
- `widget.signal.connect(self._handler)`

**Naming convention:** Handlers prefixed with `_on_` or `on_`.
