# Architectural Patterns

This document describes recurring patterns and design decisions in the codebase.

## 1. Centralized Configuration

All configuration values are centralized in `AppConfig.py` using nested static classes.

**Usage locations:**
- Application metadata: `AppConfig.py:17-62`
- Task configs: `AppConfig.py:77-111`
- Color schemes: `AppConfig.py:130-214`
- Keyboard shortcuts: `AppConfig.py:219-380`

**Pattern:** Reference `AppConfig.X.Y` instead of hardcoding values.

## 2. Task-Type Polymorphism

Two annotation tasks (ARTERY_VEIN, OPTIC_DISC) share the same UI with branching logic.

**Definition:** `AppConfig.py:77-111`

**Branching locations:**
- Canvas mask loading: `App.py:491-515`
- Paint operations: `App.py:1749-1782`
- History tracking: `App.py:1788-1795`
- Mode shortcuts: `App.py:1317-1319`

**Pattern:** Check `self.current_task` and branch using `AppConfig.Task.CONFIGS[task]`.

## 3. Callback Pattern for Canvas Events

`ImageCanvas` exposes callback hooks instead of direct method calls.

**Definition:** `App.py:469-471`
```python
self.on_paint = None
self.on_stroke_start = None
self.on_stroke_end = None
```

**Connection:** `App.py:1317-1319`

**Pattern:** Decouples canvas rendering from business logic.

## 4. Dual State Management

Separate reference and working copies of mask data.

**MaskEditor state:** `App.py:1021-1029`
- `self.artery_mask`, `self.vein_mask`, `self.disc_mask` (originals)
- `self.working_artery`, `self.working_vein`, `self.working_disc` (editable)

**Pattern:** Working copies modified during editing; originals preserved for comparison.

## 5. Compressed History (Undo/Redo)

`CompressedHistory` class manages state with zlib compression.

**Implementation:** `App.py:918-980`
- Combines masks: `np.stack(masks, axis=0)` at line 936
- Compresses: `zlib.compress()` at line 937
- Max 50 states: `AppConfig.History.MAX_SIZE`

**Pattern:** Memory-efficient history for large image masks.

## 6. Viewport Transformation

Coordinate conversion between screen and image space.

**Scale calculation:** `App.py:584-585`
```python
self.display_scale_x = scaled.width() / self.base_pixmap.width()
self.display_scale_y = scaled.height() / self.base_pixmap.height()
```

**Screen→Image:** `App.py:616-617`
```python
x_img = int(pos.x() / self.display_scale_x)
y_img = int(pos.y() / self.display_scale_y)
```

**Pattern:** All mask operations use image coordinates; display uses scaled coordinates.

## 7. Bit-Packed Mask Encoding

Two masks encoded into single byte for storage efficiency.

**Encoding:** `App.py:169-170`
- Artery in bit 0, vein in bit 1
- Values 0-3 map to: background, artery, vein, overlap

**Decoding:** `App.py:162-166`

**Pattern:** Reduces storage and simplifies history tracking.

## 8. LRU Cache for Brush Masks

Expensive brush shape computations are memoized.

**Implementation:** `App.py:229-245`
```python
@lru_cache(maxsize=256)
def get_brush_mask(size):
```

**Pattern:** Avoids redundant numpy operations during continuous painting.

## 9. Static Method Grouping

Related operations grouped as static methods in `PaintOperations` class.

**Location:** `App.py:289-417`
- `apply_brush()` at line 291
- `modify_artery()` at line 334
- `modify_vein()` at line 377

**Pattern:** Groups domain operations without instance state.

## 10. Settings Persistence (YAML)

User preferences stored in `app_settings.yaml`.

**Default template:** `App.py:1069-1096`
**Load/Save:** `App.py:1098-1129`
**Apply to UI:** `App.py:1131-1165`

**Pattern:** Human-readable configuration separate from `AppConfig` (code constants).

## 11. Resource Path Abstraction

Works in both development and PyInstaller contexts.

**Implementation:** `App.py:54-70`
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

**Implementation:** `App.py:1608-1631`
```python
masks = {os.path.splitext(f)[0]: f for f in mask_files}
imgs = {os.path.splitext(f)[0]: f for f in image_files}
common = sorted(set(masks) & set(imgs))
```

**Pattern:** O(1) lookup for file pairs with different extensions.

## 13. NumPy Dtype Validation

Explicit type checking before image I/O.

**In imwrite:** `App.py:79-97`
- Handles float32/64, bool, and uint8
- Normalizes to uint8 for saving

**In ensure_mask:** `App.py:156-159`

**Pattern:** Normalize all internal masks to uint8.

## 14. Stroke Preview Optimization

Long strokes use step-skipping for performance.

**Implementation:** `App.py:790-810`
- Thresholds at 500/1000 points: `AppConfig.py:469-470`
- Step increases to 2 or 3 for long strokes

**Pattern:** Adaptive rendering maintains responsiveness during continuous drawing.

## 15. Qt Signal-Slot Connections

Standard Qt pattern for UI event handling.

**Connection hub:** `App.py:1317-1335`
- `widget.signal.connect(self._handler)`

**Naming convention:** Handlers prefixed with `_on_` or `on_`.

## 16. Dedicated Annotation Mode (A/V Task)

Focused editing mode that isolates a single vessel type for cleaner annotation.

**Mode constants:** `AppConfig.py:67-72`
```python
class AnnotationMode:
    NORMAL = "normal"   # Combined A/V view with overlap
    ARTERY = "artery"   # Artery-only view (red)
    VEIN = "vein"       # Vein-only view (blue)
```

**Shortcuts:** `AppConfig.py:261-263`
- `Shift+C` → Enter Artery Mode
- `Shift+V` → Enter Vein Mode
- `Shift+X` → Exit to Normal Mode

**State synchronization:**
- `MaskEditor.annotation_mode` and `ImageCanvas.annotation_mode` (`App.py:455`, `App.py:1050`)
- Mode switching: `MaskEditor.set_annotation_mode()` (`App.py:1848-1877`)

**Visualization functions:**
- `mask_to_rgba_artery_only()` at `App.py:201-209`
- `mask_to_rgba_vein_only()` at `App.py:211-218`
- LUT generators: `AppConfig.Color.get_artery_only_lut()`, `get_vein_only_lut()` (`AppConfig.py:191-214`)

**Mode-aware rendering:** `ImageCanvas._build_pixmaps()` (`App.py:527-541`)
```python
if self.annotation_mode == AppConfig.AnnotationMode.ARTERY:
    self.overlay_rgba = mask_to_rgba_artery_only(...)
elif self.annotation_mode == AppConfig.AnnotationMode.VEIN:
    self.overlay_rgba = mask_to_rgba_vein_only(...)
else:  # NORMAL
    self.overlay_rgba = masks_to_rgba_av(...)
```

**Mode button filtering:** `App.py:1879-1887`
- Artery mode: only `draw_artery`, `delete_artery` enabled
- Vein mode: only `draw_vein`, `delete_vein` enabled
- Normal mode: all modes enabled

**Pattern:** State machine with synchronized UI updates across canvas and main window.
