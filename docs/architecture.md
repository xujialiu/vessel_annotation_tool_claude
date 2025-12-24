# Architecture Patterns

This document describes patterns used consistently across the codebase.

## 1. Centralized Configuration

**Where**: All modules import from `AppConfig.py`

**Example**: `app/canvas.py`, `app/main.py`, `app/rendering.py`, `app/paint_ops.py`

**Pattern**: All magic numbers, colors, shortcuts, and settings live in `AppConfig` nested static classes. Never hardcode values in application code.

```python
from AppConfig import AppConfig

# Access via nested classes
brush_size = AppConfig.Brush.DEFAULT_SIZE
color = AppConfig.Color.ARTERY
shortcut = AppConfig.Shortcut.SAVE
```

**Rationale**: Single source of truth for configuration. Easy to modify behavior without searching through code.

---

## 2. Callback-based Communication

**Where**: `app/canvas.py` → `app/main.py`

**Example**: `canvas.py:87-90`, `main.py:389-392`

**Pattern**: Canvas widget communicates with main window via callback functions instead of Qt signals.

```python
# In canvas.py
self.on_paint = None
self.on_stroke_start = None
self.on_stroke_end = None

# In main.py
self.canvas.on_paint = self._on_paint
self.canvas.on_stroke_start = self._on_stroke_start
self.canvas.on_stroke_end = self._on_stroke_end
```

**Rationale**: Decouples canvas from main window. Canvas doesn't need to know about MaskEditor internals.

---

## 3. LUT-based Color Mapping

**Where**: `app/rendering.py`

**Example**: `masks_to_rgba_av()`, `mask_to_rgba_od()`, `mask_to_rgba_artery_only()`, `mask_to_rgba_vein_only()`

**Pattern**: Use numpy lookup tables (LUT) for fast mask-to-RGBA conversion instead of per-pixel conditionals.

```python
# Define LUT once
LUT = np.array([
    [0, 0, 0, 0],      # Background
    [255, 0, 0, 255],  # Artery
    [0, 0, 255, 255],  # Vein
    [255, 255, 0, 255] # Overlap
], dtype=np.uint8)

# Apply via indexing (vectorized)
rgba = LUT[combined_mask]
```

**Rationale**: Orders of magnitude faster than loops. Critical for real-time mask overlay updates.

---

## 4. Bit-encoded Mask Storage

**Where**: `app/mask_utils.py`, `app/main.py`, `app/rendering.py`

**Example**: `mask_utils.py:17-31`

**Pattern**: Store artery/vein as separate bits in a single uint8 mask.

```python
# Encoding
combined = (artery > 0) | ((vein > 0) << 1)
# Values: 0=none, 1=artery, 2=vein, 3=overlap

# Decoding
artery = (combined & 1)
vein = ((combined >> 1) & 1)
```

**Rationale**: Compact storage (single PNG), fast bitwise operations, natural overlap representation.

---

## 5. In-Place NumPy Modification

**Where**: `app/paint_ops.py`, `app/main.py`

**Example**: `paint_ops.py:86-90`

**Pattern**: Modify mask arrays in-place using numpy slice assignment and bitwise operations.

```python
# In apply_brush
region = mask[y0:y1, x0:x1]
np.bitwise_or(region, sub_brush, out=region)  # Draw
np.bitwise_and(region, ~sub_brush, out=region)  # Erase
```

**Rationale**: Avoids memory allocation during painting. Essential for smooth brush performance.

---

## 6. Compressed History Stack

**Where**: `app/history.py`, `app/main.py`

**Example**: `history.py:46-67`

**Pattern**: Store undo/redo states as zlib-compressed byte arrays.

```python
combined = np.stack(masks, axis=0)
compressed = zlib.compress(combined.tobytes(), level=1)
self._history.append((compressed, shape, dtype))
```

**Rationale**: Masks are large (millions of pixels) but highly compressible. Compression level 1 gives good ratio with minimal CPU overhead.

---

## 7. Stroke Preview Optimization

**Where**: `app/canvas.py`

**Example**: `canvas.py:400-458`

**Pattern**: During painting, render stroke preview with QPainter overlay instead of rebuilding full mask.

```python
# During mouseMoveEvent - just collect points
self._stroke_preview_points.append((px, py, size))
self.update()  # Triggers paintEvent

# In paintEvent - draw preview circles
painter.drawEllipse(QPointF(x, y), radius, radius)

# On mouseReleaseEvent - rebuild actual overlay
self._build_pixmaps()
```

**Rationale**: QPainter drawing is fast (~1ms). Full mask-to-RGBA conversion is slow (~50ms). Preview maintains responsiveness during strokes.

---

## Data Flow

```
User Input (Mouse)
       ↓
   ImageCanvas (canvas.py)
       ↓ callbacks
   MaskEditor (main.py)
       ↓
   PaintOperations (paint_ops.py)
       ↓ modifies
   Mask Arrays (numpy)
       ↓
   CompressedHistory (history.py)
       ↓ on stroke end
   Rendering (rendering.py)
       ↓
   QPixmap Display
```

## Module Dependencies

```
AppConfig.py ← (all modules)

main.py
  ├── canvas.py
  ├── paint_ops.py
  ├── rendering.py
  ├── history.py
  ├── mask_utils.py
  ├── image_processing.py
  ├── io_utils.py
  └── dialogs.py
```
