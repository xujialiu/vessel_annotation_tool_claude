"""
Image canvas widget for the vessel annotation tool.

Provides the main drawing surface with mouse interaction,
stroke preview, and mask visualization.
"""

import numpy as np
from PySide6.QtWidgets import QLabel, QMenu
from PySide6.QtGui import QPixmap, QPainter, QColor, QPen, QBrush
from PySide6.QtCore import Qt, QPoint, QPointF

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AppConfig import AppConfig

from .rendering import (
    masks_to_rgba_av,
    mask_to_rgba_od,
    mask_to_rgba_artery_only,
    mask_to_rgba_vein_only,
    numpy_rgba_to_qimage,
    numpy_rgb_to_qpixmap,
)


class ImageCanvas(QLabel):
    """
    Custom QLabel for image rendering and mouse event handling.

    Handles:
    - Image and mask display with overlay
    - Mouse events for painting
    - Stroke preview during drawing
    - Zoom and pan
    - Brush cursor display
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(f"background-color: {AppConfig.Color.CANVAS_BACKGROUND};")
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)

        self.task_type = AppConfig.Task.ARTERY_VEIN
        self.display_mode = AppConfig.Display.ORIGINAL

        self.base_image = None
        self.original_image = None
        self.enhanced_image = None
        self.base_pixmap = None
        self.overlay_rgba = None
        self.overlay_pixmap = None
        self._composed_pixmap = None
        self._image_shape = None

        self.scale = AppConfig.Zoom.DEFAULT
        self.display_scale_x = 1.0
        self.display_scale_y = 1.0

        self.artery_mask = None
        self.vein_mask = None
        self.disc_mask = None

        self.alpha = AppConfig.Opacity.DEFAULT
        self.brush_size = AppConfig.Brush.DEFAULT_SIZE
        self.mode = "modify_artery"

        # Annotation mode state (for dedicated A/V editing)
        self.annotation_mode = AppConfig.AnnotationMode.NORMAL
        self._previous_paint_mode = "modify_artery"

        self._drawing = False
        self._dragging = False
        self._last_pos = None
        self._last_drag_pos = None

        # Stroke preview optimization
        self._stroke_preview_points = []
        self._stroke_color = QColor(*AppConfig.Color.STROKE_ARTERY)
        self._is_delete_mode = False

        # Callbacks
        self.on_paint = None
        self.on_stroke_start = None
        self.on_stroke_end = None
        self.on_brush_size_change = None
        self.parent_window = parent

        self.display_mode = AppConfig.Display.DEFAULT

    def set_task_type(self, task_type):
        """Set the current task type."""
        self.task_type = task_type
        config = AppConfig.Task.get(task_type)
        self.mode = config["default_mode"]

    def set_display_mode(self, mode):
        """Set the display mode (original or enhanced)."""
        self.display_mode = mode
        if mode == AppConfig.Display.ORIGINAL:
            self.base_image = self.original_image
        else:
            self.base_image = self.enhanced_image
        self._build_pixmaps()

    def load_image_and_masks_av(self, rgb_np, enhanced_np, artery_np, vein_np):
        """Load image and masks for artery/vein task."""
        self.task_type = AppConfig.Task.ARTERY_VEIN
        self.original_image = rgb_np
        self.enhanced_image = enhanced_np
        self.base_image = (
            rgb_np if self.display_mode == AppConfig.Display.ORIGINAL else enhanced_np
        )
        self.artery_mask = artery_np
        self.vein_mask = vein_np
        self.disc_mask = None
        self._image_shape = artery_np.shape if artery_np is not None else None
        self._stroke_preview_points = []
        self._build_pixmaps()
        self.set_scale(AppConfig.Zoom.DEFAULT)

    def load_image_and_mask_od(self, rgb_np, enhanced_np, disc_np):
        """Load image and mask for optic disc task."""
        self.task_type = AppConfig.Task.OPTIC_DISC
        self.original_image = rgb_np
        self.enhanced_image = enhanced_np
        self.base_image = (
            rgb_np if self.display_mode == AppConfig.Display.ORIGINAL else enhanced_np
        )
        self.disc_mask = disc_np
        self.artery_mask = None
        self.vein_mask = None
        self._image_shape = disc_np.shape if disc_np is not None else None
        self._stroke_preview_points = []
        self._build_pixmaps()
        self.set_scale(AppConfig.Zoom.DEFAULT)

    def _build_pixmaps(self):
        """Build the display pixmaps from current image and masks."""
        if self.base_image is None:
            return

        self.base_pixmap = numpy_rgb_to_qpixmap(self.base_image)

        if self.task_type == AppConfig.Task.ARTERY_VEIN:
            if self.artery_mask is not None and self.vein_mask is not None:
                # Choose visualization based on annotation mode
                if self.annotation_mode == AppConfig.AnnotationMode.ARTERY:
                    self.overlay_rgba = mask_to_rgba_artery_only(
                        self.artery_mask, self.alpha
                    )
                elif self.annotation_mode == AppConfig.AnnotationMode.VEIN:
                    self.overlay_rgba = mask_to_rgba_vein_only(
                        self.vein_mask, self.alpha
                    )
                else:  # NORMAL mode
                    self.overlay_rgba = masks_to_rgba_av(
                        self.artery_mask, self.vein_mask, self.alpha
                    )
                qimg = numpy_rgba_to_qimage(self.overlay_rgba)
                self.overlay_pixmap = QPixmap.fromImage(qimg)
            else:
                self.overlay_rgba = None
                self.overlay_pixmap = None
        else:
            if self.disc_mask is not None:
                self.overlay_rgba = mask_to_rgba_od(self.disc_mask, self.alpha)
                qimg = numpy_rgba_to_qimage(self.overlay_rgba)
                self.overlay_pixmap = QPixmap.fromImage(qimg)
            else:
                self.overlay_rgba = None
                self.overlay_pixmap = None

        self._composed_pixmap = None
        self._update_display_pixmap()

    def _compose_pixmap(self):
        """Compose base image and overlay into single pixmap."""
        if self._composed_pixmap is not None:
            return self._composed_pixmap

        self._composed_pixmap = QPixmap(self.base_pixmap.size())
        self._composed_pixmap.fill(Qt.transparent)
        p = QPainter(self._composed_pixmap)
        p.drawPixmap(0, 0, self.base_pixmap)
        if self.overlay_pixmap:
            p.setCompositionMode(QPainter.CompositionMode_SourceOver)
            p.drawPixmap(0, 0, self.overlay_pixmap)
        p.end()
        return self._composed_pixmap

    def _update_display_pixmap(self):
        """Update the displayed pixmap with current scale."""
        if self.base_pixmap is None:
            return

        composed = self._compose_pixmap()
        target_w = max(1, int(composed.width() * self.scale))
        target_h = max(1, int(composed.height() * self.scale))

        transform = Qt.FastTransformation
        scaled = composed.scaled(target_w, target_h, Qt.KeepAspectRatio, transform)

        self.display_scale_x = scaled.width() / self.base_pixmap.width()
        self.display_scale_y = scaled.height() / self.base_pixmap.height()
        super().setPixmap(scaled)
        self.resize(scaled.size())

    def set_scale(self, s, mouse_pos=None):
        """Set the zoom scale, optionally centering on mouse position."""
        old_scale = self.scale
        self.scale = max(AppConfig.Zoom.MIN, min(s, AppConfig.Zoom.MAX))

        if abs(self.scale - old_scale) < 0.001:
            return

        self._update_display_pixmap()

        if mouse_pos and self.parent_window:
            scroll_area = self.parent_window.ui.scrollArea
            viewport = scroll_area.viewport()
            mouse_view = viewport.mapFromGlobal(
                QPoint(int(mouse_pos.x()), int(mouse_pos.y()))
            )
            hbar, vbar = (
                scroll_area.horizontalScrollBar(),
                scroll_area.verticalScrollBar(),
            )

            img_x = (hbar.value() + mouse_view.x()) / old_scale
            img_y = (vbar.value() + mouse_view.y()) / old_scale
            hbar.setValue(int(img_x * self.scale - mouse_view.x()))
            vbar.setValue(int(img_y * self.scale - mouse_view.y()))

    def _map_event_to_image(self, ev):
        """Map mouse event position to image coordinates."""
        pos = ev.position()
        x_img = int(pos.x() / self.display_scale_x)
        y_img = int(pos.y() / self.display_scale_y)

        if self._image_shape:
            h, w = self._image_shape
            x_img = max(0, min(x_img, w - 1))
            y_img = max(0, min(y_img, h - 1))
        return x_img, y_img

    def _get_stroke_color(self):
        """Get the appropriate color for stroke preview based on current mode."""
        if self.task_type == AppConfig.Task.OPTIC_DISC:
            if "delete" in self.mode:
                return QColor(*AppConfig.Color.STROKE_DELETE)
            return QColor(*AppConfig.Color.STROKE_DISC)
        else:
            # In dedicated annotation mode, always use the mode color
            if self.annotation_mode == AppConfig.AnnotationMode.ARTERY:
                if "delete" in self.mode:
                    return QColor(*AppConfig.Color.STROKE_DELETE)
                return QColor(*AppConfig.Color.STROKE_ARTERY)
            elif self.annotation_mode == AppConfig.AnnotationMode.VEIN:
                if "delete" in self.mode:
                    return QColor(*AppConfig.Color.STROKE_DELETE)
                return QColor(*AppConfig.Color.STROKE_VEIN)
            else:
                # Normal mode - use existing logic
                if "delete" in self.mode:
                    return QColor(*AppConfig.Color.STROKE_DELETE)
                elif "artery" in self.mode:
                    return QColor(*AppConfig.Color.STROKE_ARTERY)
                elif "vein" in self.mode:
                    return QColor(*AppConfig.Color.STROKE_VEIN)
                else:
                    return QColor(*AppConfig.Color.STROKE_DEFAULT)

    def mousePressEvent(self, ev):
        """Handle mouse press events."""
        if ev.button() != Qt.LeftButton or not self.base_pixmap:
            return

        if ev.modifiers() & Qt.ControlModifier:
            self._dragging = True
            self._last_drag_pos = ev.globalPosition().toPoint()
            self.setCursor(Qt.ClosedHandCursor)
        else:
            self._drawing = True
            x, y = self._map_event_to_image(ev)
            self._last_pos = (x, y)

            # Setup stroke preview
            self._stroke_preview_points = [(x, y, self.brush_size)]
            self._stroke_color = self._get_stroke_color()
            self._is_delete_mode = "delete" in self.mode

            if self.on_stroke_start:
                self.on_stroke_start()

            # Apply paint to mask (but don't update overlay yet)
            if self.on_paint:
                self.on_paint(x, y, self.brush_size, self.mode)

            # Just trigger repaint for preview
            self.update()

    def mouseMoveEvent(self, ev):
        """Handle mouse move events."""
        if not self.base_pixmap:
            return

        if self._dragging:
            current_pos = ev.globalPosition().toPoint()
            delta = current_pos - self._last_drag_pos
            scroll_area = self.parent_window.ui.scrollArea
            scroll_area.horizontalScrollBar().setValue(
                scroll_area.horizontalScrollBar().value() - delta.x()
            )
            scroll_area.verticalScrollBar().setValue(
                scroll_area.verticalScrollBar().value() - delta.y()
            )
            self._last_drag_pos = current_pos

        elif self._drawing and self._last_pos:
            x, y = self._map_event_to_image(ev)
            lx, ly = self._last_pos
            size = self.brush_size

            # Interpolate line points
            points = self._interpolate_line(lx, ly, x, y, size)

            for px, py in points:
                # Apply paint to mask (but don't update overlay yet)
                if self.on_paint:
                    self.on_paint(px, py, size, self.mode)
                # Add to stroke preview
                self._stroke_preview_points.append((px, py, size))

            self._last_pos = (x, y)

            # Just trigger repaint for preview - this is very fast!
            self.update()
        else:
            # Not drawing, just update for cursor display
            self.update()

    def _interpolate_line(self, x1, y1, x2, y2, size):
        """Interpolate points between two positions for smooth strokes."""
        dx, dy = abs(x2 - x1), abs(y2 - y1)
        dist = max(dx, dy)

        if dist == 0:
            return [(x2, y2)]

        # Use smaller step for smoother lines
        step = max(1, size // 4)
        n_points = max(2, dist // step + 1)

        if n_points <= 2:
            return [(x1, y1), (x2, y2)]

        t = np.linspace(0, 1, n_points)
        xs = (x1 + t * (x2 - x1)).astype(np.int32)
        ys = (y1 + t * (y2 - y1)).astype(np.int32)

        # Remove duplicates
        mask = np.ones(len(xs), dtype=bool)
        mask[1:] = (xs[1:] != xs[:-1]) | (ys[1:] != ys[:-1])

        return list(zip(xs[mask], ys[mask]))

    def mouseReleaseEvent(self, ev):
        """Handle mouse release events."""
        if ev.button() == Qt.LeftButton:
            if self._dragging:
                self._dragging = False
                self._last_drag_pos = None
                self.setCursor(Qt.ArrowCursor)
            elif self._drawing:
                self._drawing = False
                self._last_pos = None

                # Clear stroke preview
                self._stroke_preview_points = []

                # Now rebuild the overlay with all changes
                if self.on_stroke_end:
                    self.on_stroke_end()

                self._build_pixmaps()

    def paintEvent(self, ev):
        """Handle paint events - draw stroke preview and cursor."""
        super().paintEvent(ev)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, False)

        # Draw stroke preview if drawing
        if self._drawing and self._stroke_preview_points:
            self._draw_stroke_preview(painter)

        # Draw cursor
        if self.underMouse():
            self._draw_cursor(painter)

        painter.end()

    def _draw_stroke_preview(self, painter):
        """Draw the stroke preview layer - this is very fast with QPainter."""
        if not self._stroke_preview_points:
            return

        # Set up pen and brush
        painter.setPen(Qt.NoPen)
        brush_color = QColor(self._stroke_color)
        brush_color.setAlpha(180)
        painter.setBrush(QBrush(brush_color))

        # Draw all stroke points as circles
        scale_x = self.display_scale_x
        scale_y = self.display_scale_y

        # For very long strokes, draw every Nth point
        points = self._stroke_preview_points
        step = 1
        if len(points) > AppConfig.UI.STROKE_POINTS_THRESHOLD_STEP_3:
            step = 3
        elif len(points) > AppConfig.UI.STROKE_POINTS_THRESHOLD_STEP_2:
            step = 2

        for i in range(0, len(points), step):
            x, y, size = points[i]

            half = (size - 1) // 2
            center_offset = size / 2.0 - half

            display_x = (x + center_offset) * scale_x
            display_y = (y + center_offset) * scale_y
            display_radius = (size / 2.0) * max(scale_x, scale_y)
            display_radius = max(display_radius, AppConfig.UI.MIN_DISPLAY_RADIUS)

            painter.drawEllipse(
                QPointF(display_x, display_y), display_radius, display_radius
            )

        # Always draw the last point
        if len(points) > 0 and step > 1:
            x, y, size = points[-1]
            half = (size - 1) // 2
            center_offset = size / 2.0 - half
            display_x = (x + center_offset) * scale_x
            display_y = (y + center_offset) * scale_y
            display_radius = (size / 2.0) * max(scale_x, scale_y)
            display_radius = max(display_radius, AppConfig.UI.MIN_DISPLAY_RADIUS)
            painter.drawEllipse(
                QPointF(display_x, display_y), display_radius, display_radius
            )

    def _draw_cursor(self, painter):
        """Draw the brush cursor."""
        if self.task_type == AppConfig.Task.OPTIC_DISC:
            c = QColor(*AppConfig.Color.CURSOR_DISC)
        elif "artery" in self.mode:
            c = QColor(*AppConfig.Color.CURSOR_ARTERY)
        elif "vein" in self.mode:
            c = QColor(*AppConfig.Color.CURSOR_VEIN)
        else:
            c = QColor(*AppConfig.Color.CURSOR_DEFAULT)

        pen = QPen(c, AppConfig.Brush.CURSOR_PEN_WIDTH)
        painter.setPen(pen)

        brush_color = QColor(c)
        brush_color.setAlpha(AppConfig.Color.CURSOR_FILL_ALPHA)
        painter.setBrush(brush_color)

        p = self.mapFromGlobal(self.cursor().pos())
        if 0 <= p.x() <= self.width() and 0 <= p.y() <= self.height():
            display_radius = (
                self.brush_size / 2.0 * max(self.display_scale_x, self.display_scale_y)
            )
            display_radius = max(display_radius, 1)
            painter.drawEllipse(p, int(display_radius), int(display_radius))

    def wheelEvent(self, ev):
        """Handle mouse wheel events for zoom and brush size."""
        if not self.base_pixmap:
            return

        delta = ev.angleDelta().y()
        if delta == 0:
            delta = ev.angleDelta().x()

        if ev.modifiers() & Qt.ControlModifier:
            # Ctrl + Scroll: Zoom
            factor = AppConfig.Zoom.FACTOR if delta > 0 else 1 / AppConfig.Zoom.FACTOR
            self.set_scale(self.scale * factor, ev.globalPosition())

        elif ev.modifiers() & Qt.ShiftModifier:
            # Shift + Scroll: Scroll viewport
            scroll_area = self.parent_window.ui.scrollArea
            bar = scroll_area.verticalScrollBar()
            bar.setValue(bar.value() - delta)

        elif ev.modifiers() & Qt.AltModifier:
            # Alt + Scroll: Horizontal scroll
            scroll_area = self.parent_window.ui.scrollArea
            bar = scroll_area.horizontalScrollBar()
            bar.setValue(bar.value() - delta)
        else:
            # Normal scroll (no modifier): Change brush size
            change = (
                AppConfig.Brush.SCROLL_STEP
                if delta > 0
                else -AppConfig.Brush.SCROLL_STEP
            )
            new_size = max(
                AppConfig.Brush.MIN_SIZE,
                min(AppConfig.Brush.MAX_SIZE, self.brush_size + change),
            )
            if new_size != self.brush_size:
                self.brush_size = new_size
                if self.on_brush_size_change:
                    self.on_brush_size_change(new_size)
                self.update()

        ev.accept()

    def show_context_menu(self, pos):
        """Show the context menu with brush modes."""
        menu = QMenu(self)
        config = AppConfig.Task.get(self.task_type)

        for mode_id, label in config["modes"]:
            # In dedicated annotation mode, only show relevant modes
            if self.annotation_mode == AppConfig.AnnotationMode.ARTERY:
                if mode_id not in ("draw_artery", "delete_artery"):
                    continue
            elif self.annotation_mode == AppConfig.AnnotationMode.VEIN:
                if mode_id not in ("draw_vein", "delete_vein"):
                    continue

            act = menu.addAction(label)
            act.triggered.connect(lambda _, m=mode_id: self.parent_window.set_mode(m))

        menu.exec(self.mapToGlobal(pos))
