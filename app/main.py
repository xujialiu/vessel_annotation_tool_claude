"""
Main window and application entry point for the vessel annotation tool.

Provides the MaskEditor main window class and main() entry point.
"""

import sys
import os
import json
import numpy as np
from PIL import Image
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QFileDialog,
    QListWidgetItem,
    QMessageBox,
)
from PySide6.QtGui import (
    QShortcut,
    QPixmap,
    QKeySequence,
    QAction,
    QIcon,
)

import yaml
import logging

# Setup path for imports from parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AppConfig import AppConfig
from MaskEditorUI import Ui_MaskEditorUI

# Import from app modules
from .io_utils import setup_logging, resource_path, imread, imwrite
from .mask_utils import ensure_mask, decode_combined_mask, encode_to_combined_mask
from .image_processing import enhance_fundus_image
from .rendering import (
    masks_to_rgba_av,
    mask_to_rgba_od,
    mask_to_rgba_artery_only,
    mask_to_rgba_vein_only,
    numpy_rgba_to_qimage,
)
from .paint_ops import PaintOperations
from .history import CompressedHistory
from .dialogs import ShortcutsDialog
from .canvas import ImageCanvas

# Module-level logger
logger = logging.getLogger(__name__)


class MaskEditor(QMainWindow):
    """Main editor class using the compiled UI."""

    def __init__(self):
        super().__init__()

        # Setup UI from compiled .ui file
        self.ui = Ui_MaskEditorUI()
        self.ui.setupUi(self)

        # Set application icon
        self._set_app_icon()

        # Create and add canvas to scroll area
        self.canvas = ImageCanvas(self)
        self.ui.scrollArea.setWidget(self.canvas)

        # Set task combo data
        self.ui.taskCombo.setItemData(0, AppConfig.Task.ARTERY_VEIN)
        self.ui.taskCombo.setItemData(1, AppConfig.Task.OPTIC_DISC)

        # Set display combo data
        self.ui.displayCombo.setItemData(0, AppConfig.Display.ORIGINAL)
        self.ui.displayCombo.setItemData(1, AppConfig.Display.ENHANCED)

        # Current task type
        self.task_type = AppConfig.Task.ARTERY_VEIN

        # Current display mode
        self.display_mode = AppConfig.Display.ORIGINAL

        # Directories
        self.mask_dir = ""
        self.image_dir = ""
        self.base_dir = ""
        self.file_names = []
        self.pairs = {}

        # Mask data (artery/vein)
        self.artery_mask = None
        self.vein_mask = None
        self.working_artery = None
        self.working_vein = None

        # Mask data (optic disc)
        self.disc_mask = None
        self.working_disc = None

        # Image data
        self.image = None
        self.original_image = None
        self.enhanced_image = None

        # History
        self.history = CompressedHistory(max_size=AppConfig.History.MAX_SIZE)

        self.record = {}
        self.modified = False
        self.current_name = None

        # Mode buttons storage
        self.brush_buttons = {}

        # Mode shortcuts storage
        self.mode_shortcuts = []

        # Annotation mode state (for dedicated A/V editing)
        self.annotation_mode = AppConfig.AnnotationMode.NORMAL

        # Setup brush slider for diameter-based sizing
        self._setup_brush_slider()

        # Setup menu bar
        self._setup_menu_bar()

        # Connect signals and initialize
        self._connect_signals()
        self._init_shortcuts()
        self._update_mode_buttons()

        # App settings
        self.app_settings = {}
        self.settings_file_path = ""

    def _get_default_settings(self):
        """Get default settings from AppConfig."""
        return {
            "brush": {
                "default_size": AppConfig.Brush.DEFAULT_SIZE,
                "min_size": AppConfig.Brush.MIN_SIZE,
                "max_size": AppConfig.Brush.MAX_SIZE,
            },
            "opacity": {
                "default": AppConfig.Opacity.DEFAULT_PERCENT,
            },
            "zoom": {
                "default": AppConfig.Zoom.DEFAULT,
            },
            "display": {
                "mode": AppConfig.Display.DEFAULT,
            },
            "task": {
                "type": AppConfig.Task.ARTERY_VEIN,
            },
            "enhancement": {
                "gaussian_sigma": AppConfig.Enhancement.GAUSSIAN_SIGMA,
            },
            "last_image": {
                "artery_vein": None,
                "optic_disc": None,
            },
        }

    def _load_app_settings(self, folder_path):
        """Load app settings from yaml file, create if not exists."""
        self.settings_file_path = os.path.join(folder_path, AppConfig.File.SETTING_FILE)

        # Setup logging with log file in the same directory
        setup_logging(folder_path)

        if os.path.exists(self.settings_file_path):
            try:
                with open(self.settings_file_path, "r", encoding="utf-8") as f:
                    self.app_settings = yaml.safe_load(f) or {}
                logger.info(f"Loaded settings from {self.settings_file_path}")
            except Exception as e:
                logger.error(f"Error loading settings: {e}")
                self.app_settings = self._get_default_settings()
        else:
            # Create default settings file
            self.app_settings = self._get_default_settings()
            self._save_app_settings()
            logger.info(f"Created default settings at {self.settings_file_path}")

        # Apply loaded settings
        self._apply_app_settings()

    def _save_app_settings(self):
        """Save current app settings to yaml file."""
        if not self.settings_file_path:
            return
        try:
            with open(self.settings_file_path, "w", encoding="utf-8") as f:
                yaml.dump(
                    self.app_settings, f, default_flow_style=False, allow_unicode=True
                )
        except Exception as e:
            logger.error(f"Error saving settings: {e}")

    def _apply_app_settings(self):
        """Apply loaded settings to the application."""
        try:
            # Apply brush settings
            brush_settings = self.app_settings.get("brush", {})
            brush_size = brush_settings.get(
                "default_size", AppConfig.Brush.DEFAULT_SIZE
            )
            self.ui.brushSlider.setValue(brush_size)
            self.canvas.brush_size = brush_size

            # Apply opacity settings
            opacity_settings = self.app_settings.get("opacity", {})
            opacity = opacity_settings.get("default", AppConfig.Opacity.DEFAULT_PERCENT)
            self.ui.opacitySlider.setValue(opacity)
            self.canvas.alpha = opacity / 100.0

            # Apply display mode
            display_settings = self.app_settings.get("display", {})
            display_mode = display_settings.get("mode", AppConfig.Display.DEFAULT)
            if display_mode == AppConfig.Display.ENHANCED:
                self.ui.displayCombo.setCurrentIndex(1)
            else:
                self.ui.displayCombo.setCurrentIndex(0)

            # Apply task type
            task_settings = self.app_settings.get("task", {})
            task_type = task_settings.get("type", AppConfig.Task.ARTERY_VEIN)
            if task_type == AppConfig.Task.OPTIC_DISC:
                self.ui.taskCombo.setCurrentIndex(1)
            else:
                self.ui.taskCombo.setCurrentIndex(0)

        except Exception as e:
            logger.error(f"Error applying settings: {e}")

    def _update_and_save_settings(self):
        """Update settings dict from current state and save."""
        self.app_settings["brush"]["default_size"] = self.canvas.brush_size
        self.app_settings["opacity"]["default"] = self.ui.opacitySlider.value()
        self.app_settings["display"]["mode"] = self.display_mode
        self.app_settings["task"]["type"] = self.task_type
        self._save_app_settings()

    def _set_app_icon(self):
        """Set the application icon."""
        icon_path = resource_path(AppConfig.App.ICON)
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

    def _setup_brush_slider(self):
        """Setup the brush slider for diameter-based brush size."""
        self.ui.brushSlider.setMinimum(AppConfig.Brush.MIN_SIZE)
        self.ui.brushSlider.setMaximum(AppConfig.Brush.MAX_SIZE)
        self.ui.brushSlider.setSingleStep(AppConfig.Brush.SINGLE_STEP)
        self.ui.brushSlider.setPageStep(AppConfig.Brush.PAGE_STEP)
        self.ui.brushSlider.setValue(AppConfig.Brush.DEFAULT_SIZE)

        self.canvas.brush_size = AppConfig.Brush.DEFAULT_SIZE

    def _setup_menu_bar(self):
        """Setup the menu bar with all menus and actions."""
        menubar = self.menuBar()
        sc = AppConfig.Shortcut

        # ==================== File Menu ====================
        file_menu = menubar.addMenu("&File")

        open_action = QAction("&Open Folder...", self)
        open_action.setShortcut(QKeySequence(sc.OPEN_FOLDER))
        open_action.setStatusTip("Open a folder containing masks and images")
        open_action.triggered.connect(self.open_folder)
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        save_action = QAction("&Save", self)
        save_action.setShortcut(QKeySequence(sc.SAVE))
        save_action.setStatusTip("Save current mask")
        save_action.triggered.connect(self.save_current)
        file_menu.addAction(save_action)

        finalize_action = QAction("&Finalize", self)
        finalize_action.setShortcut(QKeySequence(sc.FINALIZE))
        finalize_action.setStatusTip("Mark current mask as finalized")
        finalize_action.triggered.connect(self.finalize_current)
        file_menu.addAction(finalize_action)

        unfinalize_action = QAction("&Unfinalize", self)
        unfinalize_action.setStatusTip("Remove finalization mark from current mask")
        unfinalize_action.triggered.connect(self.unfinal_current)
        file_menu.addAction(unfinalize_action)

        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence(sc.QUIT))
        exit_action.setStatusTip("Exit the application")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # ==================== Edit Menu ====================
        edit_menu = menubar.addMenu("&Edit")

        undo_action = QAction("&Undo", self)
        undo_action.setShortcut(QKeySequence(sc.UNDO))
        undo_action.setStatusTip("Undo last action")
        undo_action.triggered.connect(self.undo)
        edit_menu.addAction(undo_action)

        redo_action = QAction("&Redo", self)
        redo_action.setShortcut(QKeySequence(sc.REDO))
        redo_action.setStatusTip("Redo last undone action")
        redo_action.triggered.connect(self.redo)
        edit_menu.addAction(redo_action)

        # ==================== View Menu ====================
        view_menu = menubar.addMenu("&View")

        enhanced_action = QAction("Toggle &Enhanced View", self)
        enhanced_action.setShortcut(QKeySequence(sc.TOGGLE_ENHANCED))
        enhanced_action.setStatusTip("Toggle between original and enhanced image")
        enhanced_action.triggered.connect(self.toggle_enhanced)
        view_menu.addAction(enhanced_action)

        opacity_action = QAction("Toggle Mask &Opacity", self)
        opacity_action.setShortcut(QKeySequence(sc.TOGGLE_OPACITY))
        opacity_action.setStatusTip("Toggle mask visibility")
        opacity_action.triggered.connect(self.toggle_opacity)
        view_menu.addAction(opacity_action)

        view_menu.addSeparator()

        zoom_in_action = QAction("Zoom &In", self)
        zoom_in_action.setShortcut(QKeySequence(sc.ZOOM_IN))
        zoom_in_action.setStatusTip("Zoom in")
        zoom_in_action.triggered.connect(
            lambda: self.canvas.set_scale(self.canvas.scale * AppConfig.Zoom.FACTOR)
        )
        view_menu.addAction(zoom_in_action)

        zoom_out_action = QAction("Zoom &Out", self)
        zoom_out_action.setShortcut(QKeySequence(sc.ZOOM_OUT))
        zoom_out_action.setStatusTip("Zoom out")
        zoom_out_action.triggered.connect(
            lambda: self.canvas.set_scale(self.canvas.scale / AppConfig.Zoom.FACTOR)
        )
        view_menu.addAction(zoom_out_action)

        reset_zoom_action = QAction("&Reset Zoom", self)
        reset_zoom_action.setShortcut(QKeySequence(sc.ZOOM_RESET))
        reset_zoom_action.setStatusTip("Reset zoom to 100%")
        reset_zoom_action.triggered.connect(
            lambda: self.canvas.set_scale(AppConfig.Zoom.DEFAULT)
        )
        view_menu.addAction(reset_zoom_action)

        # ==================== Help Menu ====================
        help_menu = menubar.addMenu("&Help")

        shortcuts_action = QAction("&Keyboard Shortcuts", self)
        shortcuts_action.setShortcut(QKeySequence(sc.SHOW_SHORTCUTS))
        shortcuts_action.setStatusTip("Show keyboard shortcuts")
        shortcuts_action.triggered.connect(self.show_shortcuts_dialog)
        help_menu.addAction(shortcuts_action)

        help_menu.addSeparator()

        about_action = QAction("&About", self)
        about_action.setStatusTip("About")
        about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(about_action)

    def show_shortcuts_dialog(self):
        """Show the shortcuts dialog."""
        dialog = ShortcutsDialog(self)
        dialog.exec()

    def show_about_dialog(self):
        """Show the about dialog."""
        QMessageBox.about(self, "About", AppConfig.App.get_about_html())

    def _connect_signals(self):
        """Connect all UI signals to their handlers."""
        self.canvas.on_paint = self._on_paint
        self.canvas.on_stroke_start = self._on_stroke_start
        self.canvas.on_stroke_end = self._on_stroke_end
        self.canvas.on_brush_size_change = self._on_brush_size_change

        self.ui.taskCombo.currentIndexChanged.connect(self._on_task_changed)
        self.ui.displayCombo.currentIndexChanged.connect(self._on_display_changed)

        self.ui.btnOpenFolder.clicked.connect(self.open_folder)
        self.ui.saveBtn.clicked.connect(self.save_current)
        self.ui.finalBtn.clicked.connect(self.finalize_current)
        self.ui.unfinalBtn.clicked.connect(self.unfinal_current)
        self.ui.undoBtn.clicked.connect(self.undo)
        self.ui.redoBtn.clicked.connect(self.redo)

        self.ui.listWidget.currentRowChanged.connect(self.on_list_select)

        self.ui.brushSlider.valueChanged.connect(self.on_brush_change)
        self.ui.opacitySlider.valueChanged.connect(self.on_opacity_change)

    def _on_brush_size_change(self, new_size):
        """Handle brush size change from canvas (Alt+Scroll)."""
        self.ui.brushSlider.blockSignals(True)
        self.ui.brushSlider.setValue(new_size)
        self.ui.brushSlider.blockSignals(False)

    def _init_shortcuts(self):
        """Initialize keyboard shortcuts."""
        sc = AppConfig.Shortcut
        shortcuts = {
            sc.ZOOM_IN_ALT: lambda: self.canvas.set_scale(
                self.canvas.scale * AppConfig.Zoom.FACTOR
            ),
            sc.BRUSH_DECREASE: lambda: self.ui.brushSlider.setValue(
                max(AppConfig.Brush.MIN_SIZE, self.ui.brushSlider.value() - 1)
            ),
            sc.BRUSH_INCREASE: lambda: self.ui.brushSlider.setValue(
                min(AppConfig.Brush.MAX_SIZE, self.ui.brushSlider.value() + 1)
            ),
            # Annotation mode shortcuts
            sc.ANNOTATION_ARTERY: lambda: self.set_annotation_mode(
                AppConfig.AnnotationMode.ARTERY
            ),
            sc.ANNOTATION_VEIN: lambda: self.set_annotation_mode(
                AppConfig.AnnotationMode.VEIN
            ),
            sc.ANNOTATION_EXIT: lambda: self.set_annotation_mode(
                AppConfig.AnnotationMode.NORMAL
            ),
        }
        for seq, func in shortcuts.items():
            QShortcut(QKeySequence(seq), self, func)

        # Initialize mode shortcuts
        self._setup_mode_shortcuts()

    def _setup_mode_shortcuts(self):
        """Setup mode shortcuts based on current task."""
        # Clear existing mode shortcuts
        for shortcut in self.mode_shortcuts:
            shortcut.setEnabled(False)
            shortcut.deleteLater()
        self.mode_shortcuts.clear()

        sc = AppConfig.Shortcut

        if self.task_type == AppConfig.Task.ARTERY_VEIN:
            mode_shortcut_map = {
                sc.MODE_MODIFY_ARTERY: "modify_artery",
                sc.MODE_MODIFY_VEIN: "modify_vein",
                sc.MODE_DRAW_ARTERY: "draw_artery",
                sc.MODE_DRAW_VEIN: "draw_vein",
                sc.MODE_DELETE_VESSEL: "delete_vessel",
                sc.MODE_DELETE_ARTERY: "delete_artery",
                sc.MODE_DELETE_VEIN: "delete_vein",
            }
        else:  # Optic Disc
            mode_shortcut_map = {
                sc.MODE_DRAW_DISC: "draw_disc",
                sc.MODE_DELETE_DISC: "delete_disc",
            }

        for key, mode in mode_shortcut_map.items():
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.activated.connect(lambda m=mode: self.set_mode(m))
            self.mode_shortcuts.append(shortcut)

    def _update_mode_buttons(self):
        """Update mode buttons based on current task."""
        for btn in self.brush_buttons.values():
            self.ui.modeLayout.removeWidget(btn)
            btn.deleteLater()
        self.brush_buttons.clear()

        config = AppConfig.Task.get(self.task_type)
        sc = AppConfig.Shortcut

        # Get shortcut keys for each mode
        if self.task_type == AppConfig.Task.ARTERY_VEIN:
            shortcut_map = {
                "modify_artery": sc.MODE_MODIFY_ARTERY,
                "modify_vein": sc.MODE_MODIFY_VEIN,
                "draw_artery": sc.MODE_DRAW_ARTERY,
                "draw_vein": sc.MODE_DRAW_VEIN,
                "delete_vessel": sc.MODE_DELETE_VESSEL,
                "delete_artery": sc.MODE_DELETE_ARTERY,
                "delete_vein": sc.MODE_DELETE_VEIN,
            }
        else:
            shortcut_map = {
                "draw_disc": sc.MODE_DRAW_DISC,
                "delete_disc": sc.MODE_DELETE_DISC,
            }

        for mode_id, label in config["modes"]:
            shortcut_key = shortcut_map.get(mode_id, "")
            btn_label = f"{label} ({shortcut_key})" if shortcut_key else label
            btn = QPushButton(btn_label)
            btn.clicked.connect(lambda _, m=mode_id: self.set_mode(m))
            self.brush_buttons[mode_id] = btn
            self.ui.modeLayout.addWidget(btn)

        self.canvas.mode = config["default_mode"]
        self._update_brush_button_styles(config["default_mode"])

        # Update mode shortcuts
        self._setup_mode_shortcuts()

    def _on_display_changed(self, index):
        """Handle display mode change."""
        self.display_mode = self.ui.displayCombo.currentData()
        self.canvas.set_display_mode(self.display_mode)

    def toggle_enhanced(self):
        """Toggle between original and enhanced image display."""
        if self.display_mode == AppConfig.Display.ORIGINAL:
            self.ui.displayCombo.setCurrentIndex(1)
        else:
            self.ui.displayCombo.setCurrentIndex(0)

    def _on_task_changed(self, index):
        """Handle task type change."""
        if self.modified:
            reply = QMessageBox.question(
                self,
                "Unsaved Changes",
                "Save changes before switching task?",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
            )
            if reply == QMessageBox.Yes:
                self._save_mask_to_file(self.current_name)
            elif reply == QMessageBox.Cancel:
                self.ui.taskCombo.blockSignals(True)
                self.ui.taskCombo.setCurrentIndex(
                    0 if self.task_type == AppConfig.Task.ARTERY_VEIN else 1
                )
                self.ui.taskCombo.blockSignals(False)
                return

        self.task_type = self.ui.taskCombo.currentData()
        self.canvas.set_task_type(self.task_type)
        self._update_mode_buttons()

        # Reset annotation mode when switching tasks
        self.annotation_mode = AppConfig.AnnotationMode.NORMAL
        self.canvas.annotation_mode = AppConfig.AnnotationMode.NORMAL

        self.modified = False
        self.current_name = None

        if self.base_dir:
            config = AppConfig.Task.get(self.task_type)
            new_mask_folder = self._find_subfolder(self.base_dir, config["mask_folder"])

            if not new_mask_folder:
                QMessageBox.warning(
                    self,
                    "Folder Not Found",
                    f"Could not find '{config['mask_folder']}' subfolder.\n"
                    f"Please create it or select a different folder.",
                )
                self.ui.listWidget.clear()
                self.mask_dir = ""
                self.ui.folderStatusLabel.setText(f"Missing: {config['mask_folder']}")
                self.ui.folderStatusLabel.setStyleSheet(
                    f"color: {AppConfig.Color.STATUS_ERROR}; font-size: {AppConfig.UI.STATUS_FONT_SIZE};"
                )
                return

            self.mask_dir = new_mask_folder
            self.ui.folderStatusLabel.setText(
                f"Loaded:\n• Masks: {os.path.basename(new_mask_folder)}\n• Images: {os.path.basename(self.image_dir)}"
            )
            self.ui.folderStatusLabel.setStyleSheet(
                f"color: {AppConfig.Color.STATUS_SUCCESS}; font-size: {AppConfig.UI.STATUS_FONT_SIZE};"
            )

            self._load_record()
            self._try_match()

    def _find_subfolder(self, parent_dir, target_name):
        """Find a subfolder by name (case-insensitive)."""
        target_lower = target_name.lower()
        try:
            for item in os.listdir(parent_dir):
                if (
                    os.path.isdir(os.path.join(parent_dir, item))
                    and item.lower() == target_lower
                ):
                    return os.path.join(parent_dir, item)
        except OSError:
            pass
        return None

    def open_folder(self):
        """Open a folder containing masks and images."""
        d = QFileDialog.getExistingDirectory(self, "Select folder")
        if not d:
            return

        # Load/create app settings
        self._load_app_settings(d)

        config = AppConfig.Task.get(self.task_type)
        mask_folder = self._find_subfolder(d, config["mask_folder"])
        image_folder = self._find_subfolder(d, AppConfig.File.IMAGES_FOLDER)

        if not mask_folder:
            QMessageBox.warning(
                self,
                "Folder Not Found",
                f"Could not find '{config['mask_folder']}' subfolder in:\n{d}",
            )
            return

        if not image_folder:
            QMessageBox.warning(
                self,
                "Folder Not Found",
                f"Could not find '{AppConfig.File.IMAGES_FOLDER}' subfolder in:\n{d}",
            )
            return

        self.base_dir = d
        self.mask_dir = mask_folder
        self.image_dir = image_folder

        self.ui.folderStatusLabel.setText(
            f"Loaded:\n• Masks: {os.path.basename(mask_folder)}\n• Images: {os.path.basename(image_folder)}"
        )
        self.ui.folderStatusLabel.setStyleSheet(
            f"color: {AppConfig.Color.STATUS_SUCCESS}; font-size: {AppConfig.UI.STATUS_FONT_SIZE};"
        )

        self._load_record()
        self._try_match()

    def _load_record(self):
        """Load the finalization record from file."""
        self.record = {}
        if not self.base_dir:
            return
        config = AppConfig.Task.get(self.task_type)
        p = os.path.join(self.base_dir, config["record_file"])
        if os.path.exists(p):
            try:
                with open(p) as f:
                    self.record = json.load(f)
            except Exception:
                pass

    def _save_record(self):
        """Save the finalization record to file."""
        if not self.base_dir:
            return
        config = AppConfig.Task.get(self.task_type)
        p = os.path.join(self.base_dir, config["record_file"])
        try:
            with open(p, "w") as f:
                json.dump(self.record, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving record: {e}")

    def _try_match(self):
        """Try to match mask and image files."""
        if not (self.mask_dir and self.image_dir):
            return

        valid_ext = AppConfig.File.VALID_EXTENSIONS

        masks = {
            os.path.splitext(f)[0]: f
            for f in os.listdir(self.mask_dir)
            if os.path.splitext(f)[1].lower() in valid_ext
        }
        imgs = {
            os.path.splitext(f)[0]: f
            for f in os.listdir(self.image_dir)
            if os.path.splitext(f)[1].lower() in valid_ext
        }

        common = sorted(set(masks) & set(imgs))
        if not common:
            QMessageBox.warning(self, "No Matching Files", "No matching pairs found.")
            return

        self.file_names = common
        self.pairs = {
            n: (
                os.path.join(self.mask_dir, masks[n]),
                os.path.join(self.image_dir, imgs[n]),
            )
            for n in common
        }

        last_image_dict = self.app_settings.get("last_image", {})
        last_image = last_image_dict.get(self.task_type)
        target_row = 0
        if last_image and last_image in common:
            try:
                target_row = common.index(last_image)
            except ValueError:
                target_row = 0

        self.ui.listWidget.clear()
        for n in common:
            txt = f"✅ {n}" if self.record.get(n) else n
            self.ui.listWidget.addItem(QListWidgetItem(txt))

        self.ui.listWidget.setCurrentRow(target_row)

    def on_list_select(self, new_idx):
        """Handle file list selection change."""
        if new_idx < 0:
            return

        if self.modified:
            reply = QMessageBox.question(
                self,
                "Unsaved Changes",
                "Save changes to current mask?",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
            )
            if reply == QMessageBox.Yes:
                self._save_mask_to_file(self.current_name)
            elif reply == QMessageBox.Cancel:
                return

        # Reset annotation mode when switching images
        if self.annotation_mode != AppConfig.AnnotationMode.NORMAL:
            self.annotation_mode = AppConfig.AnnotationMode.NORMAL
            self.canvas.annotation_mode = AppConfig.AnnotationMode.NORMAL
            self._update_mode_buttons_for_annotation()

        txt = self.ui.listWidget.item(new_idx).text()
        name = txt[2:] if txt.startswith("✅ ") else txt
        mask_path, img_path = self.pairs[name]

        mask = imread(mask_path)
        img = imread(img_path)

        if mask.ndim == 3:
            mask = mask[..., 0]
        mask = ensure_mask(mask)

        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.ndim == 3 and img.shape[2] == 4:
            img = img[..., :3]

        img = np.ascontiguousarray(img)
        mask = np.ascontiguousarray(mask)

        if mask.shape != img.shape[:2]:
            mask = np.array(
                Image.fromarray(mask).resize(img.shape[1::-1], Image.NEAREST)
            )

        self.original_image = img
        self.enhanced_image = enhance_fundus_image(
            img, sigma=AppConfig.Enhancement.GAUSSIAN_SIGMA
        )
        self.image = (
            self.original_image
            if self.display_mode == AppConfig.Display.ORIGINAL
            else self.enhanced_image
        )

        if self.task_type == AppConfig.Task.ARTERY_VEIN:
            artery, vein = decode_combined_mask(mask)
            self.artery_mask = artery.copy()
            self.vein_mask = vein.copy()
            self.working_artery = self.artery_mask
            self.working_vein = self.vein_mask
            self.disc_mask = None
            self.working_disc = None

            self.history.clear()
            self.history.set_num_masks(2)
            self.history.push(self.working_artery, self.working_vein)

            self.canvas.load_image_and_masks_av(
                self.original_image,
                self.enhanced_image,
                self.working_artery,
                self.working_vein,
            )
        else:
            self.disc_mask = (mask > 0).astype(np.uint8)
            self.working_disc = self.disc_mask.copy()
            self.artery_mask = None
            self.vein_mask = None
            self.working_artery = None
            self.working_vein = None

            self.history.clear()
            self.history.set_num_masks(1)
            self.history.push(self.working_disc)

            self.canvas.load_image_and_mask_od(
                self.original_image, self.enhanced_image, self.working_disc
            )

        self.modified = False
        self.current_name = name

        if "last_image" not in self.app_settings:
            self.app_settings["last_image"] = {}
        self.app_settings["last_image"][self.task_type] = name
        self._save_app_settings()

    def _on_paint(self, x, y, size, mode):
        """Handle paint events from canvas."""
        if self.task_type == AppConfig.Task.ARTERY_VEIN:
            if self.working_artery is None or self.working_vein is None:
                return

            if mode == "modify_artery":
                PaintOperations.modify_artery(
                    self.working_artery, self.working_vein, x, y, size
                )
            elif mode == "modify_vein":
                PaintOperations.modify_vein(
                    self.working_artery, self.working_vein, x, y, size
                )
            elif mode == "draw_artery":
                PaintOperations.apply_brush(self.working_artery, x, y, size, 1)
            elif mode == "draw_vein":
                PaintOperations.apply_brush(self.working_vein, x, y, size, 1)
            elif mode == "delete_artery":
                PaintOperations.apply_brush(self.working_artery, x, y, size, 0)
            elif mode == "delete_vein":
                PaintOperations.apply_brush(self.working_vein, x, y, size, 0)
            elif mode == "delete_vessel":
                PaintOperations.apply_brush(self.working_artery, x, y, size, 0)
                PaintOperations.apply_brush(self.working_vein, x, y, size, 0)

        else:
            if self.working_disc is None:
                return

            if mode == "draw_disc":
                PaintOperations.apply_brush(self.working_disc, x, y, size, 1)
            elif mode == "delete_disc":
                PaintOperations.apply_brush(self.working_disc, x, y, size, 0)

    def _on_stroke_start(self):
        """Handle stroke start event."""
        self.modified = True

    def _on_stroke_end(self):
        """Handle stroke end event."""
        if self.task_type == AppConfig.Task.ARTERY_VEIN:
            if self.working_artery is not None and self.working_vein is not None:
                self.history.push(self.working_artery, self.working_vein)
        else:
            if self.working_disc is not None:
                self.history.push(self.working_disc)

    def undo(self):
        """Undo last action."""
        result = self.history.undo()

        if self.task_type == AppConfig.Task.ARTERY_VEIN:
            artery, vein = result
            if artery is not None:
                self.working_artery[:] = artery
                self.working_vein[:] = vein
                self.canvas.artery_mask = self.working_artery
                self.canvas.vein_mask = self.working_vein
                self.canvas._build_pixmaps()
        else:
            (disc,) = result
            if disc is not None:
                self.working_disc[:] = disc
                self.canvas.disc_mask = self.working_disc
                self.canvas._build_pixmaps()

    def redo(self):
        """Redo last undone action."""
        result = self.history.redo()

        if self.task_type == AppConfig.Task.ARTERY_VEIN:
            artery, vein = result
            if artery is not None:
                self.working_artery[:] = artery
                self.working_vein[:] = vein
                self.canvas.artery_mask = self.working_artery
                self.canvas.vein_mask = self.working_vein
                self.canvas._build_pixmaps()
        else:
            (disc,) = result
            if disc is not None:
                self.working_disc[:] = disc
                self.canvas.disc_mask = self.working_disc
                self.canvas._build_pixmaps()

    def set_mode(self, mode):
        """Set the current drawing mode."""
        # Validate mode is allowed in current annotation mode
        if self.annotation_mode == AppConfig.AnnotationMode.ARTERY:
            if mode not in ("draw_artery", "delete_artery"):
                return  # Ignore invalid mode
        elif self.annotation_mode == AppConfig.AnnotationMode.VEIN:
            if mode not in ("draw_vein", "delete_vein"):
                return  # Ignore invalid mode

        self.canvas.mode = mode
        self._update_brush_button_styles(mode)

    def set_annotation_mode(self, mode):
        """Set the annotation mode (normal, artery, or vein)."""
        if self.task_type != AppConfig.Task.ARTERY_VEIN:
            return  # Only for A/V task

        old_mode = self.annotation_mode
        if old_mode == mode:
            return  # No change

        self.annotation_mode = mode
        self.canvas.annotation_mode = mode

        if mode != AppConfig.AnnotationMode.NORMAL:
            # Entering dedicated mode - save current paint mode
            self.canvas._previous_paint_mode = self.canvas.mode
            # Set appropriate default paint mode
            if mode == AppConfig.AnnotationMode.ARTERY:
                self.set_mode("draw_artery")
            else:  # VEIN
                self.set_mode("draw_vein")
        else:
            # Exiting dedicated mode - restore previous mode
            self.canvas.mode = self.canvas._previous_paint_mode
            self._update_brush_button_styles(self.canvas.mode)

        # Update mode buttons enabled state
        self._update_mode_buttons_for_annotation()

        # Rebuild display with new visualization
        self.canvas._build_pixmaps()

    def _update_mode_buttons_for_annotation(self):
        """Enable/disable mode buttons based on annotation mode."""
        for mode_id, btn in self.brush_buttons.items():
            if self.annotation_mode == AppConfig.AnnotationMode.NORMAL:
                btn.setEnabled(True)
            elif self.annotation_mode == AppConfig.AnnotationMode.ARTERY:
                btn.setEnabled(mode_id in ("draw_artery", "delete_artery"))
            elif self.annotation_mode == AppConfig.AnnotationMode.VEIN:
                btn.setEnabled(mode_id in ("draw_vein", "delete_vein"))

    def _update_brush_button_styles(self, active):
        """Update button styles to show active mode."""
        if self.task_type == AppConfig.Task.OPTIC_DISC:
            active_color = AppConfig.Color.BUTTON_ACTIVE_DISC
        elif "artery" in active:
            active_color = AppConfig.Color.BUTTON_ACTIVE_ARTERY
        elif "vein" in active:
            active_color = AppConfig.Color.BUTTON_ACTIVE_VEIN
        else:
            active_color = AppConfig.Color.BUTTON_ACTIVE_DEFAULT

        for mode, btn in self.brush_buttons.items():
            if mode == active:
                btn.setStyleSheet(
                    f"background-color: {active_color}; color: white; font-weight: bold;"
                )
            else:
                btn.setStyleSheet("")

    def on_brush_change(self, v):
        """Handle brush size change. Value is diameter in pixels."""
        self.canvas.brush_size = v
        self.canvas.update()

    def on_opacity_change(self, v):
        """Handle opacity slider change."""
        self.canvas.alpha = v / 100.0

        if self.task_type == AppConfig.Task.ARTERY_VEIN:
            if self.working_artery is not None and self.working_vein is not None:
                # Use appropriate visualization based on annotation mode
                if self.annotation_mode == AppConfig.AnnotationMode.ARTERY:
                    self.canvas.overlay_rgba = mask_to_rgba_artery_only(
                        self.working_artery, self.canvas.alpha
                    )
                elif self.annotation_mode == AppConfig.AnnotationMode.VEIN:
                    self.canvas.overlay_rgba = mask_to_rgba_vein_only(
                        self.working_vein, self.canvas.alpha
                    )
                else:
                    self.canvas.overlay_rgba = masks_to_rgba_av(
                        self.working_artery, self.working_vein, self.canvas.alpha
                    )
                qimg = numpy_rgba_to_qimage(self.canvas.overlay_rgba)
                self.canvas.overlay_pixmap = QPixmap.fromImage(qimg)
                self.canvas._composed_pixmap = None
                self.canvas._update_display_pixmap()
        else:
            if self.working_disc is not None:
                self.canvas.overlay_rgba = mask_to_rgba_od(
                    self.working_disc, self.canvas.alpha
                )
                qimg = numpy_rgba_to_qimage(self.canvas.overlay_rgba)
                self.canvas.overlay_pixmap = QPixmap.fromImage(qimg)
                self.canvas._composed_pixmap = None
                self.canvas._update_display_pixmap()

    def toggle_opacity(self):
        """Toggle mask visibility."""
        self.canvas.alpha = (
            0.0 if self.canvas.alpha > 0 else self.ui.opacitySlider.value() / 100.0
        )
        self.canvas._build_pixmaps()

    def _save_mask_to_file(self, name):
        """Save current mask to file."""
        if name is None:
            return

        if self.task_type == AppConfig.Task.ARTERY_VEIN:
            combined = encode_to_combined_mask(self.working_artery, self.working_vein)
        else:
            combined = (self.working_disc > 0).astype(np.uint8) * 255

        out = os.path.join(self.mask_dir, name + AppConfig.File.OUTPUT_FORMAT)
        imwrite(out, combined)
        self.modified = False

    def save_current(self):
        """Save current mask."""
        self._save_mask_to_file(self.current_name)

    def finalize_current(self):
        """Mark current mask as finalized and save."""
        self._save_mask_to_file(self.current_name)
        if self.current_name:
            self.record[self.current_name] = True
            self._save_record()
            current_item = self.ui.listWidget.currentItem()
            if current_item and not current_item.text().startswith("✅ "):
                current_item.setText(f"✅ {current_item.text()}")

    def unfinal_current(self):
        """Remove finalization mark from current mask."""
        idx = self.ui.listWidget.currentRow()
        if idx < 0 or not self.file_names:
            return
        item = self.ui.listWidget.item(idx)
        txt = item.text()
        name = txt[2:] if txt.startswith("✅ ") else txt
        self.record.pop(name, None)
        self._save_record()
        if txt.startswith("✅ "):
            item.setText(txt[2:])

    def closeEvent(self, event):
        """Handle window close event."""

        if self.settings_file_path:
            self._update_and_save_settings()

        if self.modified:
            reply = QMessageBox.question(
                self,
                "Unsaved Changes",
                "Save changes before exiting?",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
            )
            if reply == QMessageBox.Yes:
                self._save_mask_to_file(self.current_name)
                event.accept()
            elif reply == QMessageBox.No:
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


def main():
    """Application entry point."""
    app = QApplication(sys.argv)

    # Set application icon globally
    icon_path = AppConfig.App.ICON
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))

    w = MaskEditor()
    w.showMaximized()
    sys.exit(app.exec())
