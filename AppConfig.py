"""
AppConfig.py
"""

import numpy as np


# =====================================================================
# Application Configuration
# =====================================================================
class AppConfig:
    """Central configuration for the entire application."""

    # -----------------------------------------------------------------
    # Application Info
    # -----------------------------------------------------------------
    class App:
        """Application metadata."""

        NAME = "Vessel Annotation Tool"
        VERSION = "1.0.0"
        ICON = "icon.ico"
        DESCRIPTION = "A tool for editing and annotating fundus image masks."

        FEATURES = [
            "Artery/Vein annotation",
            "Optic Disc annotation",
            "Enhanced image view",
            "Undo/Redo support",
        ]

        DEVELOPERS = [
            ("Xujia Liu", "https://github.com/xujialiu"),
            ("Jinyue Cai", None),
        ]

        @classmethod
        def get_about_html(cls):
            """Generate HTML content for about dialog."""
            features_html = "".join(f"<li>{f}</li>" for f in cls.FEATURES)
            developers_html = ""
            for name, url in cls.DEVELOPERS:
                if url:
                    developers_html += f'<li><a href="{url}">{name}</a></li>'
                else:
                    developers_html += f"<li><b>{name}</b></li>"

            return f"""
            <h2>{cls.NAME}</h2>
            <p><b>Version:</b> {cls.VERSION}</p>
            <p>{cls.DESCRIPTION}</p>
            
            <h3>Features</h3>
            <ul>
                {features_html}
            </ul>
            
            <h3>Developers</h3>
            <ul>
                {developers_html}
            </ul>
            """

    # -----------------------------------------------------------------
    # Annotation Mode (for dedicated A/V editing)
    # -----------------------------------------------------------------
    class AnnotationMode:
        """Annotation mode constants for dedicated artery/vein editing."""

        NORMAL = "normal"
        ARTERY = "artery"
        VEIN = "vein"

    # -----------------------------------------------------------------
    # Task Configuration
    # -----------------------------------------------------------------
    class Task:
        """Task type constants and configurations."""

        ARTERY_VEIN = "artery_vein"
        OPTIC_DISC = "optic_disc"

        CONFIGS = {
            ARTERY_VEIN: {
                "name": "Artery/Vein",
                "mask_folder": "masks_artery_vein",
                "record_file": "final_record_artery_vein.json",
                "modes": [
                    ("modify_artery", "Modify Artery"),
                    ("modify_vein", "Modify Vein"),
                    ("draw_artery", "Draw Artery"),
                    ("draw_vein", "Draw Vein"),
                    ("delete_vessel", "Delete All"),
                    ("delete_artery", "Delete Artery"),
                    ("delete_vein", "Delete Vein"),
                ],
                "default_mode": "modify_artery",
                "num_masks": 2,
            },
            OPTIC_DISC: {
                "name": "Optic Disc",
                "mask_folder": "masks_optic_disc",
                "record_file": "final_record_optic_disc.json",
                "modes": [
                    ("draw_disc", "Draw Disc"),
                    ("delete_disc", "Delete Disc"),
                ],
                "default_mode": "draw_disc",
                "num_masks": 1,
            },
        }

        @classmethod
        def get(cls, task_type):
            return cls.CONFIGS.get(task_type, cls.CONFIGS[cls.ARTERY_VEIN])

    # -----------------------------------------------------------------
    # Display Mode
    # -----------------------------------------------------------------
    class Display:
        """Display mode constants."""

        ORIGINAL = "original"
        ENHANCED = "enhanced"
        DEFAULT = ENHANCED

    # -----------------------------------------------------------------
    # Colors
    # -----------------------------------------------------------------
    class Color:
        """Color definitions for the application."""

        # Mask colors (RGB tuples)
        ARTERY = (255, 0, 0)
        VEIN = (0, 0, 255)
        OVERLAP = (255, 255, 0)
        DISC = (0, 255, 0)

        # RGBA colors for stroke preview
        STROKE_ARTERY = (255, 0, 0, 180)
        STROKE_VEIN = (0, 0, 255, 180)
        STROKE_DISC = (0, 255, 0, 180)
        STROKE_DELETE = (100, 100, 100, 150)
        STROKE_DEFAULT = (255, 255, 255, 150)

        # Cursor colors (RGBA)
        CURSOR_ARTERY = (255, 0, 0, 150)
        CURSOR_VEIN = (0, 0, 255, 150)
        CURSOR_DISC = (0, 255, 0, 150)
        CURSOR_DEFAULT = (255, 255, 255, 150)
        CURSOR_FILL_ALPHA = 40

        # UI button colors (hex strings)
        BUTTON_ACTIVE_ARTERY = "#ff4444"
        BUTTON_ACTIVE_VEIN = "#4444ff"
        BUTTON_ACTIVE_DISC = "#44cc44"
        BUTTON_ACTIVE_DEFAULT = "#cccccc"

        # Canvas
        CANVAS_BACKGROUND = "black"

        # Status label colors
        STATUS_SUCCESS = "green"
        STATUS_ERROR = "red"

        # Annotation mode indicator colors (hex strings)
        ANNOTATION_MODE_ARTERY = "#ff6666"
        ANNOTATION_MODE_VEIN = "#6666ff"

        @classmethod
        def get_av_lut(cls):
            """Get the Artery/Vein color lookup table."""
            return np.array(
                [
                    [0, 0, 0, 0],  # Background
                    [cls.ARTERY[0], cls.ARTERY[1], cls.ARTERY[2], 255],  # Artery
                    [cls.VEIN[0], cls.VEIN[1], cls.VEIN[2], 255],  # Vein
                    [cls.OVERLAP[0], cls.OVERLAP[1], cls.OVERLAP[2], 255],  # Overlap
                ],
                dtype=np.uint8,
            )

        @classmethod
        def get_od_lut(cls):
            """Get the Optic Disc color lookup table."""
            return np.array(
                [
                    [0, 0, 0, 0],  # Background
                    [cls.DISC[0], cls.DISC[1], cls.DISC[2], 255],  # Disc
                ],
                dtype=np.uint8,
            )

        @classmethod
        def get_artery_only_lut(cls):
            """Get LUT for artery-only display (single mask as red)."""
            return np.array(
                [
                    [0, 0, 0, 0],  # Background
                    [cls.ARTERY[0], cls.ARTERY[1], cls.ARTERY[2], 255],  # Artery
                ],
                dtype=np.uint8,
            )

        @classmethod
        def get_vein_only_lut(cls):
            """Get LUT for vein-only display (single mask as blue)."""
            return np.array(
                [
                    [0, 0, 0, 0],  # Background
                    [cls.VEIN[0], cls.VEIN[1], cls.VEIN[2], 255],  # Vein
                ],
                dtype=np.uint8,
            )

    # -----------------------------------------------------------------
    # Keyboard Shortcuts
    # -----------------------------------------------------------------
    class Shortcut:
        """Keyboard shortcut definitions."""

        # File operations
        OPEN_FOLDER = "Ctrl+O"
        SAVE = "Ctrl+S"
        FINALIZE = "Ctrl+F"
        QUIT = "Ctrl+Q"

        # Edit operations
        UNDO = "Ctrl+Z"
        REDO = "Ctrl+Y"

        # View operations
        TOGGLE_ENHANCED = "E"
        TOGGLE_OPACITY = "O"
        ZOOM_IN = "Ctrl++"
        ZOOM_IN_ALT = "Ctrl+="
        ZOOM_OUT = "Ctrl+-"
        ZOOM_RESET = "Ctrl+0"

        # Brush operations
        BRUSH_DECREASE = "Ctrl+["
        BRUSH_INCREASE = "Ctrl+]"

        # Help
        SHOW_SHORTCUTS = "F1"

        # Mode shortcuts for Artery/Vein task
        MODE_MODIFY_ARTERY = "1"
        MODE_MODIFY_VEIN = "2"
        MODE_DRAW_ARTERY = "3"
        MODE_DRAW_VEIN = "4"
        MODE_DELETE_VESSEL = "5"
        MODE_DELETE_ARTERY = "6"
        MODE_DELETE_VEIN = "7"

        # Mode shortcuts for Optic Disc task
        MODE_DRAW_DISC = "1"
        MODE_DELETE_DISC = "2"

        # Dedicated annotation mode shortcuts
        ANNOTATION_ARTERY = "Ctrl+C"
        ANNOTATION_VEIN = "Ctrl+V"
        ANNOTATION_EXIT = "Ctrl+X"

        @classmethod
        def get_shortcuts_html(cls):
            """Generate HTML content for shortcuts dialog."""
            return f"""
            <style>
                body {{ font-family: Arial, sans-serif; padding: 10px; }}
                h2 {{ color: #333; border-bottom: 2px solid #4a90d9; padding-bottom: 5px; }}
                h3 {{ color: #4a90d9; margin-top: 20px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
                th {{ background-color: #4a90d9; color: white; padding: 10px; text-align: left; }}
                td {{ padding: 8px; border-bottom: 1px solid #ddd; }}
                tr:hover {{ background-color: #f5f5f5; }}
                .key {{ 
                    background-color: #e0e0e0; 
                    padding: 3px 8px; 
                    border-radius: 4px; 
                    font-family: monospace;
                    font-weight: bold;
                }}
            </style>
            
            <h2>⌨️ Keyboard Shortcuts</h2>
            
            <h3>📁 File Operations</h3>
            <table>
                <tr><th>Shortcut</th><th>Action</th></tr>
                <tr><td>{cls.OPEN_FOLDER}</td><td>Open Folder</td></tr>
                <tr><td>{cls.SAVE}</td><td>Save Current Mask</td></tr>
                <tr><td>{cls.FINALIZE}</td><td>Finalize Current Mask</td></tr>
                <tr><td>{cls.QUIT}</td><td>Quit Application</td></tr>
            </table>
            
            <h3>✏️ Edit Operations</h3>
            <table>
                <tr><th>Shortcut</th><th>Action</th></tr>
                <tr><td>{cls.UNDO}</td><td>Undo</td></tr>
                <tr><td>{cls.REDO}</td><td>Redo</td></tr>
            </table>
            
            <h3>🔍 View Controls</h3>
            <table>
                <tr><th>Shortcut</th><th>Action</th></tr>
                <tr><td>{cls.TOGGLE_ENHANCED}</td><td>Toggle Enhanced/Original Image</td></tr>
                <tr><td>{cls.TOGGLE_OPACITY}</td><td>Toggle Mask Opacity (Show/Hide)</td></tr>
                <tr><td>{cls.ZOOM_IN} or {cls.ZOOM_IN_ALT}</td><td>Zoom In</td></tr>
                <tr><td>{cls.ZOOM_OUT}</td><td>Zoom Out</td></tr>
                <tr><td>{cls.ZOOM_RESET}</td><td>Reset Zoom (100%)</td></tr>
            </table>
            
            <h3>🖌️ Brush Controls</h3>
            <table>
                <tr><th>Shortcut</th><th>Action</th></tr>
                <tr><td>{cls.BRUSH_DECREASE}</td><td>Decrease Brush Size by 1 pixel</td></tr>
                <tr><td>{cls.BRUSH_INCREASE}</td><td>Increase Brush Size by 1 pixel</td></tr>
                <tr><td><b>Alt + Scroll Wheel</b></td><td>Adjust Brush Size</td></tr>
            </table>
            
            <h3>🎨 Mode Shortcuts (Artery/Vein Task)</h3>
            <table>
                <tr><th>Shortcut</th><th>Mode</th></tr>
                <tr><td>{cls.MODE_MODIFY_ARTERY}</td><td>Modify Artery</td></tr>
                <tr><td>{cls.MODE_MODIFY_VEIN}</td><td>Modify Vein</td></tr>
                <tr><td>{cls.MODE_DRAW_ARTERY}</td><td>Draw Artery</td></tr>
                <tr><td>{cls.MODE_DRAW_VEIN}</td><td>Draw Vein</td></tr>
                <tr><td>{cls.MODE_DELETE_VESSEL}</td><td>Delete All</td></tr>
                <tr><td>{cls.MODE_DELETE_ARTERY}</td><td>Delete Artery</td></tr>
                <tr><td>{cls.MODE_DELETE_VEIN}</td><td>Delete Vein</td></tr>
            </table>
            
            <h3>🔵 Mode Shortcuts (Optic Disc Task)</h3>
            <table>
                <tr><th>Shortcut</th><th>Mode</th></tr>
                <tr><td>{cls.MODE_DRAW_DISC}</td><td>Draw Disc</td></tr>
                <tr><td>{cls.MODE_DELETE_DISC}</td><td>Delete Disc</td></tr>
            </table>

            <h3>🔴🔵 Dedicated Annotation Modes</h3>
            <table>
                <tr><th>Shortcut</th><th>Action</th></tr>
                <tr><td>{cls.ANNOTATION_ARTERY}</td><td>Enter Artery Mode (show only artery mask)</td></tr>
                <tr><td>{cls.ANNOTATION_VEIN}</td><td>Enter Vein Mode (show only vein mask)</td></tr>
                <tr><td>{cls.ANNOTATION_EXIT}</td><td>Exit to Normal Mode</td></tr>
            </table>

            <h3>ℹ️ Help</h3>
            <table>
                <tr><th>Shortcut</th><th>Action</th></tr>
                <tr><td>{cls.SHOW_SHORTCUTS}</td><td>Show Shortcuts (This Window)</td></tr>
            </table>
            
            <h2>🖱️ Mouse Controls</h2>
            <table>
                <tr><th>Action</th><th>Description</th></tr>
                <tr><td><b>Left Click + Drag</b></td><td>Paint with current brush mode</td></tr>
                <tr><td><b>Ctrl + Left Click + Drag</b></td><td>Pan/Drag the image</td></tr>
                <tr><td><b>Right Click</b></td><td>Show context menu (brush modes)</td></tr>
                <tr><td><b>Scroll Wheel</b></td><td>Scroll vertically</td></tr>
                <tr><td><b>Shift + Scroll Wheel</b></td><td>Scroll horizontally</td></tr>
                <tr><td><b>Ctrl + Scroll Wheel</b></td><td>Zoom in/out</td></tr>
                <tr><td><b>Scroll Wheel</b></td><td>Change brush size</td></tr>
            </table>
            
            <h2>🎨 Brush Modes (Artery/Vein Task)</h2>
            <table>
                <tr><th>Mode</th><th>Description</th></tr>
                <tr><td><b>Modify Artery</b></td><td>Convert overlapping vein to artery</td></tr>
                <tr><td><b>Modify Vein</b></td><td>Convert overlapping artery to vein</td></tr>
                <tr><td><b>Draw Artery</b></td><td>Draw new artery pixels</td></tr>
                <tr><td><b>Draw Vein</b></td><td>Draw new vein pixels</td></tr>
                <tr><td><b>Delete Artery</b></td><td>Erase artery pixels only</td></tr>
                <tr><td><b>Delete Vein</b></td><td>Erase vein pixels only</td></tr>
                <tr><td><b>Delete All</b></td><td>Erase all vessel pixels</td></tr>
            </table>
            
            <h2>🔵 Brush Modes (Optic Disc Task)</h2>
            <table>
                <tr><th>Mode</th><th>Description</th></tr>
                <tr><td><b>Draw Disc</b></td><td>Draw optic disc region</td></tr>
                <tr><td><b>Delete Disc</b></td><td>Erase optic disc region</td></tr>
            </table>
            """

    # -----------------------------------------------------------------
    # Brush Settings
    # -----------------------------------------------------------------
    class Brush:
        """Brush configuration."""

        DEFAULT_SIZE = 5
        MIN_SIZE = 1
        MAX_SIZE = 100
        SINGLE_STEP = 1
        PAGE_STEP = 5
        CURSOR_PEN_WIDTH = 2
        SCROLL_STEP = 1  # Brush size change per scroll step with Alt

    # -----------------------------------------------------------------
    # Opacity Settings
    # -----------------------------------------------------------------
    class Opacity:
        """Opacity/alpha configuration."""

        DEFAULT = 0.5
        DEFAULT_PERCENT = 50
        MIN_PERCENT = 0
        MAX_PERCENT = 100

    # -----------------------------------------------------------------
    # Zoom Settings
    # -----------------------------------------------------------------
    class Zoom:
        """Zoom configuration."""

        DEFAULT = 1.0
        MIN = 0.05
        MAX = 20.0
        FACTOR = 1.1

    # -----------------------------------------------------------------
    # History Settings
    # -----------------------------------------------------------------
    class History:
        """Undo/Redo history configuration."""

        MAX_SIZE = 50
        COMPRESSION_LEVEL = 1

    # -----------------------------------------------------------------
    # Image Enhancement Settings
    # -----------------------------------------------------------------
    class Enhancement:
        """Image enhancement parameters."""

        GAUSSIAN_SIGMA = 50.0
        WEIGHT_ORIGINAL = 4.0
        WEIGHT_BLUR = -4.0
        OFFSET = 128.0
        MORPHOLOGY_CLOSE_ITERATIONS = 2
        MORPHOLOGY_OPEN_ITERATIONS = 1
        MORPHOLOGY_KERNEL_SIZE = 5
        FUNDUS_THRESHOLD = 3

    # -----------------------------------------------------------------
    # File Settings
    # -----------------------------------------------------------------
    class File:
        """File-related configuration."""

        VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
        IMAGES_FOLDER = "images"
        OUTPUT_FORMAT = ".png"
        SETTING_FILE = "app_settings.yaml"

    # -----------------------------------------------------------------
    # UI Settings
    # -----------------------------------------------------------------
    class UI:
        """UI-related configuration."""

        SHORTCUTS_DIALOG_MIN_WIDTH = 500
        SHORTCUTS_DIALOG_MIN_HEIGHT = 600
        STATUS_FONT_SIZE = "10px"

        # Stroke preview settings
        STROKE_POINTS_THRESHOLD_STEP_2 = 500
        STROKE_POINTS_THRESHOLD_STEP_3 = 1000
        MIN_DISPLAY_RADIUS = 0.5
