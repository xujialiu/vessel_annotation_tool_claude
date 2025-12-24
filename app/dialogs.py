"""
Dialog windows for the vessel annotation tool.

Provides shortcuts help dialog and other UI dialogs.
"""

from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QTextBrowser,
    QDialogButtonBox,
)

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AppConfig import AppConfig


class ShortcutsDialog(QDialog):
    """Dialog to display all keyboard shortcuts."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Keyboard Shortcuts")
        self.setMinimumSize(
            AppConfig.UI.SHORTCUTS_DIALOG_MIN_WIDTH,
            AppConfig.UI.SHORTCUTS_DIALOG_MIN_HEIGHT,
        )
        self.setModal(True)

        layout = QVBoxLayout(self)

        text_browser = QTextBrowser()
        text_browser.setOpenExternalLinks(False)
        text_browser.setHtml(AppConfig.Shortcut.get_shortcuts_html())
        layout.addWidget(text_browser)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok)
        button_box.accepted.connect(self.accept)
        layout.addWidget(button_box)
