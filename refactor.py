import os

with open('main.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

def write_file(name, start, end, imports):
    content = imports + ''.join(lines[start:end])
    with open(name, 'w', encoding='utf-8') as f:
        f.write(content)

# models.py
models_imports = ''
write_file('models.py', 139, 156, models_imports)

# ui_components.py
ui_imports = """from PySide6.QtWidgets import QPushButton
from PySide6.QtGui import QPalette, QColor, QCursor
from PySide6.QtCore import Qt, QPropertyAnimation, QEasingCurve, QSize
from styles import get_application_stylesheet, get_modern_button_style\n\n"""
write_file('ui_components.py', 27, 138, ui_imports)

# split_view.py
split_view_imports = """from PySide6.QtWidgets import QGraphicsScene, QGraphicsView, QGraphicsLineItem
from PySide6.QtGui import QPainter, QPen, QColor, QCursor, QBrush
from PySide6.QtCore import Qt, QPointF, QRectF, Signal
from models import LineAction\n\n"""
write_file('split_view.py', 816, 1394, split_view_imports)

# split_dialog.py
split_dialog_imports = """from PySide6.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel, 
                           QHBoxLayout, QDialog, QSlider, QTabWidget, 
                           QSplitter, QFrame, QApplication, QColorDialog)
from PySide6.QtGui import QColor
from PySide6.QtCore import Qt
from styles import (get_splitter_style, get_tab_widget_style, get_slider_style, 
                   get_button_style, get_undo_redo_button_style, get_color_button_style)
from split_view import ManualSplitView\n\n"""
write_file('split_dialog.py', 1395, 1887, split_dialog_imports)

# main_window.py
main_window_imports = """import os
import sys
import json
import random
import shutil
import gc
import time
from io import BytesIO
from datetime import datetime
from PIL import Image

from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QPushButton, 
                           QFileDialog, QLabel, QMessageBox, QHBoxLayout, 
                           QProgressDialog, QProgressBar, QGroupBox, QLineEdit, 
                           QSlider, QComboBox, QTabWidget, QStatusBar, QStyle, 
                           QApplication)
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt, QSettings

from ui_components import ModernStyle, ModernButton
from split_dialog import ManualSplitDialog

# ปิดคำเตือน decompression bomb สำหรับภาพขนาดใหญ่
Image.MAX_IMAGE_PIXELS = None\n\n"""
write_file('main_window.py', 157, 815, main_window_imports)

# main.py
main_imports = """import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QImageReader
from main_window import ManualImageSplitter
from ui_components import ModernStyle\n\n"""
write_file('main.py', 1888, 1901, main_imports)
print('Refactor completed using Python script!')
