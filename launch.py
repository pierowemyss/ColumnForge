#!/usr/bin/env python3
"""
FreeColumn - Column Solver GUI Launcher

Launch script for the FreeColumn column solver application.
Provides a clean entry point to the PySide6-based GUI.
"""

import sys
import os

# Set up Python path to include our modules. `src/python` exposes core/gui;
# `src` exposes side_features (e.g. side_features.bvm).
sys.path.insert(0, os.path.abspath('src'))
sys.path.insert(0, os.path.abspath('src/python'))

def main():
    """Launch the FreeColumn GUI application."""
    try:
        from gui.main_window import MainWindow, main
        main()
    except ImportError as e:
        print(f"Error importing GUI module: {e}")
        print("Make sure PySide6 is installed: pip install PySide6")
        sys.exit(1)
    except Exception as e:
        print(f"Error launching FreeColumn: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()