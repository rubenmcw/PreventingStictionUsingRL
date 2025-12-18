# account_master.py
# Account Master v0.0 (Alpha)
# GUI to ingest an input file and export to .xlsx.
# If the target .xlsx exists, new sheets are appended instead of overwriting.

import os
import sys
import json
import traceback
from datetime import datetime

# Third-party
import pandas as pd
from openpyxl import load_workbook

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, Signal, Slot


# ------------------------- Utilities -------------------------

INVALID_SHEET_CHARS = set(r'[]:*?/\\')

def sanitize_sheet_name(name: str) -> str:
    """Remove invalid Excel sheet chars and trim whitespace; never return empty."""
    name = "".join(ch for ch in name if ch not in INVALID_SHEET_CHARS).strip()
    name = name.replace("\n", " ").replace("\r", " ")
    if not name:
        name = "Sheet"
    return name

def truncate_sheet_name(name: str, reserve: int = 0) -> str:
    """Excel sheet names must be <= 31 chars. Reserve space for suffix if needed."""
    max_len = max(0, 31 - reserve)
    return name[:max_len]

def make_unique_sheet_name(base: str, existing: set) -> str:
    """
    Make a sheet name unique among `existing`.
    Respect Excel's 31-char max, adding ' (2)', ' (3)', ...
    """
    base = sanitize_sheet_name(base)
    base = truncate_sheet_name(base)
    if base not in existing:
        return base

    # Reserve space for suffix like " (10)"
    for n in range(2, 10_000):
        suffix = f" ({n})"
        candidate = truncate_sheet_name(base, reserve=len(suffix)) + suffix
        if candidate not in existing:
            return candidate
    # Fallback (should never happen)
    return truncate_sheet_name(f"{base}_{datetime.now():%H%M%S%f}")

def derive_default_prefix(input_path: str) -> str:
    stem = os.path.splitext(os.path.basename(input_path or ""))[0]
    return sanitize_sheet_name(stem or "Data")

def read_input_file(input_path: str) -> dict:
    """
    Load the input file into one or more DataFrames.
    Returns a dict: {proposed_sheet_name: DataFrame}.
    Supports CSV, XLS(X), JSON (incl. JSON Lines), and generic delimited text.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    ext = os.path.splitext(input_path)[1].lower()
    sheet_map = {}

    if ext in {".xls", ".xlsx"}:
        # Read all sheets
        xls = pd.read_excel(input_path, sheet_name=None)
        for sheet_name, df in xls.items():
            sheet_map[sanitize_sheet_name(sheet_name or "Sheet")] = df

    elif ext == ".csv":
        # Try UTF-8; fallback to latin-1 to handle odd encodings
        try:
            df = pd.read_csv(input_path, encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(input_path, encoding="latin1")
        sheet_map["Data"] = df

    elif ext == ".json":
        # Try standard JSON; fall back to JSON Lines
        try:
            # Attempt to read as a standard JSON file
            with open(input_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
            try:
                obj = json.loads(content)
                df = pd.json_normalize(obj)
            except json.JSONDecodeError:
                # JSON Lines
                df = pd.read_json(input_path, lines=True)
            sheet_map["Data"] = df
        except Exception as e:
            raise ValueError(f"Could not parse JSON: {e}")

    else:
        # Generic delimited fallback (tab, semicolon, etc.)
        tried = []
        for sep in [",", "\t", ";", "|", "^", "~"]:
            try:
                df = pd.read_csv(input_path, sep=sep, engine="python")
                sheet_map["Data"] = df
                break
            except Exception as e:
                tried.append(f"{sep}: {e}")
        if not sheet_map:
            # If all fails, write a minimal metadata sheet
            sheet_map["FileInfo"] = pd.DataFrame(
                [{"path": input_path, "size_bytes": os.path.getsize(input_path)}]
            )

    return sheet_map


# ------------------------- Worker Thread -------------------------

class ExportWorker(QtCore.QObject):
    started = Signal()
    log = Signal(str)
    error = Signal(str)
    success = Signal(str)
    finished = Signal()

    def __init__(self, input_path: str, output_path: str, sheet_prefix: str):
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path
        self.sheet_prefix = sheet_prefix

    @Slot()
    def run(self):
        self.started.emit()
        try:
            self.log.emit(f"Reading: {self.input_path}")
            raw_map = read_input_file(self.input_path)

            # Compose candidate sheet names using prefix
            self.log.emit("Preparing sheet names...")
            pref = sanitize_sheet_name(self.sheet_prefix or derive_default_prefix(self.input_path))

            if os.path.exists(self.output_path):
                self.log.emit(f"Existing workbook found: {self.output_path}")
                try:
                    wb = load_workbook(self.output_path, read_only=False)
                    existing = {ws.title for ws in wb.worksheets}
                except Exception:
                    # If workbook can't be opened (locked, corrupted)
                    existing = set()
            else:
                existing = set()

            # Build final map with unique sheet names
            final_map = {}
            # If input had multiple sheets: prefix + "_" + input_sheet_name
            multi_input = len(raw_map) > 1
            for in_name, df in raw_map.items():
                proposed = pref if not multi_input else f"{pref}_{sanitize_sheet_name(in_name)}"
                unique = make_unique_sheet_name(proposed, existing | set(final_map.keys()))
                final_map[unique] = df

            # Ensure .xlsx extension
            xlsx_path = self.output_path
            if not xlsx_path.lower().endswith(".xlsx"):
                xlsx_path += ".xlsx"

            # Write/append using pandas + openpyxl
            mode = "a" if os.path.exists(xlsx_path) else "w"
            self.log.emit(f"Writing to: {xlsx_path} (mode: {'append' if mode=='a' else 'create'})")

            # Important: do not rely on if_sheet_exists; we guarantee unique names
            with pd.ExcelWriter(xlsx_path, engine="openpyxl", mode=mode) as writer:
                for sheet_name, df in final_map.items():
                    # Convert non-tabular objects gracefully
                    try:
                        df = pd.DataFrame(df)
                    except Exception:
                        df = pd.DataFrame({"value": [str(df)]})
                    df.to_excel(writer, sheet_name=sheet_name, index=False)

            self.success.emit(
                "Export complete.\n"
                + "\n".join(f"  • {name}" for name in final_map.keys())
            )
        except PermissionError as e:
            self.error.emit(
                "Permission error: The output file may be open in another program.\n"
                f"Details: {e}"
            )
        except Exception as e:
            tb = traceback.format_exc()
            self.error.emit(f"Unexpected error: {e}\n\n{tb}")
        finally:
            self.finished.emit()


# ------------------------- UI Widgets -------------------------

class FileDropLineEdit(QtWidgets.QLineEdit):
    """QLineEdit that accepts file drops."""
    fileDropped = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dropEvent(self, event: QtGui.QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            self.setText(path)
            self.fileDropped.emit(path)
        super().dropEvent(event)


class AccountMasterWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Account Master v0.0 (Alpha)")
        self.setMinimumSize(760, 520)
        self.setWindowIcon(self._build_icon())

        # Central layout
        cw = QtWidgets.QWidget()
        main = QtWidgets.QVBoxLayout(cw)
        main.setContentsMargins(24, 24, 24, 24)
        main.setSpacing(18)

        # Title
        title = QtWidgets.QLabel("Account Master v0.0 (Alpha)")
        title.setObjectName("appTitle")
        subtitle = QtWidgets.QLabel("Import a file and export to Excel. "
                                    "If the workbook exists, new sheets are appended.")
        subtitle.setObjectName("subtitle")

        # Input group
        in_group = QtWidgets.QGroupBox("Input File")
        in_layout = QtWidgets.QGridLayout(in_group)
        self.input_edit = FileDropLineEdit()
        self.input_edit.setPlaceholderText("Drop a file here or click Browse…")
        self.input_edit.fileDropped.connect(self._on_input_path_changed)
        self.input_edit.textChanged.connect(self._on_input_path_changed)
        in_browse = QtWidgets.QPushButton("Browse…")
        in_browse.clicked.connect(self._choose_input)

        in_layout.addWidget(self.input_edit, 0, 0, 1, 1)
        in_layout.addWidget(in_browse, 0, 1, 1, 1)

        # Output group
        out_group = QtWidgets.QGroupBox("Output Workbook (.xlsx)")
        out_layout = QtWidgets.QGridLayout(out_group)
        self.output_edit = QtWidgets.QLineEdit()
        self.output_edit.setPlaceholderText("Choose where to save the .xlsx")
        out_browse = QtWidgets.QPushButton("Browse…")
        out_browse.clicked.connect(self._choose_output)

        # Sheet prefix
        self.prefix_edit = QtWidgets.QLineEdit()
        self.prefix_edit.setPlaceholderText("Sheet name prefix (optional)")
        prefix_label = QtWidgets.QLabel("Sheet prefix")

        out_layout.addWidget(self.output_edit, 0, 0, 1, 1)
        out_layout.addWidget(out_browse, 0, 1, 1, 1)
        out_layout.addWidget(prefix_label, 1, 0, 1, 1)
        out_layout.addWidget(self.prefix_edit, 1, 0, 1, 2)

        # Info label (append behavior)
        append_info = QtWidgets.QLabel("• If the target .xlsx exists, new sheets will be appended.")
        append_info.setObjectName("hint")

        # Buttons
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch(1)
        self.export_btn = QtWidgets.QPushButton("Process & Export")
        self.export_btn.setDefault(True)
        self.export_btn.clicked.connect(self._export)
        self.clear_btn = QtWidgets.QPushButton("Clear")
        self.clear_btn.clicked.connect(self._clear)
        self.exit_btn = QtWidgets.QPushButton("Exit")
        self.exit_btn.clicked.connect(self.close)
        btn_row.addWidget(self.clear_btn)
        btn_row.addWidget(self.export_btn)
        btn_row.addWidget(self.exit_btn)

        # Progress + log
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 1)  # will switch to busy when running
        self.progress.setValue(1)
        self.progress.setTextVisible(False)
        self.progress.setObjectName("progress")

        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setPlaceholderText("Status messages will appear here…")

        # Assemble
        main.addWidget(title)
        main.addWidget(subtitle)
        main.addWidget(in_group)
        main.addWidget(out_group)
        main.addWidget(append_info)
        main.addLayout(btn_row)
        main.addWidget(self.progress)
        main.addWidget(self.log_view, 1)

        self.setCentralWidget(cw)

        # Apply modern theme
        self._apply_theme()

        # Threading
        self.thread = None
        self.worker = None

    # ------------- UI Actions -------------

    def _choose_input(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Choose input file",
            "",
            "All Files (*);;Spreadsheets (*.xlsx *.xls *.csv);;JSON (*.json)")
        if path:
            self.input_edit.setText(path)

    def _choose_output(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Choose output .xlsx",
            self._suggest_output_path(),
            "Excel Workbook (*.xlsx)")
        if path:
            # Enforce .xlsx
            if not path.lower().endswith(".xlsx"):
                path += ".xlsx"
            self.output_edit.setText(path)

    def _suggest_output_path(self) -> str:
        in_path = self.input_edit.text().strip()
        if in_path:
            base = os.path.splitext(os.path.basename(in_path))[0]
            folder = os.path.dirname(in_path)
            return os.path.join(folder, f"{base}_AccountMaster.xlsx")
        return os.path.expanduser("~/AccountMaster.xlsx")

    def _on_input_path_changed(self, *_):
        path = self.input_edit.text().strip()
        if path and not self.output_edit.text().strip():
            self.output_edit.setText(self._suggest_output_path())
        if path and not self.prefix_edit.text().strip():
            self.prefix_edit.setText(derive_default_prefix(path))

    def _log(self, msg: str):
        self.log_view.appendPlainText(msg)

    def _clear(self):
        self.input_edit.clear()
        self.output_edit.clear()
        self.prefix_edit.clear()
        self.log_view.clear()

    def _export(self):
        input_path = self.input_edit.text().strip()
        output_path = self.output_edit.text().strip()
        sheet_prefix = self.prefix_edit.text().strip()

        if not input_path:
            QtWidgets.QMessageBox.warning(self, "Missing input", "Please choose an input file.")
            return

        if not os.path.exists(input_path):
            QtWidgets.QMessageBox.warning(self, "Input not found", "The selected input file does not exist.")
            return

        if not output_path:
            output_path = self._suggest_output_path()
            self.output_edit.setText(output_path)

        # Ensure folder exists
        out_dir = os.path.dirname(output_path) or "."
        if not os.path.isdir(out_dir):
            try:
                os.makedirs(out_dir, exist_ok=True)
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Folder error", f"Cannot create output folder:\n{e}")
                return

        # Busy indicator
        self.progress.setRange(0, 0)  # busy
        self.export_btn.setEnabled(False)
        self.clear_btn.setEnabled(False)

        # Start worker thread
        self.thread = QtCore.QThread(self)
        self.worker = ExportWorker(input_path, output_path, sheet_prefix)
        self.worker.moveToThread(self.thread)

        # Connect signals
        self.thread.started.connect(self.worker.run)
        self.worker.started.connect(lambda: self._log("Starting export…"))
        self.worker.log.connect(self._log)
        self.worker.error.connect(self._on_error)
        self.worker.success.connect(self._on_success)
        self.worker.finished.connect(self._on_finished)

        # Cleanup
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    def _on_error(self, msg: str):
        self._log(msg)
        QtWidgets.QMessageBox.critical(self, "Error", msg)

    def _on_success(self, msg: str):
        self._log(msg)
        QtWidgets.QMessageBox.information(self, "Done", msg)

    def _on_finished(self):
        self.progress.setRange(0, 1)  # not busy
        self.progress.setValue(1)
        self.export_btn.setEnabled(True)
        self.clear_btn.setEnabled(True)

    # ------------- Theming -------------

    def _apply_theme(self):
        # Use Fusion style for consistency across platforms
        QtWidgets.QApplication.setStyle("Fusion")

        # Modern dark palette + stylesheet
        palette = QtGui.QPalette()
        bg = QtGui.QColor(17, 24, 39)      # slate-900
        panel = QtGui.QColor(30, 41, 59)   # slate-800
        text = QtGui.QColor(226, 232, 240) # slate-200
        accent = QtGui.QColor(79, 70, 229) # indigo-600

        palette.setColor(QtGui.QPalette.Window, bg)
        palette.setColor(QtGui.QPalette.Base, panel)
        palette.setColor(QtGui.QPalette.AlternateBase, bg)
        palette.setColor(QtGui.QPalette.Text, text)
        palette.setColor(QtGui.QPalette.WindowText, text)
        palette.setColor(QtGui.QPalette.ButtonText, text)
        palette.setColor(QtGui.QPalette.Button, panel)
        palette.setColor(QtGui.QPalette.Highlight, accent)
        palette.setColor(QtGui.QPalette.HighlightedText, Qt.white)
        self.setPalette(palette)

        self.setStyleSheet("""
            QWidget { font-family: "Segoe UI", "Inter", "Helvetica Neue", Arial; font-size: 12.5pt; color: #E2E8F0; }
            #appTitle { font-size: 20pt; font-weight: 700; padding: 2px 0 6px 0; }
            #subtitle { color: #9CA3AF; font-size: 10.5pt; padding-bottom: 8px; }
            #hint { color: #94A3B8; font-size: 10pt; }

            QGroupBox {
                border: 1px solid #334155;
                border-radius: 10px;
                margin-top: 14px;
                padding: 12px 12px 10px 12px;
                background-color: #0F172A;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                top: 2px;
                padding: 0 4px;
                background-color: transparent;
                color: #A5B4FC;
                font-weight: 600;
            }

            QLineEdit, QPlainTextEdit {
                border: 1px solid #334155;
                border-radius: 8px;
                padding: 8px 10px;
                background-color: #111827;
                selection-background-color: #4F46E5;
                selection-color: white;
            }
            QLineEdit:focus, QPlainTextEdit:focus {
                border: 1px solid #6366F1;
                box-shadow: 0 0 0 3px rgba(79,70,229,0.35);
            }

            QPushButton {
                border: 1px solid #475569;
                border-radius: 10px;
                padding: 8px 16px;
                background-color: #1F2937;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #273548;
            }
            QPushButton:pressed {
                background-color: #334155;
            }
            QPushButton:default {
                border: 1px solid #6366F1;
                background-color: #303469;
            }

            QProgressBar {
                border: 1px solid #334155;
                border-radius: 8px;
                background: #0F172A;
                min-height: 10px;
            }
            QProgressBar::chunk {
                background-color: #4F46E5;
                margin: 1px;
                border-radius: 7px;
            }

            QLabel { background-color: transparent; }
        """)

    def _build_icon(self) -> QtGui.QIcon:
        # Simple in-memory vector icon (diamond) for a clean look
        svg = """
        <svg width="256" height="256" viewBox="0 0 256 256" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0%" stop-color="#8B5CF6"/><stop offset="100%" stop-color="#4F46E5"/>
            </linearGradient>
          </defs>
          <rect width="256" height="256" rx="56" ry="56" fill="#0F172A"/>
          <path d="M128 28 L228 128 L128 228 L28 128 Z" fill="url(#g)" opacity="0.95"/>
          <text x="128" y="150" text-anchor="middle" font-size="40" fill="#E2E8F0" font-family="Segoe UI, Arial" >AM</text>
        </svg>
        """.strip().encode("utf-8")
        pixmap = QtGui.QPixmap()
        pixmap.loadFromData(svg, "SVG")
        return QtGui.QIcon(pixmap)


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = AccountMasterWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
