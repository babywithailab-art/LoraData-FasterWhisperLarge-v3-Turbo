
import sys
import os

# Create environment variable to fix the DLL issue
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import threading
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QTextEdit, QFileDialog,
    QComboBox, QMessageBox, QProgressBar, QSpinBox, QDoubleSpinBox
)
from PySide6.QtCore import Signal, QObject, Qt

import trainer_utils
import merge_utils

# --- Worker Signals for Thread Safety ---
class WorkerSignals(QObject):
    log = Signal(str)
    finished = Signal()
    error = Signal(str)

# --- Stdout/Stderr Redirector ---
class LogRedirector:
    def __init__(self, signal):
        self.signal = signal

    def write(self, text):
        if text.strip():
            self.signal.emit(text.strip())

    def flush(self):
        pass

# --- Main Window ---
class LoRAGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ThinkSub2 - LoRA Trainer & Converter")
        self.resize(800, 600)
        
        # Central Widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.layout = QVBoxLayout(central_widget)

        # 1. Manifest Selection
        manifest_layout = QHBoxLayout()
        self.manifest_input = QLineEdit()
        self.manifest_input.setPlaceholderText("Select manifest.jsonl file or directory...")
        manifest_btn = QPushButton("Browse Manifest")
        manifest_btn.clicked.connect(self.browse_manifest)
        manifest_layout.addWidget(QLabel("Manifest:"))
        manifest_layout.addWidget(self.manifest_input)
        manifest_layout.addWidget(manifest_btn)
        self.layout.addLayout(manifest_layout)

        # 2. Output Directory
        output_layout = QHBoxLayout()
        self.output_input = QLineEdit()
        self.output_input.setText(os.path.abspath("./lora-output"))
        output_btn = QPushButton("Browse Output")
        output_btn.clicked.connect(self.browse_output)
        output_layout.addWidget(QLabel("Output Dir:"))
        output_layout.addWidget(self.output_input)
        output_layout.addWidget(output_btn)
        self.layout.addLayout(output_layout)

        # 3. Unified Action Button (below Browse Output)
        self.btn_run_all = QPushButton("▶ Train → Merge → Convert")
        self.btn_run_all.setMinimumHeight(40)
        self.btn_run_all.setStyleSheet("font-size: 14px; font-weight: bold;")
        self.btn_run_all.clicked.connect(self.start_full_pipeline)
        self.layout.addWidget(self.btn_run_all)

        # 4. Settings (Language, Base Model - hidden/fixed for now)
        settings_layout = QHBoxLayout()
        
        # Language
        settings_layout.addWidget(QLabel("Lang:"))
        self.lang_input = QComboBox()
        self.lang_input.setEditable(True)
        self.lang_input.addItems(["ko", "en", "ja", "zh"]) # Common defaults
        self.lang_input.setCurrentText("ko")
        settings_layout.addWidget(self.lang_input)

        # Helper Button
        lang_help_btn = QPushButton("H")
        lang_help_btn.setFixedWidth(20)
        lang_help_btn.setToolTip("Show all supported language codes")
        lang_help_btn.clicked.connect(self.show_lang_codes)
        settings_layout.addWidget(lang_help_btn)

        # LoRA Rank (r)
        settings_layout.addWidget(QLabel(" Rank:"))
        self.rank_input = QComboBox()
        self.rank_input.addItems(["8", "16", "32"])
        self.rank_input.setCurrentText("32")
        settings_layout.addWidget(self.rank_input)

        # LoRA Dropout
        settings_layout.addWidget(QLabel(" Dropout:"))
        self.dropout_input = QDoubleSpinBox()
        self.dropout_input.setRange(0.00, 0.10)
        self.dropout_input.setSingleStep(0.01)
        self.dropout_input.setValue(0.05)
        self.dropout_input.setDecimals(2)
        settings_layout.addWidget(self.dropout_input)

        # Epochs
        settings_layout.addWidget(QLabel(" Epochs:"))
        self.epochs_input = QSpinBox()
        self.epochs_input.setRange(2, 10)
        self.epochs_input.setValue(3)
        settings_layout.addWidget(self.epochs_input)

        # Spacer
        settings_layout.addStretch()
        self.layout.addLayout(settings_layout)

        # 5. Log Area
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.layout.addWidget(QLabel("Logs:"))
        self.layout.addWidget(self.log_area)

        # State
        self.is_running = False
        self.signals = WorkerSignals()
        self.signals.log.connect(self.append_log)
        self.signals.finished.connect(self.on_task_finished)
        self.signals.error.connect(self.on_task_error)

    def append_log(self, text):
        self.log_area.append(text)
        # Auto scroll
        sb = self.log_area.verticalScrollBar()
        sb.setValue(sb.maximum())

    def show_lang_codes(self):
        codes = (
            "af, am, ar, as, az, ba, be, bg, bn, bo, br, bs, ca, cs, cy, da, de, el, en, es, et, eu, "
            "fa, fi, fo, fr, gl, gu, ha, haw, he, hi, hr, ht, hu, hy, id, is, it, ja, jw, ka, kk, km, kn, "
            "ko, la, lb, ln, lo, lt, lv, mg, mi, mk, ml, mn, mr, ms, mt, my, ne, nl, nn, no, oc, pa, pl, "
            "ps, pt, ro, ru, sa, sd, si, sk, sl, sn, so, sq, sr, su, sv, sw, ta, te, tg,th, tk, tl, tr, "
            "tt, uk, ur, uz, vi, yi, yo, yue, zh"
        )
        QMessageBox.information(self, "Supported Languages", codes)

    def browse_manifest(self):
        # Allow selecting file or directory (User request: file or folder path)
        # QFileDialog doesn't easily support both "File or Folder" in one native dialog usually.
        # We will implement a custom logic: Try to open file, if user cancels, ask if they want to select folder?
        # Or just provide a simple file picker for now as 'manifest.jsonl' is specific.
        # Let's support both via a small dialog choice or just file picker which is safer.
        # Impl: File picker. If user wants folder, they can modify text manually or pick a file inside.
        path, _ = QFileDialog.getOpenFileName(self, "Select Manifest File", "", "JSONL Files (*.jsonl);;All Files (*)")
        if path:
            self.manifest_input.setText(path)

    def browse_output(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if path:
            self.output_input.setText(path)

    def toggle_buttons(self, enable):
        self.btn_run_all.setEnabled(enable)
        self.is_running = not enable

    # --- Full Pipeline ---

    def start_full_pipeline(self):
        """Run Train → Merge → Convert sequentially"""
        if self.is_running: return
        manifest = self.manifest_input.text().strip()
        output_dir = self.output_input.text().strip()
        lang = self.lang_input.currentText().strip()
        
        # Get Hyperparams
        try:
            rank = int(self.rank_input.currentText())
            dropout = self.dropout_input.value()
            epochs = self.epochs_input.value()
        except ValueError:
            rank = 32
            dropout = 0.05
            epochs = 3

        if not manifest or not os.path.exists(manifest):
            QMessageBox.warning(self, "Error", "Invalid manifest path.")
            return

        self.toggle_buttons(False)
        self.append_log("=" * 50)
        self.append_log(f"Starting Full Pipeline: Train → Merge → Convert")
        self.append_log(f"Params: Rank={rank}, Dropout={dropout}, Epochs={epochs}, Lang={lang}")
        self.append_log("=" * 50)
        
        threading.Thread(target=self.run_full_pipeline, args=(manifest, output_dir, lang, rank, dropout, epochs), daemon=True).start()

    def run_full_pipeline(self, manifest, output_dir, lang, rank, dropout, epochs):
        try:
            # Step 1: Train LoRA
            self.signals.log.emit("\n--- Step 1/3: Training LoRA ---")
            trainer_utils.train_lora(
                manifest_path=manifest,
                output_dir=output_dir,
                language=lang,
                num_train_epochs=epochs,
                lora_r=rank,
                lora_dropout=dropout,
                progress_callback=self.signals.log.emit
            )
            self.signals.log.emit("✓ Training complete!")

            # Step 2: Merge LoRA
            self.signals.log.emit("\n--- Step 2/3: Merging LoRA ---")
            import torch
            from transformers import WhisperForConditionalGeneration, WhisperProcessor
            from peft import PeftModel
            
            base_model = "openai/whisper-large-v3-turbo"
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            base = WhisperForConditionalGeneration.from_pretrained(base_model, device_map=device)
            processor = WhisperProcessor.from_pretrained(base_model)
            
            model = PeftModel.from_pretrained(base, output_dir)
            model = model.merge_and_unload()
            model.eval()
            
            hf_save_path = os.path.join(output_dir, "hf_merged_model")
            model.save_pretrained(hf_save_path)
            processor.save_pretrained(hf_save_path)
            self.signals.log.emit(f"✓ Merged model saved to: {hf_save_path}")
            
            # Free memory
            del model, base
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            # Step 3: Convert to CT2
            self.signals.log.emit("\n--- Step 3/3: Converting to CT2 ---")
            import ctranslate2
            
            ct2_dir = os.path.join(output_dir, "ct2_model")
            converter = ctranslate2.converters.TransformersConverter(
                model_name_or_path=hf_save_path,
                copy_files=["tokenizer.json", "preprocessor_config.json"]
            )
            converter.convert(
                output_dir=ct2_dir,
                quantization="float16",
                force=True
            )
            self.signals.log.emit(f"✓ CT2 model saved to: {ct2_dir}")
            
            self.signals.log.emit("\n" + "=" * 50)
            self.signals.log.emit("🎉 Full Pipeline Complete!")
            self.signals.log.emit("=" * 50)
            self.signals.finished.emit()
            
        except Exception as e:
            self.signals.error.emit(str(e))

    # --- Individual Operations (kept for reference but not used in UI) ---

    def start_training(self):
        if self.is_running: return
        manifest = self.manifest_input.text().strip()
        output_dir = self.output_input.text().strip()
        lang = self.lang_input.currentText().strip()

        if not manifest or not os.path.exists(manifest):
            QMessageBox.warning(self, "Error", "Invalid manifest path.")
            return

        self.toggle_buttons(False)
        self.append_log(f"--- Starting Training ---\nManifest: {manifest}\nLang: {lang}\n")
        
        threading.Thread(target=self.run_training, args=(manifest, output_dir, lang), daemon=True).start()

    def run_training(self, manifest, output, lang):
        try:
            # We can override defaults here
            trainer_utils.train_lora(
                manifest_path=manifest,
                output_dir=output, # This is where adapter will be saved
                language=lang,
                progress_callback=self.signals.log.emit
            )
            self.signals.log.emit("Training finished successfully.")
            self.signals.finished.emit()
        except Exception as e:
            self.signals.error.emit(str(e))

    def start_merging(self):
        if self.is_running: return
        lora_path = self.output_input.text().strip()
        
        if not os.path.exists(lora_path):
            QMessageBox.warning(self, "Error", f"LoRA path does not exist: {lora_path}")
            return
            
        self.toggle_buttons(False)
        self.append_log(f"--- Starting Merge ---\nLoRA Path: {lora_path}\n")
        
        threading.Thread(target=self.run_merging, args=(lora_path,), daemon=True).start()

    def run_merging(self, lora_path):
        # We need to verify if we are merging AND converting or just merging?
        # The user conceptual model: "Merge -> HF Model", "CT2 Convert -> CT2 Model"
        # My previous script did both. Let's make this button just do the Merge part logically, 
        # OR we can keep the simple flow where "Merge" creates the temp HF model.
        # But wait, `merge_utils.merge_and_convert_model` does both. 
        # Let's actually separate them or just have one "Merge & Convert" button? 
        # User requested separate buttons: "Lora병합버튼", "CT2변환".
        # So I should split the util logic or handle it via flags.
        
        # Actually, `merge_and_unload` is in-memory. If we stop there, we have to save the full HF model.
        # That's fine.
        
        try:
            # We will use a modified approach to just Merge and Save HF
            # For simplicity, I will reuse the util but I might need to refactor it if strict separation is needed.
            # But wait, looking at my merge_utils, it does both. 
            # I will assume "Merge" button does the HF Merge, and "CT2" button does the Conversion from that HF Merge.
            # This requires persisting the intermediate HF model.
            
            # Let's implement a 'merge_only' function in `gui_trainer` by repurposing `merge_utils` code inline or modifying `merge_utils`.
            # Modifying `merge_utils` is better. But I already wrote it. 
            # I will just run the FULL merge_and_convert for the "Merge" button? 
            # No, user asked for separate buttons.
            # "병합(merge) → HF 단일 모델 저장"
            # "CT2 변환 → out/ct2_model 생성"
            
            # I will implement `run_merging` to call `merge_utils.merge_and_convert_model` but maybe I should just make `merge_utils` cleaner.
            # Let's just implement the logic here using the util imports.
            
            base_model = "openai/whisper-large-v3-turbo"
            
            self.signals.log.emit("Loading base model & LoRA...")
            # For this quick implementation, I'll do it in one go if the user clicks "Convert", 
            # but if they click "Merge", I'll just save the HF model.
            
            # Let's actually update `merge_utils` to support separate steps if called differently, 
            # OR just implement the specific logic here. 
            # I'll implement it here to avoid rewriting the file I just finished.
            
            import torch
            from transformers import WhisperForConditionalGeneration, WhisperProcessor
            from peft import PeftModel
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Load Base
            base = WhisperForConditionalGeneration.from_pretrained(base_model, device_map=device)
            processor = WhisperProcessor.from_pretrained(base_model)
            
            # Load LoRA
            model = PeftModel.from_pretrained(base, lora_path)
            model = model.merge_and_unload()
            model.eval()
            
            save_path = os.path.join(lora_path, "hf_merged_model")
            model.save_pretrained(save_path)
            processor.save_pretrained(save_path)
            
            self.signals.log.emit(f"Merged model saved to: {save_path}")
            self.signals.finished.emit()
            
        except Exception as e:
            self.signals.error.emit(str(e))

    def start_conversion(self):
        if self.is_running: return
        # Input is the MERGED HF MODEL, not the LoRA adapter.
        # Ideally the user points to the output dir where `hf_merged_model` is.
        
        target_dir = self.output_input.text().strip()
        hf_model_path = os.path.join(target_dir, "hf_merged_model")
        
        if not os.path.exists(hf_model_path):
             QMessageBox.warning(self, "Error", f"Merged HF model not found at: {hf_model_path}\nPlease run Merge first.")
             return

        self.toggle_buttons(False)
        self.append_log(f"--- Starting Conversion ---\nInput: {hf_model_path}\n")
        
        threading.Thread(target=self.run_conversion, args=(hf_model_path, target_dir), daemon=True).start()

    def run_conversion(self, hf_path, output_root):
        try:
            ct2_dir = os.path.join(output_root, "ct2_model")
            import ctranslate2
            
            converter = ctranslate2.converters.TransformersConverter(
                model_name_or_path=hf_path,
                copy_files=["tokenizer.json", "preprocessor_config.json"]
            )
            converter.convert(
                output_dir=ct2_dir,
                quantization="float16",
                force=True
            )
            
            self.signals.log.emit(f"Converted CT2 model saved to: {ct2_dir}")
            self.signals.finished.emit()
        except Exception as e:
            self.signals.error.emit(str(e))

    def on_task_finished(self):
        self.toggle_buttons(True)
        QMessageBox.information(self, "Done", "Task Completed Successfully!")

    def on_task_error(self, err_msg):
        self.toggle_buttons(True)
        self.append_log(f"ERROR: {err_msg}")
        QMessageBox.critical(self, "Error", err_msg)

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    
    app = QApplication(sys.argv)
    window = LoRAGUI()

    # Redirect stdout/stderr to GUI if running as frozen EXE
    if getattr(sys, 'frozen', False):
        sys.stdout = LogRedirector(window.signals.log)
        sys.stderr = LogRedirector(window.signals.log)

    window.show()
    sys.exit(app.exec())
