# deepnova_gui.py

import sys
import os
import threading
import time
import json
import re
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QTextEdit, QLineEdit, QPushButton, QLabel, QTabWidget,
        QGroupBox, QProgressBar, QTableWidget, QTableWidgetItem,
        QHeaderView, QFileDialog, QMessageBox, QSpinBox, QDoubleSpinBox,
        QCheckBox, QComboBox, QSlider, QFrame, QStatusBar,
        QMenuBar, QMenu, QListWidget, QListWidgetItem,
        QPlainTextEdit, QDialog, QDialogButtonBox, QFormLayout,
        QGridLayout, QSplitter
    )
    from PyQt6.QtCore import (
        Qt, QThread, pyqtSignal, QTimer, QSize
    )
    from PyQt6.QtGui import (
        QFont, QTextCursor, QColor, QAction
    )
except ImportError:
    print("Installing PyQt6...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "PyQt6"])
    from PyQt6.QtWidgets import *
    from PyQt6.QtCore import *
    from PyQt6.QtGui import *

from model import (
    DeepNovaAI, ModelArgs, Transformer, ProductionTokenizer,
    load_model, create_model, get_device, get_memory_info,
    cleanup_memory, logger
)

try:
    import requests
    from bs4 import BeautifulSoup
    HAS_SCRAPING = True
except ImportError:
    HAS_SCRAPING = False


class TrainingThread(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(bool, str)
    log_signal = pyqtSignal(str)
    epoch_end = pyqtSignal(int, float)
    
    def __init__(self, model, tokenizer, args, data_path: str, epochs: int = 1):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.data_path = data_path
        self.epochs = epochs
        self.is_running = True
        
    def run(self):
        try:
            self.log_signal.emit("Starting training...")
            
            from model import TextDataset, Trainer, TrainingArgs
            
            self.log_signal.emit(f"Loading data from: {self.data_path}")
            dataset = TextDataset(
                self.data_path,
                self.tokenizer,
                max_seq_len=self.args.max_seq_length,
                min_seq_len=self.args.min_seq_length
            )
            
            self.log_signal.emit(f"Loaded {len(dataset)} samples")
            
            train_args = TrainingArgs(
                epochs=self.epochs,
                train_batch_size=self.args.batch_size,
                learning_rate=self.args.learning_rate
            )
            
            trainer = Trainer(self.model, self.args, train_args, self.tokenizer)
            
            from torch.utils.data import DataLoader
            dataloader = DataLoader(
                dataset, 
                batch_size=self.args.batch_size,
                shuffle=True,
                num_workers=0
            )
            
            total_steps = len(dataloader) * self.epochs
            current_step = 0
            
            for epoch in range(self.epochs):
                if not self.is_running:
                    break
                
                self.log_signal.emit(f"\nEpoch {epoch + 1}/{self.epochs}")
                epoch_loss = 0.0
                
                for step, batch in enumerate(dataloader):
                    if not self.is_running:
                        break
                    
                    metrics = trainer.train_step(batch)
                    epoch_loss += metrics['loss']
                    
                    current_step += 1
                    progress = int(current_step / total_steps * 100)
                    self.progress.emit(progress, f"Epoch {epoch+1}/{self.epochs} - Step {step+1}/{len(dataloader)}")
                    
                    if step % 10 == 0:
                        self.log_signal.emit(
                            f"  Step {step}: loss={metrics['loss']:.4f}, lr={metrics['lr']:.2e}"
                        )
                
                avg_loss = epoch_loss / len(dataloader)
                self.epoch_end.emit(epoch + 1, avg_loss)
                self.log_signal.emit(f"  Completed epoch {epoch + 1} - Avg loss: {avg_loss:.4f}")
                
                if (epoch + 1) % 5 == 0:
                    trainer.save_checkpoint(f"./checkpoints/epoch_{epoch+1}")
                    self.log_signal.emit(f"  Saved checkpoint at epoch_{epoch+1}")
            
            if self.is_running:
                self.finished.emit(True, "Training completed successfully!")
            else:
                self.finished.emit(False, "Training stopped")
            
        except Exception as e:
            self.finished.emit(False, f"Training failed: {str(e)}")
    
    def stop(self):
        self.is_running = False


class WebLearningThread(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(bool, str, list)
    log_signal = pyqtSignal(str)
    content_learned = pyqtSignal(str, str)
    
    def __init__(self, deepnova, urls: List[str], max_depth: int = 1):
        super().__init__()
        self.deepnova = deepnova
        self.urls = urls
        self.max_depth = max_depth
        self.is_running = True
        self.learned_urls = []
        
    def _extract_text_from_url(self, url: str) -> Optional[str]:
        if not HAS_SCRAPING:
            return None
            
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            response.encoding = 'utf-8'
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
                element.decompose()
            
            title = soup.find('title')
            title_text = title.get_text().strip() if title else ""
            
            content_selectors = ['article', 'main', '.content', '#content', '.post', '.entry']
            main_content = None
            
            for selector in content_selectors:
                main_content = soup.select_one(selector)
                if main_content:
                    break
            
            if not main_content:
                main_content = soup.body if soup.body else soup
            
            text = main_content.get_text(separator='\n', strip=True)
            
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            if title_text:
                text = f"Title: {title_text}\n\n{text}"
            
            return text
            
        except Exception as e:
            self.log_signal.emit(f"Error reading {url}: {str(e)[:100]}")
            return None
    
    def run(self):
        try:
            if not HAS_SCRAPING:
                self.finished.emit(False, "requests/beautifulsoup4 not installed. Run: pip install requests beautifulsoup4", [])
                return
            
            total_urls = len(self.urls)
            learned_count = 0
            
            for i, url in enumerate(self.urls):
                if not self.is_running:
                    break
                
                self.progress.emit(
                    int((i + 1) / total_urls * 100),
                    f"Processing: {url[:60]}..."
                )
                self.log_signal.emit(f"Reading: {url}")
                
                text = self._extract_text_from_url(url)
                
                if text and len(text) > 100:
                    paragraphs = text.split('\n')
                    chunks = []
                    current_chunk = ""
                    
                    for para in paragraphs:
                        if len(current_chunk) + len(para) < 1500:
                            current_chunk += para + "\n"
                        else:
                            if current_chunk.strip():
                                chunks.append(current_chunk.strip())
                            current_chunk = para + "\n"
                    
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())
                    
                    chunk_count = 0
                    for chunk_idx, chunk in enumerate(chunks[:30]):
                        if not self.is_running:
                            break
                        
                        if len(chunk) > 50:
                            result = self.deepnova.learn(
                                chunk,
                                source=f"web:{url}"
                            )
                            
                            if result.get('success'):
                                chunk_count += 1
                                learned_count += 1
                                self.content_learned.emit(
                                    url,
                                    result.get('summary', chunk[:100])
                                )
                    
                    self.log_signal.emit(f"  Learned {chunk_count} chunks from {url}")
                    self.learned_urls.append(url)
                else:
                    self.log_signal.emit(f"  Could not extract content from {url}")
            
            self.finished.emit(
                True, 
                f"Web learning completed! Learned {learned_count} chunks from {len(self.learned_urls)} URLs",
                self.learned_urls
            )
            
        except Exception as e:
            self.finished.emit(False, f"Error: {str(e)}", [])


class ChatWorker(QThread):
    response_received = pyqtSignal(str)
    finished = pyqtSignal()
    error = pyqtSignal(str)
    
    def __init__(self, deepnova, user_input: str, max_tokens: int = 500, temperature: float = 0.7):
        super().__init__()
        self.deepnova = deepnova
        self.user_input = user_input
        self.max_tokens = max_tokens
        self.temperature = temperature
        
    def run(self):
        try:
            response = self.deepnova.chat(
                self.user_input,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature
            )
            self.response_received.emit(response)
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.finished.emit()


class KnowledgeViewer(QDialog):
    def __init__(self, item: dict, parent=None):
        super().__init__(parent)
        self.item = item
        self.setWindowTitle("Knowledge Details")
        self.setMinimumSize(600, 500)
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        summary_group = QGroupBox("Summary")
        summary_layout = QVBoxLayout(summary_group)
        summary_text = QTextEdit()
        summary_text.setPlainText(self.item.get('summary', ''))
        summary_text.setReadOnly(True)
        summary_text.setMaximumHeight(100)
        summary_layout.addWidget(summary_text)
        layout.addWidget(summary_group)
        
        text_group = QGroupBox("Full Content")
        text_layout = QVBoxLayout(text_group)
        full_text = QTextEdit()
        full_text.setPlainText(self.item.get('text', ''))
        full_text.setReadOnly(True)
        text_layout.addWidget(full_text)
        layout.addWidget(text_group)
        
        meta_group = QGroupBox("Information")
        meta_layout = QFormLayout(meta_group)
        
        meta_layout.addRow("Source:", QLabel(self.item.get('source', 'unknown')))
        meta_layout.addRow("Hash:", QLabel(self.item.get('hash', '')[:16]))
        meta_layout.addRow("Access count:", QLabel(str(self.item.get('access_count', 0))))
        
        if self.item.get('timestamp'):
            dt = datetime.fromtimestamp(self.item['timestamp'])
            meta_layout.addRow("Time:", QLabel(dt.strftime('%Y-%m-%d %H:%M:%S')))
        
        keywords = self.item.get('keywords', [])
        if keywords:
            meta_layout.addRow("Keywords:", QLabel(', '.join(keywords[:10])))
        
        layout.addWidget(meta_group)
        
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)


class DeepNovaGUI(QMainWindow):
    
    def __init__(self):
        super().__init__()
        
        self.deepnova = None
        self.model_args = None
        self.tokenizer = None
        self.model = None
        
        self.training_thread = None
        self.web_learning_thread = None
        self.chat_worker = None
        
        self.setup_ui()
        self.setup_menu()
        self.setup_status_bar()
        
        self.load_settings()
        self.init_model()
        
    def setup_ui(self):
        self.setWindowTitle("DeepNova AI - Intelligent Assistant")
        self.setGeometry(100, 100, 1400, 900)
        
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #ffffff;
                color: #2c3e50;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            
            QTabWidget::pane {
                border: 1px solid #dcdcdc;
                border-radius: 8px;
                background-color: #ffffff;
            }
            
            QTabBar::tab {
                background-color: #f0f0f0;
                padding: 10px 20px;
                margin-right: 2px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                font-size: 12pt;
            }
            
            QTabBar::tab:selected {
                background-color: #4a90d9;
                color: white;
            }
            
            QTabBar::tab:hover:!selected {
                background-color: #e0e0e0;
            }
            
            QTextEdit, QPlainTextEdit {
                background-color: #fafafa;
                border: 1px solid #dcdcdc;
                border-radius: 8px;
                padding: 10px;
                font-size: 11pt;
            }
            
            QLineEdit {
                background-color: #ffffff;
                border: 1px solid #dcdcdc;
                border-radius: 20px;
                padding: 8px 15px;
                font-size: 11pt;
            }
            
            QLineEdit:focus {
                border-color: #4a90d9;
            }
            
            QPushButton {
                background-color: #4a90d9;
                color: white;
                border: none;
                border-radius: 20px;
                padding: 8px 20px;
                font-size: 11pt;
                font-weight: 500;
            }
            
            QPushButton:hover {
                background-color: #357abd;
            }
            
            QPushButton:pressed {
                background-color: #2968a3;
            }
            
            QPushButton:disabled {
                background-color: #bdc3c7;
            }
            
            QGroupBox {
                border: 1px solid #dcdcdc;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 12px;
                font-weight: bold;
            }
            
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
            }
            
            QProgressBar {
                border: 1px solid #dcdcdc;
                border-radius: 8px;
                text-align: center;
                background-color: #f0f0f0;
                height: 22px;
            }
            
            QProgressBar::chunk {
                background-color: #4a90d9;
                border-radius: 8px;
            }
            
            QListWidget, QTableWidget {
                background-color: #fafafa;
                border: 1px solid #dcdcdc;
                border-radius: 8px;
                outline: none;
            }
            
            QListWidget::item, QTableWidget::item {
                padding: 6px;
            }
            
            QListWidget::item:selected, QTableWidget::item:selected {
                background-color: #4a90d9;
                color: white;
            }
            
            QHeaderView::section {
                background-color: #f0f0f0;
                padding: 6px;
                border: none;
                border-right: 1px solid #dcdcdc;
                border-bottom: 1px solid #dcdcdc;
                font-weight: bold;
            }
            
            QMenuBar {
                background-color: #ffffff;
                border-bottom: 1px solid #dcdcdc;
            }
            
            QMenuBar::item {
                padding: 5px 10px;
                border-radius: 4px;
            }
            
            QMenuBar::item:selected {
                background-color: #4a90d9;
                color: white;
            }
            
            QMenu {
                background-color: #ffffff;
                border: 1px solid #dcdcdc;
                border-radius: 6px;
            }
            
            QMenu::item {
                padding: 6px 20px;
            }
            
            QMenu::item:selected {
                background-color: #4a90d9;
                color: white;
            }
            
            QStatusBar {
                background-color: #f8f8f8;
                color: #666666;
            }
        """)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        self.chat_tab = self.create_chat_tab()
        self.training_tab = self.create_training_tab()
        self.learning_tab = self.create_learning_tab()
        self.knowledge_tab = self.create_knowledge_tab()
        self.settings_tab = self.create_settings_tab()
        
        self.tab_widget.addTab(self.chat_tab, "Chat")
        self.tab_widget.addTab(self.training_tab, "Training")
        self.tab_widget.addTab(self.learning_tab, "Web Learning")
        self.tab_widget.addTab(self.knowledge_tab, "Knowledge Base")
        self.tab_widget.addTab(self.settings_tab, "Settings")
        
    def create_chat_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)
        
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setPlaceholderText("Messages will appear here...")
        layout.addWidget(self.chat_display, stretch=10)
        
        input_frame = QFrame()
        input_frame.setStyleSheet("QFrame { background-color: #f8f8f8; border-radius: 25px; padding: 5px; }")
        input_layout = QHBoxLayout(input_frame)
        input_layout.setContentsMargins(10, 5, 10, 5)
        
        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("Type your message...")
        self.chat_input.returnPressed.connect(self.send_message)
        input_layout.addWidget(self.chat_input, stretch=8)
        
        self.send_button = QPushButton("Send")
        self.send_button.setMinimumWidth(80)
        self.send_button.clicked.connect(self.send_message)
        input_layout.addWidget(self.send_button)
        
        self.clear_chat_button = QPushButton("Clear")
        self.clear_chat_button.setMinimumWidth(80)
        self.clear_chat_button.clicked.connect(self.clear_chat)
        input_layout.addWidget(self.clear_chat_button)
        
        layout.addWidget(input_frame)
        
        settings_frame = QFrame()
        settings_layout = QHBoxLayout(settings_frame)
        settings_layout.setContentsMargins(0, 5, 0, 5)
        
        settings_layout.addWidget(QLabel("Max tokens:"))
        self.max_tokens_spin = QSpinBox()
        self.max_tokens_spin.setRange(50, 2000)
        self.max_tokens_spin.setValue(500)
        self.max_tokens_spin.setMinimumWidth(80)
        settings_layout.addWidget(self.max_tokens_spin)
        
        settings_layout.addSpacing(20)
        
        settings_layout.addWidget(QLabel("Temperature:"))
        self.temperature_slider = QSlider(Qt.Orientation.Horizontal)
        self.temperature_slider.setRange(0, 100)
        self.temperature_slider.setValue(70)
        self.temperature_slider.setMinimumWidth(150)
        self.temperature_slider.valueChanged.connect(self.update_temperature_label)
        settings_layout.addWidget(self.temperature_slider)
        
        self.temperature_label = QLabel("0.70")
        self.temperature_label.setMinimumWidth(40)
        settings_layout.addWidget(self.temperature_label)
        
        settings_layout.addStretch()
        
        layout.addWidget(settings_frame)
        
        return tab
    
    def create_training_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(15)
        
        settings_group = QGroupBox("Training Settings")
        settings_layout = QGridLayout(settings_group)
        settings_layout.setSpacing(12)
        
        settings_layout.addWidget(QLabel("Data directory:"), 0, 0)
        data_layout = QHBoxLayout()
        self.data_path_edit = QLineEdit()
        self.data_path_edit.setPlaceholderText("Select training data directory...")
        data_layout.addWidget(self.data_path_edit)
        
        self.browse_data_button = QPushButton("Browse")
        self.browse_data_button.setMaximumWidth(80)
        self.browse_data_button.clicked.connect(self.browse_data_path)
        data_layout.addWidget(self.browse_data_button)
        settings_layout.addLayout(data_layout, 0, 1)
        
        settings_layout.addWidget(QLabel("Batch size:"), 1, 0)
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 64)
        self.batch_size_spin.setValue(4)
        settings_layout.addWidget(self.batch_size_spin, 1, 1)
        
        settings_layout.addWidget(QLabel("Learning rate:"), 2, 0)
        self.learning_rate_spin = QDoubleSpinBox()
        self.learning_rate_spin.setRange(1e-7, 1e-3)
        self.learning_rate_spin.setDecimals(7)
        self.learning_rate_spin.setSingleStep(1e-5)
        self.learning_rate_spin.setValue(3e-4)
        self.learning_rate_spin.setMinimumWidth(120)
        settings_layout.addWidget(self.learning_rate_spin, 2, 1)
        
        settings_layout.addWidget(QLabel("Max sequence length:"), 3, 0)
        self.max_seq_len_spin = QSpinBox()
        self.max_seq_len_spin.setRange(128, 8192)
        self.max_seq_len_spin.setValue(1024)
        settings_layout.addWidget(self.max_seq_len_spin, 3, 1)
        
        settings_layout.addWidget(QLabel("Number of epochs:"), 4, 0)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 100)
        self.epochs_spin.setValue(1)
        settings_layout.addWidget(self.epochs_spin, 4, 1)
        
        self.use_amp_check = QCheckBox("Use Mixed Precision (AMP) - Faster training")
        self.use_amp_check.setChecked(True)
        settings_layout.addWidget(self.use_amp_check, 5, 0, 1, 2)
        
        layout.addWidget(settings_group)
        
        control_group = QGroupBox("Control")
        control_layout = QVBoxLayout(control_group)
        
        button_layout = QHBoxLayout()
        self.start_train_button = QPushButton("Start Training")
        self.start_train_button.setMinimumHeight(40)
        self.start_train_button.clicked.connect(self.start_training)
        self.start_train_button.setStyleSheet("background-color: #28a745;")
        button_layout.addWidget(self.start_train_button)
        
        self.stop_train_button = QPushButton("Stop Training")
        self.stop_train_button.setMinimumHeight(40)
        self.stop_train_button.clicked.connect(self.stop_training)
        self.stop_train_button.setEnabled(False)
        self.stop_train_button.setStyleSheet("background-color: #dc3545;")
        button_layout.addWidget(self.stop_train_button)
        
        control_layout.addLayout(button_layout)
        
        self.train_progress = QProgressBar()
        self.train_progress.setVisible(False)
        self.train_progress.setMinimumHeight(22)
        control_layout.addWidget(self.train_progress)
        
        layout.addWidget(control_group)
        
        log_group = QGroupBox("Training Log")
        log_layout = QVBoxLayout(log_group)
        
        self.train_log = QPlainTextEdit()
        self.train_log.setReadOnly(True)
        self.train_log.setMaximumBlockCount(500)
        log_layout.addWidget(self.train_log)
        
        layout.addWidget(log_group)
        
        return tab
    
    def create_learning_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(15)
        
        url_group = QGroupBox("URLs to Learn")
        url_layout = QVBoxLayout(url_group)
        
        self.url_list = QListWidget()
        self.url_list.setMinimumHeight(150)
        url_layout.addWidget(self.url_list)
        
        url_input_layout = QHBoxLayout()
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("Enter URL (e.g., https://example.com/article)")
        url_input_layout.addWidget(self.url_input)
        
        self.add_url_button = QPushButton("Add")
        self.add_url_button.clicked.connect(self.add_url)
        url_input_layout.addWidget(self.add_url_button)
        
        self.remove_url_button = QPushButton("Remove")
        self.remove_url_button.clicked.connect(self.remove_url)
        url_input_layout.addWidget(self.remove_url_button)
        
        self.clear_urls_button = QPushButton("Clear All")
        self.clear_urls_button.clicked.connect(self.clear_urls)
        url_input_layout.addWidget(self.clear_urls_button)
        
        url_layout.addLayout(url_input_layout)
        
        layout.addWidget(url_group)
        
        options_group = QGroupBox("Options")
        options_layout = QHBoxLayout(options_group)
        
        options_layout.addWidget(QLabel("Max depth:"))
        self.max_depth_spin = QSpinBox()
        self.max_depth_spin.setRange(0, 2)
        self.max_depth_spin.setValue(0)
        self.max_depth_spin.setToolTip("0: Current page only, 1: Follow links on same domain")
        options_layout.addWidget(self.max_depth_spin)
        
        options_layout.addStretch()
        
        layout.addWidget(options_group)
        
        control_layout = QHBoxLayout()
        
        self.start_learn_button = QPushButton("Start Web Learning")
        self.start_learn_button.setMinimumHeight(40)
        self.start_learn_button.clicked.connect(self.start_web_learning)
        self.start_learn_button.setStyleSheet("background-color: #17a2b8;")
        control_layout.addWidget(self.start_learn_button)
        
        self.stop_learn_button = QPushButton("Stop")
        self.stop_learn_button.setMinimumHeight(40)
        self.stop_learn_button.clicked.connect(self.stop_web_learning)
        self.stop_learn_button.setEnabled(False)
        control_layout.addWidget(self.stop_learn_button)
        
        layout.addLayout(control_layout)
        
        self.learn_progress = QProgressBar()
        self.learn_progress.setVisible(False)
        self.learn_progress.setMinimumHeight(22)
        layout.addWidget(self.learn_progress)
        
        log_group = QGroupBox("Learning Log")
        log_layout = QVBoxLayout(log_group)
        
        self.learn_log = QPlainTextEdit()
        self.learn_log.setReadOnly(True)
        self.learn_log.setMaximumBlockCount(300)
        log_layout.addWidget(self.learn_log)
        
        layout.addWidget(log_group)
        
        preview_group = QGroupBox("Recently Learned")
        preview_layout = QVBoxLayout(preview_group)
        
        self.learned_preview = QListWidget()
        self.learned_preview.setMinimumHeight(120)
        preview_layout.addWidget(self.learned_preview)
        
        layout.addWidget(preview_group)
        
        return tab
    
    def create_knowledge_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)
        
        search_layout = QHBoxLayout()
        self.knowledge_search = QLineEdit()
        self.knowledge_search.setPlaceholderText("Search knowledge base...")
        self.knowledge_search.returnPressed.connect(self.search_knowledge)
        search_layout.addWidget(self.knowledge_search)
        
        self.search_button = QPushButton("Search")
        self.search_button.clicked.connect(self.search_knowledge)
        search_layout.addWidget(self.search_button)
        
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh_knowledge)
        search_layout.addWidget(self.refresh_button)
        
        layout.addLayout(search_layout)
        
        self.knowledge_table = QTableWidget()
        self.knowledge_table.setColumnCount(5)
        self.knowledge_table.setHorizontalHeaderLabels(["Hash", "Summary", "Source", "Views", "Time"])
        self.knowledge_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.knowledge_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.knowledge_table.doubleClicked.connect(self.view_knowledge_item)
        layout.addWidget(self.knowledge_table)
        
        action_layout = QHBoxLayout()
        
        self.export_knowledge_button = QPushButton("Export Knowledge")
        self.export_knowledge_button.clicked.connect(self.export_knowledge)
        action_layout.addWidget(self.export_knowledge_button)
        
        self.import_knowledge_button = QPushButton("Import Knowledge")
        self.import_knowledge_button.clicked.connect(self.import_knowledge)
        action_layout.addWidget(self.import_knowledge_button)
        
        self.delete_selected_button = QPushButton("Delete Selected")
        self.delete_selected_button.clicked.connect(self.delete_selected_knowledge)
        self.delete_selected_button.setStyleSheet("background-color: #dc3545;")
        action_layout.addWidget(self.delete_selected_button)
        
        action_layout.addStretch()
        
        layout.addLayout(action_layout)
        
        stats_group = QGroupBox("Statistics")
        stats_layout = QGridLayout(stats_group)
        
        self.total_learned_label = QLabel("Total Learned: --")
        stats_layout.addWidget(self.total_learned_label, 0, 0)
        
        self.total_facts_label = QLabel("Important Facts: --")
        stats_layout.addWidget(self.total_facts_label, 0, 1)
        
        self.total_entities_label = QLabel("Entities Tracked: --")
        stats_layout.addWidget(self.total_entities_label, 1, 0)
        
        self.total_compressions_label = QLabel("Total Compressions: --")
        stats_layout.addWidget(self.total_compressions_label, 1, 1)
        
        layout.addWidget(stats_group)
        
        return tab
    
    def create_settings_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        model_group = QGroupBox("Model Configuration")
        model_layout = QGridLayout(model_group)
        
        model_layout.addWidget(QLabel("Model Size:"), 0, 0)
        self.model_size_combo = QComboBox()
        self.model_size_combo.addItems(["lite", "base", "parallel", "enhanced"])
        self.model_size_combo.setCurrentText("lite")
        model_layout.addWidget(self.model_size_combo, 0, 1)
        
        model_layout.addWidget(QLabel("Device:"), 1, 0)
        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "cuda", "cpu", "mps"])
        self.device_combo.setCurrentText("auto")
        model_layout.addWidget(self.device_combo, 1, 1)
        
        model_layout.addWidget(QLabel("Dtype:"), 2, 0)
        self.dtype_combo = QComboBox()
        self.dtype_combo.addItems(["bf16", "fp16", "fp32"])
        self.dtype_combo.setCurrentText("bf16")
        model_layout.addWidget(self.dtype_combo, 2, 1)
        
        self.parallel_moe_check = QCheckBox("Use Parallel MoE + Dense")
        self.parallel_moe_check.setChecked(False)
        model_layout.addWidget(self.parallel_moe_check, 3, 0, 1, 2)
        
        self.glm_check = QCheckBox("Use GLM (General Language Model)")
        self.glm_check.setChecked(False)
        model_layout.addWidget(self.glm_check, 4, 0, 1, 2)
        
        self.adaptive_router_check = QCheckBox("Use Adaptive Router")
        self.adaptive_router_check.setChecked(False)
        model_layout.addWidget(self.adaptive_router_check, 5, 0, 1, 2)
        
        self.dynamic_depth_check = QCheckBox("Use Dynamic Depth")
        self.dynamic_depth_check.setChecked(False)
        model_layout.addWidget(self.dynamic_depth_check, 6, 0, 1, 2)
        
        self.mtp_check = QCheckBox("Use Multi-Token Prediction")
        self.mtp_check.setChecked(False)
        model_layout.addWidget(self.mtp_check, 7, 0, 1, 2)
        
        layout.addWidget(model_group)
        
        memory_group = QGroupBox("Memory Settings")
        memory_layout = QGridLayout(memory_group)
        
        memory_layout.addWidget(QLabel("Memory file:"), 0, 0)
        self.memory_file_edit = QLineEdit()
        self.memory_file_edit.setText("deepnova_memory.json")
        memory_layout.addWidget(self.memory_file_edit, 0, 1)
        
        memory_layout.addWidget(QLabel("Max context tokens:"), 1, 0)
        self.max_context_spin = QSpinBox()
        self.max_context_spin.setRange(1024, 32768)
        self.max_context_spin.setValue(8192)
        memory_layout.addWidget(self.max_context_spin, 1, 1)
        
        layout.addWidget(memory_group)
        
        button_layout = QHBoxLayout()
        
        self.save_settings_button = QPushButton("Save Settings")
        self.save_settings_button.clicked.connect(self.save_settings)
        button_layout.addWidget(self.save_settings_button)
        
        self.load_settings_button = QPushButton("Load Settings")
        self.load_settings_button.clicked.connect(self.load_settings)
        button_layout.addWidget(self.load_settings_button)
        
        self.apply_settings_button = QPushButton("Apply and Reload Model")
        self.apply_settings_button.clicked.connect(self.apply_settings)
        self.apply_settings_button.setStyleSheet("background-color: #28a745;")
        button_layout.addWidget(self.apply_settings_button)
        
        layout.addLayout(button_layout)
        
        info_group = QGroupBox("System Information")
        info_layout = QVBoxLayout(info_group)
        
        self.system_info = QPlainTextEdit()
        self.system_info.setReadOnly(True)
        self.system_info.setMaximumHeight(200)
        info_layout.addWidget(self.system_info)
        
        self.refresh_info_button = QPushButton("Refresh Info")
        self.refresh_info_button.clicked.connect(self.refresh_system_info)
        info_layout.addWidget(self.refresh_info_button)
        
        layout.addWidget(info_group)
        
        layout.addStretch()
        
        return tab
    
    def setup_menu(self):
        menubar = self.menuBar()
        
        file_menu = menubar.addMenu("File")
        
        load_model_action = QAction("Load Model", self)
        load_model_action.triggered.connect(self.load_model_dialog)
        file_menu.addAction(load_model_action)
        
        save_model_action = QAction("Save Model", self)
        save_model_action.triggered.connect(self.save_model_dialog)
        file_menu.addAction(save_model_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        edit_menu = menubar.addMenu("Edit")
        
        clear_memory_action = QAction("Clear Short-term Memory", self)
        clear_memory_action.triggered.connect(lambda: self.clear_context(keep_important=True))
        edit_menu.addAction(clear_memory_action)
        
        clear_all_memory_action = QAction("Clear All Memory", self)
        clear_all_memory_action.triggered.connect(lambda: self.clear_context(keep_important=False))
        edit_menu.addAction(clear_all_memory_action)
        
        edit_menu.addSeparator()
        
        cleanup_memory_action = QAction("Clean GPU/RAM Memory", self)
        cleanup_memory_action.triggered.connect(self.cleanup_memory)
        edit_menu.addAction(cleanup_memory_action)
        
        view_menu = menubar.addMenu("View")
        
        show_stats_action = QAction("Show Statistics", self)
        show_stats_action.triggered.connect(self.show_stats_dialog)
        view_menu.addAction(show_stats_action)
        
        help_menu = menubar.addMenu("Help")
        
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def setup_status_bar(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        self.status_label = QLabel("Ready")
        self.status_bar.addWidget(self.status_label)
        
        self.memory_label = QLabel("")
        self.status_bar.addPermanentWidget(self.memory_label)
        
        self.memory_timer = QTimer()
        self.memory_timer.timeout.connect(self.update_memory_display)
        self.memory_timer.start(5000)
    
    def load_settings(self):
        settings_file = "deepnova_gui_settings.json"
        if os.path.exists(settings_file):
            try:
                with open(settings_file, 'r') as f:
                    settings = json.load(f)
                
                self.model_size_combo.setCurrentText(settings.get('model_size', 'lite'))
                self.device_combo.setCurrentText(settings.get('device', 'auto'))
                self.dtype_combo.setCurrentText(settings.get('dtype', 'bf16'))
                self.parallel_moe_check.setChecked(settings.get('parallel_moe', False))
                self.glm_check.setChecked(settings.get('glm', False))
                self.adaptive_router_check.setChecked(settings.get('adaptive_router', False))
                self.dynamic_depth_check.setChecked(settings.get('dynamic_depth', False))
                self.mtp_check.setChecked(settings.get('mtp', False))
                self.memory_file_edit.setText(settings.get('memory_file', 'deepnova_memory.json'))
                self.max_context_spin.setValue(settings.get('max_context_tokens', 8192))
                
            except Exception as e:
                logger.error(f"Failed to load settings: {e}")
    
    def save_settings(self):
        settings = {
            'model_size': self.model_size_combo.currentText(),
            'device': self.device_combo.currentText(),
            'dtype': self.dtype_combo.currentText(),
            'parallel_moe': self.parallel_moe_check.isChecked(),
            'glm': self.glm_check.isChecked(),
            'adaptive_router': self.adaptive_router_check.isChecked(),
            'dynamic_depth': self.dynamic_depth_check.isChecked(),
            'mtp': self.mtp_check.isChecked(),
            'memory_file': self.memory_file_edit.text(),
            'max_context_tokens': self.max_context_spin.value(),
        }
        
        try:
            with open('deepnova_gui_settings.json', 'w') as f:
                json.dump(settings, f, indent=2)
            self.status_label.setText("Settings saved")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to save settings: {e}")
    
    def load_settings_dialog(self):
        self.load_settings()
        self.status_label.setText("Settings loaded")
    
    def apply_settings(self):
        self.status_label.setText("Reloading model with new settings...")
        QTimer.singleShot(100, self.init_model)
    
    def init_model(self):
        try:
            self.status_label.setText("Initializing model...")
            
            model_size = self.model_size_combo.currentText()
            
            if model_size == "enhanced":
                self.model_args = ModelArgs.enhanced_full()
            elif model_size == "parallel":
                self.model_args = ModelArgs.parallel_moe_dense()
            elif model_size == "base":
                self.model_args = ModelArgs()
                self.model_args.dim = 2048
                self.model_args.n_layers = 24
                self.model_args.n_heads = 16
            else:
                self.model_args = ModelArgs.deepseek_v3_lite()
            
            self.model_args.use_parallel_moe_dense = self.parallel_moe_check.isChecked()
            self.model_args.use_glm = self.glm_check.isChecked()
            self.model_args.use_adaptive_router = self.adaptive_router_check.isChecked()
            self.model_args.use_dynamic_depth = self.dynamic_depth_check.isChecked()
            self.model_args.use_multi_token_prediction = self.mtp_check.isChecked()
            
            device = self.device_combo.currentText()
            if device != "auto":
                self.model_args.device = device
            
            self.model_args.dtype = self.dtype_combo.currentText()
            
            self.tokenizer = ProductionTokenizer()
            self.model = Transformer(self.model_args)
            
            self.deepnova = DeepNovaAI(
                self.model,
                self.tokenizer,
                self.model_args,
                memory_file=self.memory_file_edit.text()
            )
            
            self.deepnova.memory.max_tokens = self.max_context_spin.value()
            
            self.add_chat_message("system", f"DeepNova AI v{self.deepnova.version} loaded successfully!")
            
            if self.model_args.use_parallel_moe_dense:
                self.add_chat_message("system", "Parallel MoE+Dense enabled")
            if self.model_args.use_adaptive_router:
                self.add_chat_message("system", "Adaptive Router enabled")
            
            self.status_label.setText("Model ready")
            self.refresh_knowledge()
            self.refresh_system_info()
            
        except Exception as e:
            error_msg = f"Failed to initialize model: {str(e)}"
            self.status_label.setText(error_msg)
            QMessageBox.critical(self, "Error", error_msg)
            logger.error(error_msg, exc_info=True)
    
    def add_chat_message(self, sender: str, message: str):
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        
        if sender == "user":
            color = "#0064c8"
            prefix = "You"
        elif sender == "system":
            color = "#666666"
            prefix = "System"
        else:
            color = "#008000"
            prefix = "DeepNova"
        
        self.chat_display.setTextColor(QColor(color))
        self.chat_display.append(f"\n[{prefix}]:")
        self.chat_display.setTextColor(QColor("#333333"))
        self.chat_display.append(message)
        
        self.chat_display.ensureCursorVisible()
    
    def send_message(self):
        user_input = self.chat_input.text().strip()
        if not user_input:
            return
        
        if not self.deepnova:
            QMessageBox.warning(self, "Warning", "Model not initialized. Please wait or check settings.")
            return
        
        self.add_chat_message("user", user_input)
        self.chat_input.clear()
        
        self.send_button.setEnabled(False)
        self.chat_input.setEnabled(False)
        
        self.chat_worker = ChatWorker(
            self.deepnova,
            user_input,
            max_tokens=self.max_tokens_spin.value(),
            temperature=self.temperature_slider.value() / 100.0
        )
        self.chat_worker.response_received.connect(self.on_chat_response)
        self.chat_worker.error.connect(self.on_chat_error)
        self.chat_worker.finished.connect(self.on_chat_finished)
        self.chat_worker.start()
    
    def on_chat_response(self, response: str):
        self.add_chat_message("assistant", response)
    
    def on_chat_error(self, error: str):
        self.add_chat_message("system", f"Error: {error}")
    
    def on_chat_finished(self):
        self.send_button.setEnabled(True)
        self.chat_input.setEnabled(True)
        self.chat_input.setFocus()
    
    def clear_chat(self):
        self.chat_display.clear()
    
    def update_temperature_label(self, value: int):
        self.temperature_label.setText(f"{value/100:.2f}")
    
    def browse_data_path(self):
        path = QFileDialog.getExistingDirectory(self, "Select Training Data Directory")
        if path:
            self.data_path_edit.setText(path)
    
    def start_training(self):
        if not self.deepnova:
            QMessageBox.warning(self, "Warning", "Model not initialized")
            return
        
        data_path = self.data_path_edit.text()
        if not data_path or not os.path.exists(data_path):
            QMessageBox.warning(self, "Warning", "Please select a valid data path")
            return
        
        self.model_args.batch_size = self.batch_size_spin.value()
        self.model_args.learning_rate = self.learning_rate_spin.value()
        self.model_args.max_seq_length = self.max_seq_len_spin.value()
        self.model_args.use_amp = self.use_amp_check.isChecked()
        
        self.training_thread = TrainingThread(
            self.model,
            self.tokenizer,
            self.model_args,
            data_path,
            epochs=self.epochs_spin.value()
        )
        self.training_thread.progress.connect(self.on_training_progress)
        self.training_thread.finished.connect(self.on_training_finished)
        self.training_thread.log_signal.connect(self.on_training_log)
        self.training_thread.epoch_end.connect(self.on_epoch_end)
        self.training_thread.start()
        
        self.start_train_button.setEnabled(False)
        self.stop_train_button.setEnabled(True)
        self.train_progress.setVisible(True)
        self.train_progress.setValue(0)
        self.status_label.setText("Training in progress...")
    
    def stop_training(self):
        if self.training_thread:
            self.training_thread.stop()
            self.status_label.setText("Stopping training...")
    
    def on_training_progress(self, value: int, text: str):
        self.train_progress.setValue(value)
        self.status_label.setText(text)
    
    def on_training_log(self, message: str):
        self.train_log.appendPlainText(message)
        cursor = self.train_log.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.train_log.setTextCursor(cursor)
    
    def on_epoch_end(self, epoch: int, avg_loss: float):
        self.train_log.appendPlainText(f"  Epoch {epoch} completed - Avg loss: {avg_loss:.4f}")
    
    def on_training_finished(self, success: bool, message: str):
        self.start_train_button.setEnabled(True)
        self.stop_train_button.setEnabled(False)
        self.train_progress.setVisible(False)
        self.status_label.setText(message if success else "Training failed")
        
        if success:
            QMessageBox.information(self, "Training Complete", message)
        else:
            QMessageBox.critical(self, "Training Failed", message)
    
    def add_url(self):
        url = self.url_input.text().strip()
        if url:
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            
            for i in range(self.url_list.count()):
                if self.url_list.item(i).text() == url:
                    return
            
            self.url_list.addItem(url)
            self.url_input.clear()
    
    def remove_url(self):
        current = self.url_list.currentRow()
        if current >= 0:
            self.url_list.takeItem(current)
    
    def clear_urls(self):
        self.url_list.clear()
    
    def start_web_learning(self):
        if not self.deepnova:
            QMessageBox.warning(self, "Warning", "Model not initialized")
            return
        
        urls = []
        for i in range(self.url_list.count()):
            urls.append(self.url_list.item(i).text())
        
        if not urls:
            QMessageBox.warning(self, "Warning", "Please add at least one URL")
            return
        
        self.web_learning_thread = WebLearningThread(
            self.deepnova,
            urls,
            max_depth=self.max_depth_spin.value()
        )
        self.web_learning_thread.progress.connect(self.on_web_learning_progress)
        self.web_learning_thread.finished.connect(self.on_web_learning_finished)
        self.web_learning_thread.log_signal.connect(self.on_web_learning_log)
        self.web_learning_thread.content_learned.connect(self.on_content_learned)
        self.web_learning_thread.start()
        
        self.start_learn_button.setEnabled(False)
        self.stop_learn_button.setEnabled(True)
        self.learn_progress.setVisible(True)
        self.learn_progress.setValue(0)
        self.status_label.setText("Web learning in progress...")
    
    def stop_web_learning(self):
        if self.web_learning_thread:
            self.web_learning_thread.is_running = False
            self.status_label.setText("Stopping web learning...")
    
    def on_web_learning_progress(self, value: int, text: str):
        self.learn_progress.setValue(value)
        self.status_label.setText(text)
    
    def on_web_learning_log(self, message: str):
        self.learn_log.appendPlainText(message)
        cursor = self.learn_log.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.learn_log.setTextCursor(cursor)
    
    def on_content_learned(self, url: str, summary: str):
        self.learned_preview.addItem(f"{url[:50]}...\n   {summary[:100]}...")
        self.refresh_knowledge()
    
    def on_web_learning_finished(self, success: bool, message: str, learned_urls: list):
        self.start_learn_button.setEnabled(True)
        self.stop_learn_button.setEnabled(False)
        self.learn_progress.setVisible(False)
        self.status_label.setText(message if success else "Web learning failed")
        
        QMessageBox.information(self, "Web Learning Complete", message)
    
    def search_knowledge(self):
        query = self.knowledge_search.text().strip()
        if not query or not self.deepnova:
            return
        
        results = self.deepnova.recall(query, top_k=20)
        self.display_knowledge_results(results)
    
    def refresh_knowledge(self):
        if not self.deepnova:
            return
        
        try:
            learned = self.deepnova.list_learned(limit=100)
            self.display_knowledge_results(learned)
            
            stats = self.deepnova.get_stats()
            self.total_learned_label.setText(f"Total Learned: {stats['learning']['total_learned']}")
            self.total_facts_label.setText(f"Important Facts: {stats['memory']['important_facts']}")
            self.total_entities_label.setText(f"Entities Tracked: {stats['memory']['entities_tracked']}")
            self.total_compressions_label.setText(f"Total Compressions: {stats['memory']['total_compressions']}")
            
        except Exception as e:
            logger.error(f"Failed to refresh knowledge: {e}")
    
    def display_knowledge_results(self, results: list):
        self.knowledge_table.setRowCount(len(results))
        
        for row, item in enumerate(results):
            self.knowledge_table.setItem(row, 0, QTableWidgetItem(item.get('hash', '')[:8]))
            self.knowledge_table.setItem(row, 1, QTableWidgetItem(item.get('summary', '')[:100]))
            self.knowledge_table.setItem(row, 2, QTableWidgetItem(item.get('source', 'unknown')[:50]))
            self.knowledge_table.setItem(row, 3, QTableWidgetItem(str(item.get('access_count', 0))))
            
            ts = item.get('timestamp', 0)
            if ts:
                time_str = datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M')
            else:
                time_str = ''
            self.knowledge_table.setItem(row, 4, QTableWidgetItem(time_str))
    
    def view_knowledge_item(self):
        current_row = self.knowledge_table.currentRow()
        if current_row < 0:
            return
        
        hash_item = self.knowledge_table.item(current_row, 0)
        if not hash_item:
            return
        
        text_hash = hash_item.text()
        
        if self.deepnova:
            learned = self.deepnova.list_learned(limit=500)
            for item in learned:
                if item.get('hash', '').startswith(text_hash):
                    dialog = KnowledgeViewer(item, self)
                    dialog.exec()
                    break
    
    def delete_selected_knowledge(self):
        current_row = self.knowledge_table.currentRow()
        if current_row < 0:
            return
        
        reply = QMessageBox.question(
            self,
            "Confirm Delete",
            "Are you sure you want to delete this knowledge item?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes and self.deepnova:
            hash_item = self.knowledge_table.item(current_row, 0)
            if hash_item:
                text_hash = hash_item.text()
                learned = self.deepnova.list_learned(limit=500)
                for item in learned:
                    if item.get('hash', '').startswith(text_hash):
                        self.deepnova.forget(item['hash'])
                        break
                
                self.refresh_knowledge()
    
    def export_knowledge(self):
        if not self.deepnova:
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Knowledge",
            "knowledge_export.json",
            "JSON Files (*.json)"
        )
        
        if file_path:
            if self.deepnova.export_knowledge(file_path):
                QMessageBox.information(self, "Export Complete", f"Knowledge exported to {file_path}")
            else:
                QMessageBox.critical(self, "Export Failed", "Failed to export knowledge")
    
    def import_knowledge(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Knowledge",
            "",
            "JSON Files (*.json)"
        )
        
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if 'learned_texts' in data:
                    count = len(data['learned_texts'])
                    reply = QMessageBox.question(
                        self,
                        "Confirm Import",
                        f"This will import {count} knowledge items. Continue?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                    )
                    
                    if reply == QMessageBox.StandardButton.Yes:
                        for hash_id, text_data in data['learned_texts'].items():
                            self.deepnova.learn(
                                text_data.get('text', ''),
                                source=text_data.get('source', 'imported')
                            )
                        
                        self.refresh_knowledge()
                        QMessageBox.information(self, "Import Complete", f"Imported {count} knowledge items")
                
            except Exception as e:
                QMessageBox.critical(self, "Import Failed", str(e))
    
    def clear_context(self, keep_important: bool = True):
        if self.deepnova:
            self.deepnova.clear_context(keep_important=keep_important)
            self.status_label.setText("Context cleared" + (" (keeping important facts)" if keep_important else ""))
    
    def cleanup_memory(self):
        cleanup_memory()
        self.status_label.setText("Memory cleaned")
    
    def show_stats_dialog(self):
        if not self.deepnova:
            QMessageBox.warning(self, "Warning", "Model not initialized")
            return
        
        stats = self.deepnova.get_stats()
        
        dialog = QDialog(self)
        dialog.setWindowTitle("DeepNova Statistics")
        dialog.setMinimumSize(500, 400)
        
        layout = QVBoxLayout(dialog)
        
        text = QPlainTextEdit()
        text.setReadOnly(True)
        
        stats_text = f"""
DeepNova AI Statistics
==================================================

General:
  Version: {stats.get('version', 'N/A')}
  Uptime: {stats.get('uptime_seconds', 0):.1f} seconds
  Total Messages: {stats.get('total_messages', 0)}
  Total Tokens Generated: {stats.get('total_tokens_generated', 0)}
  Total Tokens Saved: {stats.get('total_tokens_saved', 0)}

Memory:
  Short-term Messages: {stats.get('memory', {}).get('short_term_messages', 0)}
  Important Facts: {stats.get('memory', {}).get('important_facts', 0)}
  Entities Tracked: {stats.get('memory', {}).get('entities_tracked', 0)}
  Total Compressions: {stats.get('memory', {}).get('total_compressions', 0)}

Learning:
  Total Learned: {stats.get('learning', {}).get('total_learned', 0)}
  Knowledge Graph Nodes: {stats.get('learning', {}).get('knowledge_graph_nodes', 0)}

Active Features: {', '.join(stats.get('active_features', []))}
        """
        
        text.setPlainText(stats_text)
        layout.addWidget(text)
        
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)
        
        dialog.exec()
    
    def refresh_system_info(self):
        try:
            import torch
            
            info = f"DeepNova AI System Information\n========================================\n\n"
            
            info += f"PyTorch Version: {torch.__version__}\n"
            
            if torch.cuda.is_available():
                info += f"CUDA Available: Yes\n"
                info += f"CUDA Version: {torch.version.cuda}\n"
                info += f"GPU Device: {torch.cuda.get_device_name(0)}\n"
                info += f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n"
            else:
                info += f"CUDA Available: No\n"
            
            if hasattr(torch, 'backends') and torch.backends.mps.is_available():
                info += f"MPS Available: Yes\n"
            
            info += f"\nModel Configuration:\n"
            info += f"  Model Name: {self.model_args.model_name if self.model_args else 'N/A'}\n"
            info += f"  Version: {self.model_args.model_version if self.model_args else 'N/A'}\n"
            info += f"  Device: {self.model_args.device if self.model_args else 'N/A'}\n"
            info += f"  Dtype: {self.model_args.dtype if self.model_args else 'N/A'}\n"
            
            if self.model:
                model_info = self.model.get_model_info()
                info += f"\nModel Stats:\n"
                info += f"  Total Parameters: {model_info.get('total_params_formatted', 'N/A')}\n"
                info += f"  Active Parameters: {model_info.get('active_params_formatted', 'N/A')}\n"
                info += f"  Sparsity: {model_info.get('sparsity', 0):.1%}\n"
                info += f"  Layers: {model_info.get('n_layers', 'N/A')}\n"
                info += f"  Experts: {model_info.get('n_experts', 'N/A')} (top-{model_info.get('n_activated_experts', 'N/A')})\n"
            
            self.system_info.setPlainText(info)
            
        except Exception as e:
            self.system_info.setPlainText(f"Error getting system info: {e}")
    
    def update_memory_display(self):
        try:
            info = get_memory_info()
            gpu_text = ""
            if 'gpu_allocated_gb' in info:
                gpu_pct = (info['gpu_allocated_gb'] / info['gpu_total_gb'] * 100) if info['gpu_total_gb'] > 0 else 0
                gpu_text = f"GPU: {info['gpu_allocated_gb']:.1f}/{info['gpu_total_gb']:.0f}GB ({gpu_pct:.0f}%) | "
            
            ram_text = f"RAM: {info.get('ram_used_gb', 0):.1f}/{info.get('ram_total_gb', 0):.0f}GB ({info.get('ram_percent', 0):.0f}%)"
            
            self.memory_label.setText(gpu_text + ram_text)
        except Exception:
            pass
    
    def load_model_dialog(self):
        file_path = QFileDialog.getExistingDirectory(self, "Select Model Directory")
        if file_path:
            try:
                self.model, self.model_args = load_model(file_path)
                self.tokenizer = ProductionTokenizer()
                self.deepnova = DeepNovaAI(
                    self.model,
                    self.tokenizer,
                    self.model_args,
                    memory_file=self.memory_file_edit.text()
                )
                self.status_label.setText(f"Model loaded from {file_path}")
                self.add_chat_message("system", f"Model loaded from {file_path}")
                self.refresh_knowledge()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load model: {e}")
    
    def save_model_dialog(self):
        if not self.model:
            QMessageBox.warning(self, "Warning", "No model to save")
            return
        
        file_path = QFileDialog.getExistingDirectory(self, "Select Save Directory")
        if file_path:
            try:
                from model import save_model
                save_model(self.model, file_path)
                self.status_label.setText(f"Model saved to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save model: {e}")
    
    def show_about(self):
        QMessageBox.about(
            self,
            "About DeepNova AI",
            f"""DeepNova AI v5.0.0

An advanced AI assistant with enhanced MoE+Dense architecture.

Features:
- Intelligent context memory with compression
- Web learning and knowledge extraction
- Multi-token prediction
- Adaptive router with load balancing
- Parallel MoE + Dense layers
- Dynamic depth (layer skipping)
- GLM (General Language Model) integration

(c) 2024 DeepNova Team
"""
        )


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = DeepNovaGUI()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()