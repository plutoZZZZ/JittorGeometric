#!/usr/bin/env python3
"""Simple JittorGeometric Frontend"""

import sys
import jittor as jt
from jittor import nn
import jittor_geometric as jg
from jittor_geometric.nn import GCN, GAT, GraphSAGE
from jittor_geometric.datasets import Planetoid
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QFormLayout, QComboBox, QPushButton, QTextEdit,
    QSpinBox, QDoubleSpinBox, QProgressBar, QLabel
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

class ModelThread(QThread):
    update_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(dict)
    
    def __init__(self, model_name, dataset_name, hyperparams):
        super().__init__()
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.hyperparams = hyperparams
        
    def run(self):
        try:
            self.update_signal.emit(f"Loading dataset: {self.dataset_name}")
            dataset = Planetoid(root='/tmp/' + self.dataset_name, name=self.dataset_name)
            data = dataset[0]
            
            self.update_signal.emit(f"Creating model: {self.model_name}")
            
            if self.model_name == 'GCN':
                model = GCN(
                    in_channels=dataset.num_features,
                    hidden_channels=self.hyperparams['hidden_dim'],
                    out_channels=dataset.num_classes,
                    num_layers=self.hyperparams['num_layers'],
                    dropout=self.hyperparams['dropout']
                )
            elif self.model_name == 'GAT':
                model = GAT(
                    in_channels=dataset.num_features,
                    hidden_channels=self.hyperparams['hidden_dim'],
                    out_channels=dataset.num_classes,
                    num_layers=self.hyperparams['num_layers'],
                    heads=self.hyperparams['heads'],
                    dropout=self.hyperparams['dropout']
                )
            elif self.model_name == 'GraphSAGE':
                model = GraphSAGE(
                    in_channels=dataset.num_features,
                    hidden_channels=self.hyperparams['hidden_dim'],
                    out_channels=dataset.num_classes,
                    num_layers=self.hyperparams['num_layers'],
                    dropout=self.hyperparams['dropout']
                )
            else:
                raise ValueError(f"Unknown model: {self.model_name}")
            
            self.update_signal.emit("Generating computation graph...")
            output = model(data.x, data.edge_index)
            graph_data = jt.export_graph(output)
            
            self.update_signal.emit("Starting training...")
            optimizer = jt.optim.Adam(model.parameters(), lr=self.hyperparams['lr'])
            loss_fn = nn.CrossEntropyLoss()
            
            for epoch in range(self.hyperparams['epochs']):
                model.train()
                logits = model(data.x, data.edge_index)
                loss = loss_fn(logits[data.train_mask], data.y[data.train_mask])
                optimizer.step(loss)
                
                model.eval()
                logits = model(data.x, data.edge_index)
                pred = logits.argmax(dim=1)
                acc = (pred[data.test_mask] == data.y[data.test_mask]).mean()
                
                progress = int((epoch + 1) / self.hyperparams['epochs'] * 100)
                self.progress_signal.emit(progress)
                self.update_signal.emit(f"Epoch {epoch+1}/{self.hyperparams['epochs']} - Loss: {loss.item():.4f}, Acc: {acc.item():.4f}")
            
            self.finished_signal.emit({
                'model': model,
                'graph_data': graph_data,
                'accuracy': acc.item(),
                'dataset': dataset
            })
            
        except Exception as e:
            self.update_signal.emit(f"Error: {str(e)}")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("JittorGeometric Demo")
        self.setGeometry(100, 100, 800, 600)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()
        central_widget.setLayout(layout)
        
        # Model selection
        model_group = QGroupBox("Model Selection")
        model_layout = QFormLayout()
        model_group.setLayout(model_layout)
        
        self.model_combo = QComboBox()
        self.model_combo.addItems(['GCN', 'GAT', 'GraphSAGE'])
        model_layout.addRow("Model:", self.model_combo)
        
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems(['Cora', 'Citeseer', 'Pubmed'])
        model_layout.addRow("Dataset:", self.dataset_combo)
        
        layout.addWidget(model_group)
        
        # Hyperparameters
        hyper_group = QGroupBox("Hyperparameters")
        hyper_layout = QFormLayout()
        hyper_group.setLayout(hyper_layout)
        
        self.hidden_dim_spin = QSpinBox()
        self.hidden_dim_spin.setRange(16, 512)
        self.hidden_dim_spin.setValue(128)
        hyper_layout.addRow("Hidden Dim:", self.hidden_dim_spin)
        
        self.num_layers_spin = QSpinBox()
        self.num_layers_spin.setRange(1, 5)
        self.num_layers_spin.setValue(2)
        hyper_layout.addRow("Layers:", self.num_layers_spin)
        
        self.heads_spin = QSpinBox()
        self.heads_spin.setRange(1, 8)
        self.heads_spin.setValue(4)
        hyper_layout.addRow("Heads (GAT):", self.heads_spin)
        
        self.dropout_spin = QDoubleSpinBox()
        self.dropout_spin.setRange(0.0, 0.9)
        self.dropout_spin.setValue(0.5)
        hyper_layout.addRow("Dropout:", self.dropout_spin)
        
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(1e-5, 1e-1)
        self.lr_spin.setValue(1e-2)
        hyper_layout.addRow("Learning Rate:", self.lr_spin)
        
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 100)
        self.epochs_spin.setValue(20)
        hyper_layout.addRow("Epochs:", self.epochs_spin)
        
        layout.addWidget(hyper_group)
        
        # Buttons
        btn_layout = QHBoxLayout()
        self.run_btn = QPushButton("Run Model")
        self.run_btn.clicked.connect(self.run_model)
        btn_layout.addWidget(self.run_btn)
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_model)
        self.stop_btn.setEnabled(False)
        btn_layout.addWidget(self.stop_btn)
        
        layout.addLayout(btn_layout)
        
        # Progress
        self.progress = QProgressBar()
        layout.addWidget(self.progress)
        
        # Status
        self.status_label = QLabel("Status: Ready")
        layout.addWidget(self.status_label)
        
        # Log
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)
        
        self.model_thread = None
        
    def run_model(self):
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("Status: Running...")
        self.log_text.clear()
        
        hyperparams = {
            'hidden_dim': self.hidden_dim_spin.value(),
            'num_layers': self.num_layers_spin.value(),
            'heads': self.heads_spin.value(),
            'dropout': self.dropout_spin.value(),
            'lr': self.lr_spin.value(),
            'epochs': self.epochs_spin.value()
        }
        
        self.model_thread = ModelThread(
            self.model_combo.currentText(),
            self.dataset_combo.currentText(),
            hyperparams
        )
        
        self.model_thread.update_signal.connect(self.update_log)
        self.model_thread.progress_signal.connect(self.update_progress)
        self.model_thread.finished_signal.connect(self.model_finished)
        self.model_thread.start()
        
    def stop_model(self):
        if self.model_thread:
            self.model_thread.terminate()
            self.model_thread.wait()
            self.run_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.status_label.setText("Status: Stopped")
            
    def update_log(self, message):
        self.log_text.append(message)
        self.status_label.setText(f"Status: {message[:50]}...")
        
    def update_progress(self, value):
        self.progress.setValue(value)
        
    def model_finished(self, result):
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("Status: Completed")
        
        results_text = f"Model: {self.model_combo.currentText()}\n"
        results_text += f"Dataset: {self.dataset_combo.currentText()}\n"
        results_text += f"Test Accuracy: {result['accuracy']:.4f}\n"
        results_text += f"Hidden Dim: {self.hidden_dim_spin.value()}\n"
        results_text += f"Layers: {self.num_layers_spin.value()}\n"
        
        self.log_text.append("\n" + "="*50)
        self.log_text.append(results_text)
        
        # Show computation graph info
        graph_data = result.get('graph_data', {})
        if graph_data:
            self.log_text.append("\nComputation Graph Info:")
            self.log_text.append(f"Number of nodes: {len(graph_data)}")
            
        # Show operator fusion info
        self.log_text.append("\nOperator Fusion:")
        self.log_text.append("Jittor automatically fuses operators during compilation.")
        self.log_text.append("Fusion reduces memory access and improves performance.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())