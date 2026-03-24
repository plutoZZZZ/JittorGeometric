#!/usr/bin/env python3
"""Complete JittorGeometric Frontend with Graph Visualization"""

import sys
import jittor as jt
from jittor import nn
import jittor_geometric as jg
from jittor_geometric.nn.conv import GCNConv, GATConv, SAGEConv
from jittor_geometric.datasets import Planetoid
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QGroupBox, QFormLayout, QComboBox, QPushButton, QTextEdit,
    QSpinBox, QDoubleSpinBox, QProgressBar, QLabel, QSplitter
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

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

class GraphVisualizationWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        
        self.figure = plt.figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)
        
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMaximumHeight(100)
        self.layout.addWidget(self.info_text)
        
    def plot_graph(self, graph_data):
        self.figure.clear()
        
        try:
            G = nx.DiGraph()
            
            if isinstance(graph_data, dict):
                for node, info in graph_data.items():
                    G.add_node(node)
                    if 'inputs' in info:
                        for inp in info['inputs']:
                            G.add_edge(inp, node)
            
            pos = nx.spring_layout(G, seed=42)
            nx.draw(G, pos, with_labels=True, node_size=3000, 
                   node_color='skyblue', font_size=8, font_weight='bold')
            
            self.canvas.draw()
            
            self.info_text.setText(f"Nodes: {G.number_of_nodes()}\nEdges: {G.number_of_edges()}")
            
        except Exception as e:
            self.info_text.setText(f"Error plotting graph: {str(e)}")

class OperatorFusionWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        
        self.fusion_text = QTextEdit()
        self.fusion_text.setReadOnly(True)
        self.layout.addWidget(self.fusion_text)
        
        self.fusion_text.setText("Operator fusion information will be displayed here after model execution.")
    
    def update_fusion_info(self):
        try:
            info_text = "Jittor Operator Fusion Information:\n\n"
            info_text += "="*50 + "\n"
            info_text += "Jittor automatically performs operator fusion to optimize performance.\n\n"
            info_text += "Key Fusion Techniques:\n"
            info_text += "1. Kernel Fusion - Combines multiple operations into a single CUDA kernel\n"
            info_text += "2. Memory Fusion - Reduces memory access by reusing buffers\n"
            info_text += "3. Loop Fusion - Merges nested loops to improve cache utilization\n\n"
            info_text += "Benefits:\n"
            info_text += "- Reduced memory bandwidth usage\n"
            info_text += "- Lower kernel launch overhead\n"
            info_text += "- Improved cache locality\n"
            info_text += "- Higher overall throughput\n\n"
            info_text += "Jittor's optimizer automatically analyzes the computation graph\n"
            info_text += "and applies the best fusion strategies based on hardware capabilities."
            
            self.fusion_text.setText(info_text)
            
        except Exception as e:
            self.fusion_text.setText(f"Error getting fusion info: {str(e)}")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("JittorGeometric Complete Demo")
        self.setGeometry(100, 100, 1200, 800)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Left panel (controls)
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        left_panel.setLayout(left_layout)
        left_panel.setMaximumWidth(400)
        
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
        
        left_layout.addWidget(model_group)
        
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
        
        left_layout.addWidget(hyper_group)
        
        # Buttons
        btn_layout = QHBoxLayout()
        self.run_btn = QPushButton("Run Model")
        self.run_btn.clicked.connect(self.run_model)
        btn_layout.addWidget(self.run_btn)
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_model)
        self.stop_btn.setEnabled(False)
        btn_layout.addWidget(self.stop_btn)
        
        left_layout.addLayout(btn_layout)
        
        # Progress
        self.progress = QProgressBar()
        left_layout.addWidget(self.progress)
        
        # Status
        self.status_label = QLabel("Status: Ready")
        left_layout.addWidget(self.status_label)
        
        # Log
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        left_layout.addWidget(self.log_text)
        
        # Right panel (output tabs)
        self.tab_widget = QTabWidget()
        
        # Graph visualization tab
        self.graph_widget = GraphVisualizationWidget()
        self.tab_widget.addTab(self.graph_widget, "Computation Graph")
        
        # Operator fusion tab
        self.fusion_widget = OperatorFusionWidget()
        self.tab_widget.addTab(self.fusion_widget, "Operator Fusion")
        
        # Results tab
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.tab_widget.addTab(self.results_text, "Results")
        
        main_layout.addWidget(left_panel)
        main_layout.addWidget(self.tab_widget)
        
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
        
        results_text = f"Model Training Results:\n"
        results_text += "="*50 + "\n"
        results_text += f"Model: {self.model_combo.currentText()}\n"
        results_text += f"Dataset: {self.dataset_combo.currentText()}\n"
        results_text += f"Test Accuracy: {result['accuracy']:.4f}\n"
        results_text += f"Hidden Dim: {self.hidden_dim_spin.value()}\n"
        results_text += f"Layers: {self.num_layers_spin.value()}\n"
        results_text += f"Epochs: {self.epochs_spin.value()}\n"
        
        self.results_text.setText(results_text)
        
        # Update graph visualization
        self.graph_widget.plot_graph(result.get('graph_data', {}))
        
        # Update operator fusion info
        self.fusion_widget.update_fusion_info()

if __name__ == "__main__":
    # Check if running in headless environment
    import os
    if 'DISPLAY' not in os.environ or os.environ['DISPLAY'] == '' or os.environ.get('DISPLAY', '').startswith(':'):
        print("Running in headless environment. Starting in non-interactive mode...")
        print("\nAvailable models: GCN, GAT, GraphSAGE")
        print("Available datasets: Cora, Citeseer, Pubmed")
        print("\nTo use GUI mode, please set DISPLAY environment variable or use a virtual display server.")
        print("\nRecommended for cloud servers: Use headless mode or install xvfb.")
        
        # Run a simple test instead of showing GUI
        from model_training_demo import ModelTrainer
        trainer = ModelTrainer(model_name='GCN', dataset_name='Cora')
        trainer.load_data()
        trainer.create_model()
        print("\nModel created successfully: GCN on Cora dataset")
        
        # Test training
        print("\nStarting training for 10 epochs...")
        trainer.train(epochs=10)
        acc = trainer.test()
        print(f"\nTest accuracy: {acc:.4f}")
        print("\nHeadless test completed successfully")
    else:
        print("Starting GUI mode...")
        app = QApplication(sys.argv)
        window = MainWindow()
        window.show()
        sys.exit(app.exec_())