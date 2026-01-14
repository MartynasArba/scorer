from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QComboBox, QPushButton, QFileDialog, QLabel
)

from scorer.models.scoring import score_signal

class AutoScoringWidget(QWidget):
    """
    class for automatic scoring, holds buttons
    """
    def __init__(self, meta):
        super().__init__()
        
        self.params = meta
        available_models = ['none', 'heuristic']
        layout = QVBoxLayout(self)
        
        self.file_folder = '.'
        self.state_folder = '.'
        
        #label
        self.label = QLabel('select model')
        layout.addWidget(self.label)
        
        #dropdown to select model
        self.model_selection = QComboBox()
        self.model_selection.addItems(available_models)
        layout.addWidget(self.model_selection)
        
        #button to select file folder
        button_sel_file = QPushButton('select data folder')
        button_sel_file.clicked.connect(self.select_file_folder)
        layout.addWidget(button_sel_file)
        
        #button to select state folder
        button_sel_state= QPushButton('select data folder')
        button_sel_state.clicked.connect(self.select_state_folder)
        layout.addWidget(button_sel_state)
                
        #button to run scoring
        run_button = QPushButton('run scoring')
        run_button.clicked.connect(self.run_scoring)
        layout.addWidget(run_button)
        
        
    def select_file_folder(self):
        self.file_folder = QFileDialog.getExistingDirectory(self,'select data folder containing X and y files', self.params.get('project_path', '.'), QFileDialog.ShowDirsOnly)
        if not self.file_folder:
            self.file_folder = '.'
        self.label.setText(f'selected data in {self.file_folder}')
            
    def select_state_folder(self):
        self.state_folder = QFileDialog.getExistingDirectory(self,'select folder to save in', self.params.get('project_path', '.'), QFileDialog.ShowDirsOnly)
        if not self.state_folder:
            self.state_folder = '.'
        self.label.setText(f'states will be saved in {self.state_folder}')
            
    def run_scoring(self):
        score_signal(self.file_folder, 
                     self.state_folder, 
                     meta = self.params, 
                     scorer_type = str(self.model_selection.currentText()))
        self.label.setText('scoring done')