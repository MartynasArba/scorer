from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QComboBox, QPushButton, QFileDialog, QLabel, QCheckBox, QLineEdit, QHBoxLayout
)

from scorer.models.scoring import score_signal

class AutoScoringWidget(QWidget):
    """
    class for automatic scoring, holds buttons
    """
    def __init__(self, meta: dict):
        super().__init__()
        
        self.params = meta
        available_models = ['select model','3state_GRU', 'random_forest', 'context_rf'] # '3state_pretrained', '3state_dual', '3state_SCDS', '5state_pretrained']
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
        
        #checkbox to apply corrections
        self.correction_check = QCheckBox('correct scores by common heuristics? (remove W->R transitions,  single-window NREM or REM?)')
        layout.addWidget(self.correction_check)
        
        #text to check whether some specific channel should be used for scoring
        self.scoring_ch_check = QCheckBox("select a specific channel for scoring? channel:")
        self.scoring_ch_field = QLineEdit(self)
        self.scoring_ch_field.setText("0")
        scoring_ch_layout = QHBoxLayout()
        scoring_ch_layout.addWidget(self.scoring_ch_check)
        scoring_ch_layout.addWidget(self.scoring_ch_field)
        layout.addLayout(scoring_ch_layout)
        
        #button to select file folder
        button_sel_file = QPushButton('select data folder')
        button_sel_file.clicked.connect(self.select_file_folder)
        layout.addWidget(button_sel_file)
        
        #button to select state folder
        button_sel_state= QPushButton('select scores folder')
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
        """runs imported score_signal function after validating paths"""
        if not self.file_folder or self.file_folder == '.':
            self.label.setText('select a valid data folder')
            return
        if not self.state_folder or self.state_folder == '.':
            self.label.setText('select a valid scores folder')
            return

        self.label.setText('scoring...')
        if self.scoring_ch_check.isChecked():
            try:
                selected_ch = int(self.scoring_ch_field.text())
                print(f'channel for auto scoring selected: {selected_ch}')
            except:
                print('invalid channel value passed')
        else:
            selected_ch = 0
        
        score_signal(self.file_folder,
                     self.state_folder,
                     meta=self.params,
                     selected_ch = selected_ch,
                     scorer_type = str(self.model_selection.currentText()),
                     apply_corrections=self.correction_check.isChecked())
        self.label.setText('done')