from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, 
    QPushButton, QLabel, QFileDialog, QMessageBox
)

from datetime import datetime
from data.storage import save_metadata, load_metadata

class SettingsWidget(QWidget):
    """
    class for metadata and settings, wip
    """
    def __init__(self, metadata: dict) -> None:
        """
        creates settings window
        """
        super().__init__()
        
        self.params = metadata
        self.params['scoring_started'] = str(datetime.now().strftime('%Y%m%d%H%M%S')) #only generate once per file, otherwise should reset
        self.params['project_path'] = '.'
        self.params['date'] = str(datetime.now().strftime('%Y_%m_%d'))
        
        #main layout
        layout = QVBoxLayout(self)
        
        #label
        self.mainlabel = QLabel("No data set")
        layout.addWidget(self.mainlabel)
        
        #metadata section
        self.metadata_layout = QVBoxLayout()
        layout.addLayout(self.metadata_layout)
        
        # project path selection button
        btn_proj_path = QPushButton('select project path')
        btn_proj_path.clicked.connect(self.set_project_path)
        self.metadata_layout.addWidget(btn_proj_path)
    
        for p in self.params.keys():
            if (p != 'scoring_started') & (p != 'project_path'):
                self.add_param_row(p)
        
        #buttons to save and load
        btn_save = QPushButton('save all params to file')
        btn_save.clicked.connect(self.save_metadata_func)
        self.metadata_layout.addWidget(btn_save)
        
        btn_load = QPushButton('load params from .json file')
        btn_load.clicked.connect(self.load_metadata_func)
        self.metadata_layout.addWidget(btn_load)
        
        
        btn_reset_metadata = QPushButton('reset metadata params')
        btn_reset_metadata.clicked.connect(self.reset_metadata)
        self.metadata_layout.addWidget(btn_reset_metadata)
            
        self.update_label()

    def add_param_row(self, param: str) -> None:
        """
        adds textboxes with buttons
        """
        row_layout = QHBoxLayout()
        #text box
        textbox = QLineEdit(self)
        textbox.setObjectName(f"{param}")
        row_layout.addWidget(textbox)
        #button to read
        btn = QPushButton(f"set {param}")
        btn.clicked.connect(lambda _, param = param, tbox = textbox: self.set_param(param, tbox.text()))       
        row_layout.addWidget(btn)
        #add to metadata layout
        self.metadata_layout.addLayout(row_layout)
        
    def set_project_path(self) -> str:
        """
        opens dialog, returns path to selected folder
        """
        path = QFileDialog.getExistingDirectory(self,'select project folder', ".", QFileDialog.ShowDirsOnly)
        if path:
            self.params['project_path'] = path
        self.update_label()
        
    def save_metadata_func(self) -> None:
        """
        saves metadata to json
        """
        filename, _ = QFileDialog.getSaveFileName(self, 'save file', self.params.get('project_path', '.') + f'/{self.params.get('scoring_started', '')}_meta.json', 'Text files (*.json *.txt)')
        if filename:
            save_metadata(filename, self.params)
        self.warn_if_not_set()
    
    def load_metadata_func(self) -> None:
        """
        loads metadata form json
        """
        filename, _ = QFileDialog.getOpenFileName(self, 'load file', self.params['project_path'], 'Text files (*.json *.txt)')
        if filename:
            updated_data = load_metadata(filename)
            if isinstance(updated_data, dict):
                self.params.clear() #remove old keys but keep old object so sync is not broken
                self.params.update(updated_data) #merge new values into existing dict
            self.update_label()
            self.warn_if_not_set()
        
    def warn_if_not_set(self) -> None:
        """
        warns if some values are still nto set
        """
        if None in self.params.values():
            QMessageBox.information(self, 'warning', 'some values are still not set')
        
    def update_label(self) -> None:
        """
        updates label text by current metadata values
        """
        text = ''
        for param, val in self.params.items():
            text += f'current {param} value: {val}\n'
        self.mainlabel.setText(text)
        
    def set_param(self, param: str, new_val: str) -> None:
        """
        sets metadata param to specified value
        """
        if param in self.params.keys():
            self.params[param] = new_val
        self.update_label()

    def reset_metadata(self):
        """
        resets metadata
        """
        for param in self.params.keys():
            self.params[param] = None
        self.params['scoring_started'] = str(datetime.now().strftime('%Y%m%d%H%M%S'))
        
        self.update_label()