from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, 
    QPushButton, QLabel, QFileDialog, QMessageBox
)

from datetime import datetime
from scorer.data.storage import save_metadata, load_metadata

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
        
        self.defaults = {'animal_id': '0',
                        'sample_rate': '1000',
                        'ecog_channels': '0,1',
                        'emg_channels': '2',
                        'time_channel': '3',
                        'spectral_options': None,
                        'ylim': 'infer_ephys',
                        'device': 'cuda'
                        }
        
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
        
        #set default
        btn_default = QPushButton('set default')
        btn_default.clicked.connect(self.default_metadata_func)
        self.metadata_layout.addWidget(btn_default)
        
        #buttons to save and load
        btn_save = QPushButton('save all params to file')
        btn_save.clicked.connect(self.save_metadata_func)
        self.metadata_layout.addWidget(btn_save)
        
        btn_load = QPushButton('load params from .json file')
        btn_load.clicked.connect(self.load_metadata_func)
        self.metadata_layout.addWidget(btn_load)
        
        #buttons to validate or reset
        btn_validate_metadata = QPushButton('validate metadata params')
        btn_validate_metadata.clicked.connect(self.validate_metadata)
        self.metadata_layout.addWidget(btn_validate_metadata)
        
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
            self.params['metadata_path'] = filename
            save_metadata(self.params['metadata_path'], self.params)
        self.warn_if_not_set()
    
    def load_metadata_func(self) -> None:
        """
        loads metadata form json
        """
        filename, _ = QFileDialog.getOpenFileName(self, 'load file', self.params['project_path'], 'Text files (*.json *.txt)')
        if filename:
            updated_data = load_metadata(filename)
            print(updated_data)
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
        resets metadata to nothing
        """
        for param in self.params.keys():
            self.params[param] = None
        self.params['scoring_started'] = str(datetime.now().strftime('%Y%m%d%H%M%S'))
        
        self.update_label()
        
    def default_metadata_func(self):
        """
        fills metadata with some default values
        """
        for key, val in self.defaults.items():
            if key in self.params.keys():
                self.params[key] = val
        self.update_label()
        
    def validate_metadata(self):
        """
        validates metadata, doing manual checks
        """
        params_valid = True
        
        numeric_keys = ['animal_id', 'sample_rate']
        channel_keys = ['ecog_channels', 'emg_channels']
        ylim_options = ['infer', 'infer_ephys' , 'standard']
        spectral_options = ['fourier', 'spect', None]
        
        for key in self.params.keys():
            if not self.params[key]:
                print(f'warning: {key} is empty')
            elif (key in numeric_keys):
                if not (self.params[key].isdigit()):
                    print(f'{key} should be a number')
                    params_valid = False
            elif (key in channel_keys):
                if (not self.params[key].split(',')[0].isdigit()) & (len(self.params[key]) > 1):
                    print(f'{key} is wrong; channels should be listed as numbers starting from 0, separated by commas, no spaces.')
                    params_valid = False
            elif (key == 'ylim') & (not (self.params[key] in ylim_options)):
                print(f'{key} is not an ylim option. select from {ylim_options}')
                params_valid = False
            elif (key == 'spectral_view') & (not (self.params[key] in spectral_options)):
                print(f'{key} is not a valid spectral option. select from {spectral_options}')
                params_valid = False
                
        print(f'params valid: {params_valid}')    
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("Parameter check")
        if not params_valid:
            msg.setText("Some parameters were not set correctly!")
        else:
            msg.setText("All parameters good!")
        msg.exec_()
                    
        return params_valid
        