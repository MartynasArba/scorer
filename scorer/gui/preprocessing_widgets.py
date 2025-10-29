from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel,
    QFileDialog, QLineEdit, QCheckBox
)

from data.preprocessing import from_Oslo_csv, from_non_annotated_csv, load_from_csv, load_from_csv_in_chunks
from torchaudio.functional import resample
       
# TODO: write more funcs in data.preprocessing to reflect checkmarks
# TODO: add textboxes next to checks where values are needed

class PreprocessWidget(QWidget):
    """
    class of the preprocessing widget. 
    ideally should do (most) of the preprocessing here that happens on the raw signal. 
    """
    def __init__(self, metadata) -> None:
        """
        creates the widget and buttons
        """
        super().__init__()
        #keep track of metadata
        self.params = metadata
        self.selected_file = None

        layout = QVBoxLayout(self)    
        #add main info label
        self.label = QLabel("select data to preprocess")
        layout.addWidget(self.label)
        
        #select input file, add support for other formats later if needed
        btn_select_file = QPushButton('select file to load')
        btn_select_file.clicked.connect(self.select_file)
        layout.addWidget(btn_select_file)
        
        #add checkboxes of what preprocessing steps should be done
        #add textboxes next to some of these
        self.chunk_check = QCheckBox("load in chunks? set size:") #text: chunk size
        self.chunk_size_field = QLineEdit(self)
        self.chunk_size_field.setText("100000")
        chunk_layout = QHBoxLayout()
        chunk_layout.addWidget(self.chunk_check)
        chunk_layout.addWidget(self.chunk_size_field)
                
        self.resample_check = QCheckBox("resample data? new sr:") #add option to select sample rate to resample to
        self.sr_field = QLineEdit(self)
        self.sr_field.setText("1000")
        sr_layout = QHBoxLayout()
        sr_layout.addWidget(self.resample_check)
        sr_layout.addWidget(self.sr_field)   
        
        self.filter_check = QCheckBox("bandpass filter data? limits:") #add option to select filter boundaries
        self.low_bound_field = QLineEdit(self)
        self.high_bound_field = QLineEdit(self)
        self.low_bound_field.setText("0.5")
        self.high_bound_field.setText("30")
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(self.filter_check)
        filter_layout.addWidget(self.low_bound_field)
        filter_layout.addWidget(self.high_bound_field)  
        
        self.notch_check = QCheckBox("use notch filter? freq to remove:")    #add option to select custom notch filter
        self.notch_value_field = QLineEdit(self)
        self.notch_value_field.setText("50")
        notch_layout = QHBoxLayout()
        notch_layout.addWidget(self.notch_check)
        notch_layout.addWidget(self.notch_value_field)   
        
        
        self.chop_check = QCheckBox("chop data into chunks for the next step? window size: ")   #add option to select win length
        self.win_len_field = QLineEdit(self)
        self.win_len_field.setText("1000")
        chop_layout = QHBoxLayout()
        chop_layout.addWidget(self.chop_check)
        chop_layout.addWidget(self.win_len_field)  
        
        #no param options        
        self.calculate_sum_power_check = QCheckBox("calculate sum power?") 
        self.calculate_band_power_check = QCheckBox("calculate band power?")
        self.save_processed_check = QCheckBox("save processed signal?")   
        self.testing_check = QCheckBox("use pre-scored states when chopping (for testing only)")
        self.overwrite_files_check = QCheckBox("overwrite files")
        
        for l in [chunk_layout, sr_layout, filter_layout, notch_layout, chop_layout]:
            layout.addLayout(l)
        
        for box in [self.calculate_sum_power_check, self.calculate_band_power_check, 
                    self.save_processed_check, self.testing_check, self.overwrite_files_check]:
            layout.addWidget(box)
        
        #button that runs everything
        main_btn = QPushButton("run preprocessing")
        main_btn.clicked.connect(self.run_preprocessing)
        layout.addWidget(main_btn)

    def run_preprocessing(self) -> None:
        """
        reads what steps were selected and runs preprocessing
        """
        #TODO: implement actual preprocessing
        
        if self.selected_file:      # if not set, will be None
            #load file, check if chunks are needed
            chunk_size = None if not self.chunk_check.isChecked() else int(self.chunk_size_field.text()) #set chunk size if checked
            states = None if not self.testing_check.isChecked() else -1 #mark last col as states if checked
            if not self.chunk_check.isChecked():    
                raw_ecog, raw_emg, states = load_from_csv(self.selected_file, metadata = self.params, states = states)
                print('run further preprocessing on whole file')
                self._preprocess(raw_ecog, raw_emg) #this should handle which preprocessing steps should be done 
                pass
            
            else:
                for i, (ecog_chunk, emg_chunk, states_chunk) in enumerate(load_from_csv_in_chunks(self.selected_file, metadata = self.params, states = states, chunk_size = chunk_size)):
                    print('run further preprocessing on chunks')
                    self._preprocess(ecog_chunk, emg_chunk)
                    if i > 5:
                        break
                    pass            
        else:
            self.label.setText('select a valid file!')
    
        #check that self.selected_file is set and exists
        # read checkboxes (ex. if self.filter_check.isChecked(): do stuff)   
        #maybe preprocessing in a background thread (eventually, to avoid UI freezing).
        #write pop-up messages when done
        #mark what was done in metadata
        #add progress bar
        
    def _preprocess(self, ecog, emg):
        """
        check what's checked, run corresponding funcs
        """
        self.params['preprocessing'] = []
        
        if self.resample_check.isChecked():
            print(f'before resampling: {ecog.size()}')
            ecog, emg = self._resample(ecog, emg)
            print(f'after resampling: {ecog.size()}')
        
        print(f'preprocessing steps done: {self.params['preprocessing']}')
            
        return ecog, emg
    
    def _resample(self, ecog, emg):
        sample_rate = float(self.params.get('sample_rate', 0))
        new_rate = int(self.sr_field.text())
        self.params['preprocessing']+= ['resampling']
        self.params['new_sample_rate'] = new_rate
        ecog = ecog.T #initially shape is (sample, channel) because of how its saved, resample requires time to be last dim
        emg = emg.T
        ecog = resample(ecog, sample_rate, new_rate)
        emg = resample(emg, sample_rate, new_rate)
        return ecog, emg
        
    
    def _chop(self, ecog, emg, states = None):
        """
        run chopping func
        """ 
        pass

    def select_file(self) -> None:
        """
        opens file dialog and sets path
        """
        filename, _ = QFileDialog.getOpenFileName(self, caption = "Select file to chop", directory = self.params.get('project_path', '.'), filter = "CSV files (*.csv)")
        if filename:
            self.label.setText(filename) 
            self.selected_file = filename          