from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QMessageBox,
    QFileDialog, QLineEdit, QCheckBox
)

from torchaudio.functional import resample
from data.preprocessing import load_from_csv, load_from_csv_in_chunks, bandpass_filter, notch_filter, sum_power
from data.storage import save_tensor, save_windowed

#TODO: adapt run_preprocessing to new _preprocessing output
#TODO: finish preprocessing funcs, missing sum and band powers
#TODO: add status bar instead of printing
#TODO: to avoid freezing, move to thread
#TODO: move to .h5 for chunked data (or rethink otherwise): current option might not be efficient and cause crashing if files become too large
#TODO: in storage, move path construction to a helper function
#states are saved together with processed data (except in windows) because whole file viewing is impossible in this GUI, so preprocess/raw saving is only for exporting
#print what was done and in what order is the tensor stacked, should always be ecog [num_channels], emg [num_channels], extracted features, states
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
        self.chunk_size_field.setText("1000000")
        chunk_layout = QHBoxLayout()
        chunk_layout.addWidget(self.chunk_check)
        chunk_layout.addWidget(self.chunk_size_field)
                
        self.resample_check = QCheckBox("resample data? new sr:") #add option to select sample rate to resample to
        self.sr_field = QLineEdit(self)
        self.sr_field.setText("1000")
        sr_layout = QHBoxLayout()
        sr_layout.addWidget(self.resample_check)
        sr_layout.addWidget(self.sr_field)   
        
        self.filter_check = QCheckBox("bandpass filter ecog data? limits:") #add option to select filter boundaries
        self.low_bound_field = QLineEdit(self)
        self.high_bound_field = QLineEdit(self)
        self.low_bound_field.setText("0.5")
        self.high_bound_field.setText("30")
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(self.filter_check)
        filter_layout.addWidget(self.low_bound_field)
        filter_layout.addWidget(self.high_bound_field)  
        
        self.emg_filter_check = QCheckBox("bandpass filter emg data? limits:") #add option to select filter boundaries
        self.emg_low_bound_field = QLineEdit(self)
        self.emg_high_bound_field = QLineEdit(self)
        self.emg_low_bound_field.setText("10")
        self.emg_high_bound_field.setText("100")
        emg_filter_layout = QHBoxLayout()
        emg_filter_layout.addWidget(self.emg_filter_check)
        emg_filter_layout.addWidget(self.emg_low_bound_field)
        emg_filter_layout.addWidget(self.emg_high_bound_field)  
        
        self.notch_check = QCheckBox("use notch filter? freq to remove:")    #add option to select custom notch filter
        self.notch_value_field = QLineEdit(self)
        self.notch_value_field.setText("50")
        notch_layout = QHBoxLayout()
        notch_layout.addWidget(self.notch_check)
        notch_layout.addWidget(self.notch_value_field)   
        
        #no param options        
        self.calculate_sum_power_check = QCheckBox("calculate sum power?") 
        self.calculate_band_power_check = QCheckBox("calculate band power?") 
        self.testing_check = QCheckBox("use pre-scored states when chopping (for testing only)")
        
        #checkmarks for saving options
        saving_layout = QHBoxLayout()
        self.save_raw_check = QCheckBox('save raw tensors?')
        self.save_preprocessed_check = QCheckBox('save preprocessed tensors?')
        self.save_windowed_check = QCheckBox('save windowed data for scoring? window size:')
        self.win_len_field = QLineEdit(self)
        self.win_len_field.setText("1000")
        self.save_overwrite_check = QCheckBox("overwrite files")
        saving_layout.addWidget(self.save_raw_check)
        saving_layout.addWidget(self.save_preprocessed_check)
        saving_layout.addWidget(self.save_windowed_check)
        saving_layout.addWidget(self.win_len_field)
        saving_layout.addWidget(self.save_overwrite_check)
        
        for l in [chunk_layout, sr_layout, filter_layout, emg_filter_layout, notch_layout, saving_layout]:
            layout.addLayout(l)
        
        for box in [self.calculate_sum_power_check, self.calculate_band_power_check, self.testing_check]:
            layout.addWidget(box)
        
        #button that runs everything
        main_btn = QPushButton("run preprocessing")
        main_btn.clicked.connect(self.run_preprocessing)
        layout.addWidget(main_btn)

    def run_preprocessing(self) -> None:
        """
        reads what steps were selected and runs reading, preprocessing, saving
        
        """
        if self.selected_file:      # if not set, will be None
            #load file, check if chunks are needed
            chunk_size = None if not self.chunk_check.isChecked() else int(self.chunk_size_field.text()) #set chunk size if checked
            states = None if not self.testing_check.isChecked() else -1 #mark last col as states if checked
            if not self.chunk_check.isChecked():    
                raw_ecog, raw_emg, states = load_from_csv(self.selected_file, metadata = self.params, states = states)
                tensor_seq = (raw_ecog, raw_emg) if not self.testing_check.isChecked() else (raw_ecog, raw_emg, states.unsqueeze(-1))
                print('data read done')
                
                if self.save_raw_check.isChecked():
                    save_tensor(tensor_seq = tensor_seq, 
                                metadata = self.params,
                                overwrite = self.save_overwrite_check.isChecked(),
                                chunk = None,
                                raw = True)
                    
                ecog, emg = self._preprocess(raw_ecog, raw_emg) #this should handle which preprocessing steps should be done 
                tensor_seq = (ecog, emg) if not self.testing_check.isChecked() else (ecog, emg, states.unsqueeze(-1))
                print('preprocessing done')
                
                if self.save_preprocessed_check.isChecked():
                    save_tensor(tensor_seq = tensor_seq, 
                                metadata = self.params,
                                overwrite = self.save_overwrite_check.isChecked(),
                                chunk = None,
                                raw = False)
                    
                if self.save_windowed_check.isChecked():
                    save_windowed(tensors = (ecog, emg), 
                                    states = states, 
                                    metadata = self.params, 
                                    win_len = int(self.win_len_field.text()),
                                    chunked = False, 
                                    overwrite = self.save_overwrite_check.isChecked(),
                                    testing = self.testing_check.isChecked())
    
            else:
                for i, (ecog_chunk, emg_chunk, states_chunk) in enumerate(load_from_csv_in_chunks(self.selected_file, metadata = self.params, states = states, chunk_size = chunk_size)):
                    print(f'read chunk {i}')
                    tensor_seq = (ecog_chunk, emg_chunk) if not self.testing_check.isChecked() else (ecog_chunk, emg_chunk, states_chunk.unsqueeze(-1))
                    if self.save_raw_check.isChecked():
                        save_tensor(tensor_seq = tensor_seq, 
                                metadata = self.params,
                                overwrite = self.save_overwrite_check.isChecked(),
                                chunk = i,
                                raw = True)
                    
                    ecog, emg = self._preprocess(ecog_chunk, emg_chunk)
                    tensor_seq = (ecog, emg) if not self.testing_check.isChecked() else (ecog, emg, states_chunk.unsqueeze(-1))   #change so states are saved separatelly
                    print(f'preprocessing chunk {i}')
                    
                    if self.save_preprocessed_check.isChecked():
                        save_tensor(tensor_seq = tensor_seq, 
                                metadata = self.params,
                                overwrite = self.save_overwrite_check.isChecked(),
                                chunk = i,
                                raw = False)
                        
                    if self.save_windowed_check.isChecked():
                        win_len = int(self.win_len_field.text())
                        if chunk_size % win_len != 0:
                            print('some data will be lost, chunk size is not divisible by window length')
                        save_windowed(tensors = (ecog, emg), 
                                    states = states_chunk, 
                                    metadata = self.params, 
                                    win_len = win_len,
                                    chunked = True, 
                                    overwrite = self.save_overwrite_check.isChecked(),
                                    testing = self.testing_check.isChecked())
                    #TODO: remove testing
                    if i >= 5:
                        break
        
    def _not_chunked_warning(self):
        """
        warning that the system might run out of memory because data isn't being loaded in chunks
        """
        
        if not self.chunk_check.isChecked():
                #warn that the system might run out of memory 
                response = QMessageBox.warning(self, 'warning',
                    'chunk loading is not selected, so filtering might cause the system to run out of memory\ncontinue?',
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel)
                return response
        else:
            return None
        
    def _preprocess(self, ecog, emg):
        """
        check what's checked, run corresponding funcs
        """
        self.params['preprocessing'] = []
        
        if self.resample_check.isChecked():
            print(f'before resampling: {ecog.size()}')
            ecog, emg = self._resample(ecog, emg)
            print(f'after resampling: {ecog.size()}')
            
        if ((self.filter_check.isChecked() | self.emg_filter_check.isChecked()) | self.notch_check.isChecked()) & (not self.chunk_check.isChecked()):
            warn = self._not_chunked_warning()
        else: 
            warn = None
        
        if self.filter_check.isChecked():
            if warn == QMessageBox.StandardButton.Cancel:
                print('ecog filtering cancelled')
                return ecog
            else:
                freqs = (float(self.low_bound_field.text()), float(self.high_bound_field.text()))
                ecog = self._bandpass(ecog, freqs)
                print('ecog data filtered')
                
        if self.emg_filter_check.isChecked():
            if warn == QMessageBox.StandardButton.Cancel:
                print('emg filtering cancelled')
                return emg                                
            else:
                    freqs = (float(self.emg_low_bound_field.text()), float(self.emg_high_bound_field.text()))
                    emg = self._bandpass(emg, freqs)
                    print('emg data filtered')
                    
        if self.notch_check.isChecked():
            if warn == QMessageBox.StandardButton.Cancel:
                print('notch filtering cancelled')
                return ecog, emg                                
            else:
                freq = int(self.notch_value_field.text())
                ecog, emg = self._notch(ecog, freq), self._notch(emg, freq)
                print('notch filtering done')
    
        if self.calculate_sum_power_check.isChecked():
            if warn == QMessageBox.StandardButton.Cancel:
                print('notch filtering cancelled')
            else:
                ecog_power = sum_power(ecog, smoothing = 0.2, sr = self.params['sample_rate'], device = self.params['device'], normalize = True)
                emg_power = sum_power(emg, smoothing = 0.2, sr = self.params['sample_rate'], device = self.params['device'], normalize = True)
        else:
            ecog_power, emg_power = None, None
        
        if self.calculate_band_power_check.isChecked():
            pass
    
        print(f'preprocessing steps done: {self.params["preprocessing"]}')
            
        return (ecog, emg, ecog_power, emg_power)
    
    def _resample(self, ecog, emg):
        """
        applies resample func
        """
        
        sample_rate = float(self.params.get('sample_rate', 0))
        new_rate = int(self.sr_field.text())
        self.params['preprocessing']+= ['resampling']
        self.params['old_sample_rate'] = self.params['sample_rate']
        self.params['sample_rate'] = new_rate
        ecog = ecog.T #initially shape is (sample, channel) because of how its saved, resample requires time to be last dim
        emg = emg.T
        ecog = resample(ecog.contiguous(), sample_rate, new_rate)
        emg = resample(emg.contiguous(), sample_rate, new_rate)
        return ecog, emg
        
    def _bandpass(self, signal, freqs):
        """
        applies bandpass filtering func
        """
        self.params['preprocessing']+= ['bandpass']
        if 'filter_freqs' in self.params.keys():
            self.params['filter_params'] += [freqs]
        else:
            self.params['filter_params'] = [freqs]
            
        signal = bandpass_filter(signal = signal, 
                                 sr = self.params.get('sample_rate', 250), 
                                 freqs = freqs, 
                                 metadata = self.params)
        return signal
    
    def _notch(self, signal, freq):
        """
        applies notch filtering for the specified freq
        """
        self.params['preprocessing']+= ['notch']
        if 'notch_freq' not in self.params.keys():
            self.params['notch_freq'] = [freq]
            
        signal = notch_filter(signal = signal,
                              sr = self.params.get('sample_rate', 250),
                              freq_to_remove = freq,
                              metadata = self.params)
        return signal
    
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