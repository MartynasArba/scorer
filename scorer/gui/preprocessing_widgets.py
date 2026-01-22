from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QMessageBox,
    QFileDialog, QLineEdit, QCheckBox
)

from torchaudio.functional import resample
from scorer.data.preprocessing import bandpass_filter, sum_power, band_powers, notch_filter
from scorer.data.storage import save_tensor, save_windowed, save_metadata, load_from_csv, load_from_csv_in_chunks
from pathlib import Path

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
        
        self.timechop_check = QCheckBox("load data between specific time? time (HH:MM:SS):")
        self.timechop_start = QLineEdit(self)
        self.timechop_start.setText("19:00:00")
        self.timechop_end = QLineEdit(self)
        self.timechop_end.setText("07:00:00")
        timechop_layout = QHBoxLayout()
        timechop_layout.addWidget(self.timechop_check)
        timechop_layout.addWidget(self.timechop_start)
        timechop_layout.addWidget(self.timechop_end)
    
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
        self.high_bound_field.setText("49")
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
        
        #checkmarks for saving options
        saving_layout = QHBoxLayout()
        self.save_raw_check = QCheckBox('save raw tensors?')
        self.save_preprocessed_check = QCheckBox('save preprocessed tensors?')
        self.save_windowed_check = QCheckBox('save windowed data for scoring? window size:')
        self.win_len_field = QLineEdit(self)
        self.win_len_field.setText("1000")
        self.save_overwrite_check = QCheckBox("overwrite files")
        self.add_filename_check = QCheckBox('add original file name when saving?')
        saving_layout.addWidget(self.save_raw_check)
        saving_layout.addWidget(self.save_preprocessed_check)
        saving_layout.addWidget(self.save_windowed_check)
        saving_layout.addWidget(self.win_len_field)
        saving_layout.addWidget(self.save_overwrite_check)
        saving_layout.addWidget(self.add_filename_check)
        
        for l in [chunk_layout, timechop_layout, sr_layout, filter_layout, emg_filter_layout, notch_layout, saving_layout]:
            layout.addLayout(l)
        
        for box in [self.calculate_sum_power_check, self.calculate_band_power_check]:
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
            #set original file name
            if self.add_filename_check:
                self.params['filename'] = Path(self.selected_file).stem
            #load file, check if chunks are needed
            chunk_size = None if not self.chunk_check.isChecked() else int(self.chunk_size_field.text()) #set chunk size if checked
            states = None
            if not self.chunk_check.isChecked():    
                times = (None, None)
                if self.timechop_check.isChecked():
                    times = (self.timechop_start.text(), self.timechop_end.text())
                raw_ecog, raw_emg, states = load_from_csv(self.selected_file, metadata = self.params, states = states, times = times)
                tensor_seq = (raw_ecog, raw_emg)
                self.label.setText('data read done')
                
                if self.save_raw_check.isChecked():
                    save_tensor(tensor_seq = tensor_seq, 
                                metadata = self.params,
                                overwrite = self.save_overwrite_check.isChecked(),
                                chunk = None,
                                raw = True)
                    
                tensor_seq = self._preprocess(raw_ecog, raw_emg) #this should handle which preprocessing steps should be done 
                self.label.setText('preprocessing done')
                
                if self.save_preprocessed_check.isChecked():
                    save_tensor(tensor_seq = tensor_seq, 
                                metadata = self.params,
                                overwrite = self.save_overwrite_check.isChecked(),
                                chunk = None,
                                raw = False)

                if self.save_windowed_check.isChecked():
                    save_windowed(tensors = tensor_seq, 
                                    states = states, 
                                    metadata = self.params, 
                                    win_len = int(self.win_len_field.text()),
                                    chunked = False, 
                                    overwrite = self.save_overwrite_check.isChecked(),
                                    testing = False)
                    
                #fix sample rate if resampling in the end
                if self.resample_check.isChecked():
                    self.params['preprocessing']+= ['resampling']
                    self.params['old_sample_rate'] = self.params['sample_rate']
                    self.params['sample_rate'] = int(self.sr_field.text())
            
            else:
                times = (None, None)
                if self.timechop_check.isChecked():
                    times = (self.timechop_start.text(), self.timechop_end.text())
                for i, (ecog_chunk, emg_chunk, states_chunk) in enumerate(load_from_csv_in_chunks(self.selected_file, metadata = self.params, states = states, chunk_size = chunk_size, times = times)):
                    print(f'read chunk {i}')
                    tensor_seq = (ecog_chunk, emg_chunk)
                    if self.save_raw_check.isChecked():
                        save_tensor(tensor_seq = tensor_seq, 
                                metadata = self.params,
                                overwrite = self.save_overwrite_check.isChecked(),
                                chunk = i,
                                raw = True)
                    
                    tensor_seq = self._preprocess(ecog_chunk, emg_chunk)
                                        
                    if self.save_preprocessed_check.isChecked():                        
                        save_tensor(tensor_seq = tensor_seq, 
                                metadata = self.params,
                                overwrite = self.save_overwrite_check.isChecked(),
                                chunk = i,
                                raw = False)
                        
                    if self.save_windowed_check.isChecked():
                        win_len = int(self.win_len_field.text())
                        if chunk_size % win_len != 0:
                            QMessageBox.warning(self, 'warning', 
                                                f'Data will be lost as chunk_size isn\'t divisible by window length!\n Chunk size: {chunk_size}, window length: {win_len}',
                                                QMessageBox.StandardButton.Yes)
                        
                        # append_file = True if i > 0  else False  
                            
                        save_windowed(tensors = tensor_seq, 
                                    states = states_chunk, 
                                    metadata = self.params, 
                                    win_len = win_len,
                                    chunked = True, #append_file previously
                                    chunk_id = i,
                                    overwrite = self.save_overwrite_check.isChecked(),
                                    testing = False)
                #if resampling, fix metadata
                if self.resample_check.isChecked():
                    self.params['preprocessing']+= ['resampling']
                    self.params['old_sample_rate'] = self.params['sample_rate']
                    self.params['sample_rate'] = int(self.sr_field.text())
                        
        meta_path = self.params.get('metadata_path', None)
        if meta_path is None:
            print('warning - metadata path is not set! setting to project path')
            meta_path = self.params.get('project_path', '.')
        save_metadata(meta_path, self.params)
        
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
        #construct channel explainer
        
        self.params['channels_after_preprocessing'] = ['ecog'] * len(self.params['ecog_channels'].split(',')) + ['emg'] * len(self.params['emg_channels'].split(','))
        
        self.params['preprocessing'] = []
        ecog, emg = ecog.T, emg.T       #torch usually requires channels x time
            
        if ((self.filter_check.isChecked() | self.emg_filter_check.isChecked()) | self.notch_check.isChecked() | 
            self.calculate_band_power_check.isChecked() | self.calculate_sum_power_check.isChecked()) & (not self.chunk_check.isChecked()):
            warn = self._not_chunked_warning()
        else: 
            warn = None
            
        if self.filter_check.isChecked():
            if warn == QMessageBox.StandardButton.Cancel:
                print('ecog filtering cancelled')
            else:
                freqs = (float(self.low_bound_field.text()), float(self.high_bound_field.text()))
                ecog = self._bandpass(ecog, freqs)
                print('ecog data filtered')
                
        if self.emg_filter_check.isChecked():
            if warn == QMessageBox.StandardButton.Cancel:
                print('emg filtering cancelled')                         
            else:
                    freqs = (float(self.emg_low_bound_field.text()), float(self.emg_high_bound_field.text()))
                    emg = self._bandpass(emg, freqs)
                    print('emg data filtered')
                    
        if self.notch_check.isChecked():
            if warn == QMessageBox.StandardButton.Cancel:
                print('notch filtering cancelled')                          
            else:
                freq = int(self.notch_value_field.text())
                ecog, emg = self._notch(ecog, freq), self._notch(emg, freq)
                print('notch filtering done')
    
        if self.calculate_sum_power_check.isChecked():
            if warn == QMessageBox.StandardButton.Cancel:
                print('sum pow calculation cancelled')
            else:
                ecog_power = sum_power(ecog, smoothing = 0.2, sr = int(self.params.get('sample_rate', 250)), device = self.params.get('device', 'cuda'), normalize = True, gaussian_smoothen=0.2)
                emg_power = sum_power(emg, smoothing = 0.2, sr = int(self.params.get('sample_rate', 250)), device = self.params.get('device', 'cuda'), normalize = True, gaussian_smoothen=0.2)
                self.params['preprocessing']+= ['sum_pows_calculated']
                self.params['channels_after_preprocessing'] += ['ecog_sum_power'] * len(self.params['ecog_channels'].split(',')) + ['emg_sum_power'] * len(self.params['emg_channels'].split(','))
                print('sum pows calculated')
        else:
            ecog_power, emg_power = None, None
        
        if self.calculate_band_power_check.isChecked():
            if warn == QMessageBox.StandardButton.Cancel:
                print('band pow calculation cancelled')
            else:
                bands = band_powers(signal = ecog, bands = {'delta': (0.5, 4),
                                                            'theta': (5, 9),
                                                            'alpha': (8, 13),
                                                            'sigma': (12, 15)}, 
                                    sr = int(self.params.get('sample_rate', 250)), 
                                    device= self.params['device'], smoothen = 0.2)
                self.params['preprocessing']+= ['band_pows_calculated']
                self.params['channels_after_preprocessing'] += list(bands.keys())
                print('band pows calculated')
            
        else:
            bands = {None : None}        
        
        if self.resample_check.isChecked(): #this has to be at the end as sample rate only gets overwritten after preprocessing, so filters need to use old sample rate
            print(f'before resampling: {ecog.size()}')
            ecog, emg = self._resample(ecog, emg)   #should be fine here still
            print(f'after resampling: {ecog.size()}')
        
        print('preprocessing done')
        self.label.setText('preprocessing done')
        
        return (ecog, emg, ecog_power, emg_power) + tuple(bands.values())
         
    
    def _resample(self, ecog, emg):
        """
        applies resample func
        """
        
        sample_rate = float(self.params.get('sample_rate', 250))
        new_rate = int(self.sr_field.text())
        
        if sample_rate == new_rate:
            print('not resampling: old sr = new sr')
            return ecog, emg
        
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
        print(f'in bandpass filter signal size: {signal.size()}')
        signal = bandpass_filter(signal,
                        sr = int(self.params.get('sample_rate', 250)),
                        freqs = freqs,
                        device = self.params.get('device', 'cuda'))
        return signal
    
    def _notch(self, signal, freq):
        """
        applies notch filtering for the specified freq
        """
        self.params['preprocessing']+= ['notch']
        if 'notch_freq' not in self.params.keys():
            self.params['notch_freq'] = [freq]
            
        signal = notch_filter(signal,
                        sr = int(self.params.get('sample_rate', 250)),
                        freq = freq,
                        device = self.params.get('device', 'cuda'))
        return signal

    def select_file(self) -> None:
        """
        opens file dialog and sets path
        """
        filename, _ = QFileDialog.getOpenFileName(self, caption = "Select file to preprocess", 
                                                  directory = self.params.get('project_path', '.'), 
                                                  filter = "CSV files (*.csv)")
        if filename:
            self.label.setText(f'{filename} selected for preprocessing, press run to do') 
            self.selected_file = filename          