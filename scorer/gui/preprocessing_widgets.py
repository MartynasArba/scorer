from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout,
    QPushButton, QLabel,
    QFileDialog, QRadioButton, 
    QButtonGroup, QCheckBox
)

from data.preprocessing import from_Oslo_csv, from_non_annotated_csv
       
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
        self.chunk_check = QCheckBox("load in chunks?") #text: chunk size
        self.resample_check = QCheckBox("resample data?") #add option to select sample rate to resample to
        self.filter_check = QCheckBox("filter data 0.5-30 Hz?") #add option to select filter boundaries
        self.notch_check = QCheckBox("use notch 50 filter?")    #add option to select custom notch filter
        self.calculate_sum_power_check = QCheckBox("calculate sum power?") 
        self.calculate_band_power_check = QCheckBox("calculate band power?")
        self.save_processed_check = QCheckBox("save processed signal?")   
        self.chop_check = QCheckBox("chop data into chunks for the next step")   #add option to select win length
        self.testing_check = QCheckBox("use pre-scored states when chopping (for testing only)")
        self.overwrite_files_check = QCheckBox("overwrite files")
        
        for box in [
            self.chunk_check, self.resample_check, self.filter_check,
            self.notch_check, self.calculate_sum_power_check,
            self.calculate_band_power_check, self.save_processed_check,
            self.chop_check, self.testing_check, self.overwrite_files_check
        ]:
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
        pass
    
        #check that self.selected_file is set and exists
        # read checkboxes (ex. if self.filter_check.isChecked(): do stuff)   
        #maybe preprocessing safely in a background thread (eventually, to avoid UI freezing).
        #write pop-up messages when done
        #mark what was done in metadata
        #add progress bar

    def select_file(self) -> None:
        """
        opens file dialog and sets path
        """
        filename, _ = QFileDialog.getOpenFileName(self, caption = "Select file to chop", directory = self.params.get('project_path', '.'), filter = "CSV files (*.csv)")
        if filename:
            self.label.setText(filename) 
            self.selected_file = filename          