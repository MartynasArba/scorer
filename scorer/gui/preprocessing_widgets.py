from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout,
    QPushButton, QLabel,
    QFileDialog, QRadioButton, 
    QButtonGroup
)

from data.preprocessing import from_Oslo_csv, from_non_annotated_csv
       
#another widget to run data chopping on selected data
class PreprocessWidget(QWidget):
    """
    class of the preprocessing widget. 
    ideally should do (most) of the preprocessing here that happens on the raw signal. 
    """
    def __init__(self) -> None:
        """
        creates the widget and buttons
        """
        super().__init__()

        layout = QVBoxLayout(self)
        
        self.chopper = from_Oslo_csv#default option
        
        #add radio buttons to toggle between annotated and raw data
        self.label = QLabel("Select data to preprocess")
        layout.addWidget(self.label)
        
        self.toggle_group = QButtonGroup()
        self.toggle1 = QRadioButton('load Oslo data')
        self.toggle1.setChecked(True)
        # self.toggle1.toggled.connect(self.select_chopper)
        
        self.toggle2 = QRadioButton('load non-annotated data')
        # self.toggle2.toggled.connect(self.select_chopper)
        
        self.toggle_group.addButton(self.toggle1, id = 0)
        self.toggle_group.addButton(self.toggle2, id = 1)
        self.toggle_group.setExclusive(True)
        
        layout.addWidget(self.toggle1)
        layout.addWidget(self.toggle2)
        
        self.toggle_group.buttonClicked[int].connect(self.select_chopper)

        self.label = QLabel("No file selected")
        layout.addWidget(self.label)

        btn = QPushButton("Select file to chop")
        btn.clicked.connect(self.select_file)
        layout.addWidget(btn)
    
    def select_chopper(self, value: int) -> None:
        """
        selects how to chop data: based on states (for model training) or just by time (for scoring)
        """
        funcmap = {
            0 : from_Oslo_csv,
            1 : from_non_annotated_csv
        }
        
        self.chopper = funcmap.get(value, None)

    def select_file(self) -> None:
        """
        opens file dialog and runs chopping, should split it into separate steps
        """
        file_name, _ = QFileDialog.getOpenFileName(self, caption = "Select file to chop", directory = ".", filter = "CSV files (*.csv)")
        if file_name:
            self.label.setText(file_name)            
            if self.chopper:
                print(file_name)
                self.chopper(file_name)