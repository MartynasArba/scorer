from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout,
    QPushButton, QLabel,
    QFileDialog,
    QCheckBox
)

from data.onebox_utils import run_conversion, convert_multiple_recs, get_folder_quality_report
from data.motionsensor_utils import parse_sensors

class UtilWidget(QWidget):
    """
    class to represent the tab that houses non-scoring funcs
    """
    def __init__(self, metadata: dict = None):
        super().__init__()
        #metadata - might not exist yet
        self.params = metadata
        
        layout = QVBoxLayout(self)
        
        self.label = QLabel("Ready!")
        layout.addWidget(self.label)
        
        #sub-layouts
        self.obx_layout = QVBoxLayout()
        self.sensor_layout = QVBoxLayout()
        layout.addLayout(self.obx_layout)
        layout.addLayout(self.sensor_layout)
        
        self.overwrite_check = QCheckBox("overwrite existing files? only applies to folder conversion")
        self.obx_layout.addWidget(self.overwrite_check)
        
        btn_obx_to_csv = QPushButton('convert selected onebox recording to .csv files')
        btn_obx_to_csv.clicked.connect(self.obx_to_csv)
        self.obx_layout.addWidget(btn_obx_to_csv)
        
        btn_obx_to_csv_all = QPushButton('convert all onebox recordings to .csv files')
        btn_obx_to_csv_all.clicked.connect(self.folder_to_csv)
        self.obx_layout.addWidget(btn_obx_to_csv_all)
        
        btn_quality_report = QPushButton('get quality report for all files in folder')
        btn_quality_report.clicked.connect(self.generate_quality_report)
        self.obx_layout.addWidget(btn_quality_report)
        
        self.quality_plot_check = QCheckBox("get plots of quality reports?")
        self.obx_layout.addWidget(self.quality_plot_check)
    
        btn_parse_sensors = QPushButton('split sensor files by cage')
        btn_parse_sensors.clicked.connect(self.parse_sensor_data)
        self.sensor_layout.addWidget(btn_parse_sensors)
        
        
    def obx_to_csv(self) -> None:
        """
        opens file selection and runs downsampling + conversion to csv
        """
        
        filename, _ = QFileDialog.getOpenFileName(self, caption = "Select onebox .bin file to convert", 
                                                  directory = self.params.get('project_path', '.'), 
                                                  filter = "BIN files (*.bin)")
        if filename:
            save_folder = QFileDialog.getExistingDirectory(self,'select folder to save in', self.params.get('project_path', '.'), QFileDialog.ShowDirsOnly)
            self.label.setText(f'selected {filename} for obx conversion') 
            run_conversion(filename, self.params, save_folder= save_folder, sr_new = int(self.params.get('sample_rate', '1000')))
            self.label.setText(f'{filename} obx file converted to csvs') 
            
    def folder_to_csv(self) -> None:
        """
        opens file selection and runs downsampling + conversion to csv
        """
        
        path = QFileDialog.getExistingDirectory(self,'select folder to load from', self.params.get('project_path', '.'), QFileDialog.ShowDirsOnly)
        
        if path:
            save_folder = QFileDialog.getExistingDirectory(self,'select folder to save in', self.params.get('project_path', '.'), QFileDialog.ShowDirsOnly)
            self.label.setText(f'converting all obx files from {path}') 
            convert_multiple_recs(path, self.params, save_folder= save_folder, sr_new = int(self.params.get('sample_rate', '1000')), overwrite = self.overwrite_check.isChecked())
            self.label.setText(f'{path} obx folder converted to csvs') 
            
    def generate_quality_report(self):
        """
        calculates stds for all channels, all recs in folder
        saves as csv
        """
        path = QFileDialog.getExistingDirectory(self,'select folder', self.params.get('project_path', '.'), QFileDialog.ShowDirsOnly)
        
        if path:
            self.label.setText(f'generating quality report for files in {path}') 
            get_folder_quality_report(path, savepath = self.params.get('project_path', '.'), save_fig = self.quality_plot_check.isChecked())
            self.label.setText(f'{path} report generated') 
    
    def parse_sensor_data(self):
        path = QFileDialog.getExistingDirectory(self,'select folder', self.params.get('project_path', '.'), QFileDialog.ShowDirsOnly)
        if path:
            self.label.setText(f'splitting sensor data into separate csvs in {path}') 
            parse_sensors(path)
            self.label.setText(f'{path} sensors processed') 
        