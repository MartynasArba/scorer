import torch
from PyQt5.QtWidgets import (
    QWidget
)

class AutoScoringWidget(QWidget):
    """
    class for automatic scoring, wip
    """
    def __init__(self):
        super().__init__()
        torch.set_grad_enabled(True) #should happen in training funcs
        #code goes here
        pass
        torch.set_grad_enabled(False) 
    