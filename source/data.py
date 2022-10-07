"""
Class for storing data
"""

from dataclasses import dataclass
from typing import Tuple, Union
import pandas as pd
import numpy as np

@dataclass
class Dataset:
    
    external_params: Tuple
    dataframe: pd.DataFrame

    def __init__(self, external_params: Tuple, dataframe: pd.DataFrame=None, csv_file: str=None) -> None:
        self.external_params = external_params
        if dataframe is not None:
            self.dataframe = dataframe
        elif csv_file is not None:
            self.dataframe = pd.read_csv(csv_file)
    
    def __getitem__(self, key: Union(str, int)) -> np.ndarray:
        
        if isinstance(key, str):
            return self.dataframe[key], self.external_params
        
        if isinstance(key, int):
            return self.dataframe.iloc[key], self.external_params
        