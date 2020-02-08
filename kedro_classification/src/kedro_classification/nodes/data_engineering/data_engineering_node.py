#coding: utf-8
import pandas as pd
import numpy as pd

def preprocessing(data: pd.DataFrame) -> pd.DataFrame:
    for i in range(1,40):
        var_name = 'Var.' + str(i)
        data[var_name] = 1 - data[var_name]
    return data