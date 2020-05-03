import pandas as pd
import numpy as np

'''
Doc todo
- make preprocessing function 
  for train data and test data.

'''

def preprocessing(usedata: pd.DataFrame) -> pd.DataFrame:
    for i in range(1,14):
        var_name = 'Var_' + str(i)
        usedata[var_name] = 1 - usedata[var_name]
    return usedata
