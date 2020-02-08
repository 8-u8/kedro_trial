import pandas as pd
import numpy as np


def preprocessing(usedata: pd.DataFrame) -> pd.DataFrame:
    for i in range(1,40):
        var_name = 'Var.' + str(i)
        usedata[var_name] = 1 - usedata[var_name]
    return usedata
