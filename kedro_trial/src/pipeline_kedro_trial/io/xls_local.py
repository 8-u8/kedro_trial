# coding: utf-8
from os.path import isfile
from typing import Any, Union, Dict

import pandas as pd

from kedro.io import AbstractDataSet


class ExcelLocalDataSet(AbstractDataSet):
    ''' ローカル環境でExcelデータを読み込もうという試み。
    '''

    def _describe(self) -> Dict[str, Any]:
        output = dict(filepath=self._filepath,
                      engine=self._engine,
                      load_args=self._load_args,
                      save_args=self._save_args)
        return output
    # いにっと
    # ファイル読み込みのパスとファイル名を認識
    # その後、pd.read_excelで読み込む関数_load
    # 書き込む関数_save
    # を書いた。 -> None:とかは不明。
    def __init__(
        self,
        filepath: str,
        engine: str = "xlsxwriter",
        load_args: Dict[str, Any] = None,
        save_args: Dict[str, Any] = None,
        ) -> None:

        self._filepath = filepath
        default_save_args = {}
        default_load_args = {"engine": "xlrd"}

        self._load_args = {**default_load_args, **load_args} \
            if load_args is not None else default_load_args
        self._save_args = {**default_save_args, **save_args} \
            if save_args is not None else default_save_args
        self._engine = engine

    def _load(self) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        return pd.read_excel(self._filepath, **self._load_args)

    def _save(self, data: pd.DataFrame) -> None:
        writer = pd.ExcelWriter(self._filepath, engine=self._engine)
        data.to_excel(writer, **self._save_args)
        writer.save()
        #print('successfully saved!')

    def _exists(self) -> bool:
        return isfile(self._filepath)
