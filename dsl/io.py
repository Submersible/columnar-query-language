from typing import Iterable, Dict, Optional, Union

from pandas import DataFrame as PandasDataFrame

from dsl.backend import get_backend, Backend
from dsl.expression import Identity, Source, Expr


class DataFrameReader(Identity):
    pass


class DataFrameWriter(object):
    def __init__(self, df):
        self.df = df

    def execute(self, backend: Backend):
        backend.write_dataframe(self)
        return self


class reader(object):
    @staticmethod
    def parquet(path, *, schema=None) -> Source:
        raise NotImplementedError

    @staticmethod
    def json(path, *, schema=None) -> Source:
        raise NotImplementedError

    @staticmethod
    def pandas(df: PandasDataFrame, *, partitions: Optional[int]=None, schema=None) -> Source:
        raise NotImplementedError

    @staticmethod
    def python(data: Iterable[Dict[str, any]], *, partitions: Optional[int]=None, schema=None) -> Source:
        return Source(identity=PythonDataFrameReader(data, partitions=partitions, schema=schema))


class writer(object):
    @staticmethod
    def parquet(df: Union[Expr, DataFrameReader], *, path: str, mode: str='error', compression: str=None, **kwargs):
        return ParquetDataFrameWriter(df, path=path, mode=mode, compression=compression, **kwargs).execute(get_backend())

    @staticmethod
    def csv(
        df: Union[Expr, DataFrameReader], *, path: str, mode: str='error', compression: str=None,
        separator=',', encoding='utf-8', quote='"', escape='\\', header=True, **kwargs
    ):
        return CSVDataFrameWriter(
            df, path=path, mode=mode, compression=compression,
            separator=separator, encoding=encoding, quote=quote, escape=escape, header=header, **kwargs
        ).execute(get_backend())


class PythonDataFrameReader(DataFrameReader):
    def __init__(self, data: Iterable[Dict[str, any]], *, partitions: Optional[int]=None, schema=None):
        super().__init__(schema=schema)
        self.data = data
        self.partitions = partitions


class ParquetDataFrameReader(DataFrameReader):
    def __init__(self, path: str, *, partitions=None, schema=None, **kwargs):
        super().__init__(schema=schema)
        self.path = path
        self.partitions = partitions
        self.additional_options = dict(kwargs)


class ParquetDataFrameWriter(DataFrameWriter):
    def __init__(self, df: Union[Expr, DataFrameReader], *, path: str, mode: str='error', compression: str=None, **kwargs):
        assert mode in {'error', 'overwrite', 'append'}

        super().__init__(df)
        self.path = path
        self.mode = mode
        self.compression = compression
        self.additional_options = dict(kwargs)


class CSVDataFrameWriter(DataFrameWriter):
    def __init__(
        self, df: Union[Expr, DataFrameReader], *, path: str, mode: str, compression: str,
        encoding, separator, quote, escape, header, **kwargs
    ):
        assert mode in {'error', 'overwrite', 'append'}

        super().__init__(df)
        self.path = path
        self.mode = mode
        self.compression = compression
        self.encoding = encoding
        self.separator = separator
        self.quote = quote
        self.escape = escape
        self.header = header
        self.additional_options = dict(kwargs)
