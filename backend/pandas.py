from functools import reduce
from typing import List
from weakref import WeakKeyDictionary

import pandas as pd

from dsl.backend import Backend
from dsl.expression import Identity, expr_identity, column_path, Source, Call, Operator, Scalar, Expr, Lookup, \
    CombineIdentities, MissingSourceError
from dsl.io import PythonDataFrameReader, CSVDataFrameWriter, DataFrameWriter
from dsl.pipe import Pipe

__all__ = ('PandasBackend', )


class PandasBackend(Backend):
    def __init__(self):
        self._compiled = WeakKeyDictionary()

    def collect_value(self, expr) -> List[any]:
        if isinstance(expr, Pipe):
            raise MissingSourceError(f'Missing data source for expression ? >> {expr}')
        value = self._get_value_from_expr(expr)
        if isinstance(value, pd.DataFrame):
            return value.to_dict(orient='records')
        if isinstance(value, pd.Series):
            return value.tolist()
        raise NotImplementedError

    def write_dataframe(self, writer: DataFrameWriter):
        if isinstance(writer, CSVDataFrameWriter):
            df: pd.DataFrame = self._get_value_from_expr(writer.df)
            assert writer.mode == 'overwrite', '@TODO'
            assert writer.compression is None, '@TODO'
            df.to_csv(writer.path, sep=writer.separator, header=writer.header, encoding=writer.encoding, quotechar=writer.quote, escapechar=writer.escape, index=False, **writer.additional_options)
            return
        raise NotImplementedError

    def _get_value_from_expr(self, expr):
        if expr not in self._compiled:
            value = self._create_value_from_expr(expr)
            try:
                self._compiled[expr] = value
            except TypeError:
                return value  # It's trying to weak reference ints
        return self._compiled[expr]

    def _create_value_from_expr(self, expr):
        if isinstance(expr, PythonDataFrameReader):
            return pd.DataFrame.from_records(expr.data)
        if isinstance(expr, Scalar):
            return expr.value
        if isinstance(expr, Source):
            return self._get_value_from_expr(expr_identity(expr))
        if isinstance(expr, CombineIdentities):
            return pd.DataFrame({
                field_name: self._get_value_from_expr(field_expr)
                for field_name, field_expr in expr.__select_fields__.items()
            })
        if isinstance(expr, Lookup):
            identity = expr_identity(expr)
            path = column_path(expr)
            df = self._get_value_from_expr(identity)
            return reduce(lambda _df, key: _df[key], path, df)
        if isinstance(expr, Operator):
            return expr.f(self._get_value_from_expr(expr.a), self._get_value_from_expr(expr.b))
        if isinstance(expr, Call):
            arg_values = {f'arg{i}': self._get_value_from_expr(e) for i, e in enumerate(expr.args)}
            return pd.DataFrame(arg_values).apply(lambda args: expr.f(*args), axis=1)
        if not isinstance(expr, (Expr, Identity)):
            return self._get_value_from_expr(Scalar(expr))
        raise NotImplementedError
