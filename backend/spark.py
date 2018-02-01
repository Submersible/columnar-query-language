import logging
from typing import List
from weakref import WeakKeyDictionary

import pyspark
from pyspark import SparkContext
from pyspark.sql import SparkSession, Row
from pyspark.sql import functions as SF

from dsl.backend import Backend
from dsl.expression import Identity
from dsl.io import PythonDataFrameReader, DataFrameWriter, CSVDataFrameWriter, ParquetDataFrameReader

logger = logging.getLogger(__name__)

__all__ = ('SparkBackend',)


class SparkBackend(Backend):
    spark: SparkSession
    sc: SparkContext

    def __init__(self):
        self.spark = None
        self.sc = None
        self._compiled = WeakKeyDictionary()

    def __enter__(self):
        self.spark = SparkSession.builder.getOrCreate()
        return super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.spark.stop()
        return super().__exit__(exc_type, exc_val, exc_tb)

    def collect_value(self, expr) -> List[any]:
        return self._get_value_from_expr(expr).collect()

    def write_dataframe(self, writer: DataFrameWriter):
        if isinstance(writer, CSVDataFrameWriter):
            raise NotImplementedError
        raise NotImplementedError

    def _get_value_from_expr(self, expr):
        if expr not in self._compiled:
            value = self._create_value_from_expr(expr)
            try:
                self._compiled[expr] = value
            except TypeError:
                return value  # It's trying to weak reference ints
        return self._compiled[expr]

    def get_dataframe(self, value):
        if isinstance(value, Identity):
            return self.get_identity(value)
        raise NotImplementedError

    def get_identity(self, identity: 'Identity') -> 'pyspark.sql.DataFrame':
        if identity not in self._compiled:
            if isinstance(identity, PythonDataFrameReader):
                self._compiled[identity] = self.spark.createDataFrame([Row(**item) for item in identity.data])
            else:
                raise NotImplementedError
        return self._compiled[identity]

    def _create_value_from_expr(self, value):
        if isinstance(value, ParquetDataFrameReader):
            return SparkDataFrameReader(
                format='parquet',
                partitions=value.partitions,
                **value.additional_options
            )
        raise NotImplementedError

    # NullType -> None / NaN / NaT
    # StringType -> str
    # BinaryType -> bytes
    # BooleanType -> bool
    # DateType -> date
    # TimestampType -> datetime
    # DecimalType -> Decimal
    # DoubleType -> Fraction
    # FloatType -> float
    # ByteType <- str
    # IntegerType -> int
    # LongType -> long
    # ShortType -> int
    # ArrayType[a] -> List[a]
    # MapType[a, b] -> Dict[a, b]
    # a : StructType -> __annotation__ = {k: v for StructField(k, v) for a.fields()}


class SparkDataFrame(object):
    def to_spark(self, spark: SparkSession, sc: SparkContext):
        raise NotImplementedError


class SparkDataFrameReader(SparkDataFrame):
    def __init__(self, format, *args, **kwargs):
        self.format = format
        self.reader_args = args
        self.reader_kwargs = kwargs

    def to_spark(self, spark: SparkSession, sc: SparkContext):
        reader = getattr(spark.read, self.format)
        return reader(*self.reader_args, **self.reader_kwargs)


class SparkExpr(object):
    def to_spark(self, spark: SparkSession, sc: SparkContext):
        raise NotImplementedError


class SparkCallable(object):
    pass


class SparkUDF(SparkCallable, SparkExpr):
    def __init__(self, f, *, schema=None):
        self.f = f
        self.schema = schema

    def to_spark(self, spark: SparkSession, sc: SparkContext):
        return SF.udf(self.f, returnType=self.schema)


class SparkFunction(SparkCallable, SparkExpr):
    def __init__(self, f):
        self.f = f

    def to_spark(self, spark: SparkSession, sc: SparkContext):
        return self.f


class SparkSelect(SparkExpr):
    def __init__(self, df: SparkDataFrame, exprs: List[SparkExpr]):
        self.df = df
        self.exprs = exprs

    def to_spark(self, spark: SparkSession, sc: SparkContext):
        return self.df.to_spark(spark, sc).select(*(e.to_spark(spark, sc) for e in self.exprs))


class SparkColumn(SparkExpr):
    def __init__(self, key: str):
        self.key = key

    def to_spark(self, spark: SparkSession, sc: SparkContext):
        return SF.col(self.key)


class SparkLookup(SparkExpr):
    def __init__(self, *, expr: SparkExpr, key: str):
        self.expr = expr
        self.key = key

    def to_spark(self, spark: SparkSession, sc: SparkContext):
        return self.expr.to_spark(spark, sc)[self.key]
