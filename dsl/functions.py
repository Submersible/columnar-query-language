from functools import wraps
from typing import Union, Sequence, List, TypeVar, Set

from dsl.backend import get_backend
from dsl.expression import JoinIdentity, Expr, ColumnType, Call, Source, CombineIdentities, expr_identity, Identity
from dsl.pipe import pipe_context

T, A, B = TypeVar('T'), TypeVar('A'), TypeVar('B')


def collect(value: Expr) -> List[any]:
    return get_backend().collect_value(value)


def join(left: Source, right: Source, *, on: Union[Sequence[ColumnType], Expr], how: str) -> Source:
    return Source(identity=JoinIdentity(expr_identity(left), expr_identity(right), on, how))


def explode(column):
    raise NotImplementedError


def udf(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        return Call(identity=None, f=f, args=list(args), kwargs=kwargs)
    return wrapper


def group_by(*columns):
    # def group_by(self, f: Callable[[T], A], partitions=None) -> 'RDD[Tuple[A, List[T]]]':
    raise NotImplementedError


def agg(*args, **kwargs):
    raise NotImplementedError


@pipe_context
def select(df_or_expression: Union['Source', 'Expr']=None, *args, **kwargs):
    if args:
        raise NotImplementedError
    return Source(identity=CombineIdentities(**kwargs))


def append(**kwargs):
    raise NotImplementedError


def persist(value: Expr) -> Expr:
    raise NotImplementedError


def collect_list(values: List[T]) -> List[List[T]]:
    raise NotImplementedError


def collect_set(values: List[T]) -> List[Set[T]]:
    raise NotImplementedError


def everything():
    raise NotImplementedError


def where(expr):
    raise NotImplementedError


def map_partition(f):
    raise NotImplementedError


def foreach_partitions(f):
    raise NotImplementedError


def take(n) -> List[T]:
    raise NotImplementedError


def count(self) -> int:
    raise NotImplementedError


def sample(self, replacement: bool, fraction: float, seed=None):
    raise NotImplementedError


def reduce(self, f):
    raise NotImplementedError


def repartition(expr):
    raise NotImplementedError


def partitions(self) -> int:
    raise NotImplementedError
