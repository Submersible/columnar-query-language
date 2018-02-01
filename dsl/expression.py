import logging
import operator
from collections import OrderedDict
from typing import NamedTuple, List, Optional, Union, TypeVar

import pytest

T, A, B = TypeVar('T'), TypeVar('A'), TypeVar('B')
ColumnType = Union['Column', 'str']
logger = logging.getLogger(__name__)

class IdentityTypeError(TypeError):
    pass


class FieldNotFoundTypeError(TypeError):
    pass


class MissingSourceError(TypeError):
    pass


class Identity(object):
    def __init__(self, *, schema=None):
        if schema is not None:
            raise NotImplementedError

        self.schema = schema

    def identities(self):
        return [self]

    def __repr__(self):
        params = ', '.join(f'{k}={v}' for k, v in self.__dict__.items())
        return f'{self.__class__.__name__}(<{params}>)'


class CombineIdentities(Identity):
    def __init__(self, **kwargs):
        super().__init__(schema=None)
        self.__select_fields__ = OrderedDict(kwargs.items())
        self.__select_keys__ = kwargs.keys()


class ExprMeta(type):
    class OpDefinition(NamedTuple):
        name: str
        symbol: str

    ops = {
        OpDefinition(name='add', symbol='+'),
        OpDefinition(name='sub', symbol='-'),
        OpDefinition(name='mul', symbol='*'),
        OpDefinition(name='truediv', symbol='/'),
        OpDefinition(name='floordiv', symbol='//'),
        OpDefinition(name='mod', symbol='%'),
        OpDefinition(name='pow', symbol='**'),
        # OpDefinition(name='lshift', symbol='<<'),
        # OpDefinition(name='rshift', symbol='>>'),
        OpDefinition(name='and', symbol='&'),
        OpDefinition(name='or', symbol='|'),
        OpDefinition(name='xor', symbol='^'),
    }

    def __new__(cls, name, bases, namespace, **kwds):
        def make_method(f, symbol, reverse=False):
            def op(self, other):
                a, b = (other, self) if reverse else (self, other)
                return Operator(identity=None, f=f, symbol=symbol, a=a, b=b)
            return op

        for op in cls.ops:
            name = f'__{op.name}__'
            f = getattr(operator, name)
            namespace[name] = make_method(f, op.symbol)
            namespace[f'__r{op.name}__'] = make_method(f, op.symbol, reverse=True)

        result = type.__new__(cls, name, bases, dict(namespace))
        result.members = tuple(namespace)
        return result


class Expr(metaclass=ExprMeta):
    def __init__(self, *, identity: Optional['Identity']):
        self.__expr_identity__ = identity


class Scalar(Expr):
    def __init__(self, value, *, identity=None):
        super().__init__(identity=identity)
        self.value = value

    def __repr__(self):
        return str(self.value)


class Source(Expr):
    def __init__(self, *, schema=None, identity=None):
        super().__init__(identity=identity)
        if schema is not None:
            raise NotImplementedError

        self.schema = schema

    def identities(self):
        return [self]

    def __call__(self, expression: 'Expr') -> 'WithIdentity':
        return WithIdentity(expression, identity=expr_identity(self))

    def __getattr__(self, item) -> 'SourceLookup':
        return SourceLookup(item, identity=expr_identity(self))

    def __getitem__(self, item) -> 'SourceLookup':
        return SourceLookup(item, identity=expr_identity(self))

    def __repr__(self):
        return f'Source<{expr_identity(self)}>'


class Lookup(Expr):
    def __init__(self, name: str, *, parent=None, identity=None, schema=None):
        super().__init__(identity=identity)
        self.__column_name__ = name
        self.__column_parent__ = parent
        self.__column_schema__ = schema

    def __getattr__(self, item: str) -> 'Lookup':
        return Lookup(item, parent=self, identity=self.__expr_identity__)

    def __getitem__(self, item) -> 'Lookup':
        return Lookup(item, parent=self, identity=self.__expr_identity__)

    def __repr__(self):
        return f'c.{".".join(column_path(self))}'


class SourceLookup(Lookup):
    def __getattr__(self, item: str) -> 'SourceLookup':
        return SourceLookup(item, parent=self, identity=self.__expr_identity__)

    def __getitem__(self, item) -> 'SourceLookup':
        return SourceLookup(item, parent=self, identity=self.__expr_identity__)


class Call(Expr):
    def __init__(self, *, identity, f, args=None, kwargs=None):
        super().__init__(identity=identity)
        self.f = f
        self.args = tuple(args)
        self.kwargs = kwargs or {}

    def __repr__(self):
        params = [str(x) for x in self.args]
        params += [f'{k}={str(v)}' for k, v in self.kwargs.items()]
        params = ', '.join(params)
        return f'{self.f.__name__}({params})'

    def __eq__(self, other):
        if not isinstance(other, Call):
            return False
        return (self.f == other.f and self.args == other.args
                and self.kwargs == other.kwargs)

    def __hash__(self):
        return hash((self.f.__name__, self.f.__module__, tuple(self.args), tuple((self.kwargs or {}).items())))


class Operator(Call):
    def __init__(self, *, identity, f, symbol, a, b):
        super().__init__(identity=identity, f=f, args=[a, b])
        self.symbol = symbol
        self.a = a
        self.b = b

    def __repr__(self):
        return f'{self.a} {self.symbol} {self.b}'


class WithIdentity(Expr):
    def __init__(self, expression, *, identity):
        super().__init__(identity=identity)
        self.expression = expression

    def __repr__(self):
        return f'{str(expr_identity(self))}({str(self.expression)})'


class JoinIdentity(Identity):
    def __init__(self, a: 'Identity', b: 'Identity', on, how):
        assert how in {'left', 'right', 'inner', 'outer'}

        super().__init__()
        self.a = a
        self.b = b
        self.on = tuple(on)
        self.how = how

    def identities(self):
        return [self] + self.a.identities() + self.b.identities()

    def __hash__(self):
        return hash(tuple(self.__dict__.items()))

    def __eq__(self, other):
        if other is None:
            return True
        lefts = set(hash(x) for x in self.identities())
        rights = set(hash(x) for x in other.identities())
        return bool(lefts.intersection(rights))

    def __repr__(self):
        return f'join({str(self.a)}, {str(self.b)}, on={str(list(self.on))}, how=\'{self.how}\')'


def column_name(column: Union['Source', 'Lookup']) -> str:
    return column.__column_name__


def column_path(column: Union['Source', 'Lookup']) -> List[str]:
    path = []
    while column is not None:
        path.insert(0, column.__column_name__)
        column = column.__column_parent__
    return path


def expr_identity(column: 'Expr') -> Optional['Identity']:
    try:
        return column.__expr_identity__
    except:
        raise


# move to functions
class _AbstractLookup(object):
    def __getattr__(self, item) -> 'Lookup':
        return Lookup(item)

    def __getitem__(self, item) -> 'Lookup':
        return Lookup(item)


c = _AbstractLookup()


# deprecated?
def assert_solvable_identity(expr: 'Expr'):
    raise NotImplementedError


# tests
def test_column_creator():
    assert str(c.hey) == 'c.hey'
    assert str(c['hey']) == 'c.hey'
    assert column_name(c.hey) == 'hey'
    assert column_path(c.hey) == ['hey']

    assert str(c.foo.bar.world) == 'c.foo.bar.world'
    assert column_name(c.foo.bar.world) == 'world'
    assert column_path(c.foo.bar.world) == ['foo', 'bar', 'world']


def test_column_edge_cases():
    assert str(c.__class__.__name__) == '_AbstractLookup'
    assert c.foo.__class__ == Lookup

    assert str(c['__class__']) == 'c.__class__'  # @TODO not correct output
    assert column_name(c['__class__']) == '__class__'
    assert column_path(c['__class__']) == ['__class__']

    assert str(c['__class__']['__name__']) == 'c.__class__.__name__'  # @TODO not correct output
    assert column_name(c['__class__']['__name__']) == '__name__'
    assert column_path(c['__class__']['__name__']) == ['__class__', '__name__']


def test_column_operators():
    assert str(c.foo + c.bar) == 'c.foo + c.bar'
    assert str(c.foo - c.bar) == 'c.foo - c.bar'
    assert str(c.foo * c.bar) == 'c.foo * c.bar'
    assert str(c.foo / c.bar) == 'c.foo / c.bar'
    assert str(c.foo // c.bar) == 'c.foo // c.bar'
    assert str(c.foo % c.bar) == 'c.foo % c.bar'
    assert str(c.foo ** c.bar) == 'c.foo ** c.bar'
    # assert str(c.foo << c.bar) == 'c.foo << c.bar'
    # assert str(c.foo >> c.bar) == 'c.foo >> c.bar'
    assert str(c.foo & c.bar) == 'c.foo & c.bar'
    assert str(c.foo | c.bar) == 'c.foo | c.bar'
    assert str(c.foo ^ c.bar) == 'c.foo ^ c.bar'

    assert str(c.foo.bar + c.hello.world) == 'c.foo.bar + c.hello.world'
    assert str(c.foo.bar - c.hello.world) == 'c.foo.bar - c.hello.world'
    assert str(c.foo.bar * c.hello.world) == 'c.foo.bar * c.hello.world'
    assert str(c.foo.bar / c.hello.world) == 'c.foo.bar / c.hello.world'
    assert str(c.foo.bar // c.hello.world) == 'c.foo.bar // c.hello.world'
    assert str(c.foo.bar % c.hello.world) == 'c.foo.bar % c.hello.world'
    assert str(c.foo.bar ** c.hello.world) == 'c.foo.bar ** c.hello.world'
    # assert str(c.foo.bar << c.hello.world) == 'c.foo.bar << c.hello.world'
    # assert str(c.foo.bar >> c.hello.world) == 'c.foo.bar >> c.hello.world'
    assert str(c.foo.bar & c.hello.world) == 'c.foo.bar & c.hello.world'
    assert str(c.foo.bar | c.hello.world) == 'c.foo.bar | c.hello.world'
    assert str(c.foo.bar ^ c.hello.world) == 'c.foo.bar ^ c.hello.world'


@pytest.mark.skip('@TODO order of operations')
def test_order_operations():
    assert str(c.a + c.b * c.c) == 'c.a + c.b * c.c'
    assert str((c.a + c.b) * c.c) == '(c.a + c.b) * c.c'


def test_udf():
    def double(x: str):
        return x * 2

    def double_udf(*args, **kwargs):
        return Call(identity=None, f=double, args=list(args), kwargs=kwargs)

    assert str(double_udf(c.x)) == 'double(c.x)'
    assert str(double_udf(c.x, c.y)) == 'double(c.x, c.y)'  # @TODO should error

    assert str(double_udf(c.x) * 2) == 'double(c.x) * 2'
    assert str(double_udf(c.x.y % c.z) * c.a) == 'double(c.x.y % c.z) * c.a'


def test_scalar():
    assert str(c.foo + 1337) == 'c.foo + 1337'
    assert str(c.foo - 1337) == 'c.foo - 1337'
    assert str(c.foo * 1337) == 'c.foo * 1337'
    assert str(c.foo / 1337) == 'c.foo / 1337'
    assert str(c.foo // 1337) == 'c.foo // 1337'
    assert str(c.foo % 1337) == 'c.foo % 1337'
    assert str(c.foo ** 1337) == 'c.foo ** 1337'
    # assert str(c.foo << 1337) == 'c.foo << 1337'
    # assert str(c.foo >> 1337) == 'c.foo >> 1337'
    assert str(c.foo & 1337) == 'c.foo & 1337'
    assert str(c.foo | 1337) == 'c.foo | 1337'
    assert str(c.foo ^ 1337) == 'c.foo ^ 1337'

    assert str(1337 + c.bar) == '1337 + c.bar'
    assert str(1337 - c.bar) == '1337 - c.bar'
    assert str(1337 * c.bar) == '1337 * c.bar'
    assert str(1337 / c.bar) == '1337 / c.bar'
    assert str(1337 // c.bar) == '1337 // c.bar'
    assert str(1337 % c.bar) == '1337 % c.bar'
    assert str(1337 ** c.bar) == '1337 ** c.bar'
    # assert str(1337 << c.bar) == '1337 << c.bar'
    # assert str(1337 >> c.bar) == '1337 >> c.bar'
    assert str(1337 & c.bar) == '1337 & c.bar'
    assert str(1337 | c.bar) == '1337 | c.bar'
    assert str(1337 ^ c.bar) == '1337 ^ c.bar'


@pytest.mark.skip('@TODO validate solvable expressions')
def test_identity():
    users = Identity(name='users')
    companies = Identity(name='companies')

    assert str(assert_solvable_identity(users.foo + users.bar)) == 'c.foo + c.bar'
    assert str(assert_solvable_identity(users.foo + c.hello)) == 'c.foo + c.hello'
    assert str(assert_solvable_identity((users.foo + c.hello) + c.bar)) == 'c.foo + c.hello + c.bar'
    assert str(assert_solvable_identity((users.foo + c.hello) + users.baz)) == 'c.foo + c.hello + c.baz'
    assert str(assert_solvable_identity(c.hello + users.foo)) == 'c.hello + c.foo'
    assert str(assert_solvable_identity((c.bar + c.hello) + users.foo)) == 'c.bar + c.hello + c.foo'
    assert str(assert_solvable_identity((users.baz + c.hello) + users.foo)) == 'c.baz + c.hello + c.foo'

    with pytest.raises(IdentityTypeError):
        _ = assert_solvable_identity(users.foo + companies.hello)
    with pytest.raises(IdentityTypeError):
        _ = assert_solvable_identity((users.foo + c.hello) + companies.bar)
    with pytest.raises(IdentityTypeError):
        _ = assert_solvable_identity(companies.bar + users.foo)
    with pytest.raises(IdentityTypeError):
        _ = assert_solvable_identity((c.bar + users.baz) + companies.foo)
