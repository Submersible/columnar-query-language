from abc import ABC, abstractmethod
from functools import wraps
from typing import Union, Callable, TypeVar, Generic, Sequence, Dict
import inspect

T, A, B, C = TypeVar('T'), TypeVar('A'), TypeVar('B'), TypeVar('C')


def pipe_context(f):
    """

    Must check type (must be typed with Union):
        def foo(a, *rest, *, **kwargs)
        def foo(a, b, *rest, *, **kwargs)

    Check *args count:
        def foo(a, *, **kwargs)
        def foo(a, b, *, **kwargs)
        def foo(a, b, c, *, **kwargs)

    Check first kwarg:
        def foo(*, a, **kwargs)
        def foo(*, a, b, **kwargs)
        def foo(*, a, b, c, **kwargs)

    Error:
        def foo(*rest, **kwargs)
        def foo(**kwargs)

    :param f:
    :return:
    """
    spec = inspect.getfullargspec(f)

    ctx_key = spec.args[0]
    ctx_type = spec.annotations[ctx_key]
    ctx_type = ctx_type._subs_tree()[1]

    @wraps(f)
    def wrapper(*args, **kwargs):
        if len(args) > len(spec.args):
            raise TypeError(f'Too many args for {f}')

        if len(args) > 0 and isinstance(args[0], ctx_type):
            return f(*args, **kwargs)

        all_kwargs = dict(zip(spec.args[1:], args), **kwargs)
        if ctx_key in all_kwargs:
            return f(**all_kwargs)

        @wraps(f)
        def partial(value):
            return f(**{ctx_key: value}, **all_kwargs)
        return PipeFunction(partial, args=args, kwargs=kwargs)
    return wrapper


class Pipe(Generic[A, B]):
    @abstractmethod
    def __call__(self, value: A) -> B:
        raise NotImplementedError

    def __rshift__(self, other: Union['Pipe[B, C]', Callable[[B], C]]) -> 'Pipe[A, C]':
        return PipeCompose(self, other)

    def __rrshift__(self, other: A) -> B:
        return self(other)


class PipeFunction(Pipe[A, B]):
    def __init__(self, f: Callable[[A], B], *, args: Sequence[any], kwargs: Dict[str, any]):
        self.f = f
        self.args = args
        self.kwargs = kwargs

    def __call__(self, value: A) -> B:
        return self.f(value)

    def __repr__(self):
        display_args = ', '.join([str(v) for v in list(self.args)] + [f'{k}={v}' for k, v in self.kwargs.items()])
        return f'{self.f.__name__}({display_args})'


class PipeCompose(Pipe):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __call__(self, value: A) -> B:
        return self.right(self.left(value))

    def __repr__(self):
        return f'{self.left} >> {self.right}'


def test_pipe__2_arg():
    @pipe_context
    def foo(name: Union[str, int]=None, value: int=None) -> Union[str, Callable[[str], Pipe]]:
        return f'foo({name}, {value})'

    assert str(foo('bar', 1)) == 'foo(bar, 1)'
    assert str(foo('bar', 1) >> foo(2)) == 'foo(foo(bar, 1), 2)'
    assert str(foo('wee', 1) >> foo(2) >> foo(3)) == 'foo(foo(foo(wee, 1), 2), 3)'
    foo_partial = foo(2) >> foo(3)
    assert str(foo_partial) == 'foo(2) >> foo(3)'
    assert str(foo_partial('bar')) == 'foo(foo(bar, 2), 3)'
    assert str(foo(value=2)('bar')) == 'foo(bar, 2)'


def test_pipe__1_arg__0_kwargs_any():
    @pipe_context
    def foo(name: Union[str, int]=None, **kwargs) -> Union[str, Pipe[str, str]]:
        return f'foo({name}, **{kwargs})'

    assert str(foo('hey', a=1)) == "foo(hey, **{'a': 1})"
    assert str(foo()) == 'foo()'
    assert str(foo(a=1)) == 'foo(a=1)'
    assert str(foo(a=1)('hey')) == "foo(hey, **{'a': 1})"
    assert str('hey' >> foo(a=1)) == "foo(hey, **{'a': 1})"
    assert str(foo(name='hey', a=1)) == "foo(hey, **{'a': 1})"
    assert str(foo(name='hey')) == 'foo(hey, **{})'
    assert str(foo('hey')) == 'foo(hey, **{})'


def test_pipe__0_arg__2_kwargs():
    pass


def test_pipe__0_arg__kwargs_any():
    pass
