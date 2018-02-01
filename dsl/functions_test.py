from textwrap import dedent

import pytest
from pyspark.sql import Row

from dsl import functions as F
from dsl.expression import c, MissingSourceError
from dsl.io import reader, writer


def test_pipe(backend):
    users = reader.python([{'x': x} for x in range(3)])
    a = F.select(foo=users.x * 2)
    assert str(users >> a) == "Source<CombineIdentities(<schema=None, __select_fields__=OrderedDict([('foo', c.x * 2)]), __select_keys__=dict_keys(['foo'])>)>"
    assert F.collect(users >> a) == [{'foo': 0}, {'foo': 2}, {'foo': 4}]


def test_joins():
    users = reader.python([])
    companies = reader.python([])
    user_companies = F.join(users, companies, on=['hey'], how='left')

    assert str(c.hey + user_companies.foo + users.bar + companies.baz) == 'c.hey + c.foo + c.bar + c.baz'
    assert str(user_companies(users.bar + companies.bar)) == "join(PythonDataFrameReader(<schema=None, data=[], partitions=None>), PythonDataFrameReader(<schema=None, data=[], partitions=None>), on=['hey'], how='left')(c.bar + c.bar)"


def test_backend_random_things(backend, temp_root):
    df = reader.python([{'hello': 'World', 'x': x} for x in range(5)])

    @F.udf
    def triple(x):
        return x * 3

    @F.udf
    def my_add(a, b):
        return a + b

    assert F.collect(df) == [
        {'hello': 'World', 'x': 0},
        {'hello': 'World', 'x': 1},
        {'hello': 'World', 'x': 2},
        {'hello': 'World', 'x': 3},
        {'hello': 'World', 'x': 4},
    ]

    new_select = F.select(
        rename_hello=df.hello,
        rename_x=df.x,
        x_double=df.x * 2,
        x_my_double=my_add(df.x, df.x),
        x_triple=triple(df.x)
    )
    with pytest.raises(MissingSourceError):
        F.collect(new_select)

    new_df = df >> new_select
    assert F.collect(new_df) == [
        {'rename_hello': 'World', 'rename_x': 0, 'x_double': 0, 'x_my_double': 0, 'x_triple': 0},
        {'rename_hello': 'World', 'rename_x': 1, 'x_double': 2, 'x_my_double': 2, 'x_triple': 3},
        {'rename_hello': 'World', 'rename_x': 2, 'x_double': 4, 'x_my_double': 4, 'x_triple': 6},
        {'rename_hello': 'World', 'rename_x': 3, 'x_double': 6, 'x_my_double': 6, 'x_triple': 9},
        {'rename_hello': 'World', 'rename_x': 4, 'x_double': 8, 'x_my_double': 8, 'x_triple': 12},
    ]

    writer.csv(new_df, path=f'{temp_root}/hello.csv', mode='overwrite')
    assert open(f'{temp_root}/hello.csv', 'r').read() == dedent('''
        rename_hello,rename_x,x_double,x_my_double,x_triple
        World,0,0,0,0
        World,1,2,2,3
        World,2,4,4,6
        World,3,6,6,9
        World,4,8,8,12
    ''').lstrip()


@pytest.mark.skip('@TODO')
def test_group_by():
    a = reader.python([{'foo': 'Hello', 'x': x, 'a': x * 2} for x in range(10)])
    b = reader.python([{'bar': 'World', 'x': x, 'b': x + 3} for x in range(10)])

    a_b = F.join(a, b, on=['x'], how='inner')

    combined = a_b(c.a + c.b)

    print(F.collect(combined))


@pytest.mark.skip('@TODO')
def test_group_by_piping():
    a = reader.python([{'foo': 'Hello', 'x': x, 'a': x * 2} for x in range(10)])
    b = reader.python([{'bar': 'World', 'x': x, 'b': x + 3} for x in range(10)])

    (
        a
        >> F.join(b, on=['x'], how='inner')
        >> F.append(combined=c.a + c.b)
        >> F.collect()
        >> print
    )


@pytest.mark.skip('@TODO')
def test_explode_identity():
    df = reader.python([{'foo': 'Hello', 'x': x, 'items': list(range(x))} for x in range(3)])

    item = F.explode(df, c.items)
    assert str(df.x + item)


@pytest.mark.skip('@TODO')
def test_explode():
    df = reader.python([{'foo': 'Hello', 'x': x, 'items': list(range(x))} for x in range(3)])

    (
        df
        >> F.append(item=F.explode(c.items))
        >> F.collect()
        >> print
    )


@pytest.mark.skip('@TODO')
def test_ml_pipeline():
    XGBoostClassifier: any
    word_vectorizer: any
    term_frequency: any
    idf: any

    def model_architecture(n_iterations):
        words = word_vectorizer(c.description)
        words_tf = term_frequency(words)
        words_tf_idf = idf(words_tf)
        my_new_classifier = XGBoostClassifier(x=F.select(c.position, words_tf_idf), y=c.y, n_iterations=n_iterations)
        return my_new_classifier

    pipeline = model_architecture(n_iterations=100)
    print(pipeline.input_schema())
    print(pipeline.model())
    print(pipeline.output_schema())

    df = reader.python([{'foo': 'Hello', 'x': x, 'items': list(range(x))} for x in range(3)])
    model = pipeline.fit(df)
    print(model.predict_proba(df) >> F.collect())


@pytest.mark.skip('@TODO')
def test_windowing():
    def mtcars(ctx):
        return ctx.createDataFrame([
            Row(var1=int(x / 3), var2=x) for x in range(10)
        ])

    (
        mtcars()
        >> F.group_by(c.var1)
        >> F.append(
            c(var2_new=c.var2 / F.mean(c.var2)),
            F.everything()
        )
    )
