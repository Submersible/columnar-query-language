def vectorizer(text):
    raise NotImplementedError


def term_frequency(words):
    raise NotImplementedError


def idf(values):
    raise NotImplementedError


def gradient_boosting(x, *, y, iterations, gamma, **kwargs):
    raise NotImplementedError


def random_forest(x, *, y, **kwargs):
    raise NotImplementedError


def cross_validation(model, *, strategy):
    raise NotImplementedError


def parameter_search(model, *, cv_strategy='cv_3_fold', search_strategy='bayesian'):
    raise NotImplementedError
