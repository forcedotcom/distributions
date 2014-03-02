from nose.tools import assert_equal
from distributions.models import dd_py, dd_cy, dd_cc


VERSIONS = [dd_py, dd_cy, dd_cc]
MODELS = [v.Model for v in VERSIONS]


def test_methods_run():
    EXAMPLE = MODELS[0].EXAMPLE
    raw_model = EXAMPLE['model']
    values = EXAMPLE['values']

    models = [Model.load_model(raw_model) for Model in MODELS]

    for model in models:
        group1 = model.Group()
        group2 = model.Group()
        model.group_init(group1)
        model.group_init(group2)

        for value in values:
            model.group_add_value(group1, value)
        model.group_merge(group2, group1)
        assert_equal(group1.dump(), group2.dump())
