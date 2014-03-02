from nose.tools import assert_equal
from distributions.models import dd_py, dd_cy, dd_cc


VERSIONS = [dd_py, dd_cy, dd_cc]
MODELS = [v.Model for v in VERSIONS]
EXAMPLE = MODELS[0].EXAMPLE


def test_add_remove_merge():
    raw_model = EXAMPLE['model']
    for Model in MODELS:
        model = Model.load_model(raw_model)
        yield _test_add_remove_merge, model


def _test_add_remove_merge(model):
    values = EXAMPLE['values']
    group1 = model.Group()
    group2 = model.Group()
    model.group_init(group1)
    model.group_init(group2)
    for value in values:
        model.group_add_value(group1, value)
    model.group_merge(group2, group1)
    assert_equal(group1.dump(), group2.dump())


def test_dump_group():
    raw_model = EXAMPLE['model']
    values = EXAMPLE['values']
    models = [Model.load_model(raw_model) for Model in MODELS]

    groups = []
    for model in models:
        group = model.Group()
        model.group_init(group)
        for value in values:
            model.group_add_value(group, value)
        groups.append(group)

    for i, group1 in enumerate(groups):
        for group2 in groups[i + 1:]:
            assert_equal(group1.dump(), group2.dump())
