from distributions.tests.util import (
    assert_all_close,
    list_models,
    import_model,
)


MODULES = {}
for spec in list_models():
    MODULES.setdefault(spec['name'], []).append(import_model(spec))


def test_model():
    for name in MODULES:
        yield _test_model, name


def _test_model(name):
    MODELS = [m.Model for m in MODULES[name]]
    assert_all_close([m.__name__ for m in MODELS], err_msg='Model.__name__')
    EXAMPLES = [e for m in MODELS for e in m.EXAMPLES]
    for EXAMPLE in EXAMPLES:
        raw_model = EXAMPLE['model']
        models = [Model.model_load(raw_model) for Model in MODELS]
        dumped = [m.dump() for m in models]
        assert_all_close(dumped, err_msg='model_dump')


def test_group():
    for name in MODULES:
        yield _test_group, name


def _test_group(name):
    MODELS = [m.Model for m in MODULES[name]]
    EXAMPLES = [e for m in MODELS for e in m.EXAMPLES]
    for EXAMPLE in EXAMPLES:
        raw_model = EXAMPLE['model']
        values = EXAMPLE['values'][:]
        models = [Model.model_load(raw_model) for Model in MODELS]

        groups = [model.group_create() for model in models]
        models_groups = zip(models, groups)

        for value in values:
            for model, group in models_groups:
                model.group_add_value(group, value)
            dumped = [g.dump() for g in groups]
            assert_all_close(dumped, err_msg='group_dump')

        for model, group in models_groups:
            values.append(model.sample_value(group))

        for value in values:
            scores = [
                model.score_value(group, value)
                for model, group in models_groups
            ]
            assert_all_close(scores, err_msg='score_value')

        scores = [model.score_group(group) for model, group in models_groups]
        assert_all_close(scores, err_msg='score_group')

        for model, group in models_groups:
            dumped = group.dump()
            model.group_init(group)
            group.load(dumped)

        scores = [model.score_group(group) for model, group in models_groups]
        assert_all_close(scores, err_msg='score_group')


def test_plus_group():
    for name in MODULES:
        yield _test_plus_group, name


def _test_plus_group(name):
    MODELS = [m.Model for m in MODULES[name] if hasattr(m.Model, 'plus_group')]
    EXAMPLES = [e for m in MODELS for e in m.EXAMPLES]
    for EXAMPLE in EXAMPLES:
        raw_model = EXAMPLE['model']
        values = EXAMPLE['values']
        models = [Model.model_load(raw_model) for Model in MODELS]
        groups = [model.group_create(values) for model in models]
        dumped = [m.plus_group(g).dump() for m, g in zip(models, groups)]
        assert_all_close(dumped, err_msg='model._plus_group(group)')
