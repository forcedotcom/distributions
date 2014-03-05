from distributions.tests.util import assert_all_close, import_model


MODULES = {
    'dd': ['dd_py', 'dd_cy', 'dd_cc'],
    'dpm': ['dpm_cc'],
}
for key, val in MODULES.iteritems():
    MODULES[key] = [import_model(m) for m in val]


def test_model():
    for name in MODULES:
        yield _test_model, name


def _test_model(name):
    MODELS = [m.Model for m in MODULES[name]]
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

        scores = [
            model.score_group(group)
            for model, group in models_groups
        ]
        assert_all_close(scores, err_msg='score_group')
