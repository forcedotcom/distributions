from distributions.tests.util import assert_all_close, import_model


MODULES = {
    'dd': ['dd_py', 'dd_cy', 'dd_cc'],
    'dpm': ['dpm_cc'],
}
for key, val in MODULES.iteritems():
    MODULES[key] = [import_model(m) for m in val]


def test_dump_group():
    for name in MODULES:
        yield _test_dump_group, name


def _test_dump_group(name):
    MODELS = [m.Model for m in MODULES[name]]
    EXAMPLES = [e for m in MODELS for e in m.EXAMPLES]
    for EXAMPLE in EXAMPLES:
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

        dumped = [g.dump() for g in groups]
        assert_all_close(dumped, err_msg='group')
