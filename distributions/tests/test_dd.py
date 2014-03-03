from distributions.tests.util import assert_all_close
from distributions.models import dd_py, dd_cy, dd_cc


MODULES = [dd_py, dd_cy, dd_cc]
MODELS = [m.Model for m in MODULES]
EXAMPLES = [e for m in MODELS for e in m.EXAMPLES]


def test_dump_group():
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
