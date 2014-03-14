# Copyright (c) 2014, Salesforce.com, Inc.  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# - Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# - Neither the name of Salesforce.com nor the names of its contributors
#   may be used to endorse or promote products derived from this
#   software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
# TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
# USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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
