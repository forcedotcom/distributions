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
    modules = MODULES[name]
    assert_all_close([m.NAME for m in modules], err_msg='Model.__name__')
    EXAMPLES = [e for m in modules for e in m.EXAMPLES]
    for EXAMPLE in EXAMPLES:
        raw_shared = EXAMPLE['shared']
        shareds = [module.Shared.from_dict(raw_shared) for module in modules]
        dumped = [m.dump() for m in shareds]
        assert_all_close(dumped, err_msg='shared_dump')


def test_group():
    for name in MODULES:
        yield _test_group, name


def _test_group(name):
    modules = MODULES[name]
    EXAMPLES = [e for m in modules for e in m.EXAMPLES]
    for EXAMPLE in EXAMPLES:
        values = EXAMPLE['values'][:]
        raw_shared = EXAMPLE['shared']
        temp_shared = modules[0].Shared.from_dict(raw_shared)
        for value in values:
            temp_shared.add_value(value)
        temp_shared.realize()
        raw_shared = temp_shared.dump()

        shareds = [module.Shared.from_dict(raw_shared) for module in modules]
        groups = [
            module.Group.from_values(shared)
            for module, shared in zip(modules, shareds)]
        modules_shareds_groups = zip(modules, shareds, groups)

        for value in values:
            for module, shared, group in modules_shareds_groups:
                group.add_value(shared, value)
            dumped = [g.dump() for g in groups]
            assert_all_close(dumped, err_msg='group_dump')

        for module, shared, group in modules_shareds_groups:
            value = group.sample_value(shared)
            shared.add_value(value)
            values.append(value)

        for value in values:
            scores = [
                group.score_value(shared, value)
                for module, shared, group in modules_shareds_groups
            ]
            for module, shared, group in modules_shareds_groups:
                print "------------------"
                print module
                print shared.dump()
                print group.dump()
            print value
            assert_all_close(scores, err_msg='score_value')

        scores = [
            group.score_data(shared)
            for module, shared, group in modules_shareds_groups]
        assert_all_close(scores, err_msg='score_data')

        for module, shared, group in modules_shareds_groups:
            dumped = group.dump()
            group.init(shared)
            group.load(dumped)

        scores = [
            group.score_data(shared)
            for module, shared, group in modules_shareds_groups]
        assert_all_close(scores, err_msg='score_data')


def test_plus_group():
    for name in MODULES:
        yield _test_plus_group, name


def _test_plus_group(name):
    modules = MODULES[name]
    modules = [m for m in modules if hasattr(m.Shared, 'plus_group')]
    EXAMPLES = [e for m in modules for e in m.EXAMPLES]
    for EXAMPLE in EXAMPLES:
        raw_shared = EXAMPLE['shared']
        values = EXAMPLE['values']
        shareds = [module.Shared.from_dict(raw_shared) for module in modules]
        groups = [
            module.Group.from_values(shared, values)
            for module, shared in zip(modules, shareds)]
        dumped = [m.plus_group(g).dump() for m, g in zip(shareds, groups)]
        assert_all_close(dumped, err_msg='shared._plus_group(group)')
