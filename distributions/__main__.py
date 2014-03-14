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

from collections import defaultdict
import parsable
from distributions.tests.util import list_models, import_model


@parsable.command
def flavors_by_model():
    '''
    List flavors implemented of each model.
    '''
    models = defaultdict(lambda: [])
    for spec in list_models():
        models[spec['name']].append(spec['flavor'])
    for model in sorted(models):
        print 'model {}: {}'.format(model, ' '.join(sorted(models[model])))


@parsable.command
def models_by_flavor():
    '''
    List models implemented of each flavor.
    '''
    flavors = defaultdict(lambda: [])
    for spec in list_models():
        flavors[spec['flavor']].append(spec['name'])
    for flavor in sorted(flavors):
        print 'flavor {}: {}'.format(flavor, ' '.join(sorted(flavors[flavor])))


@parsable.command
def model_apis():
    '''
    List api of each model.
    '''
    for spec in list_models():
        Model = import_model(spec).Model
        attrs = sorted(attr for attr in dir(Model) if not attr.startswith('_'))
        types = []
        methods = []
        constants = []
        for attr in attrs:
            var = getattr(Model, attr)
            if isinstance(var, type):
                types.append(attr)
            elif hasattr(var, '__call__'):
                methods.append(attr)
            else:
                constants.append(attr)
        print 'distributions.{}.models.{}.{}:'.format(
            spec['flavor'],
            spec['name'],
            Model.__name__)
        print '  types:\n    {}'.format('\n    '.join(types))
        print '  methods:\n    {}'.format('\n    '.join(methods))
        print '  constants:\n    {}'.format('\n    '.join(constants))


if __name__ == '__main__':
    parsable.dispatch()
