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
