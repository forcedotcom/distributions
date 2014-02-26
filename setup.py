from setuptools import setup, Extension, Feature
try:
    from Cython.Distutils import build_ext
    cython = True
    print 'building cython extensions'
except ImportError:
    cython = False
    print 'not building cython extensions'


def model_extension(name):
    return Extension(
        'distributions.models.{0}'.format(name),
        [
            'distributions/models/{0}.pyx'.format(name),
            'src/common.cc',
        ],
        language='c++',
        libraries=['m'],
        include_dirs=['include/distributions'],
        extra_compile_args=[
            '-std=c++0x',
            '-Wall',
            #'-Wno-sign-compare',
            #'-Wno-strict-aliasing',
            '-O3',
            #'-march=native',
            '-ffast-math',
            '-funsafe-math-optimizations',
        ])


model_feature = Feature(
    'cython models',
    standard=True,
    optional=True,
    ext_modules=[
        model_extension('dd'),
        model_extension('dd_lp'),
    ]
)


config = {
    'features': {'cython': model_feature},
    'name': 'distributions',
    'packages': [
        'distributions',
        'distributions.models',
        ],
    }


if cython:
    config['cmdclass'] = {'build_ext': build_ext}


setup(**config)
