from setuptools import setup, Extension, Feature
try:
    from Cython.Distutils import build_ext
    cython = True
    print 'building cython extensions'
except ImportError:
    cython = False
    print 'not building cython extensions'


def extension(name):
    return Extension(
        'distributions.{0}'.format(name.replace('/', '.')),
        [
            'distributions/{0}.pyx'.format(name),
            'src/common.cc',
            'src/special.cc',
            'src/random.cc',
            'src/std_wrapper.cc',
        ],
        language='c++',
        libraries=['m'],
        include_dirs=['include'],
        extra_compile_args=[
            '-std=c++0x',
            '-Wall',
            '-Werror',
            '-Wno-unused-function',
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
        extension('cSpecial'),
        extension('cRandom'),
        extension('models/dd_cy'),
        extension('models/dd_cc'),
        extension('models/dpm_cc'),
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
