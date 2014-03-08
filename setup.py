import os
import re
from setuptools import setup, Extension, Feature
try:
    from Cython.Distutils import build_ext
    cython = True
    print 'building cython extensions'
except ImportError:
    cython = False
    print 'not building cython extensions'


VERSION = None
with open(os.path.join('distributions', '__init__.py')) as f:
    for line in f:
        if re.match("__version__ = '\S+'$", line):
            VERSION = line.split()[-1].strip("'")
assert VERSION, 'could not determine version'


def extension(name):
    return Extension(
        'distributions.{0}'.format(name.replace('/', '.')),
        sources=['distributions/{0}.pyx'.format(name)],
        language='c++',
        include_dirs=['include'],
        libraries=['m', 'distributions_shared'],
        library_dirs=['build/src'],
        extra_compile_args=[
            '-std=c++0x',
            '-Wall',
            '-Werror',
            '-Wno-unused-function',
            '-Wno-sign-compare',
            '-Wno-strict-aliasing',
            '-O3',
            '-ffast-math',
            '-funsafe-math-optimizations',
            #'-fno-trapping-math',
            #'-ffinite-math-only',
            #'-fvect-cost-model',
            '-mfpmath=sse',
            '-msse4.1',
            #'-mavx',
            #'-mrecip',
            #'-march=native',
        ],
    )


model_feature = Feature(
    'cython models',
    standard=True,
    optional=True,
    ext_modules=[
        extension('hp/special'),
        extension('lp/special'),
        extension('hp/random'),
        extension('lp/random'),
        extension('hp/models/dd'),
        extension('lp/models/dd'),
        extension('lp/models/dpd'),
        extension('lp/clustering'),
    ]
)


config = {
    'version': VERSION,
    'features': {'cython': model_feature},
    'name': 'distributions',
    'packages': [
        'distributions',
        'distributions.dbg',
        'distributions.dbg.models',
        'distributions.hp',
        'distributions.hp.models',
        'distributions.lp',
        'distributions.lp.models',
        'distributions.tests',
    ],
}


if cython:
    config['cmdclass'] = {'build_ext': build_ext}


setup(**config)
