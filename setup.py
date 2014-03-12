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
    name_pyx = '{}.pyx'.format(name.replace('.', '/'))
    return Extension(
        name,
        sources=[name_pyx],
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
        extension('distributions.hp.special'),
        extension('distributions.hp.random'),
        extension('distributions.hp.models.dd'),
        extension('distributions.hp.models.gp'),
        extension('distributions.hp.models.nich'),
        extension('distributions.hp.models.dpd'),
        extension('distributions.lp.special'),
        extension('distributions.lp.random'),
        extension('distributions.lp.vector'),
        extension('distributions.lp.models.dd'),
        extension('distributions.lp.models.gp'),
        extension('distributions.lp.models.nich'),
        extension('distributions.lp.models.dpd'),
        extension('distributions.lp.clustering'),
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
