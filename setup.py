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
        libraries=['m', 'distributions'],
        library_dirs=['build/src'],
        extra_compile_args=[
            '-std=c++0x',
            '-Wall',
            '-Werror',
            '-Wno-unused-function',
            #'-Wno-sign-compare',
            #'-Wno-strict-aliasing',
            '-O3',
            '-mfpmath=sse',
            '-msse4.1',
            #'-mavx',
            '-ffast-math',
            '-funsafe-math-optimizations',
            #'-fno-trapping-math',
            #'-ffinite-math-only',
            #'-fvect-cost-model',
            #'-mrecip',
            #'-march=native',
        ],
    )


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
    'version': VERSION,
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
