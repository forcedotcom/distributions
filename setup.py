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

import os
import re
from setuptools import setup, Extension, Feature
from distutils.command.build_clib import build_clib
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


with open('README.md') as f:
    long_description = f.read()


EXTRA_COMPILE_ARGS = [
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
]


def hp_extension(name):
    name = 'distributions.' + name
    name_pyx = '{}.pyx'.format(name.replace('.', '/'))
    sources = [name_pyx]
    return Extension(
        name,
        sources=sources,
        language='c++',
        include_dirs=['include', 'distributions'],
        libraries=['m'],
        extra_compile_args=EXTRA_COMPILE_ARGS,
    )


def lp_extension(name):
    name = 'distributions.' + name
    name_pyx = '{}.pyx'.format(name.replace('.', '/'))
    sources = [name_pyx]
    sources += [
        'src/common.cc',
        'src/special.cc',
        'src/random.cc',
        'src/vector_math.cc',
    ]
    return Extension(
        name,
        sources=sources,
        language='c++',
        include_dirs=['include', 'distributions'],
        libraries=['m'],
        extra_compile_args=EXTRA_COMPILE_ARGS,
    )


high_precision = Feature(
    'high precision functions and models (cython)',
    standard=True,
    optional=True,
    ext_modules=[
        hp_extension('rng_cc'),
        hp_extension('global_rng'),
        hp_extension('hp.special'),
        hp_extension('hp.random'),
        hp_extension('hp.models.dd'),
        hp_extension('hp.models.gp'),
        hp_extension('hp.models.nich'),
        hp_extension('hp.models.dpd'),
    ],
)


low_precision = Feature(
    'low precision functions and models (cython + libdistributions)',
    standard=True,
    optional=True,
    ext_modules=[
        lp_extension('lp.special'),
        lp_extension('lp.random'),
        lp_extension('lp.vector'),
        lp_extension('lp.models.dd'),
        lp_extension('lp.models.gp'),
        lp_extension('lp.models.nich'),
        lp_extension('lp.models.dpd'),
        lp_extension('lp.clustering'),
    ],
)


config = {
    'version': VERSION,
    'name': 'distributions',
    'description': 'Primitives for Bayesian MCMC inference',
    'long_description': long_description,
    'url': 'https://github.com/forcedotcom/distributions',
    'author': 'Jonathan Glidden, Eric Jonas, Fritz Obermeyer, Cap Petschulat',
    'maintainer': 'Fritz Obermeyer',
    'maintainer_email': 'fobermeyer@salesforce.com',
    'license': 'Revised BSD',
    'features': {
        'high-precision': high_precision,
        'low-precision': low_precision,
    },
    'packages': [
        'distributions',
        'distributions.dbg',
        'distributions.dbg.models',
        'distributions.tests',
    ],
}
if cython:
    config['packages'] += [
        'distributions.hp',
        'distributions.hp.models',
        'distributions.lp',
        'distributions.lp.models',
    ]
    config['cmdclass'] = {'build_ext': build_ext}
setup(**config)
