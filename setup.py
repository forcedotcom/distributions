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
    'name': 'distributions',
    'description': 'Primitives for Bayesian MCMC inference',
    'long_description': long_description,
    'url': 'https://github.com/forcedotcom/distributions',
    'author': 'Jonathan Glidden, Eric Jonas, Fritz Obermeyer, Cap Petschulat',
    'maintainer': 'Fritz Obermeyer',
    'maintainer_email': 'fobermeyer@salesforce.com',
    'license': 'Revised BSD',
    'features': {'cython': model_feature},
    'packages': [
        'distributions',
        'distributions.dbg',
        'distributions.dbg.models',
        'distributions.tests',
    ],
}
if cython:
    config['cmdclass'] = {'build_ext': build_ext}
    config['packages'] += [
        'distributions.hp',
        'distributions.hp.models',
        'distributions.lp',
        'distributions.lp.models',
    ]


setup(**config)
