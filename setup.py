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
import sys
import numpy

if len(sys.argv) >= 2 and sys.argv[1] == 'bdist_wheel':
    # bdist_wheel needs setuptools
    import setuptools
    assert setuptools  # satisfy pyflakes

from distutils.core import setup, Extension
from distutils.version import LooseVersion

try:
    from Cython.Build import cythonize
    from cython import __version__ as cython_version
    min_cython_version = '0.20.1'
    if LooseVersion(cython_version) < LooseVersion(min_cython_version):
        raise ValueError(
            'cython support requires cython>={}'.format(min_cython_version))
    cython = True
except ImportError:
    cython = False


clang = False
if sys.platform.lower().startswith('darwin'):
    clang = True


include_dirs = ['include', 'distributions']
include_dirs.append(numpy.get_include())

if 'EXTRA_INCLUDE_PATH' in os.environ:
    include_dirs.append(os.environ['EXTRA_INCLUDE_PATH'])

extra_compile_args = [
    '-DDIST_DEBUG_LEVEL=3',
    '-DDIST_THROW_ON_ERROR=1',
    '-Wno-unused-function',
]
if clang:
    extra_compile_args.extend([
        '-mmacosx-version-min=10.7',  # for anaconda
        '-std=c++0x',
        '-stdlib=libc++',
        '-Wno-deprecated-register',
        '-Wno-#warnings',  # avoid #warning "Using deprecated NumPy API,..."
    ])
else:
    extra_compile_args.extend([
        '-std=c++0x',
        '-Wall',
        '-Werror',
        '-Wno-error=cpp',  # avoid #warning "Using deprecated NumPy API,..."
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
        #'-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION',
    ])


use_protobuf = 'DISTRIBUTIONS_USE_PROTOBUF' in os.environ


def make_extension(name):
    module = 'distributions.' + name
    sources = [
        '{}.{}'.format(module.replace('.', '/'), 'pyx' if cython else 'cpp')
    ]
    libraries = ['m']
    if use_protobuf:
        libraries.append('protobuf')
    if name.startswith('lp'):
        libraries = ['distributions_shared'] + libraries
    return Extension(
        module,
        sources=sources,
        language='c++',
        include_dirs=include_dirs,
        libraries=libraries,
        extra_compile_args=extra_compile_args,
    )


def make_extensions(names):
    return [make_extension(name) for name in names]


hp_extensions = make_extensions([
    'has_cython',
    'rng_cc',
    'global_rng',
    '_eigen',
    'hp.special',
    'hp.random',
    'hp.models.dd',
    'hp.models.gp',
    'hp.models.nich',
])


lp_extensions = make_extensions([
    'lp.special',
    'lp.random',
    'lp.vector',
    'lp.models.bb',
    'lp.models._bb',
    'lp.models.dd',
    'lp.models._dd',
    'lp.models.dpd',
    'lp.models._dpd',
    'lp.models.gp',
    'lp.models._gp',
    'lp.models.bnb',
    'lp.models._bnb',
    'lp.models.nich',
    'lp.models._nich',
    'lp.models.niw',
    'lp.models._niw',
    'lp.clustering',
    'lp.mixture',
])


if cython:
    ext_modules = cythonize(hp_extensions + lp_extensions)
else:
    ext_modules = hp_extensions + lp_extensions


version = None
with open(os.path.join('distributions', '__init__.py')) as f:
    for line in f:
        if re.match("__version__ = '\S+'$", line):
            version = line.split()[-1].strip("'")
assert version, 'could not determine version'


with open('README.md') as f:
    long_description = f.read()


config = {
    'version': version,
    'name': 'distributions',
    'description': 'Primitives for Bayesian MCMC inference',
    'long_description': long_description,
    'url': 'https://github.com/forcedotcom/distributions',
    'author': 'Jonathan Glidden, Eric Jonas, Fritz Obermeyer, Cap Petschulat',
    'maintainer': 'Fritz Obermeyer',
    'maintainer_email': 'fritz.obermeyer@gmail.com',
    'license': 'Revised BSD',
    'packages': [
        'distributions',
        'distributions.dbg',
        'distributions.dbg.models',
        'distributions.hp',
        'distributions.hp.models',
        'distributions.lp',
        'distributions.lp.models',
        'distributions.io',
        'distributions.tests',
        'distributions.vendor',
    ],
    'ext_modules': ext_modules,
}


setup(**config)
