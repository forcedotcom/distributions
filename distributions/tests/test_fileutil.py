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

from nose.tools import assert_equal
from distributions import fileutil


EXAMPLES = []
EXAMPLES.append([])
EXAMPLES.append(['asdf'])
EXAMPLES.append([
    {'a': 0, 'b': 'asdf', 'c': [0, 1, 2]},
    {'a': 1, 'b': 'asdf'},
    {'a': 2, 'b': 'asdf'},
    {'a': 3, 'b': 'asdf', 'd': {'inner': 'value'}},
    {'a': 4, 'b': 'asdf'},
    {'a': 5, 'b': 'asdf'},
    {'a': 6, 'b': 'asdf'},
    {'a': 7, 'b': 'asdf'},
    {'a': 8, 'b': 'asdf'},
    {'a': 9, 'b': 'asdf', 'c': [0]},
])


def costream_dump(stream, filename):
    costream = fileutil.json_costream_dump(filename)
    costream.next()
    for item in stream:
        costream.send(item)
    costream.close()


def test_dump_load():
    valid_pairs = [
        (fileutil.json_dump, fileutil.json_load),
        (fileutil.json_stream_dump, fileutil.json_load),
        (fileutil.json_stream_dump, fileutil.json_stream_load),
        (costream_dump, fileutil.json_load),
        (costream_dump, fileutil.json_stream_load),
    ]
    for dump, load in valid_pairs:
        for filetype in ['', '.gz', '.bz2']:
            yield _test_dump_load, dump, load, filetype


def _test_dump_load(dump, load, filetype):
    for example in EXAMPLES:
        print example
        with fileutil.tempdir() as d, fileutil.chdir(d):
            expected = example
            filename = 'test.json' + filetype
            dump(expected, filename)
            actual = list(load(filename))
            assert_equal(actual, expected)
