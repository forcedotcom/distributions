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
import bz2
import gzip
import simplejson
import struct


def mkdir_p(dirname):
    'like mkdir -p'
    if not os.path.exists(dirname):
        try:
            os.makedirs(dirname)
        except OSError as e:
            if not os.path.exists(dirname):
                raise e


def open_compressed(filename, mode='r'):
    if 'w' in mode:
        dirname = os.path.dirname(filename)
        if dirname:
            mkdir_p(dirname)
    if filename.endswith('.bz2'):
        return bz2.BZ2File(filename, mode.replace('b', ''))
    elif filename.endswith('.gz'):
        return gzip.GzipFile(filename, mode)
    else:
        return file(filename, mode)


def json_dump(data, filename, **kwargs):
    with open_compressed(filename, 'w') as f:
        simplejson.dump(data, f, **kwargs)


def json_load(filename):
    with open_compressed(filename, 'rb') as f:
        return simplejson.load(f)


def json_stream_dump(stream, filename, **kwargs):
    kwargs['separators'] = (',', ':')
    stream = iter(stream)
    with open_compressed(filename, 'w') as f:
        f.write('[')
        try:
            item = next(stream)
            f.write('\n')
            simplejson.dump(item, f, **kwargs)
            for item in stream:
                f.write(',\n')
                simplejson.dump(item, f, **kwargs)
        except StopIteration:
            pass
        f.write('\n]')


def json_costream_dump(filename, **kwargs):
    kwargs['separators'] = (',', ':')
    with open_compressed(filename, 'w') as f:
        f.write('[')
        try:
            item = (yield)
            f.write('\n')
            simplejson.dump(item, f, **kwargs)
            while True:
                item = (yield)
                f.write(',\n')
                simplejson.dump(item, f, **kwargs)
        except GeneratorExit:
            pass
        f.write('\n]')


class json_stream_load(object):
    '''
    Read json data that was created by json_stream_dump or json_costream_dump.

    Note that this exploits newline formatting in the above dumpers.
    In particular:
    - the first line is '['
    - intermediate lines are of the form '{},'.format(json_parsable_content)
    - the penultimate line is of the form '{}'.format(json_parsable_content)
    - the last line is ']'
    - there is no trailing whitespace

    An alternative would be to use ijson to streamingly load arbitrary json
    files, however in practice this is ~40x slower.
    '''
    def __init__(self, filename):
        self.fd = open_compressed(filename, 'rb')
        line = self.fd.readline(2)
        if line != '[\n':
            raise IOError(
                'Unhandled format for json_stream_load. '
                'Try recreating json file with the compatible '
                'json_stream_dump or json_costream_dump.')

    def __iter__(self):
        return self

    def next(self):
        line = self.fd.readline().rstrip(',\n')
        if line == ']':
            self.close()
            raise StopIteration
        else:
            return simplejson.loads(line)

    def close(self):
        self.fd.close()


def protobuf_stream_write(item, fd):
    assert isinstance(item, str), item
    fd.write(struct.pack('<I', len(item)))
    fd.write(item)


def protobuf_stream_read(fd):
    size_str = fd.read(4)
    if len(size_str) < 4:
        raise StopIteration
    size = struct.unpack('<I', size_str)[0]
    return fd.read(size)


def protobuf_stream_dump(stream, filename):
    with open_compressed(filename, 'wb') as f:
        for item in stream:
            protobuf_stream_write(item, f)


class protobuf_stream_load(object):
    def __init__(self, filename):
        self.fd = open_compressed(filename, 'rb')

    def __iter__(self):
        return self

    def next(self):
        return protobuf_stream_read(self.fd)

    def close(self):
        self.fd.close()
