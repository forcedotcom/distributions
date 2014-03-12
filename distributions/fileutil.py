import os
import shutil
import tempfile
import bz2
import gzip
import contextlib
import simplejson


@contextlib.contextmanager
def chdir(wd):
    oldwd = os.getcwd()
    try:
        print 'cd', wd
        os.chdir(wd)
        yield
    finally:
        print 'cd', oldwd
        os.chdir(oldwd)


@contextlib.contextmanager
def tempdir():
    oldwd = os.getcwd()
    wd = tempfile.mkdtemp()
    try:
        print 'cd', wd
        os.chdir(wd)
        yield wd
    finally:
        print 'cd', oldwd
        os.chdir(oldwd)
        shutil.rmtree(wd)


def open_compressed(filename, mode='r'):
    if filename.endswith('.bz2'):
        return bz2.BZ2File(filename, mode.replace('b', ''))
    elif filename.endswith('.gz'):
        return gzip.GzipFile(filename, mode)
    else:
        return file(filename, mode)


def json_dump(data, filename, **kwargs):
    dirname = os.path.dirname(filename)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname)
    with open_compressed(filename, 'w') as f:
        simplejson.dump(data, f, **kwargs)


def json_load(filename):
    with open_compressed(filename, 'rb') as f:
        return simplejson.load(f)


def json_stream_dump(stream, filename, **kwargs):
    kwargs['separators'] = (',', ':')
    stream = iter(stream)
    dirname = os.path.dirname(filename)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname)
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
    dirname = os.path.dirname(filename)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname)
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
