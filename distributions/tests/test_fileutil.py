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
