from __future__ import print_function
import fnmatch
import os
import re
import parsable

SYMBOLS = {
    '//': ['.c', '.cc', '.cpp', '.h', '.hpp', '.proto'],
    '#': ['.py', '.pxd', '.pyx'],
    '%': ['.tex'],
}
extensions = sum(SYMBOLS.values(), [])
SYMBOL_OF = {
    extension: symbol
    for symbol, extensions in SYMBOLS.iteritems()
    for extension in extensions
}

FILES = sorted(
    os.path.join(root, filename)
    for root, dirnames, filenames in os.walk('.')
    if '.git' not in root.split('/')
    if 'examples' not in root.split('/')
    for extension in extensions
    for filename in fnmatch.filter(filenames, '*' + extension)
    if not fnmatch.fnmatch(filename, '*.pb.h')
    if not fnmatch.fnmatch(filename, '*.pb.cc')
    if not fnmatch.fnmatch(filename, '*_pb2.py')
    if not fnmatch.fnmatch(filename, 'test_headers.cc')
)

LICENSE = []
with open('LICENSE.txt') as f:
    for line in f:
        LICENSE.append(line.rstrip())

HEADERS = {
    symbol: [symbol + ' ' + line if line else symbol for line in LICENSE]
    for symbol in SYMBOLS
}


@parsable.command
def show():
    '''
    List all files that should have a license.
    '''
    for filename in FILES:
        print(filename)


def read_and_strip_lines(filename):
    extension = re.search('\.[^.]*$', filename).group()
    symbol = SYMBOL_OF[extension]
    lines = []
    with open(filename) as i:
        writing = False
        for line in i:
            line = line.rstrip()
            if not writing and line and not line.startswith(symbol):
                writing = True
            if writing:
                lines.append(line)
    return lines


def write_lines(lines, filename):
    with open(filename, 'w') as f:
        for line in lines:
            print(line, file=f)


@parsable.command
def strip():
    '''
    Strip headers from all files.
    '''
    for filename in FILES:
        lines = read_and_strip_lines(filename)
        write_lines(lines, filename)


@parsable.command
def update():
    '''
    Update headers on all files to match LICNESE.txt.
    '''
    for filename in FILES:
        extension = re.search('\.[^.]*$', filename).group()
        symbol = SYMBOL_OF[extension]
        lines = read_and_strip_lines(filename)
        if lines and lines[0]:
            if extension == '.py' and lines[0].startswith('class '):
                lines = [''] + lines  # pep8 compliance
            write_lines(HEADERS[symbol] + [''] + lines, filename)


if __name__ == '__main__':
    parsable.dispatch()
