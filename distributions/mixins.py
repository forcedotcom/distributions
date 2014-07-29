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

import warnings
import functools


def deprecated(message='function will be removed in the future'):

    def decorator(fun):

        @functools.wraps(fun)
        def deprecated_fun(*args, **kwargs):
            warnings.warn('DEPRECATED {}: {}'.format(fun.__name__, message))
            return fun(*args, **kwargs)

        return deprecated_fun

    return decorator


class ComponentModel(object):
    pass


class SharedMixin(object):
    def add_value(self, value):
        pass

    def remove_value(self, value):
        pass

    def realize(self):
        pass


class ProtobufSerializable(object):
    @classmethod
    def to_protobuf(cls, raw, message):
        model = cls()
        model.load(raw)
        model.protobuf_dump(message)

    @classmethod
    def from_protobuf(cls, message):
        model = cls()
        model.protobuf_load(message)
        return model.dump()

    @deprecated('use protobuf_dump(message) instead')
    def dump_protobuf(self, message):
        self.protobuf_dump(message)

    @deprecated('use protobuf_load(message) instead')
    def load_protobuf(self, message):
        self.protobuf_load(message)


class GroupIoMixin(ProtobufSerializable):
    @classmethod
    def from_values(cls, model, values=[]):
        group = cls()
        group.init(model)
        for value in values:
            group.add_value(model, value)
        return group

    @classmethod
    def from_dict(cls, raw):
        group = cls()
        group.load(raw)
        return group


class SharedIoMixin(ProtobufSerializable):
    @classmethod
    def from_dict(cls, raw):
        model = cls()
        model.load(raw)
        return model
