/* 
# Copyright (c) 2013, Salesforce.com, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
#
# Neither the name of Salesforce.com nor the names of its contributors
# may be used to endorse or promote products derived from this
# software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
*/

#include <utility>
#include <unordered_map>

class SparseCounter
{
public:

    typedef int key_type;
    typedef int value_type;

private:

    struct TrivialHash
    {
        typedef key_type argument_type;
        typedef size_t result_type;

        size_t operator() (const key_type & key) const { return key; }
    };

    typedef std::unordered_map<key_type, value_type, TrivialHash> map_t;

public:

    SparseCounter ()
        : map_(),
          total_(0)
    {
    }

    void clear()
    {
        map_.clear();
        total_ = 0;
    }

    void init_count (key_type key, value_type value)
    {
        // assumes value > 0, key is not present
        map_.insert(std::make_pair(key, value));
        total_ += value;
    }

    value_type get_count (key_type key) const
    {
        auto i = map_.find(key);
        return i == map_.end() ? 0 : i->second;
    }

    int get_total () const
    {
        return total_;
    }

    void add (key_type key)
    {
        // assumes value > 0
        auto i = map_.find(key);
        if (i != map_.end()) {
            i->second += 1;
        } else {
            map_.insert(std::make_pair(key, 1));
        }
        total_ += 1;
    }

    void remove (key_type key)
    {
        auto i = map_.find(key);
        value_type new_value = i->second -= 1;
        if (new_value == 0) {
            map_.erase(i);
        }
        total_ -= 1;
    }

    typedef typename map_t::iterator iterator;
    iterator begin () { return map_.begin(); }
    iterator end () { return map_.end(); }

private:

    map_t map_;
    int total_;
};

typedef SparseCounter::iterator SparseCounter_iterator;
