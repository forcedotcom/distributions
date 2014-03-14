// Copyright (c) 2014, Salesforce.com, Inc.  All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
// - Redistributions of source code must retain the above copyright
//   notice, this list of conditions and the following disclaimer.
// - Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the distribution.
// - Neither the name of Salesforce.com nor the names of its contributors
//   may be used to endorse or promote products derived from this
//   software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE
// COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
// OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
// TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
// USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include <utility>
#include <unordered_map>

namespace distributions
{

template<class Key, class Value>
class SparseCounter
{
    struct TrivialHash
    {
        typedef Key argument_type;
        typedef size_t result_type;

        size_t operator() (const Key & key) const { return key; }
    };

    typedef std::unordered_map<Key, Value, TrivialHash> map_t;

public:

    typedef Key key_t;
    typedef Value value_t;
    typedef typename map_t::const_iterator iterator;

    void clear ()
    {
        map_.clear();
        total_ = 0;
    }

    void init_count (key_t key, value_t value)
    {
        // assumes value > 0, key is not present
        map_.insert(std::make_pair(key, value));
        total_ += value;
    }

    value_t get_count (key_t key) const
    {
        auto i = map_.find(key);
        return i == map_.end() ? 0 : i->second;
    }

    value_t get_total () const { return total_; }

    value_t add (const key_t & key, const value_t & value = 1)
    {
        // assumes value > 0
        total_ += value;
        auto i = map_.find(key);
        if (i != map_.end()) {
            return i->second += value;
        } else {
            map_.insert(std::make_pair(key, value));
            return 1;
        }
    }

    value_t remove (const key_t & key)
    {
        // assumes value > 0
        total_ -= 1;
        auto i = map_.find(key);
        value_t new_value = i->second -= 1;
        if (new_value == 0) {
            map_.erase(i);
        }
        return new_value;
    }

    void merge (const SparseCounter<key_t, value_t> & other)
    {
        for (auto i : other.map_) {
            add(i.first, i.second);
        }
        total_ += other.total_;
    }

    iterator begin () const { return map_.begin(); }
    iterator end () const { return map_.end(); }

private:

    map_t map_;
    value_t total_;
};

} // namespace distributions
