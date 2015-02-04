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
#include <distributions/common.hpp>
#include <distributions/trivial_hash.hpp>

namespace distributions {

template<class Key, class Value>
class Sparse_ {
    typedef std::unordered_map<Key, Value, TrivialHash<Key>> map_t;

    map_t map_;

 public:
    typedef Key key_t;
    typedef Value value_t;
    typedef typename map_t::iterator iterator;
    typedef typename map_t::const_iterator const_iterator;

    size_t size() const { return map_.size(); }
    void clear() { map_.clear(); }

    bool contains(const Key & key) const {
        return map_.find(key) != map_.end();
    }

    Value & add(const Key & key) {
        auto pair = map_.insert(std::make_pair(key, Value()));
        DIST_ASSERT1(pair.second, "duplicate key: " << key);
        return pair.first->second;
    }

    void add(const Key & key, const Value & value) {
        auto pair = map_.insert(std::make_pair(key, value));
        DIST_ASSERT1(pair.second, "duplicate key: " << key);
    }

    void remove(const Key & key) {
        bool removed = map_.erase(key);
        DIST_ASSERT1(removed, "missing key: " << key);
    }

    Value pop(const Key & key) {
        auto i = map_.find(key);
        DIST_ASSERT1(i != map_.end(), "missing key: " << key);
        Value result = std::move(i->second);
        map_.erase(i);
        return result;
    }

    void set(const Key & key, const Value & value) {
        auto i = map_.find(key);
        DIST_ASSERT1(i != map_.end(), "missing key: " << key);
        i->second = value;
    }

    Value & get(const Key & key) {
        auto i = map_.find(key);
        DIST_ASSERT1(i != map_.end(), "missing key: " << key);
        return i->second;
    }

    Value & get_or_add(const Key & key) {
        return map_.operator[](key);
    }

    const Value & get(const Key & key) const {
        auto i = map_.find(key);
        DIST_ASSERT1(i != map_.end(), "missing key: " << key);
        return i->second;
    }

    void unsafe_erase(iterator i) { map_.erase(i); }

    iterator begin() { return map_.begin(); }
    iterator end() { return map_.end(); }
    const_iterator begin() const { return map_.begin(); }
    const_iterator end() const { return map_.end(); }
};


template<class Key, class Value>
class SparseCounter {
    typedef std::unordered_map<Key, Value, TrivialHash<Key>> map_t;

    map_t map_;
    Value total_;

 public:
    typedef Key key_t;
    typedef Value value_t;
    typedef typename map_t::const_iterator iterator;

    void clear() {
        map_.clear();
        total_ = 0;
    }

    void init_count(key_t key, value_t value) {
        if (DIST_LIKELY(value)) {
            bool success = map_.insert(std::make_pair(key, value)).second;
            DIST_ASSERT1(success, "duplicate key: " << key);
            total_ += value;
        }
    }

    value_t get_count(key_t key) const {
        auto i = map_.find(key);
        return i == map_.end() ? 0 : i->second;
    }

    value_t get_total() const { return total_; }

    value_t add(const key_t & key, value_t value = 1) {
        static_assert(value_t(-1) < value_t(0), "value_t must be signed");
        if (DIST_LIKELY(value)) {
            total_ += value;
            auto pair = map_.insert(std::make_pair(key, value));
            bool inserted = pair.second;
            if (not inserted) {
                value = pair.first->second += value;
                if (DIST_UNLIKELY(value == 0)) {
                    map_.erase(pair.first);
                }
            }
            return value;
        } else {
            return get_count(key);
        }
    }

    value_t remove(const key_t & key) { return add(key, -1); }

    void merge(const SparseCounter<key_t, value_t> & other) {
        for (auto & i : other.map_) {
            add(i.first, i.second);
        }
        total_ += other.total_;
    }

    void rename(key_t old_key, key_t new_key) {
        auto i = map_.find(old_key);
        if (i != map_.end()) {
            value_t value = i->second;
            map_.erase(i);
            bool success = map_.insert(std::make_pair(new_key, value)).second;
            DIST_ASSERT1(success, "duplicate key: " << new_key);
        }
    }

    iterator begin() const { return map_.begin(); }
    iterator end() const { return map_.end(); }
};

}  // namespace distributions
