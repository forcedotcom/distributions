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

    void add (const key_t & key, const value_t & value = 1)
    {
        // assumes value > 0
        auto i = map_.find(key);
        if (i != map_.end()) {
            i->second += value;
        } else {
            map_.insert(std::make_pair(key, value));
        }
        total_ += value;
    }

    void remove (const key_t & key)
    {
        // assumes value > 0
        auto i = map_.find(key);
        value_t new_value = i->second -= 1;
        if (new_value == 0) {
            map_.erase(i);
        }
        total_ -= 1;
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
