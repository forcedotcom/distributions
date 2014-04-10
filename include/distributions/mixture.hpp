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

#include <vector>
#include <distributions/common.hpp>

namespace distributions
{

class MixtureIdTracker
{
public:

    typedef uint32_t Id;

    void init (size_t group_count = 0)
    {
        packed_to_global_.clear();
        global_to_packed_.clear();
        for (size_t i = 0; i < group_count; ++i) {
            add_group();
        }
    }

    void add_group ()
    {
        const Id packed = packed_to_global_.size();
        const Id global = global_to_packed_.size();
        packed_to_global_.push_back(global);
        global_to_packed_.push_back(packed);
    }

    void remove_group (Id packed)
    {
        if (DIST_DEBUG_LEVEL) {
            DIST_ASSERT(packed < packed_size(), "bad packed id: " << packed);
            const Id global = packed_to_global_[packed];
            DIST_ASSERT(global < global_size(), "bad global id: " << global);
            global_to_packed_[global] = ~Id(0);
        }
        const size_t group_count = packed_size() - 1;
        if (packed != group_count) {
            const Id global = packed_to_global_.back();
            DIST_ASSERT1(global < global_size(), "bad global id: " << global);
            packed_to_global_[packed] = global;
            global_to_packed_[global] = packed;
        }
        packed_to_global_.resize(group_count);;
    }

    Id packed_to_global (Id packed) const
    {
        DIST_ASSERT1(packed < packed_size(), "bad packed id: " << packed);
        Id global = packed_to_global_[packed];
        DIST_ASSERT1(global < global_size(), "bad global id: " << global);
        return global;
    }

    Id global_to_packed (Id global) const
    {
        DIST_ASSERT1(global < global_size(), "bad global id: " << global);
        Id packed = global_to_packed_[global];
        DIST_ASSERT1(packed < packed_size(), "bad packed id: " << packed);
        return packed;
    }

    size_t packed_size () const { return packed_to_global_.size(); }
    size_t global_size () const { return global_to_packed_.size(); }

private:

    std::vector<Id> packed_to_global_;
    std::vector<Id> global_to_packed_;
};

} // namespace distributions
