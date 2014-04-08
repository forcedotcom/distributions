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

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/gzip_stream.h>
#include <distributions/common.hpp>

namespace distributions
{
namespace protobuf
{

inline bool endswith (const char * filename, const char * suffix)
{
    return strlen(filename) >= strlen(suffix) and
        strcmp(filename + strlen(filename) - strlen(suffix), suffix) == 0;
}

class InFile
{
public:

    InFile (const char * filename) : filename_(filename)
    {
        if (strcmp(filename, "-") == 0 or strcmp(filename, "-.gz") == 0) {
            fid_ = STDIN_FILENO;
        } else {
            fid_ = open(filename, O_RDONLY | O_NOATIME);
            DIST_ASSERT(fid_ != -1, "failed to open input file " << filename);
        }

        raw_ = new google::protobuf::io::FileInputStream(fid_);

        if (endswith(filename, ".gz")) {
            gzip_ = new google::protobuf::io::GzipInputStream(raw_);
            coded_ = new google::protobuf::io::CodedInputStream(gzip_);
        } else {
            gzip_ = nullptr;
            coded_ = new google::protobuf::io::CodedInputStream(raw_);
        }
    }

    ~InFile ()
    {
        delete coded_;
        delete gzip_;
        delete raw_;
        if (fid_ != STDIN_FILENO) {
            close(fid_);
        }
    }

    template<class Message>
    void read (Message & message)
    {
        bool success = message.ParseFromCodedStream(coded_);
        DIST_ASSERT(success, "failed to parse message from " << filename_);
    }

    template<class Message>
    uint32_t try_read_stream (Message & message)
    {
        uint32_t message_size = 0;
        if (DIST_LIKELY(coded_->ReadLittleEndian32(& message_size))) {
            auto old_limit = coded_->PushLimit(message_size);
            bool success = message.ParseFromCodedStream(coded_);
            DIST_ASSERT(success, "failed to parse message from " << filename_);
            coded_->PopLimit(old_limit);
            return message_size;
        } else {
            return 0;
        }
    }

private:

    const std::string filename_;
    int fid_;
    google::protobuf::io::ZeroCopyInputStream * raw_;
    google::protobuf::io::GzipInputStream * gzip_;
    google::protobuf::io::CodedInputStream * coded_;
};


class OutFile
{
public:

    OutFile (const char * filename) : filename_(filename)
    {
        if (strcmp(filename, "-") == 0 or strcmp(filename, "-.gz") == 0) {
            fid_ = STDOUT_FILENO;
        } else {
            fid_ = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0664);
            DIST_ASSERT(fid_ != -1, "failed to open output file " << filename);
        }

        raw_ = new google::protobuf::io::FileOutputStream(fid_);

        if (endswith(filename, ".gz")) {
            gzip_ = new google::protobuf::io::GzipOutputStream(raw_);
            coded_ = new google::protobuf::io::CodedOutputStream(gzip_);
        } else {
            gzip_ = nullptr;
            coded_ = new google::protobuf::io::CodedOutputStream(raw_);
        }
    }

    ~OutFile ()
    {
        delete coded_;
        delete gzip_;
        delete raw_;
        if (fid_ != STDOUT_FILENO) {
            close(fid_);
        }
    }

    template<class Message>
    void write (Message & message)
    {
        DIST_ASSERT1(message.IsInitialized(), "message not initialized");
        bool success = message.SerializeToCodedStream(coded_);
        DIST_ASSERT(success, "failed to serialize message to " << filename_);
    }

    template<class Message>
    void write_stream (Message & message)
    {
        DIST_ASSERT1(message.IsInitialized(), "message not initialized");
        uint32_t message_size = message.ByteSize();
        DIST_ASSERT1(message_size > 0, "zero sized message is not supported");
        coded_->WriteLittleEndian32(message_size);
        message.SerializeWithCachedSizes(coded_);
    }

private:

    const std::string filename_;
    int fid_;
    google::protobuf::io::ZeroCopyOutputStream * raw_;
    google::protobuf::io::GzipOutputStream * gzip_;
    google::protobuf::io::CodedOutputStream * coded_;
};

} // namespace protobuf
} // namespace distributions
