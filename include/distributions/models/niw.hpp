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

#include <distributions/common.hpp>
#include <distributions/special.hpp>
#include <distributions/random.hpp>
#include <distributions/vector.hpp>
#include <distributions/vector_math.hpp>
#include <distributions/mixins.hpp>
#include <distributions/mixture.hpp>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Cholesky>

namespace distributions
{

template <typename Matrix>
static inline bool
is_symmetric(const Matrix &m)
{
    if (!m.isApprox(m.transpose()))
        return false;
    return true;
}

template <typename Matrix>
static inline bool
is_symmetric_positive_definite(const Matrix &m)
{
    if (!is_symmetric(m))
        return false;
    Eigen::LDLT<Matrix> ldlt;
    ldlt.compute(m);
    return ldlt.isPositive();
}

template<int dim_ = -1>
struct NormalInverseWishart
{

static_assert(dim_ == -1 || dim_ > 0, "invalid dimension");

typedef Eigen::Matrix<float, dim_, dim_> Matrix;
typedef Eigen::Matrix<float, dim_, 1> Vector;

typedef NormalInverseWishart<dim_> Model;
typedef Vector Value;
struct Group;
struct Scorer;
struct Sampler;

struct Shared : SharedMixin<Model>
{
    Vector mu;
    float kappa;
    Matrix psi;
    float nu;

    template<class Message>
    void protobuf_load (const Message & message)
    {
        // mu
        check_row_or_col_size(message.mu_size());
        mu.resize(message.mu_size(), Eigen::NoChange);
        for (size_t i = 0; i < (size_t)mu.rows(); i++)
            mu(i) = message.mu(i);

        // kappa
        DIST_ASSERT_GT(message.kappa(), 0.);
        kappa = message.kappa();

        // psi
        check_rowcol_size(message.psi_size());
        psi.resize(message.psi_size() / 2, message.psi_size() / 2);
        for (size_t i = 0; i < (size_t)psi.rows(); i++)
            for (size_t j = 0; j < (size_t)psi.cols(); j++)
                psi(i, j) = message.psi(i*size_t(psi.cols()) + j);
        DIST_ASSERT3(is_symmetric_positive_definite(psi),
                     "expected SPD matrix");
        DIST_ASSERT_EQ(mu.rows(), psi.rows());

        // nu
        DIST_ASSERT_GT(message.nu(), float(mu.rows()) - 1.);
        nu = message.nu();
    }

    template<class Message>
    void protobuf_dump (Message & message) const
    {
        message.Clear();

        for (size_t i = 0; i < (size_t)mu.size(); i++)
            message.add_mu(mu(i));

        message.set_kappa(kappa);

        for (size_t i = 0; i < (size_t)psi.rows(); i++)
            for (size_t j = 0; j < (size_t)psi.cols(); j++)
                message.add_psi(psi(i, j));

        message.set_nu(nu);
    }

    inline unsigned dim() const { return mu.rows(); }

    static Shared EXAMPLE ()
    {
        const size_t actual_dim = (dim_ == -1) ? 3 : size_t(dim_);
        Shared shared;
        shared.mu.resize(actual_dim, Eigen::NoChange);
        shared.mu.setZero();
        shared.kappa = 1.0;
        shared.psi.resize(actual_dim, actual_dim);
        shared.psi.setIdentity();
        shared.nu = float(actual_dim) + 1;
        return shared;
    }

    static inline DIST_ALWAYS_INLINE void
    check_row_or_col_size(size_t size)
    {
        if (dim_ == -1)
            return;
        DIST_ASSERT_EQ(size_t(dim_), size);
    }

    static inline DIST_ALWAYS_INLINE void
    check_rowcol_size(size_t size)
    {
        if (dim_ == -1)
            return;
        DIST_ASSERT_EQ(size_t(dim_*dim_), size);
    }
};

struct Group : GroupMixin<Model>
{
    unsigned count;
    Vector sum_x;
    Matrix sum_xxT;

    template<class Message>
    void protobuf_load (const Message & message)
    {
        // count
        count = message.count();

        // sum_x
        Shared::check_row_or_col_size(message.sum_x_size());
        sum_x.resize(message.sum_x_size(), Eigen::NoChange);
        for (size_t i = 0; i < (size_t)sum_x.rows(); i++)
            sum_x(i) = message.sum_x(i);

        // sum_xxT
        Shared::check_rowcol_size(message.sum_xxt_size());
        sum_xxT.resize(message.sum_xxt_size() / 2, message.sum_xxt_size() / 2);
        for (size_t i = 0; i < (size_t)sum_xxT.rows(); i++)
            for (size_t j = 0; j < (size_t)sum_xxT.cols(); j++)
                sum_xxT(i, j) = message.sum_xxt(i*size_t(sum_xxT.cols()) + j);
        // XXX(stephentu): should also assert positive semi-definite
        DIST_ASSERT3(is_symmetric(sum_xxT), "expected sym matrix");
        DIST_ASSERT_EQ(sum_x.rows(), sum_xxT.rows());
    }

    template<class Message>
    void protobuf_dump (Message & message) const
    {
        message.Clear();
        message.set_count(count);
        for (size_t i = 0; i < (size_t)sum_x.rows(); i++)
            message.add_sum_x(sum_x(i));
        for (size_t i = 0; i < (size_t)sum_xxT.rows(); i++)
            for (size_t j = 0; j < (size_t)sum_xxT.cols(); j++)
                message.add_sum_xxt(sum_xxT(i, j));
    }

    void init (
            const Shared & shared,
            rng_t &)
    {
        DIST_ASSERT3(shared.dim() > 0, "invalid shared");
        count = 0;
        sum_x.resize(shared.dim(), Eigen::NoChange);
        sum_x.setZero();
        sum_xxT.resize(shared.dim(), shared.dim());
        sum_xxT.setZero();
    }

    void add_value (
            const Shared & shared,
            const Value & value,
            rng_t &)
    {
        DIST_ASSERT3(shared.dim() == (size_t)value.size(), "dim mismatch");
        count++;
        sum_x += value;
        sum_xxT += value * value.transpose();
    }

    void add_repeated_value(
            const Shared & shared,
            const Value & value,
            const int & count,
            rng_t &)
    {
        DIST_ASSERT3(shared.dim() == (size_t)value.size(), "dim mismatch");
        this->count += count;
        sum_x += count * value;
        sum_xxT += count * (value * value.transpose());
    }

    void remove_value (
            const Shared & shared,
            const Value & value,
            rng_t &)
    {
        DIST_ASSERT3(shared.dim() == (size_t)value.size(), "dim mismatch");
        DIST_ASSERT3(count > 0, "nothing to remove");
        count--;
        sum_x -= value;
        sum_xxT -= value * value.transpose();
    }

    void merge (
            const Shared &,
            const Group & source,
            rng_t &)
    {
        count += source.count;
        sum_x += source.sum_x;
        sum_xxT += source.sum_xxT;
    }

    float score_value (
            const Shared & shared,
            const Value & value,
            rng_t & rng) const
    {
        Scorer scorer;
        scorer.init(shared, *this, rng);
        return scorer.eval(shared, value, rng);
    }

    float score_data (
            const Shared & shared,
            rng_t &) const
    {
        Shared post;
        advance(shared, post);
        return lmultigamma(shared.dim(), post.nu*0.5)
            + shared.nu*0.5*fast_log(shared.psi.determinant())
            - float(count*shared.dim())*0.5*1.1447298858494002 /* log(pi) */
            - lmultigamma(shared.dim(), shared.nu*0.5)
            - post.nu*0.5*fast_log(post.psi.determinant())
            + float(shared.dim())*0.5*fast_log(shared.kappa/post.kappa);
    }

    Value sample_value (
            const Shared & shared,
            rng_t & rng) const
    {
        Sampler sampler;
        sampler.init(shared, *this, rng);
        return sampler.eval(shared, rng);
    }

    void validate (const Shared & shared) const
    {

    }

    // XXX: some more cleverness is needed, but let's just
    // get this working for now
    inline void advance (
            const Shared & shared,
            Shared & post) const
    {
        const float n = count;
        Vector xbar;
        if (count)
            xbar = sum_x / n;
        else
            xbar = Vector::Zero(shared.dim());
        post.mu = shared.kappa/(shared.kappa + n)*shared.mu
            + n/(shared.kappa + n)*xbar;
        post.kappa = shared.kappa + n;
        post.nu = shared.nu + n;
        const Vector &diff = xbar - shared.mu;
        const Matrix &C_n = sum_xxT
            - sum_x*xbar.transpose()
            - xbar*sum_x.transpose()
            + n*xbar*xbar.transpose();
        post.psi = shared.psi
            + C_n
            + shared.kappa*n/(shared.kappa + n)*diff*diff.transpose();
    }
};

struct Sampler
{
    Vector mu;
    Matrix cov;

    void init (
            const Shared & shared,
            const Group & group,
            rng_t & rng)
    {
        Shared post;
        group.advance(shared, post);
        const auto p = sample_normal_inverse_wishart(
                post.mu, post.kappa, post.psi, post.nu, rng);
        mu.swap(p.first);
        cov.swap(p.second);
    }

    Value eval (
            const Shared &,
            rng_t & rng) const
    {
        return sample_multivariate_normal(mu, cov, rng);
    }
};

struct Scorer
{
    Shared post;

    void init (
            const Shared & shared,
            const Group & group,
            rng_t &)
    {
        group.advance(shared, post);
    }

    float eval (
            const Shared & shared,
            const Value & value,
            rng_t &) const
    {
        const float dof = post.nu - float(shared.dim()) + 1.;
        const Matrix sigma = post.psi * (post.kappa+1.)/(post.kappa*dof);
        return score_mv_student_t(value, dof, post.mu, sigma);
    }
};

}; // struct NormalInverseWishart

extern template struct NormalInverseWishart<-1>;
extern template struct NormalInverseWishart<2>;
extern template struct NormalInverseWishart<3>;

} // namespace distributions

/* vim: set ts=4 sw=4 sts=4 expandtab : */
