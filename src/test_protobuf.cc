#include <distributions/common.hpp>
#include <distributions/io/protobuf.hpp>

#include <distributions/models/bb.hpp>
#include <distributions/models/bnb.hpp>
#include <distributions/models/dd.hpp>
#include <distributions/models/dpd.hpp>
#include <distributions/models/gp.hpp>
#include <distributions/models/nich.hpp>
#include <distributions/models/niw.hpp>

namespace distributions {
typedef DirichletDiscrete<16> DirichletDiscrete16;
typedef NormalInverseWishart<-1> NormalInverseWishartV;
typedef NormalInverseWishart<2> NormalInverseWishart2;
typedef NormalInverseWishart<3> NormalInverseWishart3;
namespace protobuf {
typedef DirichletDiscrete_Shared DirichletDiscrete16_Shared;
typedef NormalInverseWishart_Shared NormalInverseWishartV_Shared;
typedef NormalInverseWishart_Shared NormalInverseWishart2_Shared;
typedef NormalInverseWishart_Shared NormalInverseWishart3_Shared;
typedef DirichletDiscrete_Group DirichletDiscrete16_Group;
typedef NormalInverseWishart_Group NormalInverseWishartV_Group;
typedef NormalInverseWishart_Group NormalInverseWishart2_Group;
typedef NormalInverseWishart_Group NormalInverseWishart3_Group;
}  // namespace protobuf
}  // namespace distributions

#define DIST_MODELS(x) \
    x(BetaBernoulli) \
    x(BetaNegativeBinomial) \
    x(DirichletDiscrete16) \
    x(DirichletProcessDiscrete) \
    x(GammaPoisson) \
    x(NormalInverseChiSq) \
    x(NormalInverseWishartV) \
    x(NormalInverseWishart2) \
    x(NormalInverseWishart3)

template <typename Model> struct message {};

#define DIST_SPECIALIZE_MESSAGE(name) \
    template <> struct message<distributions::name> { \
        typedef distributions::protobuf::name ## _Shared shared_message_type; \
        typedef distributions::protobuf::name ## _Group group_message_type; \
    }; \

DIST_MODELS(DIST_SPECIALIZE_MESSAGE);

#undef DIST_SPECIALIZE_MESSAGE

template <typename Model>
void test_model()
{
    // only checks for assertion errors
    auto const shared = Model::Shared::EXAMPLE();
    typename message<Model>::shared_message_type shared_message;
    shared.protobuf_dump(shared_message);
    typename Model::Shared shared1;
    shared1.protobuf_load(shared_message);
    // TODO(stephentu): check for shared equality (handling floats)

    distributions::rng_t r;
    typename Model::Group group;
    group.init(shared, r);
    typename message<Model>::group_message_type group_message;
    group.protobuf_dump(group_message);
    typename Model::Group group1;
    group1.protobuf_load(group_message);
    // TODO(stephentu): check for group equality (handling floats)
}

int main(void) {
#define DIST_TEST_MODEL(name) test_model<distributions::name>();
    DIST_MODELS(DIST_TEST_MODEL);
#undef DIST_TEST_MODEL
    return 0;
}
