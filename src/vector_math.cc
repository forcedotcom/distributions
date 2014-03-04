#ifdef USE_INTEL_MKL

#include <mkl.h>
#include <mkl_vml.h>

namespace
{

struct InitializeMKL
{
    InitializeMKL()
    {
        mkl_set_num_threads(1);
        vmlSetMode(VML_EP | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE);
    }
};

InitializeMKL initialize_mkl;

} // anonymous namespace

#endif // USE_INTEL_MKL
