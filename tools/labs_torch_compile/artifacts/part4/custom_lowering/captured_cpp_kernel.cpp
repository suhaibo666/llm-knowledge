// captured translation unit 0

#include <torch/csrc/inductor/cpp_prefix.h>
extern "C" __declspec(dllexport) void  kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(8LL); x0+=static_cast<int64_t>(1LL))
        {
            {
                float tmp_acc0 = 0;
                for(int64_t x1=static_cast<int64_t>(0LL); x1<static_cast<int64_t>(16LL); x1+=static_cast<int64_t>(1LL))
                {
                    {
                        {
                            auto tmp0 = in_ptr0[static_cast<int64_t>(x1 + 16LL*x0)];
                            auto tmp1 = static_cast<float>(2.0);
                            auto tmp2 = float(tmp0 * tmp1);
                            tmp_acc0 = tmp_acc0 + tmp2;
                        }
                    }
                }
                out_ptr0[static_cast<int64_t>(x0)] = tmp_acc0;
            }
        }
    }
}
