// captured translation unit 0

#include <torch/csrc/inductor/cpp_prefix.h>
extern "C" __declspec(dllexport) void  kernel(float* in_out_ptr0,
                       float* out_ptr0)
{
    {
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(12LL); x0+=static_cast<int64_t>(1LL))
        {
            {
                {
                    auto tmp0 = in_out_ptr0[static_cast<int64_t>(x0)];
                    auto tmp1 = static_cast<float>(0.25);
                    auto tmp2 = float(tmp0 + tmp1);
                    auto tmp3 = std::sin(tmp2);
                    in_out_ptr0[static_cast<int64_t>(x0)] = tmp3;
                }
            }
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0LL); x0<static_cast<int64_t>(3LL); x0+=static_cast<int64_t>(1LL))
        {
            {
                {
                    auto tmp0 = in_out_ptr0[static_cast<int64_t>(4LL*x0)];
                    auto tmp1 = in_out_ptr0[static_cast<int64_t>(1LL + 4LL*x0)];
                    auto tmp3 = in_out_ptr0[static_cast<int64_t>(2LL + 4LL*x0)];
                    auto tmp5 = in_out_ptr0[static_cast<int64_t>(3LL + 4LL*x0)];
                    auto tmp2 = float(tmp0 + tmp1);
                    auto tmp4 = float(tmp2 + tmp3);
                    auto tmp6 = float(tmp4 + tmp5);
                    out_ptr0[static_cast<int64_t>(x0)] = tmp6;
                }
            }
        }
    }
}
