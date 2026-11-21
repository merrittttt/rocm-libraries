// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"
#include "ck/tensor_operation/gpu/device/impl/device_pool3d_fwd_ndhwc_ndhwc.hpp"
#include "profiler/profile_pool3d_fwd_impl.hpp"
#include "test_pool_fwd_common.hpp"

// Padding expands a 2x2x2 input to a 3x3x3 output. All 27 output elements must
// fit in the descriptor's buffer bound, even though the input has only eight.
TEST(AvgPool3dFwdDescriptor, OutputLargerThanInput)
{
    using DevicePool =
        ck::tensor_operation::device::DevicePool3dFwd_NDHWC_NDHWC<float,
                                                                  float,
                                                                  int32_t,
                                                                  float,
                                                                  ck::ReduceTensorOp::AVG,
                                                                  false,
                                                                  256,
                                                                  256,
                                                                  1,
                                                                  1,
                                                                  1,
                                                                  1>;
    const auto descriptors =
        DevicePool::MakeABGridDescriptor_A_M_K_B_M({1, 1, 2, 2, 2},  // input NCDHW lengths
                                                   {1, 1, 3, 3, 3},  // output NCDHW lengths
                                                   {8, 1, 4, 2, 1},  // input strides, NDHWC
                                                   {27, 1, 9, 3, 1}, // output strides, NDHWC
                                                   {2, 2, 2},        // window
                                                   {1, 1, 1},        // stride
                                                   {1, 1, 1},        // dilation
                                                   {1, 1, 1},        // left padding
                                                   {1, 1, 1});       // right padding

    EXPECT_EQ(descriptors[ck::Number<0>{}].GetElementSpaceSize(), 8);
    EXPECT_EQ(descriptors[ck::Number<1>{}].GetElementSpaceSize(), 27);
}

static ck::index_t param_mask     = 0xffff;
static ck::index_t instance_index = -1;

template <typename Tuple>
class TestAvgPool3dFwd : public ::testing::Test
{
    protected:
    using InDataType      = std::tuple_element_t<0, Tuple>;
    using OutDataType     = std::tuple_element_t<1, Tuple>;
    using ComputeDataType = std::tuple_element_t<2, Tuple>;
    using IndexDataType   = std::tuple_element_t<3, Tuple>;

    std::vector<PoolingParam> params;

    ck::profiler::PoolFwdInputParams in_params_avg_pool{true, 2, false, false, false, 1};

    void Run()
    {
        for(size_t i = 0; i < this->params.size(); i++)
        {
            if((param_mask & (1 << i)) == 0)
            {
                continue;
            }
            auto& param = this->params[i];
            ck::profiler::PoolFwdKernelParams kernel_params{param.length_,
                                                            param.window_spatial_lengths_,
                                                            param.window_strides_,
                                                            param.window_dilations_,
                                                            param.input_left_pads_,
                                                            param.input_right_pads_};

            bool success =
                ck::profiler::profile_pool3d_fwd_impl<InDataType,
                                                      OutDataType,
                                                      ComputeDataType,
                                                      IndexDataType,
                                                      ck::tensor_layout::convolution::NDHWC,
                                                      ck::tensor_layout::convolution::NDHWC,
                                                      ck::ReduceTensorOp::AVG,
                                                      false,
                                                      false>(
                    in_params_avg_pool, kernel_params, instance_index);
            EXPECT_TRUE(success);
        }
    }
};

using KernelTypes = ::testing::Types<std::tuple<I8, I8, I32, I32>,
                                     std::tuple<F8, F8, F32, I32>,
                                     std::tuple<F16, F16, F32, I32>,
                                     std::tuple<BF16, BF16, F32, I32>,
                                     std::tuple<F32, F32, F32, I32>>;

TYPED_TEST_SUITE(TestAvgPool3dFwd, KernelTypes);
TYPED_TEST(TestAvgPool3dFwd, Test_Pool)
{
    // length, window_length, window_stride, window_dilation, left_pad, right_pad
    this->params = {{{1, 1, 1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}},
                    {{2, 16, 64, 64, 64}, {64, 64, 64}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}},
                    {{2, 16, 64, 64, 64}, {4, 4, 4}, {4, 4, 4}, {2, 2, 2}, {0, 0, 0}, {0, 0, 0}},
                    {{2, 32, 30, 30, 30}, {2, 2, 2}, {2, 2, 2}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}}};

    this->Run();
}
int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    if(argc == 1) {}
    else if(argc == 3)
    {
        param_mask     = strtol(argv[1], nullptr, 0);
        instance_index = atoi(argv[2]);
    }
    else
    {
        std::cout << "Usage of " << argv[0] << std::endl;
        std::cout << "Arg1,2: param_mask instance_index(-1 means all)" << std::endl;
    }
    return RUN_ALL_TESTS();
}
