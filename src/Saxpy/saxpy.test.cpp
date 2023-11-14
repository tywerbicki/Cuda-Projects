#include "saxpy.h"

#include "cuda_runtime.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <stdlib.h>
#include <vector>


class SaxpyTest : public testing::Test
{
protected:
    static void SetUpTestSuite() noexcept
    {
        cudaError_t result      = cudaSuccess;
        int         deviceCount = -1;

        result = cudaGetDeviceCount(&deviceCount);
        ASSERT_EQ(result, cudaSuccess);

        ASSERT_GT(deviceCount, 0) << "No CUDA-enabled device was detected";

        // Set seed once for use of SaxpyTest::GetRandFloat
        std::srand(10);
    }

    void SetUp() noexcept override final
    {
        cudaError_t result = cudaSuccess;

        result = cudaStreamCreateWithFlags(&m_saxpyStream, cudaStreamNonBlocking);
        ASSERT_EQ(result, cudaSuccess);

        result = cudaEventCreate(&m_saxpyComplete);
        ASSERT_EQ(result, cudaSuccess);
    }

    static void TearDownTestSuite() noexcept
    {

    }

    void TearDown() noexcept override final
    {
        cudaError_t result = cudaSuccess;

        result = cudaEventDestroy(m_saxpyComplete);
        EXPECT_EQ(result, cudaSuccess);

        result = cudaStreamDestroy(m_saxpyStream);
        EXPECT_EQ(result, cudaSuccess);
    }

    static float GetRandFloat() noexcept
    {
        return static_cast<float>(std::rand());
    }

    void VerifySolution(const size_t problemSize)
    {
        cudaError_t result = cudaSuccess;

        result = cudaEventRecord(m_saxpyComplete, m_saxpyStream);
        ASSERT_EQ(result, cudaSuccess);

        std::vector<float> solution(problemSize);

        saxpy::HostExecute(A, m_pXHost, m_pYHost, solution.data(), problemSize);

        // Here is where we synchronize the host with the saxpy device operations.
        result = cudaEventSynchronize(m_saxpyComplete);
        ASSERT_EQ(result, cudaSuccess);

        EXPECT_TRUE(std::equal(solution.cbegin(), solution.cend(), m_pZHost)) <<
            "Host and device saxpy execution results are not equal";
    }

    void FreeHostAllocs() noexcept
    {
        cudaError_t result = cudaSuccess;

        result = cudaFreeHost(m_pXHost);
        EXPECT_EQ(result, cudaSuccess);
        result = cudaFreeHost(m_pYHost);
        EXPECT_EQ(result, cudaSuccess);
        result = cudaFreeHost(m_pZHost);
        EXPECT_EQ(result, cudaSuccess);
    }

    cudaStream_t m_saxpyStream   = nullptr;
    cudaEvent_t  m_saxpyComplete = nullptr;
    float*       m_pXHost        = nullptr;
    float*       m_pYHost        = nullptr;
    float*       m_pZHost        = nullptr;
    float*       m_pXDevice      = nullptr;
    float*       m_pYDevice      = nullptr;
    float*       m_pZDevice      = nullptr;

    static const std::array<size_t, 6> ProblemSizes;
    static const float                 A;
};

const std::array<size_t, 6> SaxpyTest::ProblemSizes =
{
    1, 33, 1024, (1024 + 1), (1024 + 31), 1000000
};

const float SaxpyTest::A = 2.75;


TEST_F(SaxpyTest, UsingAsyncDataTransfers)
{
    for (const size_t problemSize : ProblemSizes)
    {
        cudaError_t  result             = cudaSuccess;
        const size_t problemSizeInBytes = problemSize * sizeof(float);

        result = cudaMallocAsync(&m_pXDevice, problemSizeInBytes, m_saxpyStream);
        ASSERT_EQ(result, cudaSuccess);

        result = cudaHostAlloc(&m_pXHost, problemSizeInBytes, cudaHostAllocDefault);
        ASSERT_EQ(result, cudaSuccess);

        std::generate(m_pXHost, m_pXHost + problemSize, GetRandFloat);

        result = cudaMemcpyAsync(m_pXDevice, m_pXHost, problemSizeInBytes, cudaMemcpyHostToDevice, m_saxpyStream);
        ASSERT_EQ(result, cudaSuccess);

        result = cudaMallocAsync(&m_pYDevice, problemSizeInBytes, m_saxpyStream);
        ASSERT_EQ(result, cudaSuccess);

        result = cudaHostAlloc(&m_pYHost, problemSizeInBytes, cudaHostAllocDefault);
        ASSERT_EQ(result, cudaSuccess);

        std::generate(m_pYHost, m_pYHost + problemSize, GetRandFloat);

        result = cudaMemcpyAsync(m_pYDevice, m_pYHost, problemSizeInBytes, cudaMemcpyHostToDevice, m_saxpyStream);
        ASSERT_EQ(result, cudaSuccess);

        result = cudaMallocAsync(&m_pZDevice, problemSizeInBytes, m_saxpyStream);
        ASSERT_EQ(result, cudaSuccess);

        saxpy::DeviceExecute(A, m_pXDevice, m_pYDevice, m_pZDevice, problemSize, m_saxpyStream);

        result = cudaHostAlloc(&m_pZHost, problemSizeInBytes, cudaHostAllocDefault);
        ASSERT_EQ(result, cudaSuccess);

        result = cudaMemcpyAsync(m_pZHost, m_pZDevice, problemSizeInBytes, cudaMemcpyDeviceToHost, m_saxpyStream);
        ASSERT_EQ(result, cudaSuccess);

        result = cudaFreeAsync(m_pXDevice, m_saxpyStream);
        EXPECT_EQ(result, cudaSuccess);
        result = cudaFreeAsync(m_pYDevice, m_saxpyStream);
        EXPECT_EQ(result, cudaSuccess);
        result = cudaFreeAsync(m_pZDevice, m_saxpyStream);
        EXPECT_EQ(result, cudaSuccess);

        VerifySolution(problemSize);

        FreeHostAllocs();
    }
}


TEST_F(SaxpyTest, UsingMappedHostMemory)
{
    cudaError_t result           = cudaSuccess;
    int         device           = -1;
    int         canMapHostMemory = -1;

    cudaDeviceProp prop   = {};
    prop.canMapHostMemory = 1;
    prop.integrated       = 1;

    result = cudaChooseDevice(&device, &prop);
    ASSERT_EQ(result, cudaSuccess);

    result = cudaDeviceGetAttribute(&canMapHostMemory, cudaDevAttrCanMapHostMemory, device);
    ASSERT_EQ(result, cudaSuccess);

    if (canMapHostMemory)
    {
        result = cudaSetDevice(device);
        ASSERT_EQ(result, cudaSuccess);
    }
    else
    {
        GTEST_SKIP() << "No CUDA-enabled device that can map host memory was detected";
    }

    for (const size_t problemSize : ProblemSizes)
    {
        const size_t problemSizeInBytes = problemSize * sizeof(float);

        result = cudaHostAlloc(&m_pXHost, problemSizeInBytes, cudaHostAllocMapped | cudaHostAllocWriteCombined);
        ASSERT_EQ(result, cudaSuccess);

        result = cudaHostAlloc(&m_pYHost, problemSizeInBytes, cudaHostAllocMapped | cudaHostAllocWriteCombined);
        ASSERT_EQ(result, cudaSuccess);

        result = cudaHostAlloc(&m_pZHost, problemSizeInBytes, cudaHostAllocMapped);
        ASSERT_EQ(result, cudaSuccess);

        std::generate(m_pXHost, m_pXHost + problemSize, GetRandFloat);
        std::generate(m_pYHost, m_pYHost + problemSize, GetRandFloat);

        result = cudaHostGetDevicePointer(&m_pXDevice, m_pXHost, 0);
        ASSERT_EQ(result, cudaSuccess);

        result = cudaHostGetDevicePointer(&m_pYDevice, m_pYHost, 0);
        ASSERT_EQ(result, cudaSuccess);

        result = cudaHostGetDevicePointer(&m_pZDevice, m_pZHost, 0);
        ASSERT_EQ(result, cudaSuccess);

        saxpy::DeviceExecute(A, m_pXDevice, m_pYDevice, m_pZDevice, problemSize, m_saxpyStream);

        VerifySolution(problemSize);
        
        FreeHostAllocs();
    }
}
