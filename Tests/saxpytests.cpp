#include "pch.h"

#include <algorithm>
#include <array>
#include <vector>

#include "cuda_runtime.h"

#include "device.h"
#include "saxpy.h"


class SaxpyTest : public ::testing::Test
{
protected:
    // Per-test-suite set-up.
    static void SetUpTestSuite()
    {
        cudaError_t result      = cudaSuccess;
        int         deviceCount = -1;

        result = cudaGetDeviceCount(&deviceCount);
        ASSERT_EQ(result, cudaSuccess);
        ASSERT_GT(deviceCount, 0);

        result = device::GetMostMultiProcessors(deviceCount, s_selectedDevice);
        ASSERT_EQ(result, cudaSuccess);

        result = cudaSetDevice(s_selectedDevice);
        ASSERT_EQ(result, cudaSuccess);
    }

    // Per-test-suite tear-down.
    static void TearDownTestSuite()
    {
        cudaError_t result = cudaSuccess;

        result = cudaDeviceReset();
        ASSERT_EQ(result, cudaSuccess);
    }

    // Per-test set-up.
    void SetUp() override
    {
        m_result = cudaSuccess;

        m_result = cudaStreamCreateWithFlags(&m_saxpyStream, cudaStreamNonBlocking);
        ASSERT_EQ(m_result, cudaSuccess);

        m_result = cudaEventCreate(&m_saxpyComplete);
        ASSERT_EQ(m_result, cudaSuccess);
    }

    // Per-test tear-down.
    void TearDown() override
    {
        m_result = cudaEventDestroy(m_saxpyComplete);
        ASSERT_EQ(m_result, cudaSuccess);

        m_result = cudaStreamDestroy(m_saxpyStream);
        ASSERT_EQ(m_result, cudaSuccess);

        m_saxpyStream   = nullptr;
        m_saxpyComplete = nullptr;
    }

    // Resources that are used for all tests.
    static int s_selectedDevice;

    // Resources that are created and destroyed for each test.
    cudaStream_t m_saxpyStream;
    cudaEvent_t  m_saxpyComplete;

    cudaError_t m_result;
};


TEST_F(SaxpyTest, AsynchronousTransfers)
{
    const std::array<size_t, 7> sizes = { 0, 1, 3, 33, 65, 1025, 1048577 };

    for (const size_t size : sizes)
    {
        const float  a           = 3.0f;
        const size_t sizeInBytes = size * sizeof(float);
        float*       pXHost      = nullptr;
        float*       pYHost      = nullptr;
        float*       pXDevice    = nullptr;
        float*       pYDevice    = nullptr;

        std::vector<float> solution(size);

        m_result = cudaMallocAsync(&pXDevice, sizeInBytes, m_saxpyStream);
        ASSERT_EQ(m_result, cudaSuccess);

        m_result = cudaHostAlloc(&pXHost, sizeInBytes, cudaHostAllocDefault);
        ASSERT_EQ(m_result, cudaSuccess);

        std::fill(pXHost, pXHost + size, 4.0f);

        m_result = cudaMemcpyAsync(pXDevice, pXHost, sizeInBytes, cudaMemcpyHostToDevice, m_saxpyStream);
        ASSERT_EQ(m_result, cudaSuccess);

        m_result = cudaMallocAsync(&pYDevice, sizeInBytes, m_saxpyStream);
        ASSERT_EQ(m_result, cudaSuccess);

        m_result = cudaHostAlloc(&pYHost, sizeInBytes, cudaHostAllocDefault);
        ASSERT_EQ(m_result, cudaSuccess);

        std::fill(pYHost, pYHost + size, 1.5f);

        std::transform(pXHost, pXHost + size,
                       pYHost,
                       solution.begin(),
                       [=](float x, float y) { return (a * x) + y; });

        m_result = cudaMemcpyAsync(pYDevice, pYHost, sizeInBytes, cudaMemcpyHostToDevice, m_saxpyStream);
        ASSERT_EQ(m_result, cudaSuccess);

        saxpy::DeviceExecute(m_saxpyStream, size, a, pXDevice, pYDevice);

        m_result = cudaMemcpyAsync(pYHost, pYDevice, sizeInBytes, cudaMemcpyDeviceToHost, m_saxpyStream);
        ASSERT_EQ(m_result, cudaSuccess);

        m_result = cudaFreeAsync(pXDevice, m_saxpyStream);
        ASSERT_EQ(m_result, cudaSuccess);
        m_result = cudaFreeAsync(pYDevice, m_saxpyStream);
        ASSERT_EQ(m_result, cudaSuccess);

        m_result = cudaEventRecord(m_saxpyComplete, m_saxpyStream);
        ASSERT_EQ(m_result, cudaSuccess);

        // Here is where we synchronize the host with the saxpy device operations.
        m_result = cudaEventSynchronize(m_saxpyComplete);
        ASSERT_EQ(m_result, cudaSuccess);

        EXPECT_TRUE(std::equal(solution.cbegin(), solution.cend(), pYHost));
    }
}