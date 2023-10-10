#include "pch.h"

#include "cuda_runtime.h"

#include "device.h"


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
        cudaError_t result = cudaSuccess;

        result = cudaStreamCreateWithFlags(&m_saxpyStream, cudaStreamNonBlocking);
        ASSERT_EQ(result, cudaSuccess);

        result = cudaEventCreate(&m_saxpyComplete);
        ASSERT_EQ(result, cudaSuccess);
    }

    // Per-test tear-down.
    void TearDown() override
    {
        cudaError_t result = cudaSuccess;

        result = cudaEventDestroy(m_saxpyComplete);
        ASSERT_EQ(result, cudaSuccess);

        result = cudaStreamDestroy(m_saxpyStream);
        ASSERT_EQ(result, cudaSuccess);
    }

	// Device that is used for all tests.
	static int s_selectedDevice;

    // Resources that are created and destroyed for each test.
    cudaStream_t m_saxpyStream;
    cudaEvent_t  m_saxpyComplete;
};


TEST(TestCaseName, TestName) {
  EXPECT_EQ(1, 1);
  EXPECT_TRUE(true);
}