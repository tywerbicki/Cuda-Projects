#pragma once

#include "cuda_runtime.h"


class StreamTimer final
{
private:
    cudaError_t  m_status;
    cudaStream_t m_stream;
    cudaEvent_t  m_startEvent;
    cudaEvent_t  m_stopEvent;

public:
    explicit StreamTimer(cudaStream_t stream = static_cast<cudaStream_t>(0));
    ~StreamTimer();

    cudaError_t GetStatus() const noexcept { return m_status; }

    cudaError_t Start() noexcept;
    cudaError_t Stop()  noexcept;

    cudaError_t GetElapsedTime(float& elapsedTime) noexcept;
};