#ifndef UTILITIES_STREAMTIMER_H
#define UTILITIES_STREAMTIMER_H

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

    [[nodiscard]] cudaError_t GetStatus() const noexcept { return m_status; }

    [[nodiscard]] cudaError_t Start() noexcept;
    [[nodiscard]] cudaError_t Stop()  noexcept;

    [[nodiscard]] cudaError_t GetElapsedTimeInMs(float& elapsedTime) noexcept;
};


#endif // UTILITIES_STREAMTIMER_H