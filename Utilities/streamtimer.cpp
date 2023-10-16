#include "debug.h"
#include "streamtimer.h"

#include "cuda_runtime.h"


StreamTimer::StreamTimer(cudaStream_t stream)
	: m_stream(stream)
{
	m_status = cudaEventCreateWithFlags(&m_startEvent, cudaEventDefault);
	DBG_PRINT_ON_CUDA_ERROR(m_status);

	m_status = cudaEventCreateWithFlags(&m_stopEvent, cudaEventDefault);
	DBG_PRINT_ON_CUDA_ERROR(m_status);
}


StreamTimer::~StreamTimer()
{
	m_status = cudaEventDestroy(m_startEvent);
	DBG_PRINT_ON_CUDA_ERROR(m_status);

	m_status = cudaEventDestroy(m_stopEvent);
	DBG_PRINT_ON_CUDA_ERROR(m_status);
}


cudaError_t StreamTimer::Start() noexcept
{
	m_status = cudaEventRecord(m_startEvent, m_stream);
	DBG_PRINT_ON_CUDA_ERROR(m_status);

	return m_status;
}


cudaError_t StreamTimer::Stop() noexcept
{
	m_status = cudaEventRecord(m_stopEvent, m_stream);
	DBG_PRINT_ON_CUDA_ERROR(m_status);

	return m_status;
}


cudaError_t StreamTimer::GetElapsedTime(float& elapsedTime) noexcept
{
	m_status = cudaEventSynchronize(m_stopEvent);
	DBG_PRINT_RETURN_ON_CUDA_ERROR(m_status);

	m_status = cudaEventElapsedTime(&elapsedTime, m_startEvent, m_stopEvent);
	DBG_PRINT_ON_CUDA_ERROR(m_status);

	return m_status;
}