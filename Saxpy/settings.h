#pragma once


namespace saxpy
{
	enum class MemoryStrategy
	{
		dynamic,
		forceMapped,
		forceAsync
	};

	namespace settings
	{
#ifdef _DEBUG
		inline extern const MemoryStrategy memoryStrategy = MemoryStrategy::forceMapped;
#else // _DEBUG
		inline extern const MemoryStrategy memoryStrategy = MemoryStrategy::dynamic;
#endif // _DEBUG
	}
}