#pragma once

#define Tid threadIdx.x

#if defined(__CUDACC__)
	#if ARCH >= 300
		#if ARCH == 300
			#pragma message("\n\nCompute Capability: 3\n")
			const int SMem_Per_SM =	49152;
		#elif ARCH == 350
			#pragma message("\n\nCompute Capability: 3.5\n")
			const int SMem_Per_SM =	49152;
		#elif ARCH == 370
			#pragma message("\n\nCompute Capability: 3.7\n")
			const int SMem_Per_SM =	114688;
		#elif ARCH == 500
			#pragma message("\n\nCompute Capability: 5.0\n")
			const int SMem_Per_SM =	65536;
		#elif ARCH == 520
			#pragma message("\n\nCompute Capability: 5.2\n")
			const int SMem_Per_SM = 98304;
		#endif
		const int      Thread_Per_SM  =  2048;
		const int    SMem_Per_Thread  =  SMem_Per_SM / Thread_Per_SM;
		const int IntSMem_Per_Thread  =  SMem_Per_Thread / 4;
		const int      SMem_Per_Warp  =  SMem_Per_Thread * 32;
		const int   IntSMem_Per_Warp  =  IntSMem_Per_Thread * 32;
		const int        MaxBlockDim  =  1024;
		const int         MemoryBank  =  32;
	#else
		#error message("\n\nCompute Capability NOT supported (<3)\n")
	#endif
#endif

#define                  MIN_V(a, b)	((a) > (b) ? (b) : (a))
#define     MAX_CONCURR_BL(BlockDim)	( MAX_CONCURR_TH / (BlockDim) )
#define     SMem_Per_Block(BlockDim)	( MIN_V( SMem_Per_Thread * (BlockDim) , 49152) )
#define  IntSMem_Per_Block(BlockDim)	( MIN_V( IntSMem_Per_Thread * (BlockDim) , 49152 / 4) )
