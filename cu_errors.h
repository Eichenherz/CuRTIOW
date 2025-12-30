#ifndef __CU_ERRORS_H__
#define __CU_ERRORS_H__

#include "core_types.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <assert.h>

#define CUDA_CHECK( val ) CudaCheckErr( ( val ), __FILE__, __LINE__ )

inline void CudaCheckErr( cudaError_t err, const char const* file, const int line ) 
{
    if( err )
    {
        fprintf( stderr, "CUDA ERROR %s:%d: %s (%d)\n", file, line, cudaGetErrorString( err ), err );
        assert( false );
        cudaDeviceReset();
        std::exit( 99 );
    }
}

#endif // !__CU_ERRORS_H__
