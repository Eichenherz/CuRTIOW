#ifndef __CU_TEX2D_H__
#define __CU_TEX2D_H__

#include <cuda_runtime.h>
#include "core_types.h"
#include "cu_errors.h"
#include "cu_allocator.h"

#include <memory>

enum cuda_texture_usage_flags : u64
{
    READONLY = 1,
    WRITE = 2,
};

// TODO: there's a color attach flag too !
struct cuda_tex2d
{
    cudaSurfaceObject_t surf;
    cudaTextureObject_t tex;
    cudaArray_t array;
    cudaChannelFormatDesc channelFormatDesc;
    u16 width;
    u16 height;
    u16 elemPitch;
};

template<typename pixel_t>
cuda_tex2d CudaCreateTex2D( u16 width, u16 height, u64 usageFlags )
{
    cudaChannelFormatDesc channelFormatDesc = cudaCreateChannelDesc<pixel_t>();
    constexpr u32 arrayAccessFlags = cudaArraySurfaceLoadStore | cudaArrayTextureGather;

    cudaArray_t array;
    CUDA_CHECK( cudaMallocArray( &array, &channelFormatDesc, width, height, arrayAccessFlags ) );

    cudaResourceDesc rscDesc = {};
    rscDesc.resType = cudaResourceTypeArray;
    rscDesc.res.array.array = array;

    cudaSurfaceObject_t surf = 0;
    if( cuda_texture_usage_flags::WRITE & usageFlags )
    {
        CUDA_CHECK( cudaCreateSurfaceObject( &surf, &rscDesc ) );
    }

    cudaTextureObject_t tex = 0;
    // TODO: add more stuff like addresing ?
    if( cuda_texture_usage_flags::READONLY & usageFlags )
    {
        cudaTextureDesc texDesc = {
            .filterMode = cudaFilterModePoint, .readMode = cudaReadModeElementType, .normalizedCoords = 0 };
        CUDA_CHECK( cudaCreateTextureObject( &tex, &rscDesc, &texDesc, nullptr ) );
    }

    u16 elemPitch = ( channelFormatDesc.x + channelFormatDesc.y + channelFormatDesc.z + channelFormatDesc.w ) / 8;
    assert( sizeof( pixel_t ) == elemPitch );

    return {
        .surf = surf,
        .tex = tex,
        .array = array,
        .channelFormatDesc = channelFormatDesc,
        .width = width,
        .height = height,
        .elemPitch = elemPitch
    };
}

inline void CudaDestroyTex2D( cuda_tex2d* tex2d )
{
    if( !tex2d ) return;
    if( tex2d->surf ) CUDA_CHECK( cudaDestroySurfaceObject( tex2d->surf ) );
    if( tex2d->tex ) CUDA_CHECK( cudaDestroyTextureObject( tex2d->tex ) );
    if( tex2d->array ) CUDA_CHECK( cudaFreeArray( tex2d->array ) );
}

template<typename pixel_t>
inline auto MakeSharedCudaTex2D( u16 width, u16 height, u64 usageFlags ) 
{
    auto* raw = new cuda_tex2d{};
    *raw = CudaCreateTex2D<pixel_t>( width, height, usageFlags );
    return std::shared_ptr<cuda_tex2d>( raw, CudaDestroyTex2D );
}


inline auto CudaCopyImageToHostSync( const cuda_tex2d& tex2d )
{
    const u64 rowSizeInBytes = tex2d.width * tex2d.elemPitch;

    pagelocked_host_vector<u8> out;
    out.resize( rowSizeInBytes * tex2d.height );

    CUDA_CHECK( cudaMemcpy2DFromArray(
        std::data( out ), rowSizeInBytes, tex2d.array, 0, 0, rowSizeInBytes, tex2d.height, cudaMemcpyDeviceToHost ) );

    return out;
}

#endif // !__CU_TEX2D_H__
