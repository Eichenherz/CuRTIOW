#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "core_types.h"
#include <stdio.h>


#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <vector>

#include "cu_errors.h"
#include "cu_allocator.h"


constexpr u16 width = 1280;
constexpr u16 height = 720;


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
    
    //inline ~cuda_tex2d()
    //{
    //    if( surf ) CUDA_CHECK( cudaDestroySurfaceObject( surf ) );
    //    if( tex ) CUDA_CHECK( cudaDestroyTextureObject( tex ) );
    //    if( array ) CUDA_CHECK( cudaFreeArray( array ) );
    //}
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

inline auto CudaCopyImageToHostSync( const cuda_tex2d& tex2d )
{
    const u64 rowSizeInBytes = tex2d.width * tex2d.elemPitch;

    pagelocked_host_vector<u8> out;
    out.resize( rowSizeInBytes * tex2d.height );

    CUDA_CHECK( cudaMemcpy2DFromArray(
        std::data( out ), rowSizeInBytes, tex2d.array, 0, 0, rowSizeInBytes, tex2d.height, cudaMemcpyDeviceToHost ) );

    return out;
}

__global__ void GradientSurfaceKernel( cudaSurfaceObject_t fbSurf, u16 width, u16 height ) 
{
    u32 xi = threadIdx.x + blockIdx.x * blockDim.x;
    u32 yi = threadIdx.y + blockIdx.y * blockDim.y;

    if( ( xi >= width ) || ( yi >= height ) ) return;

    auto pixel = make_uchar4( 
        (u8)( ( xi * 255 ) / width ), // Red increases left → right
        (u8)( ( yi * 255 ) / height ), 0, 255 ); // Green increases top → bottom
    
    // NOTE: cuda 12.9 with err with interleaved source in ptx for this line ! on Pascal
    surf2Dwrite( pixel, fbSurf, xi * sizeof( pixel ), yi );
}

int main()
{
    using pixel_t = uchar4;

    constexpr u64 usageFlags = cuda_texture_usage_flags::READONLY | cuda_texture_usage_flags::WRITE;
    cuda_tex2d gradTex = CudaCreateTex2D<pixel_t>( width, height, usageFlags );

    u32 tx = 8;
    u32 ty = 8;
    
    dim3 blocks( ( width + tx - 1 ) / tx, ( height + ty - 1 ) / ty );
    GradientSurfaceKernel<<<blocks, dim3( tx, ty )>>>( gradTex.surf, width, height );
    CUDA_CHECK( cudaGetLastError() );
    CUDA_CHECK( cudaDeviceSynchronize() );

    auto hostTex = CudaCopyImageToHostSync( gradTex );
    stbi_write_png( "gradient.png", width, height, 4, std::data( hostTex ), width * sizeof( pixel_t ) );

    return 0;
}
