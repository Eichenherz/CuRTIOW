#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <memory>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include "core_types.h"
#include "cu_errors.h"
#include "cu_allocator.h"
#include "cu_math.h"

constexpr u16 width = 1280;
constexpr u16 height = 720;

struct ray
{
    alignas( 16 ) float3 origin;
    alignas( 16 ) float3 dir;

    __device__ float3 At( float t ) const
    {
        return origin + t * dir;
    }
};

// NOTE: the vecs are initialized to RH ( look at you RH palm )
struct camera
{
    float4x4 view;
    float4x4 proj;
    float4x4 invVP;

    float3 pos;
    float fovYRad;

    float3 fwd;
    float aspect;

    float3 up;
    float zNear;
};

inline camera MakeCamRH( u32 width, u32 height, float fovYRad = 1.570f, float zNear = 0.1f )
{
    camera cam = {
        .pos = {},
        .fovYRad = fovYRad,
        .fwd = { 0.0f, 0.0f, -1.0f },
        .aspect = ( float ) width / ( float ) height,
        .up = { 0.0f, 1.0f, 0.0f },
        .zNear = zNear
    };

    float3 lookAt = cam.pos + cam.fwd;
    cam.view = LookAtRH( cam.pos, lookAt, cam.up );
    cam.proj = PerspectiveInfFarRH( fovYRad, cam.aspect, zNear );

    float4x4 vp = mul( cam.proj, cam.view );

    float vpDet = det( vp );
    assert( 0.0f != vpDet );

    cam.invVP = ( 1.0f / vpDet ) * adj( vp );

    return cam;
}

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


__constant__ camera mainCam;


__device__ float3 GetRayCol( const ray& r ) 
{
    float t = 0.5f * r.dir.y + 0.5f;
    return lerp( float3{ 1.0f, 1.0f, 1.0f }, float3{ 0.5f, 0.7f, 1.0f }, t );
};

__global__ void RenderKernel( cudaSurfaceObject_t fbSurf, u32 width, u32 height ) 
{
    u32 xi = threadIdx.x + blockIdx.x * blockDim.x;
    u32 yi = threadIdx.y + blockIdx.y * blockDim.y;

    if( ( xi >= width ) || ( yi >= height ) ) return;

    float4 clipSpacePos = GetClipSpaceVector( { xi, yi }, { width, height } );
    float4 worldHmgPos = mul( mainCam.invVP, clipSpacePos );

    float3 rayDir = normalize( float3{ worldHmgPos.x, worldHmgPos.y, worldHmgPos.z } - mainCam.pos );

    ray r = { .origin = mainCam.pos, .dir = rayDir };

    float3 rayCol = GetRayCol( r );
    // NOTE: to avoid artifacts
    rayCol = saturatef3( rayCol ) * 255.99f;

    auto pixel = make_uchar4( ( u8 ) rayCol.x, ( u8 ) rayCol.y, ( u8 ) rayCol.z, 255 );
    
    // NOTE: cuda 12.9 with err with interleaved source in ptx for this line ! on Pascal
    surf2Dwrite( pixel, fbSurf, xi * sizeof( pixel ), yi );
}

struct cuda_context
{
    inline cuda_context()
    {
        // NOTE: Choose which GPU to run on, change this on a multi-GPU system.
        CUDA_CHECK( cudaSetDevice( 0 ) );
    }

    inline ~cuda_context()
    {
        // NOTE: cudaDeviceReset must be called before exiting in order for profiling and
        // tracing tools such as Nsight and Visual Profiler to show complete traces.
        CUDA_CHECK( cudaDeviceReset() );
    }
};

int main()
{
    using pixel_t = uchar4;

    cuda_context cudaCtx = {};

    constexpr u64 usageFlags = cuda_texture_usage_flags::READONLY | cuda_texture_usage_flags::WRITE;
    auto pGradTex = MakeSharedCudaTex2D<pixel_t>( width, height, usageFlags );

    camera cam = MakeCamRH( width, height );

    CUDA_CHECK( cudaMemcpyToSymbol( mainCam, &cam, sizeof( cam ), 0, cudaMemcpyHostToDevice ) );


    u32 tx = 8;
    u32 ty = 8;
    
    dim3 blocks( ( width + tx - 1 ) / tx, ( height + ty - 1 ) / ty );
    RenderKernel<<<blocks, dim3( tx, ty )>>>( pGradTex->surf, width, height );
    CUDA_CHECK( cudaGetLastError() );
    CUDA_CHECK( cudaDeviceSynchronize() );

    auto hostTex = CudaCopyImageToHostSync( *pGradTex );
    stbi_write_png( "sky.png", width, height, 4, std::data( hostTex ), width * sizeof( pixel_t ) );

    return 0;
}
