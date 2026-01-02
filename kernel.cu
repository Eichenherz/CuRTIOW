#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <memory>

#include <thrust/device_vector.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include "core_types.h"
#include "cu_errors.h"
#include "cu_allocator.h"
#include "cu_math.h"
#include "cu_rand.h"

#include "ray_tracing.h"

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

struct world
{
    const sphere_t* hittables;
    u32 count;
};

struct globals
{
    camera               mainCam;
    alignas( 8 ) u32     width;
    alignas( 8 ) u32     height;
    alignas( 8 ) u32     samplesPerPixel;
    alignas( 8 ) u32     maxBounces;
};

// NOTE: these must be properly alligned
__constant__ globals globs;

__device__ float3 Shade( const ray& r, float attenuation )
{
    float t = 0.5f * r.dir.y + 0.5f;
    float3 col = lerp( float3{ 1.0f, 1.0f, 1.0f }, float3{ 0.5f, 0.7f, 1.0f }, t );
    return col * attenuation;
};

__global__ void RenderKernel( cudaSurfaceObject_t fbSurf, const world w, u32 frameIdx ) 
{
    u32 xi = threadIdx.x + blockIdx.x * blockDim.x;
    u32 yi = threadIdx.y + blockIdx.y * blockDim.y;

    if( ( xi >= globs.width ) || ( yi >= globs.height ) ) return;

    const camera mainCam = globs.mainCam;

    u32 pixelIdx = yi * globs.width + xi;
    u32 seed = pixelIdx ^ ( frameIdx * PRIME1 );

    float3 accColor = {};
    for( u32 si = 0; si < globs.samplesPerPixel; ++si )
    {
        float2 jitter = RandVec2( si, seed ) - float2{ 0.5f, 0.5f };
        float4 clipSpacePos = GetClipSpaceVector( float2( xi, yi ) + jitter, float2( globs.width, globs.height ) );
        float4 worldHmgPos = mul( mainCam.invVP, clipSpacePos );

        float3 rayDir = normalize( xyz( worldHmgPos ) - mainCam.pos );

        ray currentRay = { .origin = mainCam.pos, .dir = rayDir };
        float currentAttenuation = 1.0f;
        constexpr float reflectionFactor = 0.3f;

        u32 bi = 0;
        for( ; bi < globs.maxBounces; ++bi )
        {
            hit_record rec = INVALID_HIT;
            float closestSoFar = CUDART_INF_F;
            for( u32 hi = 0; hi < w.count; ++hi )
            {
                sphere_t hittable = w.hittables[ hi ];
                hit_record hit = HitRayVsSphere( currentRay, hittable, RAY_EPSILON, closestSoFar );
                if( NO_HIT != hit.t )
                {
                    closestSoFar = hit.t;
                    rec = hit;
                }
            }

            if( IsValidHit( rec ) )
            {
                u32 diffSeed = si ^ ( seed * PRIME2 );
                float3 unitRand = normalize( RandVec3( bi, diffSeed ) );
                float3 lamberitanReflectionDistr = rec.normal + unitRand;

                currentRay = { .origin = rec.point, .dir = normalize( lamberitanReflectionDistr ) };
                currentAttenuation *= reflectionFactor;
            }
            else
            {
                accColor += Shade( currentRay, currentAttenuation );
                break;
            }
        }   
        if( bi >= globs.maxBounces )
        {
            accColor = {};
        }
    }

    float3 linearCol = accColor / ( float ) globs.samplesPerPixel;
    float3 srgbCol = LinearToSrgb( linearCol );
    // NOTE: to avoid artifacts
    float3 pixelCol = saturatef3( srgbCol ) * 255.99f;
    u8x4 pixel = { ( u8 ) pixelCol.x, ( u8 ) pixelCol.y, ( u8 ) pixelCol.z, 255 };
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

    globals g = {
        .mainCam = MakeCamRH( width, height ),
        .width = width,
        .height = height,
        .samplesPerPixel = 50,
        .maxBounces = 5
    };

    CUDA_CHECK( cudaMemcpyToSymbol( globs, &g, sizeof( g ) ) );

    std::vector<sphere_t> spheres;
    spheres.push_back( { .center = { 0.0f, 0.0f, -1.0f }, .radius = 0.5f } );
    spheres.push_back( { .center = { 0.0f, -100.5f, -1.0f }, .radius = 100.0f } );

    thrust::device_vector<sphere_t> dSpheres = { std::cbegin( spheres ), std::cend( spheres ) };

    world w = { .hittables = thrust::raw_pointer_cast( std::data( dSpheres ) ), .count = ( u32 ) std::size( dSpheres ) };

    u32 frameIdx = 0;

    u32 tx = 8;
    u32 ty = 8;
    
    dim3 blocks( ( width + tx - 1 ) / tx, ( height + ty - 1 ) / ty );
    RenderKernel<<<blocks, dim3( tx, ty )>>>( pGradTex->surf, w, frameIdx );
    CUDA_CHECK( cudaGetLastError() );
    CUDA_CHECK( cudaDeviceSynchronize() );

    auto hostTex = CudaCopyImageToHostSync( *pGradTex );
    stbi_write_png( "sky.png", width, height, 4, std::data( hostTex ), width * sizeof( pixel_t ) );

    return 0;
}
