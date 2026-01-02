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
#include "cu_tex2d.h"

#include "ray_tracing.h"

constexpr u16 width = 1280;
constexpr u16 height = 720;


constexpr float3 LAMBERT = {};
constexpr float3 METAL = {};
constexpr float3 DIELECTRIC = {};

struct material
{
    float3 albedo;
    float ior;

};

struct world
{
    const sphere_t*      hittables;
    const material*      materials;
    u32                  count;
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

inline __device__ float3 Shade( const ray& r )
{
    float t = 0.5f * r.dir.y + 0.5f;
    float3 col = lerp( float3{ 1.0f, 1.0f, 1.0f }, float3{ 0.5f, 0.7f, 1.0f }, t );
    return col;
};

//__device__ Scatter()
//{
//
//}

__global__ void RenderKernel( cudaSurfaceObject_t fbSurf, const world w, u32 frameIdx ) 
{
    u32 xi = threadIdx.x + blockIdx.x * blockDim.x;
    u32 yi = threadIdx.y + blockIdx.y * blockDim.y;

    if( ( xi >= globs.width ) || ( yi >= globs.height ) ) return;

    const camera mainCam = globs.mainCam;

    u32 pixelIdx = yi * globs.width + xi;
    u32 baseSeed = pixelIdx ^ ( frameIdx * PRIME1 );

    float3 accColor = {};
    for( u32 si = 0; si < globs.samplesPerPixel; ++si )
    {
        float2 jitter = RandUnitFloat2( si, baseSeed ) - float2{ 0.5f, 0.5f };
        float4 clipSpacePos = GetClipSpaceVector( float2( xi, yi ) + jitter, float2( globs.width, globs.height ) );
        float4 worldHmgPos = mul( mainCam.invVP, clipSpacePos );
        float3 worldPos = xyz( worldHmgPos ) / worldHmgPos.w;
        float3 rayDir = normalize( worldPos - mainCam.pos );

        ray currentRay = { .origin = mainCam.pos, .dir = rayDir };
        constexpr float reflectionFactor = 0.3f;

        float3 attenuationFactor = { 1.0f, 1.0f, 1.0f };
        u32 pathSeed = baseSeed ^ ( si * PRIME3 );
        for( u32 bi = 0; bi < globs.maxBounces; ++bi )
        {
            // NOTE: Russian Roulette
            if( bi > 3 )
            {
                float p = max( attenuationFactor.x, max( attenuationFactor.y, attenuationFactor.z ) );
                float shootChance = RandUnitFloat( bi, pathSeed );
                if( shootChance > p ) break;
                attenuationFactor /= p; // NOTE: Energy conservation boost
            }

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
                float3 unitRand = RandUnitFloat3( bi, pathSeed ) * 2.0f - 1.0f; // NOTE: need float3 in [-1; 1] 
                float3 lamberitanReflectionDistr = rec.normal + unitRand;

                currentRay = { .origin = rec.point, .dir = normalize( lamberitanReflectionDistr ) };
                attenuationFactor *= reflectionFactor;
            }
            else
            {
                accColor += Shade( currentRay ) * attenuationFactor;
                break;
            }
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
        //CUDA_CHECK( cudaDeviceSetLimit( cudaLimitStackSize, 16000 ) );
        //CUDA_CHECK( cudaDeviceSetCacheConfig( cudaFuncCachePreferL1 ) );
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
