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
#include "warps.h"
#include "cpp_helpers.h"

#include "sdl_platform.h"
#include "dx11.h"
#include "cu_dx11_interop.h"

#include "ray_tracing.h"
#include "materials.h"

constexpr u16 width = 1280;
constexpr u16 height = 720;

constexpr char WINDOW_TITLE[] = "CuRTIOW";


struct worldref
{
    const sphere_t*      hittables;
    const material*      materials;
    u32                  count;
};

struct world
{
    thrust::device_vector<sphere_t> spheres;
    thrust::device_vector<material> materials;

    world()
    {
        std::vector<sphere_t> spheres;
        spheres.push_back( { .center = { 0.0f, 0.0f, -1.2f }, .radius = 0.5f } );
        spheres.push_back( { .center = { 0.0f, -100.5f, -1.0f }, .radius = 100.0f } );
        spheres.push_back( { .center = { 1.0f, 0.0f, -1.0f }, .radius = 0.5f } );
        spheres.push_back( { .center = {-1.0f, 0.0f, -1.0f }, .radius = 0.5f } );
        spheres.push_back( { .center = {-1.0f, 0.0f, -1.0f }, .radius = 0.4f } );

        std::vector<material> materials;
        materials.push_back( { .albedo = { 0.8f, 0.8f, 0.0f }, .type = material_type::LAMBERT } );
        materials.push_back( { .albedo = { 0.1f, 0.2f, 0.5f }, .type = material_type::LAMBERT } );
        materials.push_back( { .albedo = { 0.9f, 0.7f, 0.1f }, .fuzz = 0.666f, .type = material_type::METAL } );
        materials.push_back( { .ior = 1.5f, .type = material_type::DIELECTRIC } );
        materials.push_back( { .ior = 1.0f / 1.5f, .type = material_type::DIELECTRIC } );

        assert( std::size( spheres ) == std::size( materials ) );

        this->spheres = { std::cbegin( spheres ), std::cend( spheres ) };
        this->materials = { std::cbegin( materials ), std::cend( materials ) };
    }

    inline worldref GetRef() const
    {
        assert( std::size( spheres ) == std::size( materials ) );
        return {
            .hittables = thrust::raw_pointer_cast( std::data( spheres ) ),
            .materials = thrust::raw_pointer_cast( std::data( materials ) ),
            .count = ( u32 ) std::size( spheres )
        };
    }
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

inline __device__ float3 Shade( const ray_t& r )
{
    float t = 0.5f * r.dir.y + 0.5f;
    float3 col = lerp( float3{ 1.0f, 1.0f, 1.0f }, float3{ 0.5f, 0.7f, 1.0f }, t );
    return col;
};

__global__ void RenderKernel( cudaSurfaceObject_t fbSurf, const worldref w, u32 frameIdx ) 
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

        ray_t currentRay = { .origin = mainCam.pos, .dir = rayDir };

        float3 energyFactor = make_float3( 1.0f );
        u32 pathSeed = baseSeed ^ ( si * PRIME3 );
        for( u32 bi = 0; bi < globs.maxBounces; ++bi )
        {
            // NOTE: Russian Roulette
            if( bi > 3 )
            {
                float survivalChance = max( energyFactor.x, max( energyFactor.y, energyFactor.z ) );
                float shootChance = RandUnitFloat( bi, pathSeed );
                if( shootChance > survivalChance ) break;
                energyFactor /= survivalChance; // NOTE: Energy conservation boost
            }

            u32 primitiveIdx = 0;
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
                    primitiveIdx = hi;
                }
            }

            if( IsValidHit( rec ) )
            {
                const material& mat = w.materials[ primitiveIdx ];

                auto[ scatterDir, attenuation ] = Scatter( mat, currentRay.dir, rec.worldNormal, bi, pathSeed );

                currentRay = { .origin = rec.point, .dir = scatterDir };
                energyFactor *= attenuation;
            }
            else
            {
                accColor += Shade( currentRay ) * energyFactor;
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

struct cuda_context
{
    u64              luid;
    cudaStream_t     mainStream;
    u32              deviceIdx;

    cuda_context()
    {
        i32  deviceCount;
        CUDA_CHECK( cudaGetDeviceCount( &deviceCount ) );
        CUDA_CHECK( ( 0 == deviceCount ) ? cudaError( -1 ) : cudaError::cudaSuccess );

        cudaDeviceProp deviceProp;
        // TODO: check device props ?
        for( u32 di = 0; di < deviceCount; ++di )
        {
            CUDA_CHECK( cudaGetDeviceProperties( &deviceProp, di ) );
            CUDA_CHECK( ( WARP_SIZE != deviceProp.warpSize ) ? cudaError( -1 ) : cudaError::cudaSuccess );

            std::memcpy( &luid, deviceProp.luid, sizeof( deviceProp.luid ) );
            deviceIdx = di;

            break;
        }

        printf( "CUDA GPU %d: %s\n", deviceIdx, deviceProp.name );

        CUDA_CHECK( cudaSetDevice( deviceIdx ) );

        CUDA_CHECK( cudaStreamCreate( &mainStream ) );

        //CUDA_CHECK( cudaDeviceSetLimit( cudaLimitStackSize, 16000 ) );
        //CUDA_CHECK( cudaDeviceSetCacheConfig( cudaFuncCachePreferL1 ) );
    }

    inline ~cuda_context()
    {
        CUDA_CHECK( cudaStreamDestroy( mainStream ) );
        // NOTE: cudaDeviceReset must be called before exiting in order for profiling and
        // tracing tools such as Nsight and Visual Profiler to show complete traces.
        CUDA_CHECK( cudaDeviceReset() );
    }
};


#include <chrono>

struct frame_timer 
{
    using steady_clk_t = std::chrono::steady_clock;

    steady_clk_t::time_point start;
    std::chrono::microseconds& duration;

    inline frame_timer( std::chrono::microseconds& duration ) : duration{ duration }, start{ steady_clk_t::now() } {}
    inline ~frame_timer()
    {
        auto end = steady_clk_t::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>( end - start );
    }
};

inline auto ToMiliseconds( std::chrono::microseconds duration )
{
    return std::chrono::duration<float, std::milli>( duration ).count();
}

// NOTE: DBG flag used to trigger between the surface "in-place" vs dedicated + copy impls !
constexpr bool DBG_INPLACE_WORK = false;

int main()
{
    using pixel_t = uchar4;

    std::ios::sync_with_stdio( false );

    // NOTE: init sub-systems
    auto pCu = std::make_unique<cuda_context>();
    auto pPlatform = std::make_unique<sdl_platform>( width, height, WINDOW_TITLE );

    auto[ widthInPixels, heightInPixels ] = pPlatform->GetWindowSizeInPixels();
    auto pDx11 = std::make_unique<dx11_context>( 
        ( HWND ) pPlatform->GetWin32WindowHandle(), pCu->luid, widthInPixels, heightInPixels );

    dx11_texture dx11Tex = pDx11->CreateTexture2D( 
        width, height, dx11_context::SWAPCHAIN_FORMAT, D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX /* NOTE: NEEDED FOR INTEROP */ );
    auto pInteropTex = std::shared_ptr<interop_tex2d>{ new interop_tex2d{ dx11Tex }, DestroyInteropTex2D };

    // NOTE: resources
    constexpr u64 usageFlags = cuda_texture_usage_flags::READONLY | cuda_texture_usage_flags::WRITE;
    auto pGradTex = MakeSharedCudaTex2D<pixel_t>( width, height, usageFlags );

    // TODO: might wanna chenge these numbers
    u32 tx = 8;
    u32 ty = 8;

    dim3 blocks( ( width + tx - 1 ) / tx, ( height + ty - 1 ) / ty );

    // NOTE: engine loop
    static bool quit = false;
    for( u64 frameIdx = 0; !quit; ++frameIdx )
    {
        std::chrono::microseconds frameTime = {};
        {
            frame_timer frameTimer = { frameTime };
            for( SDL_Event e; SDL_PollEvent( &e );)
            {
                quit = ( SDL_EVENT_QUIT == e.type ) || ( SDL_EVENT_TERMINATING == e.type );
                if( quit ) break;
            }

            auto RenderLoop = [&] ( cudaSurfaceObject_t fbSurf ) {
                GradientSurfaceKernel<<<blocks, dim3( tx, ty ), 0, pCu->mainStream>>>( fbSurf, width, height );
                CUDA_CHECK( cudaGetLastError() );
            };

            if constexpr( !DBG_INPLACE_WORK )
            {
                RenderLoop( pGradTex->surf );
                //CUDA_CHECK( cudaStreamSynchronize( pCu->mainStream ) );
            }
            // NOTE: interop copy stuff
            {
                auto scopedMtx = pInteropTex->GetScopedMutex<CUDA_ACQUIRE>();
                auto mappedTex = pInteropTex->GetMapped<DBG_INPLACE_WORK>( pCu->mainStream );
                if constexpr( DBG_INPLACE_WORK )
                {
                    RenderLoop( mappedTex.surf );
                }
                else
                {
                    // TODO: check it's the same size format/type
                    u64 widthInBytes = pGradTex->width * pGradTex->elemPitch;
                    CudaCopyTextureDeviceSync( pGradTex->array, mappedTex.mem, widthInBytes, pGradTex->height );
                    CUDA_CHECK( cudaGetLastError() );
                }

            }

            auto scopedMtx = pInteropTex->GetScopedMutex<DX11_ACQUIRE>();
            pDx11->GetNextSwapchainImageCopyAndPresent( pInteropTex->dx11, false );
        }

        float frameTimeMs = ToMiliseconds( frameTime );
        std::cout << "\rFrame " << frameIdx << " time ms " << frameTimeMs << std::flush;
    }

    //globals g = {
    //    .mainCam = MakeCamRH( width, height ),
    //    .width = width,
    //    .height = height,
    //    .samplesPerPixel = 50,
    //    .maxBounces = 32
    //};
    //
    //CUDA_CHECK( cudaMemcpyToSymbol( globs, &g, sizeof( g ) ) );
    //
    //auto pWorld = std::make_shared<world>();
    //
    //worldref worldView = pWorld->GetRef();
    //
    //
    //RenderKernel<<<blocks, dim3( tx, ty )>>>( pGradTex->surf, worldView, frameIdx );
    //CUDA_CHECK( cudaGetLastError() );
    //CUDA_CHECK( cudaDeviceSynchronize() );
    //
    //auto hostTex = CudaCopyImageToHostSync( *pGradTex );
    //stbi_write_png( "sky.png", width, height, 4, std::data( hostTex ), width * sizeof( pixel_t ) );

    return 0;
}
