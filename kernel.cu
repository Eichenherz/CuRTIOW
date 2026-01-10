#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <memory>

#include <thrust/device_vector.h>

#include <SDL3/SDL.h>

#include "win32_include.h"
#include <d3d11.h>
#include <dxgi.h>

#include <cuda_d3d11_interop.h>


#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include "core_types.h"
#include "cu_errors.h"
#include "cu_allocator.h"
#include "cu_math.h"
#include "cu_rand.h"
#include "cu_tex2d.h"

#include "ray_tracing.h"
#include "materials.h"

constexpr u16 width = 1280;
constexpr u16 height = 720;

constexpr char WINDOW_TITLE[] = "CuRTIOW";

void SdlCheck( bool failed, const char* file, const int line )
{
    if( failed )
    {
        SDL_Log( "Failed at %s:%d with: %s", file, line, SDL_GetError() );
        abort();
    }
}

#define SDL_CHECK( val ) SdlCheck( val, __FILE__, __LINE__ )

using engine_loop_t = void( * )( );

struct sdl_platform
{
    SDL_Window*     wnd;

    inline sdl_platform( i32 windowWidth, i32 windowHeight, const char* windowTitle )
    {
        SDL_CHECK( !SDL_Init( SDL_INIT_VIDEO | SDL_INIT_EVENTS ) );

       wnd = SDL_CreateWindow( windowTitle, windowWidth, windowHeight, SDL_WINDOW_HIGH_PIXEL_DENSITY );
       SDL_CHECK( nullptr == wnd );
    }

    inline ~sdl_platform()
    {
        if( wnd ) SDL_DestroyWindow( wnd );
        SDL_Quit();
    }

    inline void RunLoop( engine_loop_t EngineLoop )
    {
        static bool quit = false;
        while( !quit )
        {
            for( SDL_Event e; SDL_PollEvent( &e );)
            {
                quit = ( SDL_EVENT_QUIT == e.type ) || ( SDL_EVENT_TERMINATING == e.type );
                if( quit ) break;

                EngineLoop();
            }
        }
    }
};

// NOTE: ComPtr bullshit
template<typename T>
struct dx_releaser
{
    inline void operator()( T* ptr ) const noexcept
    {
        if( ptr ) ptr->Release();
    }
};

template<typename T>
using dx_unique = std::unique_ptr<T, dx_releaser<T>>;
template<typename T>
using dx_shared = std::shared_ptr<T>;

template<typename T>
inline dx_shared<T> MakeDxSharedFromRaw( T* raw )
{
    return dx_shared<T>{ raw, dx_releaser<T>{} };
}

void HresultCheck( HRESULT hr )
{
    if( SUCCEEDED( hr ) ) return;

    char* msg = nullptr;

    DWORD dwFlags = FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_IGNORE_INSERTS;
    FormatMessageA(
        dwFlags, nullptr, hr, MAKELANGID( LANG_NEUTRAL, SUBLANG_DEFAULT ), ( LPSTR ) &msg, 0, nullptr );

    if( msg )
    {
        std::cout << "HRESULT Failed: 0x" << std::hex << hr << " - " << msg;
        LocalFree( msg );
    }
    else
    {
        std::cout << "HRESULT Failed: 0x" << std::hex << hr << " (Unknown error)\n";
    }
}

#define HR_CHECK( hr ) HresultCheck( hr )

// TODO: make sure we create on the same device as CUDA !
struct dx11_context
{
    static constexpr u32                      SWAPCHAIN_IMG_COUNT = 3;
    static constexpr DXGI_FORMAT              SWAPCHAIN_FORMAT = DXGI_FORMAT_R8G8B8A8_UNORM;

    dx_unique<ID3D11Device>                   device;
    dx_unique<ID3D11DeviceContext>            context;
    dx_unique<IDXGISwapChain>                 swapchain;

    dx_shared<ID3D11Texture2D>                backbuffers[ SWAPCHAIN_IMG_COUNT ];
    dx_shared<ID3D11RenderTargetView>         rtvs[ SWAPCHAIN_IMG_COUNT ];

    D3D_FEATURE_LEVEL                         featureLevel;

    // TODO: dynamically get machine correct SC desc !
    dx11_context( SDL_Window* wnd )
    {
        assert( nullptr != wnd );

        auto winProperties = SDL_GetWindowProperties( wnd );
        auto hwnd = ( HWND ) SDL_GetPointerProperty( winProperties, SDL_PROP_WINDOW_WIN32_HWND_POINTER, nullptr );

        i32 widthInPixels, heightInPixels;
        SDL_CHECK( !SDL_GetWindowSizeInPixels( wnd, &widthInPixels, &heightInPixels ) );

        DXGI_SWAP_CHAIN_DESC scDesc = {
            .BufferDesc = {
                .Width = ( u32 ) widthInPixels,
                .Height = ( u32 ) heightInPixels,
                .RefreshRate = {.Numerator = 60, .Denominator = 1 },
                .Format = SWAPCHAIN_FORMAT,
            },
            .SampleDesc = {.Count = 1, .Quality = 0 },
            .BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT,
            .BufferCount = SWAPCHAIN_IMG_COUNT,
            .OutputWindow = hwnd,
            .Windowed = TRUE,
            .SwapEffect = DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL
        };

        UINT createFlags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;
    #ifdef _DEBUG
        createFlags |= D3D11_CREATE_DEVICE_DEBUG;
    #endif

        ID3D11Device* rawDevice = nullptr;
        ID3D11DeviceContext* rawContext = nullptr;
        IDXGISwapChain* rawSwapchain = nullptr;
        HR_CHECK( D3D11CreateDeviceAndSwapChain(
            nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, createFlags, nullptr, 0, D3D11_SDK_VERSION,
            &scDesc, &rawSwapchain, &rawDevice, &featureLevel, &rawContext
        ) );

        device = dx_unique<ID3D11Device>{ rawDevice };
        context = dx_unique<ID3D11DeviceContext>{ rawContext };
        swapchain = dx_unique<IDXGISwapChain>{ rawSwapchain };

        for( u32 sci = 0; sci < SWAPCHAIN_IMG_COUNT; ++sci )
        {
            ID3D11Texture2D* thisScImg;
            HR_CHECK( swapchain->GetBuffer( sci, IID_PPV_ARGS( &thisScImg ) ) );
            ID3D11RenderTargetView* thisScView;
            HR_CHECK( device->CreateRenderTargetView( thisScImg, nullptr, &thisScView ) );

            backbuffers[ sci ] = MakeDxSharedFromRaw( thisScImg );
            rtvs[ sci ] = MakeDxSharedFromRaw( thisScView );
        }

        D3D11_VIEWPORT vp = {
            .TopLeftX = 256,
            .TopLeftY = 256,
            .Width = ( float ) widthInPixels,
            .Height = ( float ) heightInPixels,
            .MinDepth = 0.0f,
            .MaxDepth = 1.0f
        };
        context->RSSetViewports( 1, &vp );
    }
};

inline cudaGraphicsResource* Dx11RegisterCudaInterOpTex( ID3D11Texture2D* dx11Tex )
{
    cudaGraphicsResource* cudaRes = nullptr;
    CUDA_CHECK( cudaGraphicsD3D11RegisterResource( &cudaRes, dx11Tex, cudaGraphicsRegisterFlagsWriteDiscard ) );

    return cudaRes;
}

struct cuda_dx11_mapped_array
{
    cudaGraphicsResource* dx11Mapping;
    cudaStream_t stream;
    cudaArray_t mem;

    inline cuda_dx11_mapped_array( cudaGraphicsResource* dx11Rsc, cudaStream_t s )
    {
        dx11Mapping = dx11Rsc;
        stream = s;

        CUDA_CHECK( cudaGraphicsMapResources( 1, &dx11Mapping, stream ) );
        CUDA_CHECK( cudaGraphicsSubResourceGetMappedArray( &mem, dx11Mapping, 0, 0 ) );
    }

    inline ~cuda_dx11_mapped_array()
    {
        CUDA_CHECK( cudaGraphicsUnmapResources( 1, &dx11Mapping, stream ) );
    }
};

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

struct renderer_state
{
    dx11_context              dx11;
    sdl_platform              platform;
    cuda_context              cu;
    cudaGraphicsResource*     swapchainViews[ dx11_context::SWAPCHAIN_IMG_COUNT ] = {};

    renderer_state() : platform{ width, height, WINDOW_TITLE }, dx11{ platform.wnd }, cu{}
    {
        for( u32 sci = 0; sci < dx11_context::SWAPCHAIN_IMG_COUNT; ++sci )
        {
            swapchainViews[ sci ] = Dx11RegisterCudaInterOpTex( dx11.backbuffers[ sci ].get() );
        }
    }

    ~renderer_state()
    {
        for( cudaGraphicsResource*& pScView : swapchainViews )
        {
            if( nullptr == pScView ) continue;
            CUDA_CHECK( cudaGraphicsUnregisterResource( pScView ) );
            pScView = nullptr;
        }

        cu.~cuda_context();
        dx11.~dx11_context();
        platform.~sdl_platform();
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
        .maxBounces = 32
    };

    CUDA_CHECK( cudaMemcpyToSymbol( globs, &g, sizeof( g ) ) );

    auto pWorld = std::make_shared<world>();

    worldref worldView = pWorld->GetRef();

    u32 frameIdx = 0;

    u32 tx = 8;
    u32 ty = 8;
    
    dim3 blocks( ( width + tx - 1 ) / tx, ( height + ty - 1 ) / ty );
    RenderKernel<<<blocks, dim3( tx, ty )>>>( pGradTex->surf, worldView, frameIdx );
    CUDA_CHECK( cudaGetLastError() );
    CUDA_CHECK( cudaDeviceSynchronize() );

    auto hostTex = CudaCopyImageToHostSync( *pGradTex );
    stbi_write_png( "sky.png", width, height, 4, std::data( hostTex ), width * sizeof( pixel_t ) );

    return 0;
}
