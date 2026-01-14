#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <memory>

#include <thrust/device_vector.h>

#include <SDL3/SDL.h>

#include "win32_include.h"
#include <d3d11.h>
#include <dxgi1_6.h>

#include <cuda_d3d11_interop.h>


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
inline auto MakeDxSharedFromRaw( T* raw )
{
    return std::shared_ptr<T>{ raw, dx_releaser<T>{} };
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
    abort();
}

#define HR_CHECK( hr ) HresultCheck( hr )

inline u64 LuidToU64( const LUID& luid )
{
    return ( u64( u32( luid.HighPart ) ) << 32 ) | u32( luid.LowPart );
}

struct dx11_texture
{
    ID3D11Texture2D*     rsc;
    DXGI_FORMAT          format;
    u16                  width;
    u16                  height;

    inline auto GetKeyedMutex() const
    {
        IDXGIKeyedMutex* keyedMutex = nullptr;
        HR_CHECK( rsc->QueryInterface( IID_PPV_ARGS( &keyedMutex ) ) );

        return dx_unique<IDXGIKeyedMutex>{ keyedMutex };
    }
};

// TODO: make sure we create on the same device as CUDA !
struct dx11_context
{
    static constexpr u32                      SWAPCHAIN_IMG_COUNT = 3;
    static constexpr DXGI_FORMAT              SWAPCHAIN_FORMAT = DXGI_FORMAT_R8G8B8A8_UNORM;

    dx_unique<ID3D11Device>                   device;
    dx_unique<ID3D11DeviceContext>            context;
    dx_unique<IDXGISwapChain1>                swapchain;

    std::shared_ptr<ID3D11Texture2D>          backbuffer;

    // NOTE: fuck ctors
    dx11_context() = default;

    // TODO: dynamically get machine correct SC desc !
    dx11_context( SDL_Window* wnd, const u64 desiredLuid )
    {
        assert( nullptr != wnd );

        // Create factory
        IDXGIFactory2* rawFactory;
        HR_CHECK( CreateDXGIFactory2( 0, IID_PPV_ARGS( &rawFactory ) ) );
        auto factory = dx_unique<IDXGIFactory2>{ rawFactory };

        // Get adapter
        dx_unique<IDXGIAdapter1> gpu = nullptr;
        u32 ai = 0;
        for( 
            IDXGIAdapter1* adapter = nullptr; 
            factory->EnumAdapters1( ai, &adapter ) != DXGI_ERROR_NOT_FOUND; 
            adapter->Release(), ++ai
        ) {
            DXGI_ADAPTER_DESC1 desc;
            adapter->GetDesc1( &desc );

            if( desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE ) continue;

            u64 luid64 = LuidToU64( desc.AdapterLuid );
            if( desiredLuid == luid64 )
            {
                gpu = dx_unique<IDXGIAdapter1>{ adapter };
                break;
            }
        }

        HR_CHECK( ( nullptr == gpu ) ? E_FAIL : 0 );

        // Create device
        UINT createFlags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;
    #ifdef _DEBUG
        createFlags |= D3D11_CREATE_DEVICE_DEBUG;
    #endif

        ID3D11Device* rawDevice = nullptr;
        ID3D11DeviceContext* rawContext = nullptr;

        D3D_FEATURE_LEVEL featureLevel = D3D_FEATURE_LEVEL_11_1;
        HR_CHECK( D3D11CreateDevice( gpu.get(), D3D_DRIVER_TYPE_UNKNOWN, nullptr, createFlags, nullptr, 0,
                                     D3D11_SDK_VERSION, &rawDevice, &featureLevel, &rawContext ) );

        device = dx_unique<ID3D11Device>{ rawDevice };
        context = dx_unique<ID3D11DeviceContext>{ rawContext };

        // Get hwnd
        auto winProperties = SDL_GetWindowProperties( wnd );
        auto hwnd = ( HWND ) SDL_GetPointerProperty( winProperties, SDL_PROP_WINDOW_WIN32_HWND_POINTER, nullptr );

        // Create swapchain
        i32 widthInPixels, heightInPixels;
        SDL_CHECK( !SDL_GetWindowSizeInPixels( wnd, &widthInPixels, &heightInPixels ) );

        DXGI_SWAP_CHAIN_DESC1 scDesc = {
            .Width = ( u32 ) widthInPixels,
            .Height = ( u32 ) heightInPixels,
            .Format = SWAPCHAIN_FORMAT,
            .SampleDesc = { .Count = 1, .Quality = 0 },
            .BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT,
            .BufferCount = SWAPCHAIN_IMG_COUNT,
            .SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD,
            .Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING
        };

        IDXGISwapChain1* rawSwapchain = nullptr;
        HR_CHECK( factory->CreateSwapChainForHwnd( device.get(), hwnd, &scDesc, nullptr, nullptr, &rawSwapchain ) );
        swapchain = dx_unique<IDXGISwapChain1>{ rawSwapchain };

        // NOTE: unlike Vk/DX12 we don't manually acquire SC images, we only use img0 in dx11
        ID3D11Texture2D* thisScImg = nullptr;
        HR_CHECK( swapchain->GetBuffer( 0, IID_PPV_ARGS( &thisScImg ) ) );
        backbuffer = MakeDxSharedFromRaw( thisScImg );

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

    dx11_texture CreateTexture2D( u16 width, u16 height, DXGI_FORMAT format, u32 miscFlags )
    {
        D3D11_TEXTURE2D_DESC texDesc = {
            .Width            = ( u32 ) width,
            .Height           = ( u32 ) height,
            .MipLevels        = 1,
            .ArraySize        = 1,
            .Format           = format,
            .SampleDesc       = { .Count = 1 },
            .Usage            = D3D11_USAGE_DEFAULT,
            .BindFlags        = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE,
            .CPUAccessFlags   = 0,
            .MiscFlags        = miscFlags
        };

        ID3D11Texture2D* tex = nullptr;
        HR_CHECK( device->CreateTexture2D( &texDesc, nullptr, &tex ) );
        return { .rsc = tex, .format = texDesc.Format, .width = ( u16 ) texDesc.Width, .height = ( u16 ) texDesc.Height };
    }
};

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
    u64 luid;
    u32 deviceIdx;

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

struct interop_tex2d
{
    dx11_texture                  dx11;
    cudaGraphicsResource*         cudaView;
    dx_unique<IDXGIKeyedMutex>    keyedMutex;
};

enum interop_dir_t : u32
{
    DX11_TO_CUDA = 0,
    CUDA_TO_DX11 = 1,
};

template<interop_dir_t INTEROP_DIR>
struct scoped_interop_mutex
{
    IDXGIKeyedMutex* mtx;
    scoped_interop_mutex( IDXGIKeyedMutex* mtx ) : mtx{ mtx }
    {
        HR_CHECK( mtx->AcquireSync( INTEROP_DIR, INFINITE ) );
    }

    ~scoped_interop_mutex()
    {
        u64 flippedOwnership = INTEROP_DIR ^ 1;
        HR_CHECK( mtx->ReleaseSync( flippedOwnership ) );
    }

    NON_COPYABLE( scoped_interop_mutex );
};

inline auto CreateInteropTex2D( const dx11_texture& dx11Tex )
{
    assert( nullptr != dx11Tex.rsc );

    interop_tex2d tex2d = { .dx11 = dx11Tex, .keyedMutex = dx11Tex.GetKeyedMutex() };
    CUDA_CHECK( cudaGraphicsD3D11RegisterResource( &tex2d.cudaView, tex2d.dx11.rsc, cudaGraphicsRegisterFlagsNone ) );

    return tex2d;
}

inline void DestroyInteropTex2D( interop_tex2d& interopTex )
{
    if( interopTex.cudaView )
    {
        CUDA_CHECK( cudaGraphicsUnregisterResource( interopTex.cudaView ) );
        interopTex.cudaView = nullptr;
    }
    if( interopTex.dx11.rsc )
    {
        u64 refs = interopTex.dx11.rsc->Release();
        interopTex.dx11.rsc = nullptr;
        assert( 0 == refs );
    }
}

struct renderer_state
{
    dx11_context              dx11;
    interop_tex2d             cudaDx11InteropTex;
    cuda_context              cu;
    sdl_platform              platform;

    renderer_state() : cu{}, platform{ width, height, WINDOW_TITLE }  
    {
        dx11 = { platform.wnd, cu.luid };
        dx11_texture dx11Tex = dx11.CreateTexture2D( 
            width, height, dx11_context::SWAPCHAIN_FORMAT, D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX /* NOTE: NEEDED FOR INTEROP */ );
        cudaDx11InteropTex = CreateInteropTex2D( dx11Tex );
    }

    ~renderer_state()
    {
        DestroyInteropTex2D( cudaDx11InteropTex );
        cu.~cuda_context();
        dx11.~dx11_context();
        platform.~sdl_platform();
    }
};

int main()
{
    using pixel_t = uchar4;

    auto pRenderer = std::make_unique<renderer_state>();

    const dx11_context& dx11 = pRenderer->dx11;

    static bool quit = false;
    while( !quit )
    {
        for( SDL_Event e; SDL_PollEvent( &e );)
        {
            quit = ( SDL_EVENT_QUIT == e.type ) || ( SDL_EVENT_TERMINATING == e.type );
            if( quit ) break;
        }

        HR_CHECK( dx11.swapchain->Present( 1, 0 ) );
    }

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
