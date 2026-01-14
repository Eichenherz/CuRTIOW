#ifndef __DX11_H__
#define __DX11_H__

#include "win32_include.h"
#include <d3d11.h>
#include <dxgi1_6.h>

#include <memory>
#include <iostream>
#include <assert.h>

#include "core_types.h"
#include "dx11_tex.h"

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

// TODO: make sure we create on the same device as CUDA !
struct dx11_context
{
    static constexpr u32                      SWAPCHAIN_IMG_COUNT = 3;
    static constexpr DXGI_FORMAT              SWAPCHAIN_FORMAT = DXGI_FORMAT_R8G8B8A8_UNORM;

    dx_unique<ID3D11Device>                   device;
    dx_unique<ID3D11DeviceContext>            context;
    dx_unique<IDXGISwapChain1>                swapchain;

    // NOTE: fuck ctors
    dx11_context() = default;

    // TODO: dynamically get machine correct SC desc !
    dx11_context( HWND hwnd, const u64 desiredLuid, u32 widthInPixels, u32 heightInPixels )
    {
        assert( IsWindow( hwnd ) );

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

        // Create swapchain
        DXGI_SWAP_CHAIN_DESC1 scDesc = {
            .Width = widthInPixels,
            .Height = heightInPixels,
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

        ID3D11RenderTargetView* rtv = nullptr;

        constexpr u32 D3D11_RESOURCE_MISC_SHARED_FLAGS_BITS = 
            D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX | D3D11_RESOURCE_MISC_SHARED | D3D11_RESOURCE_MISC_SHARED_NTHANDLE;
        bool hasSharedFalgs = D3D11_RESOURCE_MISC_SHARED_FLAGS_BITS & texDesc.MiscFlags;
        if( !hasSharedFalgs )
        {
            D3D11_RENDER_TARGET_VIEW_DESC rtvDesc = {
                .Format = texDesc.Format,
                .ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D,
                .Texture2D = { .MipSlice = 0 }
            };
            HR_CHECK( device->CreateRenderTargetView( tex, &rtvDesc, &rtv ) );
        }

        IDXGIKeyedMutex* keyedMutex = nullptr;
        if( D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX & texDesc.MiscFlags )
        {
            HR_CHECK( tex->QueryInterface( IID_PPV_ARGS( &keyedMutex ) ) );
        }

        return { 
            .rsc        = tex, 
            .rtv        = rtv, 
            .keyedMutex = keyedMutex,
            .format     = texDesc.Format, 
            .width      = ( u16 ) texDesc.Width, 
            .height     = ( u16 ) texDesc.Height
        };
    }

    void GetNextSwapchainImageCopyAndPresent( const dx11_texture& srcTex, bool vsync )
    {
        assert( srcTex.rsc );

        ID3D11Texture2D* thisScImg = nullptr;
        HR_CHECK( swapchain->GetBuffer( 0, IID_PPV_ARGS( &thisScImg ) ) );

        context->CopyResource( thisScImg, srcTex.rsc );

        u32 syncInterval = vsync ? 1 : 0;
        HR_CHECK( swapchain->Present( syncInterval, 0 ) );

        thisScImg->Release();
    }

    void ClearRenderTargetView( const dx11_texture& tex, float4 color )
    {
        assert( tex.rtv );
        context->ClearRenderTargetView( tex.rtv, ( float* ) &color );
    }
};

#endif // !__DX11_H__
