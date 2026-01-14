#ifndef __CU_DX11_INTEROP_H__
#define __CU_DX11_INTEROP_H__

#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>

#include "cu_errors.h"
#include "core_types.h"
#include "dx11_tex.h"

// TODO: is this more efficient than a copy ?
struct interop_mapped_resource
{
    cudaGraphicsResource* dx11Mapping;
    cudaStream_t stream;
    cudaArray* mem;

    cudaSurfaceObject_t surf;


    inline interop_mapped_resource( cudaGraphicsResource* dx11Rsc, cudaStream_t s )
    {
        dx11Mapping = dx11Rsc;
        stream = s;

        CUDA_CHECK( cudaGraphicsMapResources( 1, &dx11Mapping, stream ) );
        CUDA_CHECK( cudaGraphicsSubResourceGetMappedArray( &mem, dx11Mapping, 0, 0 ) );

        cudaResourceDesc resDesc = { .resType = cudaResourceTypeArray };
        resDesc.res.array.array = mem;
        CUDA_CHECK( cudaCreateSurfaceObject( &surf, &resDesc ) );
    }

    inline ~interop_mapped_resource()
    {
        CUDA_CHECK( cudaDestroySurfaceObject( surf ) );
        CUDA_CHECK( cudaGraphicsUnmapResources( 1, &dx11Mapping, stream ) );
    }
};

enum interop_mutex_key_t : u64
{
    CUDA_ACQUIRE = 0,
    DX11_ACQUIRE = 1,
};

template<interop_mutex_key_t INTEROP_DIR>
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

struct interop_tex2d
{
    dx11_texture                  dx11;
    cudaGraphicsResource*         cudaView;

    interop_tex2d( const dx11_texture& dx11Tex )
    {
        assert( nullptr != dx11Tex.rsc );

        dx11 = dx11Tex;
        CUDA_CHECK( cudaGraphicsD3D11RegisterResource( &cudaView, dx11.rsc, cudaGraphicsRegisterFlagsNone ) );
    }

    inline interop_mapped_resource GetMapped( cudaStream_t stream ) const
    {
        return { cudaView, stream };
    }

    template<interop_mutex_key_t INTEROP_DIR>
    inline auto GetScopedMutex()
    {
        assert( dx11.keyedMutex );
        return scoped_interop_mutex<INTEROP_DIR>{ dx11.keyedMutex };
    }
};

inline void DestroyInteropTex2D( interop_tex2d* interopTex )
{
    if( !interopTex ) return;

    if( interopTex->cudaView )
    {
        CUDA_CHECK( cudaGraphicsUnregisterResource( interopTex->cudaView ) );
        interopTex->cudaView = nullptr;
    }
    interopTex->dx11.Release();
}

#endif // !__CU_DX11_INTEROP_H__
