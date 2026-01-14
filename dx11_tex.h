#ifndef __DX11_TEX_H__
#define __DX11_TEX_H__

#include <d3d11.h>

#include "core_types.h"

struct dx11_texture
{
    ID3D11Texture2D*            rsc;
    ID3D11RenderTargetView*     rtv;
    IDXGIKeyedMutex*            keyedMutex;
    DXGI_FORMAT                 format;
    u16                         width;
    u16                         height;

    // TODO: check for refcounting
    inline void Release()
    {
        if( keyedMutex ) keyedMutex->Release();
        if( rtv ) rtv->Release();
        if( rsc ) rsc->Release();
    }
};

#endif // !__DX11_TEX_H__
