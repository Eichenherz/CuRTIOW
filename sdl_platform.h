#ifndef __SDL_PLATFORM_H__
#define __SDL_PLATFORM_H__

#include <SDL3/SDL.h>

#include "core_types.h"

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

    inline void* GetWin32WindowHandle() const
    {
        auto winProperties = SDL_GetWindowProperties( wnd );
        void* hwnd = SDL_GetPointerProperty( winProperties, SDL_PROP_WINDOW_WIN32_HWND_POINTER, nullptr );
        SDL_CHECK( nullptr == hwnd );

        return hwnd;
    }

    struct sdl_window_size
    {
        u32 widthInPixels;
        u32 heightInPixels;
    };
    inline sdl_window_size GetWindowSizeInPixels() const
    {
        i32 width, height;
        SDL_CHECK( !SDL_GetWindowSizeInPixels( wnd, &width, &height ) );
        return { .widthInPixels = ( u32 ) width, .heightInPixels = ( u32 ) height };
    }
};

#endif // !__SDL_PLATFORM_H__
