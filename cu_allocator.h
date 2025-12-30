#ifndef __CU_ALLOCATOR_H__
#define __CU_ALLOCATOR_H__

#include <cuda_runtime.h>
#include "cu_errors.h"

#include <vector>

// NOTE: inspired by https://committhis.github.io/2020/10/06/cuda-abstractions.html
template <typename T>
class pagelocked_host_allocator
{
public:
    using value_type = T;
    using ptr_type = value_type*;
    using size_type = std::size_t;

    pagelocked_host_allocator() noexcept = default;

    template <typename U>
    pagelocked_host_allocator( pagelocked_host_allocator<U> const& ) noexcept {}

    inline value_type* allocate( size_type n, const void* = 0 )
    {
        value_type* tmp;
        CUDA_CHECK( cudaMallocHost( ( void** ) &tmp, n * sizeof( T ) ) );
        return tmp;
    }

    inline void deallocate( ptr_type p, size_type n )
    {
        if( !p ) return;
        CUDA_CHECK( cudaFreeHost( p ) );
    }
};

template <typename T, typename U>
bool operator==( const pagelocked_host_allocator<T>&, const pagelocked_host_allocator<U>& )
{
    return true;
}

template <typename T, typename U>
bool operator!=( const pagelocked_host_allocator<T>&, const pagelocked_host_allocator<U>& )
{
    return false;
}

template <typename T>
using pagelocked_host_vector = std::vector<T, pagelocked_host_allocator<T>>;

#endif // !__CU_ALLOCATOR_H__
