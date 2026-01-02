#ifndef __CU_RAND_H__
#define __CU_RAND_H__

#include <cuda_runtime.h>
#include "core_types.h"

// NOTE: from https://github.com/PaulUlanovskij/squirrel_noise_5.rs
inline __host__ __device__ u32 SquirrelNoise5( u32 idx, u32 seed = 0 )
{
    constexpr u32 SQ5_BIT_NOISE1 = 0xd2a80a3f; // 11010010101010000000101000111111
    constexpr u32 SQ5_BIT_NOISE2 = 0xa884f197; // 10101000100001001111000110010111
    constexpr u32 SQ5_BIT_NOISE3 = 0x6C736F4B; // 01101100011100110110111101001011
    constexpr u32 SQ5_BIT_NOISE4 = 0xB79F3ABB; // 10110111100111110011101010111011
    constexpr u32 SQ5_BIT_NOISE5 = 0x1B56C4F5; // 00011011010101101100010011110101
    
    u32 mangled = idx;
    mangled *= SQ5_BIT_NOISE1;
    mangled += seed;
    mangled ^= mangled >> 9;
    mangled += SQ5_BIT_NOISE2;
    mangled ^= mangled >> 11;
    mangled *= SQ5_BIT_NOISE3;
    mangled ^= mangled >> 13;
    mangled += SQ5_BIT_NOISE4;
    mangled ^= mangled >> 15;
    mangled *= SQ5_BIT_NOISE5;
    mangled ^= mangled >> 17;
    return mangled;
}

constexpr u32 PRIME1 = 198491317; // Large prime number with non-boring bits
constexpr u32 PRIME2 = 6542989; // Large prime number with distinct and non-boring bits
constexpr u32 PRIME3 = 357239; // Large prime number with distinct and non-boring bits

__forceinline__ __host__ __device__ u32 RandInt( u32 idx, u32 seed = 0 )
{
    return SquirrelNoise5( idx, seed );
}
__forceinline__ __host__ __device__ float RandUnitFloat( u32 idx, u32 seed = 0 )
{
    // NOTE: Fast float generation from masked bits
    return ( float ) ( RandInt( idx, seed ) & 0x00FFFFFF ) / ( float ) ( 0x01000000 );
}
__forceinline__ __host__ __device__ float2 RandUnitFloat2( u32 idx, u32 seed = 0 )
{
    return { RandUnitFloat( idx, seed ), RandUnitFloat( idx + PRIME1, seed ) };
}
__forceinline__ __host__ __device__ float3 RandUnitFloat3( u32 idx, u32 seed = 0 )
{
    return { RandUnitFloat( idx + PRIME1, seed ), RandUnitFloat( idx + PRIME2, seed ), RandUnitFloat( idx + PRIME3, seed ) };
}

#endif // !__CU_RAND_H__
