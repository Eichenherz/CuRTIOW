#ifndef __CORE_TYPES_H__
#define __CORE_TYPES_H__

#include <stdint.h>
#include <vector_types.h>

using u8 = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;

using i8 = int8_t;
using i16 = int16_t;
using i32 = int32_t;
using i64 = int64_t;

using u8x2 = uchar2;
using u8x3 = uchar3;
using u8x4 = uchar4;

using u16x2 = ushort2;
using u16x3 = ushort3;
using u16x4 = ushort4;

using u32x2 = uint2;
using u32x3 = uint3;
using u32x4 = uint4;

using u64x2 = ulong2;
using u64x3 = ulong3;
using u64x4 = ulong4;

using i8x2 = char2;
using i8x3 = char3;
using i8x4 = char4;

using i16x2 = short2;
using i16x3 = short3;
using i16x4 = short4;

using i32x2 = int2;
using i32x3 = int3;
using i32x4 = int4;

using i64x2 = long2;
using i64x3 = long3;
using i64x4 = long4;

template<typename T>
concept Number32BitsMax = ( sizeof( T ) <= 4 );

#endif // !__CORE_TYPES_H__
