#ifndef __RAY_TRACING_H__
#define __RAY_TRACING_H__

#include <cuda_runtime.h>
#include <assert.h>

#include "core_types.h"
#include "cu_math.h"

constexpr float RAY_EPSILON = 0.001f;

// NOTE: the vecs are initialized to RH ( look at you RH palm )
struct camera
{
    float4x4 view;
    float4x4 proj;
    float4x4 invVP;

    float3 pos;
    float fovYRad;

    float3 fwd;
    float aspect;

    float3 up;
    float zNear;
};

inline __host__ camera MakeCamRH( u32 width, u32 height, float fovYRad = 1.570f, float zNear = 0.1f )
{
    camera cam = {
        .pos = {},
        .fovYRad = fovYRad,
        .fwd = { 0.0f, 0.0f, -1.0f },
        .aspect = ( float ) width / ( float ) height,
        .up = { 0.0f, 1.0f, 0.0f },
        .zNear = zNear
    };

    float3 lookAt = cam.pos + cam.fwd;
    cam.view = LookAtRH( cam.pos, lookAt, cam.up );
    cam.proj = PerspectiveInfFarRH( fovYRad, cam.aspect, zNear );

    float4x4 vp = mul( cam.proj, cam.view );

    float vpDet = det( vp );
    assert( 0.0f != vpDet );

    cam.invVP = ( 1.0f / vpDet ) * adj( vp );

    return cam;
}

struct ray
{
    alignas( 16 ) float3 origin;
    alignas( 16 ) float3 dir;
};

inline __host__ __device__ float3 RayAt( const ray& r, float t )
{
    return r.origin + t * r.dir;
}

struct alignas( 16 ) sphere_t
{
    float3 center;
    float radius;
};

constexpr float NO_HIT = -1.0f;

struct alignas( 16 ) hit_record
{
    float3 point;
    float t;
    float3 normal;
    bool frontFace;
};

constexpr hit_record INVALID_HIT = { .t = NO_HIT };

// TODO: also check normal to not be zero
inline __host__ __device__ bool IsValidHit( const hit_record& hit )
{
    return NO_HIT != hit.t;
}


__host__ __device__ hit_record HitRayVsSphere( const ray& r, sphere_t sphere, float rayTMin, float rayTMax )
{
    float3 oc = ( r.origin - sphere.center ) * -1.0f; // NOTE: bc of Z dir 
    float a = dot( r.dir, r.dir );
    float c = dot( oc, oc ) - sphere.radius * sphere.radius;
    float h = dot( r.dir, oc );
    float discriminant = h * h - a * c;

    if( discriminant < 0 ) return INVALID_HIT;

    float invA = 1.0f / a;
    float sqrtDiscrInvA = sqrtf( discriminant ) * invA;

    float root0 = h * invA - sqrtDiscrInvA;
    float root1 = h * invA + sqrtDiscrInvA;

    bool r0Valid = ( root0 > rayTMin ) && ( root0 < rayTMax );
    bool r1Valid = ( root1 > rayTMin ) && ( root1 < rayTMax );

    if( !r0Valid && !r1Valid)
    {
       return INVALID_HIT;
    }

    float root = r0Valid ? root0 : root1;
    float3 p = RayAt( r, root );
    float3 n = ( p - sphere.center ) / sphere.radius;
    bool isFrontFace = dot( r.dir, n ) < 0;

    return { .point = p, .t = root, .normal = isFrontFace ? n : -n, .frontFace = isFrontFace };
}

#endif // !__RAY_TRACING_H__
