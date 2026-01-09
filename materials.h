#ifndef __MATERIALS_H__
#define __MATERIALS_H__

#include <cuda_runtime.h>
#include "cu_math.h"
#include "cu_rand.h"
#include "core_types.h"

struct alignas( 16 ) pbr_material
{
    float3 baseColor;
    float metallic;
    float roughness;
    float transmission;
    float ior;
    float specular;
};

enum material_type : u32
{
    LAMBERT,
    METAL,
    DIELECTRIC
};

struct alignas( 16 ) material
{
    float3        albedo;
    float         ior;
    float         fuzz;
    material_type type;
};

struct scatter_res_t
{
    alignas( 16 ) float3 dir;
    alignas( 16 ) float3 attenuation;
};

inline __host__ __device__ float ReflectanceSchlick( float cosTheta, float refIdx )
{
    float r0 = ( 1.0f - refIdx ) / ( 1.0f + refIdx );
    r0 = r0 * r0;
    float inv = 1.0f - cosTheta;
    float inv5 = inv * inv * inv * inv * inv;
    return ( 1.0f - r0 ) * inv5 + r0;
}

inline __host__ __device__ float3 Refract( float3 rayDir, float3 surfNormal, float snellCoef, float cosTheta )
{
    float3 rPerp = snellCoef * ( rayDir + cosTheta * surfNormal );
    float3 rParallel = -std::sqrtf( std::fabsf( 1.0f - length_sq( rPerp ) ) ) * surfNormal;
    return rPerp + rParallel;
}

// TODO: check for degenerate scatter dir
__device__ scatter_res_t Scatter( const material& mat, float3 rayDir, float3 surfNormal, u32 randSeqIdx, u32 seed )
{
    float3 noise = RandUnitFloat3( randSeqIdx, seed );
    float randVar = noise.x;
    noise = noise * 2.0f - 1.0f; // NOTE: need float3 in [-1; 1] 

    float3 lambertDir = normalize( surfNormal + noise );

    float3 scatterDir = lambertDir;
    if( material_type::METAL == mat.type )
    {
        float3 reflDir = reflect( rayDir, surfNormal );
        float3 metalDir = normalize( reflDir + mat.fuzz * noise );
        scatterDir = metalDir;
    }
    else if( material_type::DIELECTRIC == mat.type )
    {
        bool isFrontFace = dot( rayDir, surfNormal ) < 0;
        float3 outwardNormal = isFrontFace ? surfNormal : -surfNormal;

        float cosTheta = fminf( dot( -rayDir, outwardNormal ), 1.0f );
        float sinTheta = sqrtf( fmaxf( 0.0f, 1.0f - cosTheta * cosTheta ) );

        float refractionIndex = isFrontFace ? ( 1.0f / mat.ior ) : mat.ior;

        bool cannotRefract = refractionIndex * sinTheta > 1.0f;
        bool isReflectance = ReflectanceSchlick( cosTheta, refractionIndex ) > randVar;

        float3 dielecDir = ( isReflectance || cannotRefract ) ? 
            reflect( rayDir, outwardNormal ) : 
            Refract( rayDir, outwardNormal, refractionIndex, cosTheta );
        scatterDir = normalize( dielecDir );
    }

    float3 attenuation = ( material_type::DIELECTRIC == mat.type ) ? make_float3( 1.0f ) : mat.albedo;

    return { .dir = scatterDir, .attenuation = attenuation };
}

#endif // !__MATERIALS_H__
