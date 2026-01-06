#ifndef __MATERIALS_H__
#define __MATERIALS_H__

#include <cuda_runtime.h>
#include "cu_math.h"
#include "cu_rand.h"

constexpr float3 LAMBERT = { 1.0f, 0.0f, 0.0f };
constexpr float3 METAL = { 0.0f, 1.0f, 0.0f };
constexpr float3 DIELECTRIC = { 0.0f, 0.0f, 1.0f };

struct alignas( 16 ) material
{
    float3 albedo;
    float ior;
    float3 materialMask;
    float fuzz;
};

struct scatter_res_t
{
    alignas( 16 ) float3 dir;
    alignas( 16 ) float3 attenuation;
};

inline __host__ __device__ float Schlick( float cosTheta, float refIdx )
{
    float r0 = ( 1.0f - refIdx ) / ( 1.0f + refIdx );
    r0 = r0 * r0;
    float inv = 1.0f - cosTheta;
    float inv5 = inv * inv * inv * inv * inv;
    return ( 1.0f - r0 ) * inv5 + r0;
}

// NOTE: Branchless refract candidate: sqrtk is zero if total internal reflection
inline __host__ __device__ float3 RefractBranchless( float3 dir, float3 surfNormal, float snellCoef )
{
    float dt = dot( dir, surfNormal );
    float k = 1.0f - snellCoef * snellCoef * ( 1.0f - dt * dt );
    float sqrtk = sqrtf( fmaxf( k, 0.0f ) );
    return snellCoef * ( dir - surfNormal * dt ) - surfNormal * sqrtk;
}

// TODO: check for degenerate scatter dir
__device__ scatter_res_t Scatter( const material& mat, float3 rayDir, float3 surfNormal, u32 randSeqIdx, u32 seed )
{
    float sumWeightsEps = mat.materialMask.x + mat.materialMask.y + mat.materialMask.z + 1e-6f;
    float3 materialTypeWeights = mat.materialMask / sumWeightsEps;

    float3 unitSquareRand = RandUnitFloat3( randSeqIdx, seed ) * 2.0f - 1.0f; // NOTE: need float3 in [-1; 1] 

    float3 lambertDir = normalize( surfNormal + unitSquareRand );

    float3 reflDir = reflect( rayDir, surfNormal );

    float3 metalDir = normalize( reflDir + mat.fuzz * unitSquareRand );

    //float cosTheta = fminf( dot( -rayDir, surfNormal ), 1.0f );
    //float sinTheta = sqrtf( fmaxf( 0.0f, 1.0f - cosTheta * cosTheta ) );
    //bool isFrontFace = dot( rayDir, surfNormal ) < 0;
    //float snellLawCoef = isFrontFace ? ( 1.0f / mat.ior ) : mat.ior;
    //float cannotRefract = snellLawCoef * sinTheta > 1.0f;
    //float reflectProb = cannotRefract + ( 1.0f - cannotRefract ) * Schlick( cosTheta, mat.ior );
    //float isReflect = unitSquareRand.x < reflectProb;
    //
    //float3 refracted = RefractBranchless( rayDir, surfNormal, snellLawCoef );
    //float3 dielecDir = isReflect * reflDir + ( 1.0f - isReflect ) * refracted;


    float3 scatterDir = lambertDir * materialTypeWeights.x + metalDir * materialTypeWeights.y;// +dielecDir * materialTypeWeights.z;

    float lambertMetalMask = materialTypeWeights.x + materialTypeWeights.y;
    float3 att = materialTypeWeights.z * float3{ 1.0f, 1.0f, 1.0f } + mat.albedo * lambertMetalMask;
    float3 attenuation = att * sumWeightsEps;

    return { .dir = scatterDir, .attenuation =  mat.albedo };
}

#endif // !__MATERIALS_H__
