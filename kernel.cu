#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <memory>
#include <chrono>

#include <thrust/device_vector.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include "core_types.h"
#include "cu_errors.h"
#include "cu_allocator.h"
#include "cu_math.h"
#include "cu_rand.h"
#include "cu_tex2d.h"
#include "warps.h"
#include "cpp_helpers.h"

#include "sdl_platform.h"
#include "dx11.h"
#include "cu_dx11_interop.h"

#include "ray_tracing.h"
#include "materials.h"

#include <hp_serialization.h>
#include <hp_error.h>
#include <hp_mesh.h>

constexpr u16 width = 1280;
constexpr u16 height = 720;

constexpr char WINDOW_TITLE[] = "CuRTIOW";

struct world_data
{
    thrust::device_vector<instance> instances;
    thrust::device_vector<gpu_bvh2_node> globalTlasBuffer;
    thrust::device_vector<gpu_bvh2_node> globalClasBuffer;
    thrust::device_vector<rt_meshlet_info> meshletInfoBuffer;
    thrust::device_vector<float3> globalVertexPosBuffer;
    thrust::device_vector<packed_vtx> globalPackedVertexBuffer;
    thrust::device_vector<u8> globalTriangleBuffer;

    void UploadAssetFileData( const hellpack_view& hellpackView )
    {
        auto instView = hellpackView.Typed<instance>( hellpack_entry_slot::INST );
        auto tlasView = hellpackView.Typed<gpu_bvh2_node>( hellpack_entry_slot::TLAS );
        auto clasView = hellpackView.Typed<gpu_bvh2_node>( hellpack_entry_slot::CLAS );
        auto mletView = hellpackView.Typed<rt_meshlet_info>( hellpack_entry_slot::MLET );
        auto posView = hellpackView.Typed<float3>( hellpack_entry_slot::VPOS );
        auto pvtxView = hellpackView.Typed<packed_vtx>( hellpack_entry_slot::PVTX );
        auto triView = hellpackView.Typed<u8>( hellpack_entry_slot::TRI );

        instances.resize( std::size( instView ) );
        globalTlasBuffer.resize( std::size( tlasView ) );
        globalClasBuffer.resize( std::size( clasView ) );
        meshletInfoBuffer.resize( std::size( mletView ) );
        globalVertexPosBuffer.resize( std::size( posView ) );
        globalPackedVertexBuffer.resize( std::size( pvtxView ) );
        globalTriangleBuffer.resize( std::size( triView ) );

        CUDA_CHECK( cudaMemcpyAsync(
            thrust::raw_pointer_cast( std::data( instances ) ),
            std::data( instView ),
            std::size( instView ) * sizeof( instView[ 0 ] ),
            cudaMemcpyHostToDevice, 0 ) );

        CUDA_CHECK( cudaMemcpyAsync(
            thrust::raw_pointer_cast( std::data( globalTlasBuffer ) ),
            std::data( tlasView ),
            std::size( tlasView ) * sizeof( tlasView[ 0 ] ),
            cudaMemcpyHostToDevice, 0 ) );

        CUDA_CHECK( cudaMemcpyAsync(
            thrust::raw_pointer_cast( std::data( globalClasBuffer ) ),
            std::data( clasView ),
            std::size( clasView ) * sizeof( clasView[ 0 ] ),
            cudaMemcpyHostToDevice, 0 ) );

        CUDA_CHECK( cudaMemcpyAsync(
            thrust::raw_pointer_cast( std::data( meshletInfoBuffer ) ),
            std::data( mletView ),
            std::size( mletView ) * sizeof( mletView[ 0 ] ),
            cudaMemcpyHostToDevice, 0 ) );

        CUDA_CHECK( cudaMemcpyAsync(
            thrust::raw_pointer_cast( std::data( globalVertexPosBuffer ) ),
            std::data( posView ),
            std::size( posView ) * sizeof( posView[ 0 ] ),
            cudaMemcpyHostToDevice, 0 ) );

        CUDA_CHECK( cudaMemcpyAsync(
            thrust::raw_pointer_cast( std::data( globalPackedVertexBuffer ) ),
            std::data( pvtxView ),
            std::size( pvtxView ) * sizeof( pvtxView[ 0 ] ),
            cudaMemcpyHostToDevice, 0 ) );

        CUDA_CHECK( cudaMemcpyAsync(
            thrust::raw_pointer_cast( std::data( globalTriangleBuffer ) ),
            std::data( triView ),
            std::size( triView ) * sizeof( triView[ 0 ] ),
            cudaMemcpyHostToDevice, 0 ) );
    }
};

struct world_ref
{
    const instance*         instances;
    const gpu_bvh2_node*    globalTlasBuffer;
    const gpu_bvh2_node*    globalClasBuffer;
    const rt_meshlet_info*  meshletInfoBuffer;
    const float3*           globalVertexPosBuffer;
    const packed_vtx*       globalPackedVertexBuffer;
    const u8*               globalTriangleBuffer;

    u32                     instanceCount;
    u32                     tlasNodeCount;
    u32                     clasNodeCount;
    u32                     meshletCount;
    u32                     vertexCount;
    u32                     triangleByteCount;

    inline world_ref( const world_data& w )
    {
        instances = thrust::raw_pointer_cast( std::data( w.instances ) );
        instanceCount = ( u32 ) std::size( w.instances );

        globalTlasBuffer = thrust::raw_pointer_cast( std::data( w.globalTlasBuffer ) );
        tlasNodeCount = ( u32 ) std::size( w.globalTlasBuffer );

        globalClasBuffer = thrust::raw_pointer_cast( std::data( w.globalClasBuffer ) );
        clasNodeCount = ( u32 ) std::size( w.globalClasBuffer );

        meshletInfoBuffer = thrust::raw_pointer_cast( std::data( w.meshletInfoBuffer ) );
        meshletCount = ( u32 ) std::size( w.meshletInfoBuffer );

        globalVertexPosBuffer = thrust::raw_pointer_cast( std::data( w.globalVertexPosBuffer ) );
        globalPackedVertexBuffer = thrust::raw_pointer_cast( std::data( w.globalPackedVertexBuffer ) );
        vertexCount = ( u32 ) std::size( w.globalVertexPosBuffer );

        globalTriangleBuffer = thrust::raw_pointer_cast( std::data( w.globalTriangleBuffer ) );
        triangleByteCount = ( u32 ) std::size( w.globalTriangleBuffer );
    }
};

struct globals
{
    camera               mainCam;
    alignas( 8 ) u32     width;
    alignas( 8 ) u32     height;
    alignas( 8 ) u32     samplesPerPixel;
    alignas( 8 ) u32     maxBounces;
};

// NOTE: these must be properly alligned
__constant__ globals globs;

__device__ inline float3 Shade( const ray_t& r )
{
    float t = 0.5f * r.dir.y + 0.5f;
    float3 col = lerp( float3{ 1.0f, 1.0f, 1.0f }, float3{ 0.5f, 0.7f, 1.0f }, t );
    return col;
};

constexpr u32 VGPR_STACK_SIZE = 48;

__host__ __device__ float RayVsAABB( float3 aabbMin, float3 aabbMax, float3 rayOrigin, float3 rayInvDir )
{
    float3 tA3 = ( aabbMin - rayOrigin ) * rayInvDir;
    float3 tB3 = ( aabbMax - rayOrigin ) * rayInvDir;

    float3 tMin3 = fminf( tA3, tB3 );
    float3 tMax3 = fmaxf( tA3, tB3 );

    float tEnter = fmaxf( tMin3.x, fmaxf( tMin3.y, tMin3.z ) );
    float tExit = fminf( tMax3.x, fminf( tMax3.y, tMax3.z ) );

    return ( tExit >= tEnter ) ? tEnter : CUDART_INF_F;
}

__host__ __device__ hit_record RayVsCluster( 
    float3 rayOrigin, 
    float3 rayInvDir, 
    const packed_trs& trs,
    const rt_meshlet_info& clusterInfo, 
    const float3* positions, 
    const u8* triangleIndices 
) {
    for( u32 ti = 0; ti < clusterInfo.triangleCount; ti += 3 )
    {
        float3 v0 = positions[ clusterInfo.vertexOffset + triangleIndices[ clusterInfo.triangleOffset + ti + 0 ] ];
        float3 v1 = positions[ clusterInfo.vertexOffset + triangleIndices[ clusterInfo.triangleOffset + ti + 1 ] ];
        float3 v2 = positions[ clusterInfo.vertexOffset + triangleIndices[ clusterInfo.triangleOffset + ti + 2 ] ];
    }

    float3 tA3 = ( aabbMin - rayOrigin ) * rayInvDir;
    float3 tB3 = ( aabbMax - rayOrigin ) * rayInvDir;

    float3 tMin3 = fminf( tA3, tB3 );
    float3 tMax3 = fmaxf( tA3, tB3 );

    float tEnter = fmaxf( tMin3.x, fmaxf( tMin3.y, tMin3.z ) );
    float tExit = fminf( tMax3.x, fminf( tMax3.y, tMax3.z ) );

    return ( tExit >= tEnter ) ? tEnter : CUDART_INF_F;
}
constexpr u32 CLUSTER_PAYLOAD_FLAG_BITS = 0x80000000u;

struct hit_t
{
    float hitDist;
    u32 primitiveIdx;
};

__device__ __forceinline bool IsMiss( const hit_t& hit )
{
    return hit.hitDist == CUDART_INF_F;
}

struct aabb_t
{
    float3 min;
    float3 max;
};

struct tlas_policy
{
    const instance* instances;

    __device__ __forceinline aabb_t GetAabb( u32 idx )
    {
        const instance& inst = instances[ idx ];
        return { .min = inst.aabbMin, .max = inst.aabbMax };
    }
};

// NOTE: this needs to offset by the root idx !
struct clas_policy
{
    const rt_meshlet_info* clusters;

    __device__ __forceinline aabb_t GetAabb( u32 idx )
    {
        const rt_meshlet_info& mlet = clusters[ idx ];
        return { .min = mlet.aabbMin, .max = mlet.aabbMax };
    }
};

template<typename TraversalPolicy>
__device__ hit_t TraverseBVH( 
    float3                rayOrigin, 
    float3                rayInvDir, 
    const gpu_bvh2_node*  bvh, 
    TraversalPolicy       traversalPolicy
) {
    // NOTE: assume bvh is never empty !
    u32 vgprStack[ VGPR_STACK_SIZE ] = {};
    vgprStack[ 0 ] = 0;
    
    float hitDist = CUDART_INF_F;
    u32 primitiveIdx = u32( -1 );

    for( u32 stackCount = 1; stackCount > 0; --stackCount )
    {
        u32 nodePtr = vgprStack[ stackCount - 1 ];
        const gpu_bvh2_node& node = bvh[ nodePtr ];
        
        u32 childIdx0 = node.childIdx[ 0 ];
        if( Bvh2IsLeaf( childIdx0 ) )
        {
            u32 leafIdx = Bvh2LeafBase( childIdx0 );
            auto[ aabbMin, aabbMax ] = traversalPolicy.GetAabb( leafIdx );

            float instDist = RayVsAABB( aabbMin, aabbMax, rayOrigin, rayInvDir );

            bool closerHit = hitDist > instDist;
 
            primitiveIdx = closerHit ? leafIdx : primitiveIdx;
            hitDist = fminf( hitDist, instDist );
            
            continue;
        }
        
        childIdx0 = Bvh2NodeIdx( childIdx0 );
        u32 childIdx1 = Bvh2NodeIdx( node.childIdx[ 1 ] );

        float tChildDist0 = RayVsAABB( node.min[ 0 ], node.max[ 0 ], rayOrigin, rayInvDir );
        float tChildDist1 = RayVsAABB( node.min[ 1 ], node.max[ 1 ], rayOrigin, rayInvDir );

        bool swapChildren = tChildDist0 > tChildDist1;

        u32 nearChildIdx = swapChildren ? childIdx1 : childIdx0;
        u32 farChildIdx  = swapChildren ? childIdx0 : childIdx1;
        float nearT = swapChildren ? tChildDist1 : tChildDist0;
        float farT  = swapChildren ? tChildDist0 : tChildDist1;

        vgprStack[ stackCount ] = farChildIdx;
        stackCount += ( u32 ) ( farT < hitDist );
        vgprStack[ stackCount ] = nearChildIdx;
        stackCount += ( u32 ) ( nearT < hitDist );
    }

    return { .hitDist = hitDist, .primitiveIdx = primitiveIdx };
}

// NOTE: don't pass as a ref unless it's a gpu ref !
__global__ void KernelPrimaryVisibility( cudaSurfaceObject_t fbSurf, const world_ref worldRef, u32 frameIdx ) 
{
    u32 xi = threadIdx.x + blockIdx.x * blockDim.x;
    u32 yi = threadIdx.y + blockIdx.y * blockDim.y;

    if( ( xi >= globs.width ) || ( yi >= globs.height ) ) return;

    const camera& mainCam = globs.mainCam;
    const instance* instances = worldRef.instances;
    const rt_meshlet_info* meshlets = worldRef.meshletInfoBuffer;

    u32 pixelIdx = yi * globs.width + xi;
    u32 baseSeed = pixelIdx ^ ( frameIdx * PRIME1 );

    float2 jitter = RandUnitFloat2( 0, baseSeed ) - float2{ 0.5f, 0.5f };
    float4 clipSpacePos = GetClipSpaceVector( float2( xi, yi ) + jitter, float2( globs.width, globs.height ) );
    float4 worldHmgPos = mul( mainCam.invVP, clipSpacePos );
    float3 worldPos = xyz( worldHmgPos ) / worldHmgPos.w;
    float3 rayDir = normalize( worldPos - mainCam.pos );

    float3 rayOrigin = mainCam.pos;
    float3 invRayDir = 1.0f / rayDir;

    hit_t instanceHit = TraverseBVH( rayOrigin, invRayDir, worldRef.globalTlasBuffer, tlas_policy{ instances } );

    bool isMiss = IsMiss( instanceHit );
    if( isMiss )
    {
        surf2Dwrite( CUDART_INF_F, fbSurf, xi * sizeof( float ), yi );
        return;
    }

    const instance thisInst = instances[ instanceHit.primitiveIdx ];
    bool isCluster = Bvh2RefIsInvalid( thisInst.clasBvhRoot );

    u32 clusterIdx = thisInst.baseMeshletOffset;

    if( !isCluster )
    {
        hit_t clusterHit = TraverseBVH( rayOrigin, invRayDir, thisInst.clasBvhRoot, clas_policy{ meshlets } );
        bool isMiss = IsMiss( clusterHit );
        if( isMiss )
        {
            surf2Dwrite( CUDART_INF_F, fbSurf, xi * sizeof( float ), yi );
            return;
        }
        clusterIdx = clusterHit.primitiveIdx;
    }

    const rt_meshlet_info& cluster = meshlets[ clusterIdx ];

    hit_record hit = RayVsCluster( 
        rayOrigin, invRayDir, thisInst.toWorld, cluster, worldRef.globalVertexPosBuffer, worldRef.globalTriangleBuffer );

    float viewDepth = hit.t * dot( mainCam.fwd, rayDir );
   
    // NOTE: cuda 12.9 with err with interleaved source in ptx for this line ! on Pascal
    surf2Dwrite( viewDepth, fbSurf, xi * sizeof( viewDepth ), yi );
}

__global__ void GradientSurfaceKernel( cudaSurfaceObject_t fbSurf, u16 width, u16 height ) 
{
    u32 xi = threadIdx.x + blockIdx.x * blockDim.x;
    u32 yi = threadIdx.y + blockIdx.y * blockDim.y;

    if( ( xi >= width ) || ( yi >= height ) ) return;

    auto pixel = make_uchar4( 
        (u8)( ( xi * 255 ) / width ), // Red increases left → right
        (u8)( ( yi * 255 ) / height ), 0, 255 ); // Green increases top → bottom

    // NOTE: cuda 12.9 with err with interleaved source in ptx for this line ! on Pascal
    surf2Dwrite( pixel, fbSurf, xi * sizeof( pixel ), yi );
}

struct cuda_context
{
    u64              luid;
    cudaStream_t     mainStream;
    u32              deviceIdx;

    cuda_context()
    {
        i32  deviceCount;
        CUDA_CHECK( cudaGetDeviceCount( &deviceCount ) );
        CUDA_CHECK( ( 0 == deviceCount ) ? cudaError( -1 ) : cudaError::cudaSuccess );

        cudaDeviceProp deviceProp;
        // TODO: check device props ?
        for( u32 di = 0; di < deviceCount; ++di )
        {
            CUDA_CHECK( cudaGetDeviceProperties( &deviceProp, di ) );
            CUDA_CHECK( ( WARP_SIZE != deviceProp.warpSize ) ? cudaError( -1 ) : cudaError::cudaSuccess );

            std::memcpy( &luid, deviceProp.luid, sizeof( deviceProp.luid ) );
            deviceIdx = di;

            break;
        }

        printf( "CUDA GPU %d: %s\n", deviceIdx, deviceProp.name );

        CUDA_CHECK( cudaSetDevice( deviceIdx ) );

        CUDA_CHECK( cudaStreamCreate( &mainStream ) );

        //CUDA_CHECK( cudaDeviceSetLimit( cudaLimitStackSize, 16000 ) );
        //CUDA_CHECK( cudaDeviceSetCacheConfig( cudaFuncCachePreferL1 ) );
    }

    inline ~cuda_context()
    {
        CUDA_CHECK( cudaStreamDestroy( mainStream ) );
        // NOTE: cudaDeviceReset must be called before exiting in order for profiling and
        // tracing tools such as Nsight and Visual Profiler to show complete traces.
        CUDA_CHECK( cudaDeviceReset() );
    }
};

struct frame_timer 
{
    using steady_clk_t = std::chrono::steady_clock;

    steady_clk_t::time_point start;
    std::chrono::microseconds& duration;

    inline frame_timer( std::chrono::microseconds& duration ) : duration{ duration }, start{ steady_clk_t::now() } {}
    inline ~frame_timer()
    {
        auto end = steady_clk_t::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>( end - start );
    }
};

inline auto ToMiliseconds( std::chrono::microseconds duration )
{
    return std::chrono::duration<float, std::milli>( duration ).count();
}

// NOTE: DBG flag used to trigger between the surface "in-place" vs dedicated + copy impls !
constexpr bool DBG_INPLACE_WORK = false;

// TODO: this is already duplicated in HellPack
inline auto ReadFileBinary( const char* path )
{
    FILE* f = nullptr;
    HP_ASSERT( ::fopen_s( &f, path, "rb" ) == 0 );
    HP_ASSERT( f );

    HP_ASSERT( ::fseek( f, 0, SEEK_END ) == 0 );
    i32 sz = ::ftell( f );
    HP_ASSERT( sz >= 0 );
    HP_ASSERT( ::fseek( f, 0, SEEK_SET ) == 0 );

    std::vector<u8> out( sz );
    u64 read = ::fread( std::data( out ), 1, std::size( out ), f );
    HP_ASSERT( std::size( out ) == read );

    HP_ASSERT( ::fclose( f ) == 0 );
    return out;
}

int main()
{
    using pixel_t = uchar4;

    std::ios::sync_with_stdio( false );

    std::vector<u8> hlpBinaryData = ReadFileBinary( "D:/3d models/nightclub_futuristic_pub_ambience_asset.hllp" );
    hellpack_view hellpackView = { hlpBinaryData };

    world_data gpuWorldData;
    gpuWorldData.UploadAssetFileData( hellpackView );

    // NOTE: init sub-systems
    auto pPlatform = std::make_unique<sdl_platform>( width, height, WINDOW_TITLE );

    auto[ widthInPixels, heightInPixels ] = pPlatform->GetWindowSizeInPixels();

    auto pCu = std::make_unique<cuda_context>();
    auto pDx11 = std::make_unique<dx11_context>( 
        ( HWND ) pPlatform->GetWin32WindowHandle(), pCu->luid, widthInPixels, heightInPixels );

    dx11_texture dx11Tex = pDx11->CreateTexture2D( 
        width, height, dx11_context::SWAPCHAIN_FORMAT, D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX /* NOTE: NEEDED FOR INTEROP */ );
    auto pInteropTex = std::shared_ptr<interop_tex2d>{ new interop_tex2d{ dx11Tex }, DestroyInteropTex2D };

    // NOTE: resources
    constexpr u64 usageFlags = cuda_texture_usage_flags::READONLY | cuda_texture_usage_flags::WRITE;
    auto pGradTex = MakeSharedCudaTex2D<pixel_t>( width, height, usageFlags );

    // TODO: might wanna chenge these numbers
    u32 tx = 8;
    u32 ty = 8;

    dim3 blocks( ( width + tx - 1 ) / tx, ( height + ty - 1 ) / ty );

    // NOTE: engine loop
    static bool quit = false;
    for( u64 frameIdx = 0; !quit; ++frameIdx )
    {
        std::chrono::microseconds frameTime = {};
        {
            frame_timer frameTimer = { frameTime };
            for( SDL_Event e; SDL_PollEvent( &e ); )
            {
                quit = ( SDL_EVENT_QUIT == e.type ) || ( SDL_EVENT_TERMINATING == e.type );
                if( quit ) break;
            }

            auto RenderLoop = [ & ]( cudaSurfaceObject_t fbSurf ) 
            {
                GradientSurfaceKernel<<<blocks, dim3( tx, ty ), 0, pCu->mainStream>>>( fbSurf, width, height );
                CUDA_CHECK( cudaGetLastError() );
            };

            if constexpr( !DBG_INPLACE_WORK )
            {
                RenderLoop( pGradTex->surf );
                //CUDA_CHECK( cudaStreamSynchronize( pCu->mainStream ) );
            }
            // NOTE: interop stuff
            {
                auto scopedMtx = pInteropTex->GetScopedMutex<CUDA_ACQUIRE>();
                auto mappedTex = pInteropTex->GetMapped<DBG_INPLACE_WORK>( pCu->mainStream );
                if constexpr( DBG_INPLACE_WORK )
                {
                    RenderLoop( mappedTex.surf );
                }
                else
                {
                    // TODO: check it's the same size format/type
                    u64 widthInBytes = pGradTex->width * pGradTex->elemPitch;
                    CudaCopyTextureDeviceSync( pGradTex->array, mappedTex.mem, widthInBytes, pGradTex->height );
                    CUDA_CHECK( cudaGetLastError() );
                }
            }

            auto scopedMtx = pInteropTex->GetScopedMutex<DX11_ACQUIRE>();
            pDx11->GetNextSwapchainImageCopyAndPresent( pInteropTex->dx11, false );
        }

        float frameTimeMs = ToMiliseconds( frameTime );
        std::cout << "\rFrame " << frameIdx << " time ms " << frameTimeMs << std::flush;
    }

    //globals g = {
    //    .mainCam = MakeCamRH( width, height ),
    //    .width = width,
    //    .height = height,
    //    .samplesPerPixel = 50,
    //    .maxBounces = 32
    //};
    //
    //CUDA_CHECK( cudaMemcpyToSymbol( globs, &g, sizeof( g ) ) );
    //
    //auto pWorld = std::make_shared<world>();
    //
    //worldref worldView = pWorld->GetRef();
    //
    //
    //
    //auto hostTex = CudaCopyImageToHostSync( *pGradTex );
    //stbi_write_png( "sky.png", width, height, 4, std::data( hostTex ), width * sizeof( pixel_t ) );

    return 0;
}
