#include <optix_device.h>

#include "gdt/random/random.h"
#include "PostProcessing.h"
#include "MyMaterial.h"
#include "LaunchParams.h"

using namespace osc;

#define NUM_LIGHT_SAMPLES 4
#define PI 3.1415926

namespace osc {

    /*! launch parameters in constant memory, filled in by optix upon
        optixLaunch (this gets filled in from the buffer we pass to
        optixLaunch) */
    extern "C" __constant__ LaunchParams optixLaunchParams;

    /*! per-ray data now captures random number generator, so programs
        can access RNG state */

    static __device__ vec2f sampling_equirectangular_map(vec3f n) {
        float u = atan(n.z / n.x);
        u = (u + PI) / (2.0 * PI);

        float v = asin(n.y);
        v = (v * 2.0 + PI) / (2.0 * PI);
        v = 1.0f - v;

        return vec2f(u, v);
    }

    static __forceinline__ __device__
        void* unpackPointer(uint32_t i0, uint32_t i1)
    {
        const uint64_t uptr = static_cast<uint64_t>(i0) << 32 | i1;
        void* ptr = reinterpret_cast<void*>(uptr);
        return ptr;
    }

    static __forceinline__ __device__
        void  packPointer(void* ptr, uint32_t& i0, uint32_t& i1)
    {
        const uint64_t uptr = reinterpret_cast<uint64_t>(ptr);
        i0 = uptr >> 32;
        i1 = uptr & 0x00000000ffffffff;
    }

    template<typename T>
    static __forceinline__ __device__ T* getPRD()
    {
        const uint32_t u0 = optixGetPayload_0();
        const uint32_t u1 = optixGetPayload_1();
        return reinterpret_cast<T*>(unpackPointer(u0, u1));
    }

    //------------------------------------------------------------------------------
    // closest hit and anyhit programs for radiance-type rays.
    //
    // Note eventually we will have to create one pair of those for each
    // ray type and each geometry type we want to render; but this
    // simple example doesn't use any actual geometries yet, so we only
    // create a single, dummy, set of them (we do have to have at least
    // one group of them to set up the SBT)
    //------------------------------------------------------------------------------

    extern "C" __global__ void __closesthit__shadow()
    {
        /* not going to be used ... */
    }

    extern "C" __global__ void __closesthit__radiance()
    {
        uint32_t isectPtr0 = optixGetPayload_0();
        uint32_t isectPtr1 = optixGetPayload_1();
        Interaction* interaction = reinterpret_cast<Interaction*>(unpackPointer(isectPtr0, isectPtr1));
        const TriangleMeshSBTData& sbtData
            = *(const TriangleMeshSBTData*)optixGetSbtDataPointer();

        const int   primID = optixGetPrimitiveIndex();
        const vec3i index = sbtData.index[primID];
        const float u = optixGetTriangleBarycentrics().x;
        const float v = optixGetTriangleBarycentrics().y;

        const vec3f& A = sbtData.vertex[index.x];
        const vec3f& B = sbtData.vertex[index.y];
        const vec3f& C = sbtData.vertex[index.z];
        const vec3f pos = (1.f - u - v) * A + u * B + v * C;
        interaction->position = pos;

        vec3f Ng = cross(B - A, C - A);
        vec3f Ns = (sbtData.normal)
            ? ((1.f - u - v) * sbtData.normal[index.x]
                + u * sbtData.normal[index.y]
                + v * sbtData.normal[index.z])
            : Ng;
        interaction->realNormal = Ng;
        const vec3f rayDir = optixGetWorldRayDirection();

        if (dot(rayDir, Ng) > 0.f) Ng = -Ng;
        Ng = normalize(Ng);

        if (dot(Ng, Ns) < 0.f)
            Ns -= 2.f * dot(Ng, Ns) * Ng;
        Ns = normalize(Ns);

        interaction->geomNormal = Ns;

        if (sbtData.texcoord) {
            interaction->texcoord = (1.f - u - v) * sbtData.texcoord[index.x]
                + u * sbtData.texcoord[index.y]
                + v * sbtData.texcoord[index.z];
        }
        interaction->mat_mes = sbtData.mat_mes;
    }

    extern "C" __global__ void __anyhit__radiance()
    { /*! for this simple example, this will remain empty */
    }

    extern "C" __global__ void __anyhit__shadow()
    { /*! not going to be used */
    }

    //------------------------------------------------------------------------------
    // miss program that gets called for any ray that did not have a
    // valid intersection
    //
    // as with the anyhit/closest hit programs, in this example we only
    // need to have _some_ dummy function to set up a valid SBT
    // ------------------------------------------------------------------------------

    extern "C" __global__ void __miss__radiance()
    {
        uint32_t isectPtr0 = optixGetPayload_0();
        uint32_t isectPtr1 = optixGetPayload_1();
        Interaction* interaction = reinterpret_cast<Interaction*>(unpackPointer(isectPtr0, isectPtr1));
        interaction->distance = FLT_MAX;

        const cudaTextureObject_t& sbtData
            = *(const cudaTextureObject_t*)optixGetSbtDataPointer();
        vec3f ray_dir = optixGetWorldRayDirection();
        vec2f uv = sampling_equirectangular_map(ray_dir);
        vec4f fromTexture = tex2D<float4>(sbtData, uv.x, uv.y);
        interaction->mat_mes.emitter = (vec3f)fromTexture;
    }

    extern "C" __global__ void __miss__shadow()
    {

    }

    //------------------------------------------------------------------------------
    // ray gen program - the actual rendering happens in here
    //------------------------------------------------------------------------------
    extern "C" __global__ void __raygen__renderFrame()
    {
        // compute a test pattern based on pixel ID
        const int ix = optixGetLaunchIndex().x;
        const int iy = optixGetLaunchIndex().y;
        const auto& camera = optixLaunchParams.camera;

        int numPixelSamples = optixLaunchParams.numPixelSamples;

        PRD prd;

        vec3f pixelColor = 0.f;
        // normalized screen plane position, in [0,1]^2
        vec2f screen(vec2f(ix, iy) / vec2f(optixLaunchParams.frame.size));

        // generate ray direction
        vec3f rayDir = normalize(camera.direction
            + (screen.x - 0.5f) * camera.horizontal
            + (screen.y - 0.5f) * camera.vertical);

        for (int sampleID = 0; sampleID < numPixelSamples; sampleID++) {
            Ray myRay;
            myRay.origin = camera.position;
            myRay.direction = rayDir;
            vec3f radiance = 0.0f;
            vec3f accum = 1.0f;
            for (int bounces = 0; ; ++bounces)
            {
                if (bounces >= optixLaunchParams.maxBounce) {
                    //radiance = 0.0f;
                    break;
                }
                Interaction isect;
                isect.distance = 0;
                unsigned int isectPtr0, isectPtr1;
                packPointer(&isect, isectPtr0, isectPtr1);
                optixTrace(optixLaunchParams.traversable,
                    myRay.origin,
                    myRay.direction,
                    0,    // tmin
                    1e20f,  // tmax
                    0.0f,   // rayTime
                    OptixVisibilityMask(255),
                    OPTIX_RAY_FLAG_DISABLE_ANYHIT,//OPTIX_RAY_FLAG_NONE,
                    RADIANCE_RAY_TYPE,            // SBT offset
                    RAY_TYPE_COUNT,               // SBT stride
                    RADIANCE_RAY_TYPE,            // missSBTIndex 
                    isectPtr0, isectPtr1);
                if (isect.distance == FLT_MAX)
                {
                    if (bounces > 0)
                        radiance += isect.mat_mes.emitter * accum;
                    else
                        radiance += isect.mat_mes.emitter * accum / 2.0f;
                    break;
                }
                if (isect.mat_mes.emitterTextureID != -1) {
                    float u = isect.texcoord.x;
                    float v = isect.texcoord.y;
                    vec4f fromTexture = tex2D<float4>(isect.mat_mes.emitter_texture, u, v);
                    radiance += (vec3f)fromTexture * isect.mat_mes.emitter * accum / 3;
                }
                else
                    radiance += isect.mat_mes.emitter * accum;
                vec3f wo;
                float pdf = 0.0f;
                vec3f bsdf = cal_bsdf(isect, myRay.direction, &wo, &pdf, ix, iy, optixLaunchParams.frame.frameID);
                float cosine = fabsf(dot(isect.geomNormal, myRay.direction));
                accum *= bsdf * cosine / pdf;
                myRay = isect.spawn_ray(wo);
            }
            pixelColor += radiance;
        }

        vec4f rgba(pixelColor / numPixelSamples, 1.f);
        rgba.x = powf(rgba.x, 1 / 2.2f);
        rgba.y = powf(rgba.y, 1 / 2.2f);
        rgba.z = powf(rgba.z, 1 / 2.2f);
        if (rgba.x > 1)rgba.x = 1.0f;
        if (rgba.y > 1)rgba.y = 1.0f;
        if (rgba.z > 1)rgba.z = 1.0f;
        if (rgba.w > 1)rgba.w = 1.0f;

        // and write/accumulate to frame buffer ...
        const uint32_t fbIndex = ix + iy * optixLaunchParams.frame.size.x;
        if (optixLaunchParams.frame.frameID > 0) {
            rgba
                += float(optixLaunchParams.frame.frameID)
                * vec4f(optixLaunchParams.frame.colorBuffer[fbIndex]);
            rgba /= (optixLaunchParams.frame.frameID + 1.0f);
        }
        optixLaunchParams.frame.colorBuffer[fbIndex] = (float4)rgba;
        HSV hsv; BGR bgr; bgr.r = rgba.x; bgr.g = rgba.y; bgr.b = rgba.z;
        BGR2HSV(bgr, hsv);
        hsv.v = hsv.v * (1 + optixLaunchParams.lightness_change);
        if (hsv.s >= 0.05f)
            hsv.s += optixLaunchParams.saturate_change;
        HSV2BGR(hsv, bgr);
        Contrast(bgr, optixLaunchParams.contrast_change, 0.5f);
        rgba.x = bgr.r; rgba.y = bgr.g; rgba.z = bgr.b;
        optixLaunchParams.frame.renderBuffer[fbIndex] = (float4)rgba;
    }

} // ::osc
