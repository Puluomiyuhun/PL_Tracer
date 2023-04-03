#pragma once

#include "optix7.h"
#include <vector>
#include <assert.h>

namespace osc {

    /*! CUDABuffer就是申请的一段Cuda端的内存 */
    struct CUDABuffer {
        inline CUdeviceptr d_pointer() const
        {
            return (CUdeviceptr)d_ptr;
        }

        //GPU端中重新规范内存的大小
        void resize(size_t size)
        {
            if (d_ptr) free();
            alloc(size);
        }

        //在GPU设备上申请一段内存
        void alloc(size_t size)
        {
            this->sizeInBytes = size;
            CUDA_CHECK(Malloc((void**)&d_ptr, sizeInBytes));
        }

        //释放内存
        void free()
        {
            CUDA_CHECK(Free(d_ptr));
            d_ptr = nullptr;
            sizeInBytes = 0;
        }

        //在Gpu端申请一段内存并将一串vt数据上传到Gpu端
        template<typename T>
        void alloc_and_upload(const std::vector<T>& vt)
        {
            alloc(vt.size() * sizeof(T));
            upload((const T*)vt.data(), vt.size());
        }

        //从Cpu端传输数据到Gpu端
        template<typename T>
        void upload(const T* t, size_t count)
        {
            assert(d_ptr != nullptr);
            assert(sizeInBytes == count * sizeof(T));
            CUDA_CHECK(Memcpy(d_ptr, (void*)t,
                count * sizeof(T), cudaMemcpyHostToDevice));
        }

        //从Gpu端传输数据到Cpu端
        template<typename T>
        void download(T* t, size_t count)
        {
            assert(d_ptr != nullptr);
            assert(sizeInBytes == count * sizeof(T));
            CUDA_CHECK(Memcpy((void*)t, d_ptr,
                count * sizeof(T), cudaMemcpyDeviceToHost));
        }

        inline size_t size() const { return sizeInBytes; }
        size_t sizeInBytes{ 0 };
        void* d_ptr{ nullptr };
    };

}