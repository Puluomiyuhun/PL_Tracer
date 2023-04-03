#pragma once

#include "optix7.h"
#include <vector>
#include <assert.h>

namespace osc {

    /*! CUDABuffer���������һ��Cuda�˵��ڴ� */
    struct CUDABuffer {
        inline CUdeviceptr d_pointer() const
        {
            return (CUdeviceptr)d_ptr;
        }

        //GPU�������¹淶�ڴ�Ĵ�С
        void resize(size_t size)
        {
            if (d_ptr) free();
            alloc(size);
        }

        //��GPU�豸������һ���ڴ�
        void alloc(size_t size)
        {
            this->sizeInBytes = size;
            CUDA_CHECK(Malloc((void**)&d_ptr, sizeInBytes));
        }

        //�ͷ��ڴ�
        void free()
        {
            CUDA_CHECK(Free(d_ptr));
            d_ptr = nullptr;
            sizeInBytes = 0;
        }

        //��Gpu������һ���ڴ沢��һ��vt�����ϴ���Gpu��
        template<typename T>
        void alloc_and_upload(const std::vector<T>& vt)
        {
            alloc(vt.size() * sizeof(T));
            upload((const T*)vt.data(), vt.size());
        }

        //��Cpu�˴������ݵ�Gpu��
        template<typename T>
        void upload(const T* t, size_t count)
        {
            assert(d_ptr != nullptr);
            assert(sizeInBytes == count * sizeof(T));
            CUDA_CHECK(Memcpy(d_ptr, (void*)t,
                count * sizeof(T), cudaMemcpyHostToDevice));
        }

        //��Gpu�˴������ݵ�Cpu��
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