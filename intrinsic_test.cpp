#include <hip/hip_runtime.h>

#define CHECK(cmd)                                                                                 \
    {                                                                                              \
        hipError_t error = cmd;                                                                    \
        if (error != hipSuccess) {                                                                 \
            fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), error,         \
                    __FILE__, __LINE__);                                                           \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    }


typedef int __attribute__((ext_vector_type(4))) intx4;

__device__
float __llvm_amdgcn_buffer_load(intx4 rsrc, unsigned vindex, unsigned offset, bool glc, bool slc) __asm("llvm.amdgcn.buffer.load");

__global__ void kernel(float* A_d) {
    unsigned index = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);
    unsigned offset = index * sizeof(float);

    intx4 input {0};
    // fill in byte 0 - 1
    *reinterpret_cast<float**>(&input) = A_d;
    // fill in byte 2
    reinterpret_cast<int*>(&input)[2] = -1;
    // fill in byte 3
    reinterpret_cast<int*>(&input)[3] = 0x00027000;

    // original logic in C:
    A_d[index] = __llvm_amdgcn_buffer_load(input, index, offset, false, false);
}

template<typename T>
void launchTestKernel(void (*kernel)(T*), size_t N, unsigned threadsPerBlock, unsigned unrollFactor, unsigned vectorizeFactor, T* ptr) {
    unsigned blocks = N/threadsPerBlock/unrollFactor/vectorizeFactor;

    //printf("N: %zu unrollFactor: %u vectorizeFactor: %u threadsPerBlock: %u blocks: %u\n", N, unrollFactor, vectorizeFactor, threadsPerBlock, blocks);
    hipLaunchKernelGGL(kernel, dim3(blocks), dim3(threadsPerBlock), 0, 0, ptr);
}

int main() {
    float *A_d;
    float *A_h;
    size_t N = 1024 * 1024;
    size_t Nbytes = N * sizeof(float);
    static int device = 0;
    CHECK(hipSetDevice(device));
    hipDeviceProp_t props;
    CHECK(hipGetDeviceProperties(&props, device /*deviceID*/));
    printf("info: running on device %s\n", props.name);
    printf("info: architecture on AMD GPU device is: %d\n", props.gcnArch);
    printf("info: allocate host mem (%6.2f MB)\n", Nbytes / 1024.0 / 1024.0);
    A_h = (float*)malloc(Nbytes);
    CHECK(A_h == 0 ? hipErrorMemoryAllocation : hipSuccess);
    // Fill with 1.0f + i
    for (size_t i = 0; i < N; i++) {
        A_h[i] = 1.0f + i;
    }

    printf("info: allocate device mem (%6.2f MB)\n", Nbytes / 1024.0 / 1024.0);
    CHECK(hipMalloc(&A_d, Nbytes));

    printf("info: copy Host2Device %p->%p %zu bytes\n", A_h, A_d, Nbytes);
    CHECK(hipMemcpy(A_d, A_h, Nbytes, hipMemcpyHostToDevice));

    launchTestKernel(kernel, N, 256, 1, 1, A_d);
    hipStreamSynchronize(nullptr);

    printf("info: copy Device2Host %p->%p %zu bytes\n", A_d, A_h, Nbytes);
    CHECK(hipMemcpy(A_h, A_d, Nbytes, hipMemcpyDeviceToHost));

    hipFree(A_d);
    return 0;
}
