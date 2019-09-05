#include <stdio.h>
#include "hip/hip_runtime.h"

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
float __buffer_load_dword(float* ptr, unsigned offset) {
  float output;
  intx4 input {0};
  // fill in byte 0 - 1
  *reinterpret_cast<float**>(&input) = ptr;
  // fill in byte 2
  reinterpret_cast<int*>(&input)[2] = -1;
  // fill in byte 3
  reinterpret_cast<int*>(&input)[3] = 0x00027000;

  asm volatile("\n \
    buffer_load_dword %0, %1, %2, 0 offen offset:0 \n \
    s_waitcnt 0 \n \
    " : "=v"(output)
      : "v"(offset), "s"(input));

  return output;
}

__device__
__global__ void vector_plus1(float* A_d) {
    unsigned index = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);
    unsigned offset = index * sizeof(float);

    // original logic in C:
    // A_d[index] = A_d[index] + 1.0f;
    //
    // __buffer_load_dword(ptr, offset) = *(ptr + offset)
    A_d[index] = __buffer_load_dword(A_d, offset);
}

__device__
float __buffer_load_dword_generic(float* ptr, unsigned p0, unsigned p1, unsigned Oi) {
  float output;
  intx4 input {0};
  // fill in byte 0 - 1
  *reinterpret_cast<float**>(&input) = ptr + p0;
  // fill in byte 2
  reinterpret_cast<int*>(&input)[2] = -1;
  // fill in byte 3
  reinterpret_cast<int*>(&input)[3] = 0x00027000;

  asm volatile("\n \
    buffer_load_dword %0, %1, %2, %3 offen offset:0 \n \
    s_waitcnt 0 \n \
    " : "=v"(output)
      : "v"(p1), "s"(input), "s"(Oi));

  return output;
}

__device__
void __buffer_load_dword_generic_unroll_16(float* dest, float* ptr_in, unsigned p0, unsigned p1, unsigned* O) {
  intx4 input {0};
  // fill in byte 0 - 1
  *reinterpret_cast<float**>(&input) = ptr_in + p0;
  // fill in byte 2
  reinterpret_cast<int*>(&input)[2] = -1;
  // fill in byte 3
  reinterpret_cast<int*>(&input)[3] = 0x00027000;

  asm volatile("\n \
    buffer_load_dword %0, %16, %17, %18 offen offset:0 \n \
    buffer_load_dword %1, %16, %17, %19 offen offset:0 \n \
    buffer_load_dword %2, %16, %17, %20 offen offset:0 \n \
    buffer_load_dword %3, %16, %17, %21 offen offset:0 \n \
    buffer_load_dword %4, %16, %17, %22 offen offset:0 \n \
    buffer_load_dword %5, %16, %17, %23 offen offset:0 \n \
    buffer_load_dword %6, %16, %17, %24 offen offset:0 \n \
    buffer_load_dword %7, %16, %17, %25 offen offset:0 \n \
    buffer_load_dword %8, %16, %17, %26 offen offset:0 \n \
    buffer_load_dword %9, %16, %17, %27 offen offset:0 \n \
    buffer_load_dword %10, %16, %17, %28 offen offset:0 \n \
    buffer_load_dword %11, %16, %17, %29 offen offset:0 \n \
    buffer_load_dword %12, %16, %17, %30 offen offset:0 \n \
    buffer_load_dword %13, %16, %17, %31 offen offset:0 \n \
    buffer_load_dword %14, %16, %17, %32 offen offset:0 \n \
    buffer_load_dword %15, %16, %17, %33 offen offset:0 \n \
    s_waitcnt 0 \n \
    " : "=v"(dest[0]), "=v"(dest[1]), "=v"(dest[2]), "=v"(dest[3]), "=v"(dest[4]), "=v"(dest[5]), "=v"(dest[6]), "=v"(dest[7]), "=v"(dest[8]), "=v"(dest[9]), "=v"(dest[10]), "=v"(dest[11]), "=v"(dest[12]), "=v"(dest[13]), "=v"(dest[14]), "=v"(dest[15])
      : "v"(p1), "s"(input), "s"(O[0]), "s"(O[1]), "s"(O[2]), "s"(O[3]), "s"(O[4]), "s"(O[5]), "s"(O[6]), "s"(O[7]), "s"(O[8]), "s"(O[9]), "s"(O[10]), "s"(O[11]), "s"(O[12]), "s"(O[13]), "s"(O[14]), "s"(O[15]));
}

__device__
__global__ void vector_plus1_generic_unroll_16(float* A_d) {
    constexpr unsigned N_per_thread = 16;
    unsigned p0 = hipBlockIdx_x * hipBlockDim_x;
    unsigned p1 = hipThreadIdx_x * N_per_thread;
    unsigned O[N_per_thread] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };

    float dest[N_per_thread];

    // original logic in C
    //for (unsigned i = 0; i < N_per_thread; ++i) {
    //  dest[i] = A_d[p0 + p1 + O[i]] + 1.0f;
    //}

    // Use inline assembly
    // Undesirable effect: s_waitcnt 0 after every load
    //
    //__buffer_load_dword_genric(ptr, offset) = *(ptr + offset)
    //for (unsigned i = 0; i < N_per_thread; ++i) {
    //  dest[i] = __buffer_load_dword_generic(A_d, p0 * sizeof(float), p1 * sizeof(float), O[i] * sizeof(float)) + 1.0f;
    //}

    // Use manually unrolled inline assembly
    unsigned O2[N_per_thread];
    for (unsigned i = 0; i < N_per_thread; ++i) {
      O2[i] = O[i] * sizeof(float);
    }
    __buffer_load_dword_generic_unroll_16(dest, A_d, p0 * sizeof(float), p1 * sizeof(float), O2);
    for (unsigned i = 0; i < N_per_thread; ++i) {
      dest[i] = dest[i] + 1.0f;
    }


    // Store results back
    for (unsigned i = 0; i < N_per_thread; ++i) {
      A_d[p0 + p1 + O[i]] = dest[i];
    }
}

__device__
__global__ void vector_plus1_generic(float* A_d) {
    constexpr unsigned N_per_thread = 16;
    unsigned p0 = hipBlockIdx_x * hipBlockDim_x;
    unsigned p1 = hipThreadIdx_x * N_per_thread;
    unsigned O[N_per_thread] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };

    float dest[N_per_thread];

    // original logic in C
    //for (unsigned i = 0; i < N_per_thread; ++i) {
    //  dest[i] = A_d[p0 + p1 + O[i]] + 1.0f;
    //}

    // Use inline assembly
    // Undesirable effect: s_waitcnt 0 after every load
    //
    //__buffer_load_dword_genric(ptr, offset) = *(ptr + offset)
    for (unsigned i = 0; i < N_per_thread; ++i) {
      dest[i] = __buffer_load_dword_generic(A_d, p0 * sizeof(float), p1 * sizeof(float), O[i] * sizeof(float)) + 1.0f;
    }

    // Store results back
    for (unsigned i = 0; i < N_per_thread; ++i) {
      A_d[p0 + p1 + O[i]] = dest[i];
    }
}

int main(int argc, char* argv[]) {
    float *A_d, *C_d;
    float *A_h, *C_h;
    size_t N = 1024;
    size_t Nbytes = N * sizeof(float);
    static int device = 0;
    CHECK(hipSetDevice(device));
    hipDeviceProp_t props;
    CHECK(hipGetDeviceProperties(&props, device /*deviceID*/));
    printf("info: running on device %s\n", props.name);
    printf("info: architecture on AMD GPU device is: %d\n", props.gcnArch);
    printf("info: allocate host mem (%6.2f MB)\n", 2 * Nbytes / 1024.0 / 1024.0);
    A_h = (float*)malloc(Nbytes);
    CHECK(A_h == 0 ? hipErrorMemoryAllocation : hipSuccess);
    C_h = (float*)malloc(Nbytes);
    CHECK(C_h == 0 ? hipErrorMemoryAllocation : hipSuccess);
    // Fill with Phi + i
    for (size_t i = 0; i < N; i++) {
        A_h[i] = 1.0f + i;
        C_h[i] = A_h[i];
    }

    printf("info: allocate device mem (%6.2f MB)\n", 2 * Nbytes / 1024.0 / 1024.0);
    CHECK(hipMalloc(&A_d, Nbytes));
    //CHECK(hipMalloc(&C_d, Nbytes));

    printf("info: copy Host2Device\n");
    CHECK(hipMemcpy(A_d, A_h, Nbytes, hipMemcpyHostToDevice));

    //const unsigned blocks = 1;
    //const unsigned threadsPerBlock = 16;
    //printf("info: launch 'vector_plus1' kernel\n");
    //hipLaunchKernelGGL(vector_plus1, dim3(blocks), dim3(threadsPerBlock), 0, 0, A_d);

    printf("info: launch 'vector_plus1_generic' kernel\n");
    hipLaunchKernelGGL(vector_plus1_generic, dim3(1), dim3(64), 0, 0, A_d);

    //printf("info: launch 'vector_plus1_generic_unroll_16' kernel\n");
    //hipLaunchKernelGGL(vector_plus1_generic_unroll_16, dim3(1), dim3(64), 0, 0, A_d);

    //printf("info: launch 'vector_plus1_store' kernel\n");
    //hipLaunchKernelGGL(vector_plus1_store, dim3(N/64), dim3(64), 0, 0, A_d);

    //printf("info: launch 'vector_plus1_generic_store' kernel\n");
    //hipLaunchKernelGGL(vector_plus1_generic_store, dim3(1), dim3(64), 0, 0, A_d);

    printf("info: copy Device2Host\n");
    CHECK(hipMemcpy(A_h, A_d, Nbytes, hipMemcpyDeviceToHost));

//#define CHECK_RESULT (0)

#ifndef CHECK_RESULT
#define CHECK_RESULT (1)
#endif
    printf("GPU CPU\n");
#if CHECK_RESULT
    printf("info: check result\n");
#else
#endif
    for (size_t i = 0; i < N; i++) {
        printf("%f %f\n", A_h[i], C_h[i]);
#if CHECK_RESULT
        if (A_h[i] != C_h[i] + 1.0f) {
            CHECK(hipErrorUnknown);
        }
#endif
    }
#if CHECK_RESULT
    printf("PASSED!\n");
#endif
}
