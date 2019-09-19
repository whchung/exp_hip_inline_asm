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

typedef float __attribute__((ext_vector_type(4))) floatx4;


// ptr[offset] = value
__device__
void __buffer_store_dword(float* ptr, unsigned offset, float value) {
  intx4 input {0};
  // fill in byte 0 - 1
  *reinterpret_cast<float**>(&input) = ptr;
  // fill in byte 2
  reinterpret_cast<int*>(&input)[2] = -1;
  // fill in byte 3
  reinterpret_cast<int*>(&input)[3] = 0x00027000;

  asm volatile("\n \
    buffer_store_dword %1, %2, %0, 0 offen offset:0 \n \
    s_waitcnt 0 \n \
    " :
      : "s"(input), "v"(value), "v"(offset));
}

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

// ptr[offset] = value
__device__
void __global_store_dword(float* ptr, unsigned off, float value) {
  unsigned long long offset = off;
  asm volatile("\n \
    global_store_dword %0, %1, %2, offset:0 \n \
    s_waitcnt 0 \n \
    " :
      : "v"(offset), "v"(value), "s"(ptr));
}

__device__
void __global_store_dwordx4(floatx4* ptr, unsigned off, floatx4 value) {
  unsigned long long offset = off;
  asm volatile("\n \
    global_store_dwordx4 %0, %1, %2, offset:0 \n \
    s_waitcnt 0 \n \
    " :
      : "v"(offset), "v"(value), "s"(ptr));
}

__device__
float __global_load_dword(float* ptr, unsigned off) {
  float output;
  unsigned long long offset = off;

  asm volatile("\n \
    global_load_dword %0, %1, %2, offset:0 \n \
    s_waitcnt 0 \n \
    " : "=v"(output)
      : "v"(offset), "s"(ptr));

  return output;
}

__device__
floatx4 __global_load_dwordx4(floatx4* ptr, unsigned off) {
  floatx4 output;
  unsigned long long offset = off;

  asm volatile("\n \
    global_load_dwordx4 %0, %1, %2, offset:0 \n \
    s_waitcnt 0 \n \
    " : "=v"(output)
      : "v"(offset), "s"(ptr));

  return output;
}

__device__
__global__ void vector_plus1_naive(float* A_d) {
    unsigned index = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);
    unsigned offset = index * sizeof(float);

    // original logic in C:
    A_d[index] = A_d[index] + 1.0f;
}

__device__
__global__ void vector_plus1_buffer_load(float* A_d) {
    unsigned index = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);
    unsigned offset = index * sizeof(float);

    // original logic in C:
    // A_d[index] = A_d[index] + 1.0f;
    //
    // __buffer_load_dword(ptr, offset) = *(ptr + offset)
    A_d[index] = __buffer_load_dword(A_d, offset) + 1.0f;
}

__device__
__global__ void vector_plus1_global_load(float* A_d) {
    unsigned index = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);
    unsigned offset = index * sizeof(float);

    // original logic in C:
    // A_d[index] = A_d[index] + 1.0f;
    //
    // __buffer_load_dword(ptr, offset) = *(ptr + offset)
    A_d[index] = __global_load_dword(A_d, offset) + 1.0f;
}

__device__
__global__ void vector_plus1_global_loadx4(floatx4* A_d) {
    unsigned index = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);
    unsigned offset = index * sizeof(floatx4);

    // original logic in C:
    // A_d[index] = A_d[index] + 1.0f;
    //
    // __buffer_load_dword(ptr, offset) = *(ptr + offset)
    A_d[index] = __global_load_dwordx4(A_d, offset) + 1.0f;
}

__device__
__global__ void vector_plus1_buffer_store(float* A_d) {
    unsigned index = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);
    unsigned offset = index * sizeof(float);

    // original logic in C:
    // A_d[index] = A_d[index] + 1.0f;
    //
    float result = __buffer_load_dword(A_d, offset) + 1.0f;
    __buffer_store_dword(A_d, offset, result);
}

__device__
__global__ void vector_plus1_global_store(float* A_d) {
    unsigned index = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);
    unsigned offset = index * sizeof(float);

    // original logic in C:
    // A_d[index] = A_d[index] + 1.0f;
    //
    float result = __global_load_dword(A_d, offset) + 1.0f;
    __global_store_dword(A_d, offset, result);
}

__device__
__global__ void vector_plus1_global_storex4(floatx4* A_d) {
    unsigned index = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);
    unsigned offset = index * sizeof(floatx4);

    // original logic in C:
    // A_d[index] = A_d[index] + 1.0f;
    //
    floatx4 result = __global_load_dwordx4(A_d, offset) + 1.0f;
    __global_store_dwordx4(A_d, offset, result);
    //A_d[offset] = result;
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
void __buffer_store_dword_generic(float* ptr, unsigned p0, unsigned p1, unsigned Oi, float value) {
  intx4 input {0};
  // fill in byte 0 - 1
  *reinterpret_cast<float**>(&input) = ptr + p0;
  // fill in byte 2
  reinterpret_cast<int*>(&input)[2] = -1;
  // fill in byte 3
  reinterpret_cast<int*>(&input)[3] = 0x00027000;

  asm volatile("\n \
    buffer_store_dword %1, %2, %0, %3 offen offset:0 \n \
    s_waitcnt 0 \n \
    " :
      : "s"(input), "v"(value), "v"(p1), "s"(Oi));
}

__device__
void __buffer_load_dword_generic_unroll_2(float* dest, float* ptr, unsigned p0, unsigned p1, unsigned* O) {
  intx4 input {0};
  // fill in byte 0 - 1
  *reinterpret_cast<float**>(&input) = ptr + p0;
  // fill in byte 2
  reinterpret_cast<int*>(&input)[2] = -1;
  // fill in byte 3
  reinterpret_cast<int*>(&input)[3] = 0x00027000;

  asm volatile("\n \
    buffer_load_dword %0, %2, %3, %4 offen offset:0 \n \
    buffer_load_dword %1, %2, %3, %5 offen offset:0 \n \
    s_waitcnt 0 \n \
    " : "=v"(dest[0]), "=v"(dest[1])
      : "v"(p1), "s"(input), "s"(O[0]), "s"(O[1]));
}

__device__
void __buffer_load_dword_generic_unroll_4(float* dest, float* ptr, unsigned p0, unsigned p1, unsigned* O) {
  intx4 input {0};
  // fill in byte 0 - 1
  *reinterpret_cast<float**>(&input) = ptr + p0;
  // fill in byte 2
  reinterpret_cast<int*>(&input)[2] = -1;
  // fill in byte 3
  reinterpret_cast<int*>(&input)[3] = 0x00027000;

  asm volatile("\n \
    buffer_load_dword %0, %4, %5, %6 offen offset:0 \n \
    buffer_load_dword %1, %4, %5, %7 offen offset:0 \n \
    buffer_load_dword %2, %4, %5, %8 offen offset:0 \n \
    buffer_load_dword %3, %4, %5, %9 offen offset:0 \n \
    s_waitcnt 0 \n \
    " : "=v"(dest[0]), "=v"(dest[1]), "=v"(dest[2]), "=v"(dest[3])
      : "v"(p1), "s"(input), "s"(O[0]), "s"(O[1]), "s"(O[2]), "s"(O[3]));
}

__device__
void __buffer_load_dword_generic_unroll_16(float* dest, float* ptr, unsigned p0, unsigned p1, unsigned* O) {
  intx4 input {0};
  // fill in byte 0 - 1
  *reinterpret_cast<float**>(&input) = ptr + p0;
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
void __buffer_store_dword_generic_unroll_2(float* ptr, unsigned p0, unsigned p1, unsigned* O, float* value) {
  intx4 input {0};
  // fill in byte 0 - 1
  *reinterpret_cast<float**>(&input) = ptr + p0;
  // fill in byte 2
  reinterpret_cast<int*>(&input)[2] = -1;
  // fill in byte 3
  reinterpret_cast<int*>(&input)[3] = 0x00027000;

  asm volatile("\n \
    buffer_store_dword %0, %2, %3, %4 offen offset:0 \n \
    buffer_store_dword %1, %2, %3, %5 offen offset:0 \n \
    s_waitcnt 0 \n \
    " :
      : "v"(value[0]), "v"(value[1]), "v"(p1), "s"(input), "s"(O[0]), "s"(O[1]));
}

__device__
void __buffer_store_dword_generic_unroll_4(float* ptr, unsigned p0, unsigned p1, unsigned* O, float* value) {
  intx4 input {0};
  // fill in byte 0 - 1
  *reinterpret_cast<float**>(&input) = ptr + p0;
  // fill in byte 2
  reinterpret_cast<int*>(&input)[2] = -1;
  // fill in byte 3
  reinterpret_cast<int*>(&input)[3] = 0x00027000;

  asm volatile("\n \
    buffer_store_dword %0, %4, %5, %6 offen offset:0 \n \
    buffer_store_dword %1, %4, %5, %7 offen offset:0 \n \
    buffer_store_dword %2, %4, %5, %8 offen offset:0 \n \
    buffer_store_dword %3, %4, %5, %9 offen offset:0 \n \
    s_waitcnt 0 \n \
    " :
      : "v"(value[0]), "v"(value[1]), "v"(value[2]), "v"(value[3]), "v"(p1), "s"(input), "s"(O[0]), "s"(O[1]), "s"(O[2]), "s"(O[3]));
}

__device__
void __buffer_store_dword_generic_unroll_16(float* ptr, unsigned p0, unsigned p1, unsigned* O, float* value) {
  intx4 input {0};
  // fill in byte 0 - 1
  *reinterpret_cast<float**>(&input) = ptr + p0;
  // fill in byte 2
  reinterpret_cast<int*>(&input)[2] = -1;
  // fill in byte 3
  reinterpret_cast<int*>(&input)[3] = 0x00027000;

  asm volatile("\n \
    buffer_store_dword %0, %16, %17, %18 offen offset:0 \n \
    buffer_store_dword %1, %16, %17, %19 offen offset:0 \n \
    buffer_store_dword %2, %16, %17, %20 offen offset:0 \n \
    buffer_store_dword %3, %16, %17, %21 offen offset:0 \n \
    buffer_store_dword %4, %16, %17, %22 offen offset:0 \n \
    buffer_store_dword %5, %16, %17, %23 offen offset:0 \n \
    buffer_store_dword %6, %16, %17, %24 offen offset:0 \n \
    buffer_store_dword %7, %16, %17, %25 offen offset:0 \n \
    buffer_store_dword %8, %16, %17, %26 offen offset:0 \n \
    buffer_store_dword %9, %16, %17, %27 offen offset:0 \n \
    buffer_store_dword %10, %16, %17, %28 offen offset:0 \n \
    buffer_store_dword %11, %16, %17, %29 offen offset:0 \n \
    buffer_store_dword %12, %16, %17, %30 offen offset:0 \n \
    buffer_store_dword %13, %16, %17, %31 offen offset:0 \n \
    buffer_store_dword %14, %16, %17, %32 offen offset:0 \n \
    buffer_store_dword %15, %16, %17, %33 offen offset:0 \n \
    s_waitcnt 0 \n \
    " :
      : "v"(value[0]), "v"(value[1]), "v"(value[2]), "v"(value[3]), "v"(value[4]), "v"(value[5]), "v"(value[6]), "v"(value[7]), "v"(value[8]), "v"(value[9]), "v"(value[10]), "v"(value[11]), "v"(value[12]), "v"(value[13]), "v"(value[14]), "v"(value[15]), "v"(p1), "s"(input), "s"(O[0]), "s"(O[1]), "s"(O[2]), "s"(O[3]), "s"(O[4]), "s"(O[5]), "s"(O[6]), "s"(O[7]), "s"(O[8]), "s"(O[9]), "s"(O[10]), "s"(O[11]), "s"(O[12]), "s"(O[13]), "s"(O[14]), "s"(O[15]));
}

__device__
__global__ void vector_plus1_naive_unroll_16(float* A_d) {
    constexpr unsigned N_per_thread = 16;
    unsigned p0 = hipBlockIdx_x * hipBlockDim_x * N_per_thread;
    unsigned p1 = hipThreadIdx_x * N_per_thread;
    unsigned O[N_per_thread] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };

    float dest[N_per_thread];

    // original logic in C
    for (unsigned i = 0; i < N_per_thread; ++i) {
      dest[i] = A_d[p0 + p1 + O[i]] + 1.0f;
    }

    // Store results back
    for (unsigned i = 0; i < N_per_thread; ++i) {
      A_d[p0 + p1 + O[i]] = dest[i];
    }
}

__device__
__global__ void vector_plus1_buffer_load_generic_unroll_2(float* A_d) {
    constexpr unsigned N_per_thread = 2;
    unsigned p0 = hipBlockIdx_x * hipBlockDim_x * N_per_thread;
    unsigned p1 = hipThreadIdx_x * N_per_thread;
    unsigned O[N_per_thread] = { 0, 1 };

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
    //  dest[i] = __buffer_load_dword_generic(A_d, p0, p1 * sizeof(float), O[i] * sizeof(float)) + 1.0f;
    //}

    // Use manually unrolled inline assembly
    unsigned O2[N_per_thread];
    for (unsigned i = 0; i < N_per_thread; ++i) {
      O2[i] = O[i] * sizeof(float);
    }
    __buffer_load_dword_generic_unroll_2(dest, A_d, p0, p1 * sizeof(float), O2);
    for (unsigned i = 0; i < N_per_thread; ++i) {
      dest[i] = dest[i] + 1.0f;
    }


    // Store results back
    for (unsigned i = 0; i < N_per_thread; ++i) {
      A_d[p0 + p1 + O[i]] = dest[i];
    }
}

__device__
__global__ void vector_plus1_buffer_load_generic_unroll_4(float* A_d) {
    constexpr unsigned N_per_thread = 4;
    unsigned p0 = hipBlockIdx_x * hipBlockDim_x * N_per_thread;
    unsigned p1 = hipThreadIdx_x * N_per_thread;
    unsigned O[N_per_thread] = { 0, 1, 2, 3 };

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
    //  dest[i] = __buffer_load_dword_generic(A_d, p0, p1 * sizeof(float), O[i] * sizeof(float)) + 1.0f;
    //}

    // Use manually unrolled inline assembly
    unsigned O2[N_per_thread];
    for (unsigned i = 0; i < N_per_thread; ++i) {
      O2[i] = O[i] * sizeof(float);
    }
    __buffer_load_dword_generic_unroll_4(dest, A_d, p0, p1 * sizeof(float), O2);
    for (unsigned i = 0; i < N_per_thread; ++i) {
      dest[i] = dest[i] + 1.0f;
    }


    // Store results back
    for (unsigned i = 0; i < N_per_thread; ++i) {
      A_d[p0 + p1 + O[i]] = dest[i];
    }
}

__device__
__global__ void vector_plus1_buffer_load_generic_unroll_16(float* A_d) {
    constexpr unsigned N_per_thread = 16;
    unsigned p0 = hipBlockIdx_x * hipBlockDim_x * N_per_thread;
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
    //  dest[i] = __buffer_load_dword_generic(A_d, p0, p1 * sizeof(float), O[i] * sizeof(float)) + 1.0f;
    //}

    // Use manually unrolled inline assembly
    unsigned O2[N_per_thread];
    for (unsigned i = 0; i < N_per_thread; ++i) {
      O2[i] = O[i] * sizeof(float);
    }
    __buffer_load_dword_generic_unroll_16(dest, A_d, p0, p1 * sizeof(float), O2);
    for (unsigned i = 0; i < N_per_thread; ++i) {
      dest[i] = dest[i] + 1.0f;
    }

    // Store results back
    for (unsigned i = 0; i < N_per_thread; ++i) {
      A_d[p0 + p1 + O[i]] = dest[i];
    }
}

__device__
__global__ void vector_plus1_buffer_store_generic_unroll_2(float* A_d) {
    constexpr unsigned N_per_thread = 2;
    unsigned p0 = hipBlockIdx_x * hipBlockDim_x * N_per_thread;
    unsigned p1 = hipThreadIdx_x * N_per_thread;
    unsigned O[N_per_thread] = { 0, 1 };

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
    //  dest[i] = __buffer_load_dword_generic(A_d, p0, p1 * sizeof(float), O[i] * sizeof(float)) + 1.0f;
    //}

    // Use manually unrolled inline assembly
    unsigned O2[N_per_thread];
    for (unsigned i = 0; i < N_per_thread; ++i) {
      O2[i] = O[i] * sizeof(float);
    }
    __buffer_load_dword_generic_unroll_2(dest, A_d, p0, p1 * sizeof(float), O2);
    for (unsigned i = 0; i < N_per_thread; ++i) {
      dest[i] = dest[i] + 1.0f;
    }
    __buffer_store_dword_generic_unroll_2(A_d, p0, p1 * sizeof(float), O2, dest);
}

__device__
__global__ void vector_plus1_buffer_store_generic_unroll_4(float* A_d) {
    constexpr unsigned N_per_thread = 4;
    unsigned p0 = hipBlockIdx_x * hipBlockDim_x * N_per_thread;
    unsigned p1 = hipThreadIdx_x * N_per_thread;
    unsigned O[N_per_thread] = { 0, 1, 2, 3 };

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
    //  dest[i] = __buffer_load_dword_generic(A_d, p0, p1 * sizeof(float), O[i] * sizeof(float)) + 1.0f;
    //}

    // Use manually unrolled inline assembly
    unsigned O2[N_per_thread];
    for (unsigned i = 0; i < N_per_thread; ++i) {
      O2[i] = O[i] * sizeof(float);
    }
    __buffer_load_dword_generic_unroll_4(dest, A_d, p0, p1 * sizeof(float), O2);
    for (unsigned i = 0; i < N_per_thread; ++i) {
      dest[i] = dest[i] + 1.0f;
    }
    __buffer_store_dword_generic_unroll_4(A_d, p0, p1 * sizeof(float), O2, dest);
}

__device__
__global__ void vector_plus1_buffer_store_generic_unroll_16(float* A_d) {
    constexpr unsigned N_per_thread = 16;
    unsigned p0 = hipBlockIdx_x * hipBlockDim_x * N_per_thread;
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
    //  dest[i] = __buffer_load_dword_generic(A_d, p0, p1 * sizeof(float), O[i] * sizeof(float)) + 1.0f;
    //}

    // Use manually unrolled inline assembly
    unsigned O2[N_per_thread];
    for (unsigned i = 0; i < N_per_thread; ++i) {
      O2[i] = O[i] * sizeof(float);
    }
    __buffer_load_dword_generic_unroll_16(dest, A_d, p0, p1 * sizeof(float), O2);
    for (unsigned i = 0; i < N_per_thread; ++i) {
      dest[i] = dest[i] + 1.0f;
    }
    __buffer_store_dword_generic_unroll_16(A_d, p0, p1 * sizeof(float), O2, dest);
}

__device__
__global__ void vector_plus1_buffer_load_generic_1(float* A_d) {
    constexpr unsigned N_per_thread = 1;
    unsigned p0 = hipBlockIdx_x * hipBlockDim_x * N_per_thread;
    unsigned p1 = hipThreadIdx_x * N_per_thread;
    unsigned O[N_per_thread] = { 0 };

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
      dest[i] = __buffer_load_dword_generic(A_d, p0, p1 * sizeof(float), O[i] * sizeof(float)) + 1.0f;
    }

    // Store results back
    for (unsigned i = 0; i < N_per_thread; ++i) {
      A_d[p0 + p1 + O[i]] = dest[i];
    }
}

__device__
__global__ void vector_plus1_buffer_load_generic_2(float* A_d) {
    constexpr unsigned N_per_thread = 2;
    unsigned p0 = hipBlockIdx_x * hipBlockDim_x * N_per_thread;
    unsigned p1 = hipThreadIdx_x * N_per_thread;
    unsigned O[N_per_thread] = { 0, 1 };

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
      dest[i] = __buffer_load_dword_generic(A_d, p0, p1 * sizeof(float), O[i] * sizeof(float)) + 1.0f;
    }

    // Store results back
    for (unsigned i = 0; i < N_per_thread; ++i) {
      A_d[p0 + p1 + O[i]] = dest[i];
    }
}

__device__
__global__ void vector_plus1_buffer_load_generic_4(float* A_d) {
    constexpr unsigned N_per_thread = 4;
    unsigned p0 = hipBlockIdx_x * hipBlockDim_x * N_per_thread;
    unsigned p1 = hipThreadIdx_x * N_per_thread;
    unsigned O[N_per_thread] = { 0, 1, 2, 3 };

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
      dest[i] = __buffer_load_dword_generic(A_d, p0, p1 * sizeof(float), O[i] * sizeof(float)) + 1.0f;
    }

    // Store results back
    for (unsigned i = 0; i < N_per_thread; ++i) {
      A_d[p0 + p1 + O[i]] = dest[i];
    }
}

__device__
__global__ void vector_plus1_buffer_load_generic(float* A_d) {
    constexpr unsigned N_per_thread = 16;
    unsigned p0 = hipBlockIdx_x * hipBlockDim_x * N_per_thread;
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
      dest[i] = __buffer_load_dword_generic(A_d, p0, p1 * sizeof(float), O[i] * sizeof(float)) + 1.0f;
    }

    // Store results back
    for (unsigned i = 0; i < N_per_thread; ++i) {
      A_d[p0 + p1 + O[i]] = dest[i];
    }
}

__device__
__global__ void vector_plus1_buffer_store_generic(float* A_d) {
    constexpr unsigned N_per_thread = 16;
    unsigned p0 = hipBlockIdx_x * hipBlockDim_x * N_per_thread;
    unsigned p1 = hipThreadIdx_x * N_per_thread;
    unsigned O[N_per_thread] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };

    //float dest[N_per_thread];

    // original logic in C
    //for (unsigned i = 0; i < N_per_thread; ++i) {
    //  dest[i] = A_d[p0 + p1 + O[i]] + 1.0f;
    //}

    // Use inline assembly
    // Undesirable effect: s_waitcnt 0 after every load

    // __buffer_load_dword_genric(ptr, offset) = *(ptr + offset)
    for (unsigned i = 0; i < N_per_thread; ++i) {
      float result = __buffer_load_dword_generic(A_d, p0, p1 * sizeof(float), O[i] * sizeof(float)) + 1.0f;
      __buffer_store_dword_generic(A_d, p0, p1 * sizeof(float), O[i] * sizeof(float), result);
    }
}

__device__
void __global_load_dwordx4_unroll_2(floatx4* dest, floatx4* ptr, unsigned long long* O) {
  asm volatile("\n \
    global_load_dwordx4 %0, %2, %4, offset:0 \n \
    global_load_dwordx4 %1, %3, %4, offset:0 \n \
    s_waitcnt 0 \n \
    " : "=v"(dest[0]), "=v"(dest[1])
      : "v"(O[0]), "v"(O[1]), "s"(ptr));
}

__device__
void __global_load_dwordx4_unroll_4(floatx4* dest, floatx4* ptr, unsigned long long* O) {
  asm volatile("\n \
    global_load_dwordx4 %0, %4, %8, offset:0 \n \
    global_load_dwordx4 %1, %5, %8, offset:0 \n \
    global_load_dwordx4 %2, %6, %8, offset:0 \n \
    global_load_dwordx4 %3, %7, %8, offset:0 \n \
    s_waitcnt 0 \n \
    " : "=v"(dest[0]), "=v"(dest[1]), "=v"(dest[2]), "=v"(dest[3])
      : "v"(O[0]), "v"(O[1]), "v"(O[2]), "v"(O[3]), "s"(ptr));
}

//__global_load_dword_unroll_16(dest, A_d, O);
__device__
void __global_load_dword_unroll_16(float* dest, float* ptr, unsigned long long* O) {
  asm volatile("\n \
    global_load_dword %0, %16, %32, offset:0 \n \
    global_load_dword %1, %17, %32, offset:0 \n \
    global_load_dword %2, %18, %32, offset:0 \n \
    global_load_dword %3, %19, %32, offset:0 \n \
    global_load_dword %4, %20, %32, offset:0 \n \
    global_load_dword %5, %21, %32, offset:0 \n \
    global_load_dword %6, %22, %32, offset:0 \n \
    global_load_dword %7, %23, %32, offset:0 \n \
    global_load_dword %8, %24, %32, offset:0 \n \
    global_load_dword %9, %25, %32, offset:0 \n \
    global_load_dword %10, %26, %32, offset:0 \n \
    global_load_dword %11, %27, %32, offset:0 \n \
    global_load_dword %12, %28, %32, offset:0 \n \
    global_load_dword %13, %29, %32, offset:0 \n \
    global_load_dword %14, %30, %32, offset:0 \n \
    global_load_dword %15, %31, %32, offset:0 \n \
    s_waitcnt 0 \n \
    " : "=v"(dest[0]), "=v"(dest[1]), "=v"(dest[2]), "=v"(dest[3]), "=v"(dest[4]), "=v"(dest[5]), "=v"(dest[6]), "=v"(dest[7]), "=v"(dest[8]), "=v"(dest[9]), "=v"(dest[10]), "=v"(dest[11]), "=v"(dest[12]), "=v"(dest[13]), "=v"(dest[14]), "=v"(dest[15])
      : "v"(O[0]), "v"(O[1]), "v"(O[2]), "v"(O[3]), "v"(O[4]), "v"(O[5]), "v"(O[6]), "v"(O[7]), "v"(O[8]), "v"(O[9]), "v"(O[10]), "v"(O[11]), "v"(O[12]), "v"(O[13]), "v"(O[14]), "v"(O[15]), "s"(ptr));
}

__device__
__global__ void vector_plus1_global_load_unroll_16(float* A_d) {
    constexpr unsigned N_per_thread = 16;
    unsigned p0 = hipBlockIdx_x * hipBlockDim_x * N_per_thread;
    unsigned p1 = hipThreadIdx_x * N_per_thread;
    unsigned O[N_per_thread] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };

    float dest[N_per_thread];

    // original logic in C
    //for (unsigned i = 0; i < N_per_thread; ++i) {
    //  dest[i] = A_d[p0 + p1 + O[i]] + 1.0f;
    //}

    // Use inline assembly
    // Undesirable effect: s_waitcnt 0 after every load
    //unsigned baseOffset = p0 + p1;
    //for (unsigned i = 0; i < N_per_thread; ++i) {
    //  dest[i] = __global_load_dword(A_d, (baseOffset + O[i]) * sizeof(float)) + 1.0f;
    //}

    // Use manually unrolled inline assembly
    unsigned long long O2[N_per_thread];
    for (unsigned i = 0; i < N_per_thread; ++i) {
      O2[i] = (p0 + p1 + O[i]) * sizeof(float);
    }
    __global_load_dword_unroll_16(dest, A_d, O2);
    for (unsigned i = 0; i < N_per_thread; ++i) {
      dest[i] = dest[i] + 1.0f;
    }

    // Store results back
    for (unsigned i = 0; i < N_per_thread; ++i) {
      A_d[p0 + p1 + O[i]] = dest[i];
    }
}

__device__
void __global_store_dwordx4_unroll_2(floatx4* ptr, unsigned long long* O, floatx4* value) {
  asm volatile("\n \
    global_store_dwordx4 %0, %2, %4, offset:0 \n \
    global_store_dwordx4 %1, %3, %4, offset:0 \n \
    s_waitcnt 0 \n \
    " :
      : "v"(O[0]), "v"(O[1]), "v"(value[0]), "v"(value[1]), "s"(ptr));
}

__device__
void __global_store_dwordx4_unroll_4(floatx4* ptr, unsigned long long* O, floatx4* value) {
  asm volatile("\n \
    global_store_dwordx4 %0, %4, %8, offset:0 \n \
    global_store_dwordx4 %1, %5, %8, offset:0 \n \
    global_store_dwordx4 %2, %6, %8, offset:0 \n \
    global_store_dwordx4 %3, %7, %8, offset:0 \n \
    s_waitcnt 0 \n \
    " :
      : "v"(O[0]), "v"(O[1]), "v"(O[2]), "v"(O[3]), "v"(value[0]), "v"(value[1]), "v"(value[2]), "v"(value[3]), "s"(ptr));
}

// ptr[offset] = value
__device__
void __global_store_dword_unroll_16(float* ptr, unsigned long long* O, float* value) {
  asm volatile("\n \
    global_store_dword %0, %16, %32, offset:0 \n \
    global_store_dword %1, %17, %32, offset:0 \n \
    global_store_dword %2, %18, %32, offset:0 \n \
    global_store_dword %3, %19, %32, offset:0 \n \
    global_store_dword %4, %20, %32, offset:0 \n \
    global_store_dword %5, %21, %32, offset:0 \n \
    global_store_dword %6, %22, %32, offset:0 \n \
    global_store_dword %7, %23, %32, offset:0 \n \
    global_store_dword %8, %24, %32, offset:0 \n \
    global_store_dword %9, %25, %32, offset:0 \n \
    global_store_dword %10, %26, %32, offset:0 \n \
    global_store_dword %11, %27, %32, offset:0 \n \
    global_store_dword %12, %28, %32, offset:0 \n \
    global_store_dword %13, %29, %32, offset:0 \n \
    global_store_dword %14, %30, %32, offset:0 \n \
    global_store_dword %15, %31, %32, offset:0 \n \
    s_waitcnt 0 \n \
    " :
      : "v"(O[0]), "v"(O[1]), "v"(O[2]), "v"(O[3]), "v"(O[4]), "v"(O[5]), "v"(O[6]), "v"(O[7]), "v"(O[8]), "v"(O[9]), "v"(O[10]), "v"(O[11]), "v"(O[12]), "v"(O[13]), "v"(O[14]), "v"(O[15]), "v"(value[0]), "v"(value[1]), "v"(value[2]), "v"(value[3]), "v"(value[4]), "v"(value[5]), "v"(value[6]), "v"(value[7]), "v"(value[8]), "v"(value[9]), "v"(value[10]), "v"(value[11]), "v"(value[12]), "v"(value[13]), "v"(value[14]), "v"(value[15]), "s"(ptr));
}

__device__
__global__ void vector_plus1_global_storex4_unroll_2(floatx4* A_d) {
    constexpr unsigned N_per_thread = 2;
    unsigned p0 = hipBlockIdx_x * hipBlockDim_x * N_per_thread;
    unsigned p1 = hipThreadIdx_x * N_per_thread;
    unsigned O[N_per_thread] = { 0, 1 };

    floatx4 dest[N_per_thread];

    // original logic in C
    //for (unsigned i = 0; i < N_per_thread; ++i) {
    //  dest[i] = A_d[p0 + p1 + O[i]] + 1.0f;
    //}

    // Use inline assembly
    // Undesirable effect: s_waitcnt 0 after every load
    //unsigned baseOffset = p0 + p1;
    //for (unsigned i = 0; i < N_per_thread; ++i) {
    //  dest[i] = __global_load_dword(A_d, (baseOffset + O[i]) * sizeof(float)) + 1.0f;
    //}

    // Use manually unrolled inline assembly
    unsigned long long O2[N_per_thread];
    for (unsigned i = 0; i < N_per_thread; ++i) {
      O2[i] = (p0 + p1 + O[i]) * sizeof(floatx4);
    }
    __global_load_dwordx4_unroll_2(dest, A_d, O2);
    for (unsigned i = 0; i < N_per_thread; ++i) {
      dest[i] = dest[i] + 1.0f;
    }
    __global_store_dwordx4_unroll_2(A_d, O2, dest);
}

__device__
__global__ void vector_plus1_global_storex4_unroll_4(floatx4* A_d) {
    constexpr unsigned N_per_thread = 4;
    unsigned p0 = hipBlockIdx_x * hipBlockDim_x * N_per_thread;
    unsigned p1 = hipThreadIdx_x * N_per_thread;
    unsigned O[N_per_thread] = { 0, 1, 2, 3 };

    floatx4 dest[N_per_thread];

    // original logic in C
    //for (unsigned i = 0; i < N_per_thread; ++i) {
    //  dest[i] = A_d[p0 + p1 + O[i]] + 1.0f;
    //}

    // Use inline assembly
    // Undesirable effect: s_waitcnt 0 after every load
    //unsigned baseOffset = p0 + p1;
    //for (unsigned i = 0; i < N_per_thread; ++i) {
    //  dest[i] = __global_load_dword(A_d, (baseOffset + O[i]) * sizeof(float)) + 1.0f;
    //}

    // Use manually unrolled inline assembly
    unsigned long long O2[N_per_thread];
    for (unsigned i = 0; i < N_per_thread; ++i) {
      O2[i] = (p0 + p1 + O[i]) * sizeof(floatx4);
    }
    __global_load_dwordx4_unroll_4(dest, A_d, O2);
    for (unsigned i = 0; i < N_per_thread; ++i) {
      dest[i] = dest[i] + 1.0f;
    }
    __global_store_dwordx4_unroll_4(A_d, O2, dest);
}

__device__
__global__ void vector_plus1_global_store_unroll_16(float* A_d) {
    constexpr unsigned N_per_thread = 16;
    unsigned p0 = hipBlockIdx_x * hipBlockDim_x * N_per_thread;
    unsigned p1 = hipThreadIdx_x * N_per_thread;
    unsigned O[N_per_thread] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };

    float dest[N_per_thread];

    // original logic in C
    //for (unsigned i = 0; i < N_per_thread; ++i) {
    //  dest[i] = A_d[p0 + p1 + O[i]] + 1.0f;
    //}

    // Use inline assembly
    // Undesirable effect: s_waitcnt 0 after every load
    //unsigned baseOffset = p0 + p1;
    //for (unsigned i = 0; i < N_per_thread; ++i) {
    //  dest[i] = __global_load_dword(A_d, (baseOffset + O[i]) * sizeof(float)) + 1.0f;
    //}

    // Use manually unrolled inline assembly
    unsigned long long O2[N_per_thread];
    for (unsigned i = 0; i < N_per_thread; ++i) {
      O2[i] = (p0 + p1 + O[i]) * sizeof(float);
    }
    __global_load_dword_unroll_16(dest, A_d, O2);
    for (unsigned i = 0; i < N_per_thread; ++i) {
      dest[i] = dest[i] + 1.0f;
    }
    __global_store_dword_unroll_16(A_d, O2, dest);
}

template<typename T>
void launchTestKernel(void (*kernel)(T*), size_t N, unsigned threadsPerBlock, unsigned unrollFactor, unsigned vectorizeFactor, T* ptr) {
    unsigned blocks = N/threadsPerBlock/unrollFactor/vectorizeFactor;

    //printf("N: %zu unrollFactor: %u vectorizeFactor: %u threadsPerBlock: %u blocks: %u\n", N, unrollFactor, vectorizeFactor, threadsPerBlock, blocks);
    hipLaunchKernelGGL(kernel, dim3(blocks), dim3(threadsPerBlock), 0, 0, ptr);
}

int main(int argc, char* argv[]) {
    float *A_d, *C_d;
    float *A_h, *C_h;
    size_t N = 1024 * 1024 * 1024;
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
    // Fill with 1.0f + i
    for (size_t i = 0; i < N; i++) {
        A_h[i] = 1.0f + i;
        C_h[i] = A_h[i];
    }

    printf("info: allocate device mem (%6.2f MB)\n", Nbytes / 1024.0 / 1024.0);
    CHECK(hipMalloc(&A_d, Nbytes));

    printf("info: copy Host2Device %p->%p %zu bytes\n", A_h, A_d, Nbytes);
    CHECK(hipMemcpy(A_d, A_h, Nbytes, hipMemcpyHostToDevice));

// benchmarking
for (int i = 0; i < 1024; ++i) {
#if 1
    //printf("info: launch 'vector_plus1_naive' kernel\n");
    launchTestKernel(vector_plus1_naive, N, 256, 1, 1, A_d);
#endif

#if 0
    //printf("info: launch 'vector_plus1_naive_unroll_16' kernel\n");
    launchTestKernel(vector_plus1_naive_unroll_16, N, 256, 16, 1, A_d);
#endif

#if 0
    //printf("info: launch 'vector_plus1_buffer_load' kernel\n");
    launchTestKernel(vector_plus1_buffer_load, N, 256, 1, 1, A_d);
#endif

#if 0
    //printf("info: launch 'vector_plus1_global_load' kernel\n");
    launchTestKernel(vector_plus1_global_load, N, 256, 1, 1, A_d);
#endif
    
#if 0
    // slower than other
    //printf("info: launch 'vector_plus1_global_loadx4' kernel\n");
    launchTestKernel<floatx4>(vector_plus1_global_loadx4, N, 256, 1, 4, reinterpret_cast<floatx4*>(A_d));
#endif
    
#if 0
    //printf("info: launch 'vector_plus1_buffer_load_generic_1' kernel\n");
    launchTestKernel(vector_plus1_buffer_load_generic_1, N, 256, 1, 1, A_d);
#endif

#if 0
    //printf("info: launch 'vector_plus1_buffer_load_generic_2' kernel\n");
    launchTestKernel(vector_plus1_buffer_load_generic_2, N, 256, 2, 1, A_d);
#endif

#if 0
    //printf("info: launch 'vector_plus1_buffer_load_generic_4' kernel\n");
    launchTestKernel(vector_plus1_buffer_load_generic_4, N, 256, 4, 1, A_d);
#endif

#if 0
    // slow compared to others
    //printf("info: launch 'vector_plus1_buffer_load_generic' kernel\n");
    launchTestKernel(vector_plus1_buffer_load_generic, N, 256, 16, 1, A_d);
#endif

#if 0
    //printf("info: launch 'vector_plus1_buffer_load_generic_unroll_2' kernel\n");
    launchTestKernel(vector_plus1_buffer_load_generic_unroll_2, N, 256, 2, 1, A_d);
#endif

#if 0
    // XXX TBD
    // looks like mixing buffer_load_dword and global_store_dwordx4 would have issue.
    //printf("info: launch 'vector_plus1_buffer_load_generic_unroll_4' kernel\n");
    launchTestKernel(vector_plus1_buffer_load_generic_unroll_4, N, 256, 4, 1, A_d);
#endif

#if 0
    // XXX TBD
    // looks like mixing buffer_load_dword and global_store_dwordx4 would have issue.
    //printf("info: launch 'vector_plus1_buffer_load_generic_unroll_16' kernel\n");
    launchTestKernel(vector_plus1_buffer_load_generic_unroll_16, N, 256, 16, 1, A_d);
#endif

#if 0
    // XXX TBD
    // looks like mixing global_load_dword and global_store_dwordx4 would have issue.
    //printf("info: launch 'vector_plus1_global_load_unroll_16' kernel\n");
    launchTestKernel(vector_plus1_global_load_unroll_16, N, 256, 16, 1, A_d);
#endif

#if 0
    //printf("info: launch 'vector_plus1_buffer_store' kernel\n");
    launchTestKernel(vector_plus1_buffer_store, N, 256, 1, 1, A_d);
#endif

#if 0
    //printf("info: launch 'vector_plus1_global_store' kernel\n");
    launchTestKernel(vector_plus1_global_store, N, 256, 1, 1, A_d);
#endif

#if 0
    //printf("info: launch 'vector_plus1_global_storex4' kernel\n");
    launchTestKernel<floatx4>(vector_plus1_global_storex4, N, 256, 1, 4, reinterpret_cast<floatx4*>(A_d));
#endif
    
#if 0
    // slow compared to others
    //printf("info: launch 'vector_plus1_buffer_store_generic' kernel\n");
    launchTestKernel(vector_plus1_buffer_store_generic, N, 256, 16, 1, A_d);
#endif

#if 0
    //printf("info: launch 'vector_plus1_buffer_store_generic_unroll_2' kernel\n");
    launchTestKernel(vector_plus1_buffer_store_generic_unroll_2, N, 256, 2, 1, A_d);
#endif

#if 0
    //printf("info: launch 'vector_plus1_buffer_store_generic_unroll_4' kernel\n");
    launchTestKernel(vector_plus1_buffer_store_generic_unroll_4, N, 256, 4, 1, A_d);
#endif

#if 0
    // slow compared to others
    //printf("info: launch 'vector_plus1_buffer_store_generic_unroll_16' kernel\n");
    launchTestKernel(vector_plus1_buffer_store_generic_unroll_16, N, 256, 16, 1, A_d);
#endif

#if 0
    // slow compared to others
    //printf("info: launch 'vector_plus1_global_store_unroll_16' kernel\n");
    launchTestKernel(vector_plus1_global_store_unroll_16, N, 256, 16, 1, A_d);
#endif

#if 0
    //printf("info: launch 'vector_plus1_global_storex4_unroll_2' kernel\n");
    launchTestKernel<floatx4>(vector_plus1_global_storex4_unroll_2, N, 256, 2, 4, reinterpret_cast<floatx4*>(A_d));
#endif

#if 0
    //printf("info: launch 'vector_plus1_global_storex4_unroll_4' kernel\n");
    launchTestKernel<floatx4>(vector_plus1_global_storex4_unroll_4, N, 256, 4, 4, reinterpret_cast<floatx4*>(A_d));
#endif

}

    hipStreamSynchronize(nullptr);

    printf("info: copy Device2Host %p->%p %zu bytes\n", A_d, A_h, Nbytes);
    CHECK(hipMemcpy(A_h, A_d, Nbytes, hipMemcpyDeviceToHost));

    printf("info: check result\n");
    bool passed = true;
    for (size_t i = 0; i < N; i++) {
        //printf("%zu %f %f\n", i, A_h[i], C_h[i]);
        if (A_h[i] != C_h[i] + 1.0f) {
            //CHECK(hipErrorUnknown);
            printf("INCORRECT %zu %f %f\n", i, A_h[i], C_h[i]);
            passed = false;
            break;
        }
    }
    if (passed)
      printf("PASSED!\n");

    hipFree(A_d);
    return 0;
}
