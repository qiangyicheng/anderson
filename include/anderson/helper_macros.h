#pragma once

#define ANDERSON_OMP_SIMD_ALIGNMENT 64
#define ANDERSON_OMP_SIMD_LEN 8
#define ANDERSON_OMP_SIMD_ALIGNED(...) simd simdlen(ANDERSON_OMP_SIMD_LEN) aligned(__VA_ARGS__:ANDERSON_OMP_SIMD_ALIGNMENT)
#define ANDERSON_OMP_SUM(...) reduction(+:__VA_ARGS__)
#define ANDERSON_OMP_MAX(...) reduction(max:__VA_ARGS__)
#define ANDERSON_OMP_SCHEDULE schedule(static)
#define ANDERSON_CLASS_CTORS_ALL_DEFAULT(name) \
    name(const name &) = default; \
    name(name &&) = default; \
    name &operator=(const name &) = default; \
    name &operator=(name &&) = default;
