#define _POSIX_C_SOURCE 200809L

#include <stdatomic.h>
#include <stdint.h>
#include <time.h>

static uint64_t monotonic_nanoseconds(void) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (uint64_t)now.tv_sec * UINT64_C(1000000000) + (uint64_t)now.tv_nsec;
}

static uint64_t alu_step(uint64_t state, uint64_t index) {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state * UINT64_C(0xd1342543de82ef95) + index;
}

uint64_t hyperloader_alu_spin(_Atomic uint32_t *stop) {
    uint64_t state = UINT64_C(0x9e3779b97f4a7c15);
    uint64_t iterations = 0;
    while (atomic_load_explicit(stop, memory_order_relaxed) == 0) {
        for (uint32_t index = 0; index < 65536; ++index) {
            state = alu_step(state, index);
        }
        iterations += 65536;
    }
    return state ^ iterations;
}

uint64_t hyperloader_alu_pulse(
    _Atomic uint32_t *stop,
    uint64_t active_nanoseconds,
    uint64_t period_nanoseconds
) {
    if (active_nanoseconds == 0 || active_nanoseconds > period_nanoseconds) {
        return 0;
    }
    uint64_t state = UINT64_C(0x9e3779b97f4a7c15);
    uint64_t iterations = 0;
    uint64_t period_start = monotonic_nanoseconds();
    while (atomic_load_explicit(stop, memory_order_relaxed) == 0) {
        const uint64_t active_until = period_start + active_nanoseconds;
        while (monotonic_nanoseconds() < active_until) {
            state = alu_step(state, iterations);
            ++iterations;
        }
        period_start += period_nanoseconds;
        const struct timespec wake_at = {
            .tv_sec = (time_t)(period_start / UINT64_C(1000000000)),
            .tv_nsec = (long)(period_start % UINT64_C(1000000000)),
        };
        clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &wake_at, NULL);
    }
    return state ^ iterations;
}

void hyperloader_alu_store_stop(_Atomic uint32_t *stop, uint32_t value) {
    atomic_store_explicit(stop, value, memory_order_relaxed);
}
