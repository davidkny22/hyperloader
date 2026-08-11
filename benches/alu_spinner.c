#include <stdatomic.h>
#include <stdint.h>

uint64_t hyperloader_alu_spin(_Atomic uint32_t *stop) {
    uint64_t state = UINT64_C(0x9e3779b97f4a7c15);
    uint64_t iterations = 0;
    while (atomic_load_explicit(stop, memory_order_relaxed) == 0) {
        for (uint32_t index = 0; index < 65536; ++index) {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state = state * UINT64_C(0xd1342543de82ef95) + index;
        }
        iterations += 65536;
    }
    return state ^ iterations;
}

void hyperloader_alu_store_stop(_Atomic uint32_t *stop, uint32_t value) {
    atomic_store_explicit(stop, value, memory_order_relaxed);
}
