#include "csrc/engine/sequence.hpp"
#include <pybind11/stl.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("pages", [](const Sequence& seq, int count) {
        TORCH_CHECK(count >= 0 && static_cast<size_t>(count) <= seq.cache.block_table.size());
        return std::vector<int>(seq.cache.block_table.begin(), seq.cache.block_table.begin() + count);
    });
}
