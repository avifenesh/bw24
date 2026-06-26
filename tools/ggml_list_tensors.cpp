// List tensors in a GGUF, optionally filtered to a ggml type name.
// Prints: name  type  ne0xne1[xne2]  n_elems  type_size
// Build: g++ -O2 -std=c++17 ggml_list_tensors.cpp -o ggml_list_tensors \
//        -I<llama>/include -I<llama>/ggml/include -L<llama>/build/bin -lllama -lggml -lggml-base
#include "ggml.h"
#include "gguf.h"
#include <cstdio>
#include <cstring>
#include <string>

static const char* type_name(enum ggml_type t) {
    return ggml_type_name(t);
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s gguf [TYPE_NAME_FILTER]\n", argv[0]); return 1; }
    const char* filter = (argc >= 3) ? argv[2] : nullptr;

    struct gguf_init_params p = { /*no_alloc=*/true, /*ctx=*/nullptr };
    struct gguf_context* c = gguf_init_from_file(argv[1], p);
    if (!c) { fprintf(stderr, "gguf load fail: %s\n", argv[1]); return 1; }

    int64_t nt = gguf_get_n_tensors(c);
    for (int64_t i = 0; i < nt; ++i) {
        enum ggml_type ty = gguf_get_tensor_type(c, i);
        const char* tn = type_name(ty);
        if (filter && strcmp(tn, filter) != 0) continue;
        const char* name = gguf_get_tensor_name(c, i);
        size_t sz = gguf_get_tensor_size(c, i);
        printf("%-44s %-8s size=%zu type_size=%zu blck=%ld\n",
               name, tn, sz, ggml_type_size(ty), (long)ggml_blck_size(ty));
    }
    gguf_free(c);
    return 0;
}
