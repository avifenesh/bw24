// Ground-truth logit dumper: feed explicit token IDs, print argmax + top-k of the
// last-token logits. Used to validate bw24-engine's forward against llama.cpp exactly
// (same token IDs => no tokenizer ambiguity).
//
// build: see tools/build_llama_logits.sh
#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <algorithm>

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <model.gguf> <tok0> [tok1 ...]\n", argv[0]); return 1; }
    const char* model_path = argv[1];
    std::vector<llama_token> tokens;
    for (int i = 2; i < argc; i++) tokens.push_back((llama_token)atoi(argv[i]));

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    // n_gpu_layers from env LLG_NGL (default 999). For models too big for 24GB (e.g. 35B MoE),
    // set LLG_NGL=0 to run on CPU and get ground-truth argmax for comparison.
    const char* ngl_env = getenv("LLG_NGL");
    mp.n_gpu_layers = ngl_env ? atoi(ngl_env) : 999;
    llama_model* model = llama_model_load_from_file(model_path, mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }
    const llama_vocab* vocab = llama_model_get_vocab(model);
    int n_vocab = llama_vocab_n_tokens(vocab);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = tokens.size() + 8;
    cp.n_batch = tokens.size() + 8;
    llama_context* ctx = llama_init_from_model(model, cp);

    llama_batch batch = llama_batch_get_one(tokens.data(), (int)tokens.size());
    if (llama_decode(ctx, batch)) { fprintf(stderr, "decode failed\n"); return 1; }

    const float* logits = llama_get_logits_ith(ctx, (int)tokens.size() - 1);

    // argmax + top-5
    std::vector<int> idx(n_vocab);
    for (int i = 0; i < n_vocab; i++) idx[i] = i;
    std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(),
                      [&](int a, int b){ return logits[a] > logits[b]; });

    printf("argmax=%d logit=%.5f\n", idx[0], logits[idx[0]]);
    printf("top5:");
    for (int i = 0; i < 5; i++) printf(" (%d,%.5f)", idx[i], logits[idx[i]]);
    printf("\n");
    // dump a fixed set of ids for cross-check (first 8 vocab logits)
    printf("logit[0..8]:");
    for (int i = 0; i < 8 && i < n_vocab; i++) printf(" %.5f", logits[i]);
    printf("\n");

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
