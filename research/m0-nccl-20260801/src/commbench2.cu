// m0-nccl commbench2: extension — CUDA-graph-captured and peer-copy all-to-all variants,
// plus graph-captured ping-pongs (launch-free hop floor).
//
// Modes:
//   commbench2 ga2a  <d0> <d1> [...]   NCCL a2a captured in a CUDA graph (1 a2a/graph), replayed
//   commbench2 pa2a  <d0> <d1> [...]   peer-copy a2a (cudaMemcpyPeerAsync, event-barrier per iter)
//   commbench2 gpa2a <d0> <d1> [...]   peer-copy a2a captured in a CUDA graph, replayed
//   commbench2 gppp  <devA> <devB>     peer ping-pong, 50 round trips per graph, replayed
//   commbench2 gpp   <devA> <devB>     NCCL ping-pong, 50 round trips per graph, replayed

#include <nccl.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CHECK_CUDA(cmd) do { cudaError_t _e = (cmd); if (_e != cudaSuccess) { \
  fprintf(stderr, "CUDA error %s:%d '%s'\n", __FILE__, __LINE__, cudaGetErrorString(_e)); exit(1); } } while(0)
#define CHECK_NCCL(cmd) do { ncclResult_t _r = (cmd); if (_r != ncclSuccess) { \
  fprintf(stderr, "NCCL error %s:%d '%s'\n", __FILE__, __LINE__, ncclGetErrorString(_r)); exit(1); } } while(0)

static double now_ms(void) {
  struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

static const size_t PAIR_SIZES[] = {4096, 65536, 1048576, 16777216, 268435456};
static const int N_PAIR_SIZES = 5;
static const size_t A2A_SIZES[] = {65536, 262144, 1048576, 4194304, 16777216};
static const int N_A2A_SIZES = 5;
static const int REPS = 5;
static int g_ncclver = 0;

// ---------- shared a2a state ----------
static int n_, devs_[8];
static ncclComm_t comms_[8];
static cudaStream_t streams_[8];
static char *sbuf_[8], *rbuf_[8];
static cudaEvent_t evA_[8], evB_[8];

static void a2a_init(int n, int* devs, int want_nccl) {
  n_ = n; memcpy(devs_, devs, n * sizeof(int));
  size_t maxp = A2A_SIZES[N_A2A_SIZES-1];
  if (want_nccl) CHECK_NCCL(ncclCommInitAll(comms_, n, devs));
  for (int i = 0; i < n; i++) {
    CHECK_CUDA(cudaSetDevice(devs[i]));
    CHECK_CUDA(cudaStreamCreateWithFlags(&streams_[i], cudaStreamNonBlocking));
    CHECK_CUDA(cudaMalloc((void**)&sbuf_[i], maxp * n));
    CHECK_CUDA(cudaMalloc((void**)&rbuf_[i], maxp * n));
    CHECK_CUDA(cudaMemset(sbuf_[i], 0x11 * (i+1), maxp * n));
    CHECK_CUDA(cudaEventCreateWithFlags(&evA_[i], cudaEventDisableTiming));
    CHECK_CUDA(cudaEventCreateWithFlags(&evB_[i], cudaEventDisableTiming));
    CHECK_CUDA(cudaDeviceSynchronize());
    if (!want_nccl) {
      for (int j = 0; j < n; j++) if (i != j) {
        cudaError_t e = cudaDeviceEnablePeerAccess(devs[j], 0);
        if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) CHECK_CUDA(e);
        cudaGetLastError();
      }
    }
  }
}

static void a2a_sync_all(void) {
  for (int i = 0; i < n_; i++) { CHECK_CUDA(cudaSetDevice(devs_[i])); CHECK_CUDA(cudaStreamSynchronize(streams_[i])); }
}

static void nccl_a2a_once(size_t sz) {
  CHECK_NCCL(ncclGroupStart());
  for (int i = 0; i < n_; i++) for (int j = 0; j < n_; j++) if (i != j) {
    CHECK_NCCL(ncclSend(sbuf_[i] + (size_t)j * sz, sz, ncclChar, j, comms_[i], streams_[i]));
    CHECK_NCCL(ncclRecv(rbuf_[i] + (size_t)j * sz, sz, ncclChar, j, comms_[i], streams_[i]));
  }
  CHECK_NCCL(ncclGroupEnd());
}

static void peer_a2a_enqueue(size_t sz) {
  // UVA peer copy: capturable into CUDA graphs (cudaMemcpyPeerAsync is not)
  for (int i = 0; i < n_; i++) for (int j = 0; j < n_; j++) if (i != j)
    CHECK_CUDA(cudaMemcpyAsync(rbuf_[j] + (size_t)i * sz,
                               sbuf_[i] + (size_t)j * sz, sz, cudaMemcpyDefault, streams_[i]));
}

// fork all streams from streams_[0], run body, join back to streams_[0]
static cudaGraphExec_t capture_a2a(void (*body)(size_t), size_t sz) {
  cudaGraph_t graph;
  CHECK_CUDA(cudaSetDevice(devs_[0]));
  CHECK_CUDA(cudaStreamBeginCapture(streams_[0], cudaStreamCaptureModeThreadLocal));
  CHECK_CUDA(cudaEventRecord(evA_[0], streams_[0]));
  for (int i = 1; i < n_; i++) CHECK_CUDA(cudaStreamWaitEvent(streams_[i], evA_[0], 0));
  body(sz);
  for (int i = 1; i < n_; i++) {
    CHECK_CUDA(cudaEventRecord(evB_[i], streams_[i]));
    CHECK_CUDA(cudaStreamWaitEvent(streams_[0], evB_[i], 0));
  }
  CHECK_CUDA(cudaStreamEndCapture(streams_[0], &graph));
  cudaGraphExec_t exec;
  CHECK_CUDA(cudaGraphInstantiate(&exec, graph, NULL, NULL, 0));
  CHECK_CUDA(cudaGraphDestroy(graph));
  return exec;
}

static void run_graph_a2a(const char* name, void (*body)(size_t), int n, int* devs, int want_nccl) {
  a2a_init(n, devs, want_nccl);
  for (int si = 0; si < N_A2A_SIZES; si++) {
    size_t sz = A2A_SIZES[si];
    // uncaptured warmup (NCCL lazy init must happen outside capture)
    for (int w = 0; w < 10; w++) body(sz);
    a2a_sync_all();
    cudaGraphExec_t exec = capture_a2a(body, sz);
    CHECK_CUDA(cudaSetDevice(devs_[0]));
    for (int w = 0; w < 10; w++) CHECK_CUDA(cudaGraphLaunch(exec, streams_[0]));
    CHECK_CUDA(cudaStreamSynchronize(streams_[0]));
    int iters = 100;
    for (int rep = 0; rep < REPS; rep++) {
      double t0 = now_ms();
      for (int it = 0; it < iters; it++) CHECK_CUDA(cudaGraphLaunch(exec, streams_[0]));
      CHECK_CUDA(cudaStreamSynchronize(streams_[0]));
      double t1 = now_ms();
      double per_us = (t1 - t0) * 1000.0 / (double)iters;
      double aggbw = (double)n_ * (double)(n_-1) * (double)sz * (double)iters / ((t1 - t0) * 1e-3) / 1e9;
      printf("{\"test\":\"%s\",\"impl\":\"%s\",\"ngpus\":%d,\"devs\":[", name, want_nccl ? "nccl-graph" : "peer-graph", n_);
      for (int i = 0; i < n_; i++) printf("%d%s", devs_[i], i < n_-1 ? "," : "");
      printf("],\"size_bytes_per_peer\":%zu,\"iters\":%d,\"rep\":%d,\"total_ms\":%.3f,"
             "\"per_a2a_us\":%.3f,\"agg_bw_GBps\":%.2f,\"nccl\":%d}\n",
             sz, iters, rep, t1-t0, per_us, aggbw, g_ncclver);
    }
    CHECK_CUDA(cudaGraphExecDestroy(exec));
  }
}

static void run_peer_a2a(int n, int* devs) {  // eager, event barrier per iter
  a2a_init(n, devs, 0);
  for (int si = 0; si < N_A2A_SIZES; si++) {
    size_t sz = A2A_SIZES[si];
    int iters = 100;
    for (int w = 0; w < 10; w++) peer_a2a_enqueue(sz);
    a2a_sync_all();
    for (int rep = 0; rep < REPS; rep++) {
      double t0 = now_ms();
      for (int it = 0; it < iters; it++) {
        cudaEvent_t* cur = (it & 1) ? evB_ : evA_;
        cudaEvent_t* prev = (it & 1) ? evA_ : evB_;
        if (it > 0)
          for (int i = 0; i < n_; i++) for (int j = 0; j < n_; j++) if (i != j)
            CHECK_CUDA(cudaStreamWaitEvent(streams_[i], prev[j], 0));
        peer_a2a_enqueue(sz);
        for (int i = 0; i < n_; i++) CHECK_CUDA(cudaEventRecord(cur[i], streams_[i]));
      }
      a2a_sync_all();
      double t1 = now_ms();
      double per_us = (t1 - t0) * 1000.0 / (double)iters;
      double aggbw = (double)n_ * (double)(n_-1) * (double)sz * (double)iters / ((t1 - t0) * 1e-3) / 1e9;
      printf("{\"test\":\"alltoall\",\"impl\":\"peer\",\"ngpus\":%d,\"devs\":[", n_);
      for (int i = 0; i < n_; i++) printf("%d%s", devs_[i], i < n_-1 ? "," : "");
      printf("],\"size_bytes_per_peer\":%zu,\"iters\":%d,\"rep\":%d,\"total_ms\":%.3f,"
             "\"per_a2a_us\":%.3f,\"agg_bw_GBps\":%.2f}\n",
             sz, iters, rep, t1-t0, per_us, aggbw);
    }
  }
}

// ---------- graphed ping-pongs (pair) ----------
static void run_graph_pp(int devA, int devB, int use_nccl) {
  int devs[2] = {devA, devB};
  ncclComm_t comms[2];
  if (use_nccl) CHECK_NCCL(ncclCommInitAll(comms, 2, devs));
  cudaStream_t sA, sB;
  char *bufA, *bufB, *bufA2, *bufB2;
  size_t maxsz = PAIR_SIZES[N_PAIR_SIZES-1];
  CHECK_CUDA(cudaSetDevice(devA));
  CHECK_CUDA(cudaStreamCreateWithFlags(&sA, cudaStreamNonBlocking));
  CHECK_CUDA(cudaMalloc((void**)&bufA, maxsz)); CHECK_CUDA(cudaMalloc((void**)&bufA2, maxsz));
  cudaEvent_t evF, evJ;
  CHECK_CUDA(cudaEventCreateWithFlags(&evF, cudaEventDisableTiming));
  CHECK_CUDA(cudaSetDevice(devB));
  CHECK_CUDA(cudaStreamCreateWithFlags(&sB, cudaStreamNonBlocking));
  CHECK_CUDA(cudaMalloc((void**)&bufB, maxsz)); CHECK_CUDA(cudaMalloc((void**)&bufB2, maxsz));
  CHECK_CUDA(cudaEventCreateWithFlags(&evJ, cudaEventDisableTiming));
  if (!use_nccl) {
    CHECK_CUDA(cudaSetDevice(devA));
    cudaError_t e = cudaDeviceEnablePeerAccess(devB, 0);
    if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) CHECK_CUDA(e);
    cudaGetLastError();
    CHECK_CUDA(cudaSetDevice(devB));
    e = cudaDeviceEnablePeerAccess(devA, 0);
    if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) CHECK_CUDA(e);
    cudaGetLastError();
  }
  const int K = 50;  // round trips per graph
  for (int si = 0; si < N_PAIR_SIZES; si++) {
    size_t sz = PAIR_SIZES[si];
    // warmup uncaptured
    for (int w = 0; w < 5; w++) {
      if (use_nccl) {
        CHECK_NCCL(ncclGroupStart());
        CHECK_NCCL(ncclSend(bufA, sz, ncclChar, 1, comms[0], sA));
        CHECK_NCCL(ncclRecv(bufB2, sz, ncclChar, 0, comms[1], sB));
        CHECK_NCCL(ncclGroupEnd());
        CHECK_NCCL(ncclGroupStart());
        CHECK_NCCL(ncclSend(bufB, sz, ncclChar, 0, comms[1], sB));
        CHECK_NCCL(ncclRecv(bufA2, sz, ncclChar, 1, comms[0], sA));
        CHECK_NCCL(ncclGroupEnd());
      } else {
        CHECK_CUDA(cudaMemcpyAsync(bufB2, bufA, sz, cudaMemcpyDefault, sA));
        CHECK_CUDA(cudaMemcpyAsync(bufA2, bufB, sz, cudaMemcpyDefault, sA));
      }
    }
    CHECK_CUDA(cudaSetDevice(devA)); CHECK_CUDA(cudaStreamSynchronize(sA));
    CHECK_CUDA(cudaSetDevice(devB)); CHECK_CUDA(cudaStreamSynchronize(sB));
    // capture K round trips
    cudaGraph_t graph;
    CHECK_CUDA(cudaSetDevice(devA));
    CHECK_CUDA(cudaStreamBeginCapture(sA, cudaStreamCaptureModeThreadLocal));
    if (use_nccl) {
      CHECK_CUDA(cudaEventRecord(evF, sA));
      CHECK_CUDA(cudaStreamWaitEvent(sB, evF, 0));
      for (int k = 0; k < K; k++) {
        CHECK_NCCL(ncclGroupStart());
        CHECK_NCCL(ncclSend(bufA, sz, ncclChar, 1, comms[0], sA));
        CHECK_NCCL(ncclRecv(bufB2, sz, ncclChar, 0, comms[1], sB));
        CHECK_NCCL(ncclGroupEnd());
        CHECK_NCCL(ncclGroupStart());
        CHECK_NCCL(ncclSend(bufB, sz, ncclChar, 0, comms[1], sB));
        CHECK_NCCL(ncclRecv(bufA2, sz, ncclChar, 1, comms[0], sA));
        CHECK_NCCL(ncclGroupEnd());
      }
      CHECK_CUDA(cudaEventRecord(evJ, sB));
      CHECK_CUDA(cudaStreamWaitEvent(sA, evJ, 0));
    } else {
      for (int k = 0; k < K; k++) {
        CHECK_CUDA(cudaMemcpyAsync(bufB2, bufA, sz, cudaMemcpyDefault, sA));
        CHECK_CUDA(cudaMemcpyAsync(bufA2, bufB, sz, cudaMemcpyDefault, sA));
      }
    }
    CHECK_CUDA(cudaStreamEndCapture(sA, &graph));
    cudaGraphExec_t exec;
    CHECK_CUDA(cudaGraphInstantiate(&exec, graph, NULL, NULL, 0));
    CHECK_CUDA(cudaGraphDestroy(graph));
    for (int w = 0; w < 3; w++) CHECK_CUDA(cudaGraphLaunch(exec, sA));
    CHECK_CUDA(cudaStreamSynchronize(sA));
    for (int rep = 0; rep < REPS; rep++) {
      int launches = (sz >= 268435456) ? 1 : 2;  // 50 or 100 RTs per rep
      double t0 = now_ms();
      for (int l = 0; l < launches; l++) CHECK_CUDA(cudaGraphLaunch(exec, sA));
      CHECK_CUDA(cudaStreamSynchronize(sA));
      double t1 = now_ms();
      int rts = K * launches;
      double lat_us = (t1 - t0) * 1000.0 / ((double)rts * 2.0);
      double bw = (double)sz / (lat_us * 1e-6) / 1e9;
      printf("{\"test\":\"pingpong-graph\",\"impl\":\"%s\",\"devA\":%d,\"devB\":%d,\"size_bytes\":%zu,"
             "\"round_trips\":%d,\"transfers\":%d,\"rep\":%d,\"total_ms\":%.3f,"
             "\"lat_us_oneway\":%.3f,\"bw_GBps_oneway\":%.2f}\n",
             use_nccl ? "nccl-graph" : "peer-graph", devA, devB, sz, rts, rts*2, rep, t1-t0, lat_us, bw);
    }
    CHECK_CUDA(cudaGraphExecDestroy(exec));
  }
}

int main(int argc, char** argv) {
  if (argc < 4) { fprintf(stderr, "usage: %s <ga2a|pa2a|gpa2a|gppp|gpp> <dev...>\n", argv[0]); return 1; }
  ncclGetVersion(&g_ncclver);
  setvbuf(stdout, NULL, _IOLBF, 0);
  const char* mode = argv[1];
  if (!strcmp(mode, "gppp")) { run_graph_pp(atoi(argv[2]), atoi(argv[3]), 0); return 0; }
  if (!strcmp(mode, "gpp"))  { run_graph_pp(atoi(argv[2]), atoi(argv[3]), 1); return 0; }
  int n = argc - 2; int devs[8];
  if (n < 2 || n > 8) { fprintf(stderr, "need 2-8 devs\n"); return 1; }
  for (int i = 0; i < n; i++) devs[i] = atoi(argv[i+2]);
  if      (!strcmp(mode, "ga2a"))  run_graph_a2a("alltoall-graph", nccl_a2a_once, n, devs, 1);
  else if (!strcmp(mode, "pa2a"))  run_peer_a2a(n, devs);
  else if (!strcmp(mode, "gpa2a")) run_graph_a2a("alltoall-graph", peer_a2a_enqueue, n, devs, 0);
  else { fprintf(stderr, "unknown mode %s\n", mode); return 1; }
  return 0;
}
