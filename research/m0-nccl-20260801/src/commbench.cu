// m0-nccl commbench: NCCL send/recv + all-to-all + cudaMemcpyPeerAsync control
// Single-process, single-thread, multi-device (NCCL grouped ops).
// Output: JSONL rows on stdout, one per (test, size, rep).
//
// Modes:
//   commbench pp   <devA> <devB>          NCCL ping-pong (latency + effective one-way BW)
//   commbench uni  <devA> <devB>          NCCL unidirectional A->B saturated BW
//   commbench bidir <devA> <devB>         NCCL simultaneous both-directions aggregate BW
//   commbench ppp  <devA> <devB>          cudaMemcpyPeerAsync ping-pong (control)
//   commbench puni <devA> <devB>          cudaMemcpyPeerAsync unidirectional A->B (control)
//   commbench pbidir <devA> <devB>        cudaMemcpyPeerAsync both directions (control)
//   commbench a2a  <d0> <d1> [...]        NCCL all-to-all (grouped send/recv), 2-5 GPUs

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

// ---------- pair tests (NCCL) ----------
struct Pair {
  int devA, devB;
  ncclComm_t comms[2];      // rank0=devA rank1=devB
  cudaStream_t sA, sB;
  char *sA_buf, *rA_buf, *sB_buf, *rB_buf;
};

static void pair_init(struct Pair* p, int devA, int devB, size_t maxsz, int want_nccl) {
  p->devA = devA; p->devB = devB;
  int devs[2] = {devA, devB};
  if (want_nccl) CHECK_NCCL(ncclCommInitAll(p->comms, 2, devs));
  CHECK_CUDA(cudaSetDevice(devA));
  CHECK_CUDA(cudaStreamCreateWithFlags(&p->sA, cudaStreamNonBlocking));
  CHECK_CUDA(cudaMalloc((void**)&p->sA_buf, maxsz));
  CHECK_CUDA(cudaMalloc((void**)&p->rA_buf, maxsz));
  CHECK_CUDA(cudaMemset(p->sA_buf, 0xA5, maxsz));
  CHECK_CUDA(cudaSetDevice(devB));
  CHECK_CUDA(cudaStreamCreateWithFlags(&p->sB, cudaStreamNonBlocking));
  CHECK_CUDA(cudaMalloc((void**)&p->sB_buf, maxsz));
  CHECK_CUDA(cudaMalloc((void**)&p->rB_buf, maxsz));
  CHECK_CUDA(cudaMemset(p->sB_buf, 0x5A, maxsz));
  CHECK_CUDA(cudaDeviceSynchronize());
}

static void pair_sync(struct Pair* p) {
  CHECK_CUDA(cudaSetDevice(p->devA)); CHECK_CUDA(cudaStreamSynchronize(p->sA));
  CHECK_CUDA(cudaSetDevice(p->devB)); CHECK_CUDA(cudaStreamSynchronize(p->sB));
}

static void nccl_rt(struct Pair* p, size_t sz) {  // one round trip A->B->A
  CHECK_NCCL(ncclGroupStart());
  CHECK_NCCL(ncclSend(p->sA_buf, sz, ncclChar, 1, p->comms[0], p->sA));
  CHECK_NCCL(ncclRecv(p->rB_buf, sz, ncclChar, 0, p->comms[1], p->sB));
  CHECK_NCCL(ncclGroupEnd());
  CHECK_NCCL(ncclGroupStart());
  CHECK_NCCL(ncclSend(p->sB_buf, sz, ncclChar, 0, p->comms[1], p->sB));
  CHECK_NCCL(ncclRecv(p->rA_buf, sz, ncclChar, 1, p->comms[0], p->sA));
  CHECK_NCCL(ncclGroupEnd());
}

static void run_nccl_pp(int devA, int devB) {
  struct Pair p; pair_init(&p, devA, devB, PAIR_SIZES[N_PAIR_SIZES-1], 1);
  for (int si = 0; si < N_PAIR_SIZES; si++) {
    size_t sz = PAIR_SIZES[si];
    int iters = (sz >= 16777216) ? 50 : 100;  // round trips -> 2*iters transfers
    for (int w = 0; w < 10; w++) nccl_rt(&p, sz);
    pair_sync(&p);
    for (int rep = 0; rep < REPS; rep++) {
      double t0 = now_ms();
      for (int i = 0; i < iters; i++) nccl_rt(&p, sz);
      pair_sync(&p);
      double t1 = now_ms();
      double lat_us = (t1 - t0) * 1000.0 / ((double)iters * 2.0);
      double bw = (double)sz / (lat_us * 1e-6) / 1e9;
      printf("{\"test\":\"pingpong\",\"impl\":\"nccl\",\"devA\":%d,\"devB\":%d,\"size_bytes\":%zu,"
             "\"round_trips\":%d,\"transfers\":%d,\"rep\":%d,\"total_ms\":%.3f,"
             "\"lat_us_oneway\":%.3f,\"bw_GBps_oneway\":%.2f,\"nccl\":%d}\n",
             devA, devB, sz, iters, iters*2, rep, t1-t0, lat_us, bw, g_ncclver);
    }
  }
}

static void run_nccl_uni(int devA, int devB) {
  struct Pair p; pair_init(&p, devA, devB, PAIR_SIZES[N_PAIR_SIZES-1], 1);
  for (int si = 0; si < N_PAIR_SIZES; si++) {
    size_t sz = PAIR_SIZES[si];
    int iters = (sz >= 268435456) ? 50 : 100;
    for (int w = 0; w < 10; w++) {
      CHECK_NCCL(ncclGroupStart());
      CHECK_NCCL(ncclSend(p.sA_buf, sz, ncclChar, 1, p.comms[0], p.sA));
      CHECK_NCCL(ncclRecv(p.rB_buf, sz, ncclChar, 0, p.comms[1], p.sB));
      CHECK_NCCL(ncclGroupEnd());
    }
    pair_sync(&p);
    for (int rep = 0; rep < REPS; rep++) {
      double t0 = now_ms();
      for (int i = 0; i < iters; i++) {
        CHECK_NCCL(ncclGroupStart());
        CHECK_NCCL(ncclSend(p.sA_buf, sz, ncclChar, 1, p.comms[0], p.sA));
        CHECK_NCCL(ncclRecv(p.rB_buf, sz, ncclChar, 0, p.comms[1], p.sB));
        CHECK_NCCL(ncclGroupEnd());
      }
      pair_sync(&p);
      double t1 = now_ms();
      double per_us = (t1 - t0) * 1000.0 / (double)iters;
      double bw = (double)sz * (double)iters / ((t1 - t0) * 1e-3) / 1e9;
      printf("{\"test\":\"uni\",\"impl\":\"nccl\",\"devA\":%d,\"devB\":%d,\"dir\":\"%d->%d\","
             "\"size_bytes\":%zu,\"iters\":%d,\"rep\":%d,\"total_ms\":%.3f,"
             "\"per_iter_us\":%.3f,\"bw_GBps\":%.2f,\"nccl\":%d}\n",
             devA, devB, devA, devB, sz, iters, rep, t1-t0, per_us, bw, g_ncclver);
    }
  }
}

static void run_nccl_bidir(int devA, int devB) {
  struct Pair p; pair_init(&p, devA, devB, PAIR_SIZES[N_PAIR_SIZES-1], 1);
  for (int si = 0; si < N_PAIR_SIZES; si++) {
    size_t sz = PAIR_SIZES[si];
    int iters = (sz >= 268435456) ? 50 : 100;
    for (int w = 0; w < 10; w++) {
      CHECK_NCCL(ncclGroupStart());
      CHECK_NCCL(ncclSend(p.sA_buf, sz, ncclChar, 1, p.comms[0], p.sA));
      CHECK_NCCL(ncclRecv(p.rA_buf, sz, ncclChar, 1, p.comms[0], p.sA));
      CHECK_NCCL(ncclSend(p.sB_buf, sz, ncclChar, 0, p.comms[1], p.sB));
      CHECK_NCCL(ncclRecv(p.rB_buf, sz, ncclChar, 0, p.comms[1], p.sB));
      CHECK_NCCL(ncclGroupEnd());
    }
    pair_sync(&p);
    for (int rep = 0; rep < REPS; rep++) {
      double t0 = now_ms();
      for (int i = 0; i < iters; i++) {
        CHECK_NCCL(ncclGroupStart());
        CHECK_NCCL(ncclSend(p.sA_buf, sz, ncclChar, 1, p.comms[0], p.sA));
        CHECK_NCCL(ncclRecv(p.rA_buf, sz, ncclChar, 1, p.comms[0], p.sA));
        CHECK_NCCL(ncclSend(p.sB_buf, sz, ncclChar, 0, p.comms[1], p.sB));
        CHECK_NCCL(ncclRecv(p.rB_buf, sz, ncclChar, 0, p.comms[1], p.sB));
        CHECK_NCCL(ncclGroupEnd());
      }
      pair_sync(&p);
      double t1 = now_ms();
      double per_us = (t1 - t0) * 1000.0 / (double)iters;
      double agg = 2.0 * (double)sz * (double)iters / ((t1 - t0) * 1e-3) / 1e9;
      printf("{\"test\":\"bidir\",\"impl\":\"nccl\",\"devA\":%d,\"devB\":%d,\"size_bytes\":%zu,"
             "\"iters\":%d,\"rep\":%d,\"total_ms\":%.3f,\"per_iter_us\":%.3f,"
             "\"bw_GBps_aggregate\":%.2f,\"nccl\":%d}\n",
             devA, devB, sz, iters, rep, t1-t0, per_us, agg, g_ncclver);
    }
  }
}

// ---------- pair tests (cudaMemcpyPeerAsync control) ----------
static void peer_enable(int devA, int devB) {
  int can = 0;
  CHECK_CUDA(cudaDeviceCanAccessPeer(&can, devA, devB));
  fprintf(stderr, "canAccessPeer %d->%d = %d\n", devA, devB, can);
  CHECK_CUDA(cudaSetDevice(devA));
  cudaError_t e = cudaDeviceEnablePeerAccess(devB, 0);
  if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) CHECK_CUDA(e);
  cudaGetLastError();
  CHECK_CUDA(cudaSetDevice(devB));
  e = cudaDeviceEnablePeerAccess(devA, 0);
  if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) CHECK_CUDA(e);
  cudaGetLastError();
}

static void run_peer_pp(int devA, int devB) {
  struct Pair p; pair_init(&p, devA, devB, PAIR_SIZES[N_PAIR_SIZES-1], 0);
  peer_enable(devA, devB);
  CHECK_CUDA(cudaSetDevice(devA));
  for (int si = 0; si < N_PAIR_SIZES; si++) {
    size_t sz = PAIR_SIZES[si];
    int iters = (sz >= 16777216) ? 50 : 100;  // round trips
    for (int w = 0; w < 10; w++) {
      CHECK_CUDA(cudaMemcpyPeerAsync(p.rB_buf, devB, p.sA_buf, devA, sz, p.sA));
      CHECK_CUDA(cudaMemcpyPeerAsync(p.rA_buf, devA, p.sB_buf, devB, sz, p.sA));
    }
    CHECK_CUDA(cudaStreamSynchronize(p.sA));
    for (int rep = 0; rep < REPS; rep++) {
      double t0 = now_ms();
      for (int i = 0; i < iters; i++) {
        CHECK_CUDA(cudaMemcpyPeerAsync(p.rB_buf, devB, p.sA_buf, devA, sz, p.sA));
        CHECK_CUDA(cudaMemcpyPeerAsync(p.rA_buf, devA, p.sB_buf, devB, sz, p.sA));
      }
      CHECK_CUDA(cudaStreamSynchronize(p.sA));
      double t1 = now_ms();
      double lat_us = (t1 - t0) * 1000.0 / ((double)iters * 2.0);
      double bw = (double)sz / (lat_us * 1e-6) / 1e9;
      printf("{\"test\":\"pingpong\",\"impl\":\"peer\",\"devA\":%d,\"devB\":%d,\"size_bytes\":%zu,"
             "\"round_trips\":%d,\"transfers\":%d,\"rep\":%d,\"total_ms\":%.3f,"
             "\"lat_us_oneway\":%.3f,\"bw_GBps_oneway\":%.2f}\n",
             devA, devB, sz, iters, iters*2, rep, t1-t0, lat_us, bw);
    }
  }
}

static void run_peer_uni(int devA, int devB) {
  struct Pair p; pair_init(&p, devA, devB, PAIR_SIZES[N_PAIR_SIZES-1], 0);
  peer_enable(devA, devB);
  CHECK_CUDA(cudaSetDevice(devA));
  for (int si = 0; si < N_PAIR_SIZES; si++) {
    size_t sz = PAIR_SIZES[si];
    int iters = (sz >= 268435456) ? 50 : 100;
    for (int w = 0; w < 10; w++)
      CHECK_CUDA(cudaMemcpyPeerAsync(p.rB_buf, devB, p.sA_buf, devA, sz, p.sA));
    CHECK_CUDA(cudaStreamSynchronize(p.sA));
    for (int rep = 0; rep < REPS; rep++) {
      double t0 = now_ms();
      for (int i = 0; i < iters; i++)
        CHECK_CUDA(cudaMemcpyPeerAsync(p.rB_buf, devB, p.sA_buf, devA, sz, p.sA));
      CHECK_CUDA(cudaStreamSynchronize(p.sA));
      double t1 = now_ms();
      double per_us = (t1 - t0) * 1000.0 / (double)iters;
      double bw = (double)sz * (double)iters / ((t1 - t0) * 1e-3) / 1e9;
      printf("{\"test\":\"uni\",\"impl\":\"peer\",\"devA\":%d,\"devB\":%d,\"dir\":\"%d->%d\","
             "\"size_bytes\":%zu,\"iters\":%d,\"rep\":%d,\"total_ms\":%.3f,"
             "\"per_iter_us\":%.3f,\"bw_GBps\":%.2f}\n",
             devA, devB, devA, devB, sz, iters, rep, t1-t0, per_us, bw);
    }
  }
}

static void run_peer_bidir(int devA, int devB) {
  struct Pair p; pair_init(&p, devA, devB, PAIR_SIZES[N_PAIR_SIZES-1], 0);
  peer_enable(devA, devB);
  for (int si = 0; si < N_PAIR_SIZES; si++) {
    size_t sz = PAIR_SIZES[si];
    int iters = (sz >= 268435456) ? 50 : 100;
    for (int w = 0; w < 10; w++) {
      CHECK_CUDA(cudaMemcpyPeerAsync(p.rB_buf, devB, p.sA_buf, devA, sz, p.sA));
      CHECK_CUDA(cudaMemcpyPeerAsync(p.rA_buf, devA, p.sB_buf, devB, sz, p.sB));
    }
    pair_sync(&p);
    for (int rep = 0; rep < REPS; rep++) {
      double t0 = now_ms();
      for (int i = 0; i < iters; i++) {
        CHECK_CUDA(cudaMemcpyPeerAsync(p.rB_buf, devB, p.sA_buf, devA, sz, p.sA));
        CHECK_CUDA(cudaMemcpyPeerAsync(p.rA_buf, devA, p.sB_buf, devB, sz, p.sB));
      }
      pair_sync(&p);
      double t1 = now_ms();
      double per_us = (t1 - t0) * 1000.0 / (double)iters;
      double agg = 2.0 * (double)sz * (double)iters / ((t1 - t0) * 1e-3) / 1e9;
      printf("{\"test\":\"bidir\",\"impl\":\"peer\",\"devA\":%d,\"devB\":%d,\"size_bytes\":%zu,"
             "\"iters\":%d,\"rep\":%d,\"total_ms\":%.3f,\"per_iter_us\":%.3f,"
             "\"bw_GBps_aggregate\":%.2f}\n",
             devA, devB, sz, iters, rep, t1-t0, per_us, agg);
    }
  }
}

// ---------- all-to-all (NCCL grouped send/recv) ----------
static void run_a2a(int n, int* devs) {
  ncclComm_t comms[8];
  cudaStream_t streams[8];
  char *sbuf[8], *rbuf[8];
  size_t maxp = A2A_SIZES[N_A2A_SIZES-1];
  CHECK_NCCL(ncclCommInitAll(comms, n, devs));
  for (int i = 0; i < n; i++) {
    CHECK_CUDA(cudaSetDevice(devs[i]));
    CHECK_CUDA(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
    CHECK_CUDA(cudaMalloc((void**)&sbuf[i], maxp * n));
    CHECK_CUDA(cudaMalloc((void**)&rbuf[i], maxp * n));
    CHECK_CUDA(cudaMemset(sbuf[i], 0x11 * (i+1), maxp * n));
    CHECK_CUDA(cudaDeviceSynchronize());
  }
  for (int si = 0; si < N_A2A_SIZES; si++) {
    size_t sz = A2A_SIZES[si];
    int iters = 100;
    for (int w = 0; w < 10 + (si==0?10:0); w++) {
      CHECK_NCCL(ncclGroupStart());
      for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) if (i != j) {
        CHECK_NCCL(ncclSend(sbuf[i] + (size_t)j * sz, sz, ncclChar, j, comms[i], streams[i]));
        CHECK_NCCL(ncclRecv(rbuf[i] + (size_t)j * sz, sz, ncclChar, j, comms[i], streams[i]));
      }
      CHECK_NCCL(ncclGroupEnd());
    }
    for (int i = 0; i < n; i++) { CHECK_CUDA(cudaSetDevice(devs[i])); CHECK_CUDA(cudaStreamSynchronize(streams[i])); }
    for (int rep = 0; rep < REPS; rep++) {
      double t0 = now_ms();
      for (int it = 0; it < iters; it++) {
        CHECK_NCCL(ncclGroupStart());
        for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) if (i != j) {
          CHECK_NCCL(ncclSend(sbuf[i] + (size_t)j * sz, sz, ncclChar, j, comms[i], streams[i]));
          CHECK_NCCL(ncclRecv(rbuf[i] + (size_t)j * sz, sz, ncclChar, j, comms[i], streams[i]));
        }
        CHECK_NCCL(ncclGroupEnd());
      }
      for (int i = 0; i < n; i++) { CHECK_CUDA(cudaSetDevice(devs[i])); CHECK_CUDA(cudaStreamSynchronize(streams[i])); }
      double t1 = now_ms();
      double per_us = (t1 - t0) * 1000.0 / (double)iters;
      double aggbytes = (double)n * (double)(n-1) * (double)sz;
      double aggbw = aggbytes * (double)iters / ((t1 - t0) * 1e-3) / 1e9;
      double perrank = aggbw / n;
      printf("{\"test\":\"alltoall\",\"impl\":\"nccl\",\"ngpus\":%d,\"devs\":[", n);
      for (int i = 0; i < n; i++) printf("%d%s", devs[i], i < n-1 ? "," : "");
      printf("],\"size_bytes_per_peer\":%zu,\"iters\":%d,\"rep\":%d,\"total_ms\":%.3f,"
             "\"per_a2a_us\":%.3f,\"agg_bw_GBps\":%.2f,\"per_rank_bw_GBps\":%.2f,\"nccl\":%d}\n",
             sz, iters, rep, t1-t0, per_us, aggbw, perrank, g_ncclver);
    }
  }
}

int main(int argc, char** argv) {
  if (argc < 4) { fprintf(stderr, "usage: %s <pp|uni|bidir|ppp|puni|pbidir|a2a> <dev...>\n", argv[0]); return 1; }
  ncclGetVersion(&g_ncclver);
  setvbuf(stdout, NULL, _IOLBF, 0);
  const char* mode = argv[1];
  if (!strcmp(mode, "a2a")) {
    int n = argc - 2; int devs[8];
    if (n < 2 || n > 8) { fprintf(stderr, "a2a needs 2-8 devs\n"); return 1; }
    for (int i = 0; i < n; i++) devs[i] = atoi(argv[i+2]);
    run_a2a(n, devs);
    return 0;
  }
  int devA = atoi(argv[2]), devB = atoi(argv[3]);
  if      (!strcmp(mode, "pp"))     run_nccl_pp(devA, devB);
  else if (!strcmp(mode, "uni"))    run_nccl_uni(devA, devB);
  else if (!strcmp(mode, "bidir"))  run_nccl_bidir(devA, devB);
  else if (!strcmp(mode, "ppp"))    run_peer_pp(devA, devB);
  else if (!strcmp(mode, "puni"))   run_peer_uni(devA, devB);
  else if (!strcmp(mode, "pbidir")) run_peer_bidir(devA, devB);
  else { fprintf(stderr, "unknown mode %s\n", mode); return 1; }
  return 0;
}
