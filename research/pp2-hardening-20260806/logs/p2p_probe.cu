// P2P bandwidth/latency probe (p2pBandwidthLatencyTest-class), pp2-hardening lane.
// Measures: cudaMemcpyPeerAsync D2D unidirectional + bidirectional BW, host-bounce
// (D2H+H2D via pinned) for the same payload, and one-way latency at PP-boundary sizes.
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  printf("CUDA_ERR %s @ %s:%d -> %s\n",#x,__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)

static double bw_gbs(size_t bytes, double sec, int n){ return (double)bytes*n/sec/1e9; }

int main(){
  int nd=0; CK(cudaGetDeviceCount(&nd));
  printf("devices=%d\n", nd);
  for(int i=0;i<nd;i++){ cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,i));
    printf("dev%d name=\"%s\" cc=%d.%d pciBus=%02x:%02x.%d memBW_theo=%.0fGB/s\n",
      i,p.name,p.major,p.minor,p.pciBusID,p.pciDeviceID,p.pciDomainID,
      2.0*p.memoryClockRate*(p.memoryBusWidth/8)/1.0e6);
  }
  // canAccessPeer matrix
  printf("--- canAccessPeer ---\n");
  for(int i=0;i<nd;i++) for(int j=0;j<nd;j++) if(i!=j){ int ca=0;
    CK(cudaDeviceCanAccessPeer(&ca,i,j)); printf("canAccessPeer %d->%d = %d\n",i,j,ca); }
  if(nd<2){ printf("SKIP: need 2 devices\n"); return 0; }

  // enable peer both ways
  int ca01=0,ca10=0; CK(cudaDeviceCanAccessPeer(&ca01,0,1)); CK(cudaDeviceCanAccessPeer(&ca10,1,0));
  bool peer = ca01 && ca10;
  if(peer){ CK(cudaSetDevice(0)); cudaDeviceEnablePeerAccess(1,0);
            CK(cudaSetDevice(1)); cudaDeviceEnablePeerAccess(0,0); }
  printf("peer_enabled=%d\n", (int)peer);

  // payload sizes: PP boundary (n_embd f32) up to bulk
  // 4096*4=16KB (q27), 12KB (122B n_embd=3072? actual 12KB per assessment), plus bulk
  size_t sizes[] = {4*1024, 12*1024, 16*1024, 64*1024, 256*1024,
                    1ul<<20, 4ul<<20, 16ul<<20, 64ul<<20, 256ul<<20};
  int nsz = sizeof(sizes)/sizeof(sizes[0]);
  size_t maxsz = sizes[nsz-1];

  void *d0=nullptr,*d1=nullptr,*d0b=nullptr,*d1b=nullptr,*hpin=nullptr;
  CK(cudaSetDevice(0)); CK(cudaMalloc(&d0,maxsz)); CK(cudaMalloc(&d0b,maxsz)); CK(cudaMemset(d0,1,maxsz));
  CK(cudaSetDevice(1)); CK(cudaMalloc(&d1,maxsz)); CK(cudaMalloc(&d1b,maxsz)); CK(cudaMemset(d1,2,maxsz));
  CK(cudaHostAlloc(&hpin,maxsz,cudaHostAllocDefault));

  cudaStream_t s0,s1,s0b,s1b;
  CK(cudaSetDevice(0)); CK(cudaStreamCreate(&s0)); CK(cudaStreamCreate(&s0b));
  CK(cudaSetDevice(1)); CK(cudaStreamCreate(&s1)); CK(cudaStreamCreate(&s1b));
  cudaEvent_t e0,e1; CK(cudaSetDevice(0)); CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));

  printf("--- bandwidth: bytes, uni_p2p_GBs, bidir_p2p_GBs, bounce_GBs (host-staged D2H+H2D) ---\n");
  for(int k=0;k<nsz;k++){
    size_t B=sizes[k];
    int iters = B<(1ul<<20) ? 2000 : (B<(16ul<<20)?300:40);
    // warmup + uni p2p 0->1
    CK(cudaSetDevice(0));
    for(int i=0;i<20;i++) CK(cudaMemcpyPeerAsync(d1,1,d0,0,B,s0));
    CK(cudaStreamSynchronize(s0));
    CK(cudaEventRecord(e0,s0));
    for(int i=0;i<iters;i++) CK(cudaMemcpyPeerAsync(d1,1,d0,0,B,s0));
    CK(cudaEventRecord(e1,s0)); CK(cudaEventSynchronize(e1));
    float ms=0; CK(cudaEventElapsedTime(&ms,e0,e1));
    double uni = bw_gbs(B, ms/1e3, iters);
    // bidirectional: 0->1 on s0, 1->0 on s1b concurrently
    CK(cudaSetDevice(0)); for(int i=0;i<10;i++) CK(cudaMemcpyPeerAsync(d1,1,d0,0,B,s0));
    CK(cudaSetDevice(1)); for(int i=0;i<10;i++) CK(cudaMemcpyPeerAsync(d0b,0,d1b,1,B,s1b));
    CK(cudaSetDevice(0)); CK(cudaStreamSynchronize(s0)); CK(cudaSetDevice(1)); CK(cudaStreamSynchronize(s1b));
    cudaEvent_t bs,be; CK(cudaSetDevice(0)); CK(cudaEventCreate(&bs)); CK(cudaEventCreate(&be));
    CK(cudaEventRecord(bs,s0));
    for(int i=0;i<iters;i++){
      CK(cudaSetDevice(0)); CK(cudaMemcpyPeerAsync(d1,1,d0,0,B,s0));
      CK(cudaSetDevice(1)); CK(cudaMemcpyPeerAsync(d0b,0,d1b,1,B,s1b));
    }
    CK(cudaSetDevice(0)); CK(cudaEventRecord(be,s0));
    CK(cudaSetDevice(1)); CK(cudaStreamSynchronize(s1b));
    CK(cudaSetDevice(0)); CK(cudaEventSynchronize(be));
    float bms=0; CK(cudaEventElapsedTime(&bms,bs,be));
    double bidir = bw_gbs(B, bms/1e3, iters)*2.0;
    // host bounce: dev0 -> pinned host -> dev1 (what a non-P2P PP boundary pays)
    CK(cudaSetDevice(0));
    for(int i=0;i<10;i++){ CK(cudaMemcpyAsync(hpin,d0,B,cudaMemcpyDeviceToHost,s0)); }
    CK(cudaStreamSynchronize(s0));
    CK(cudaEventRecord(e0,s0));
    for(int i=0;i<iters;i++){
      CK(cudaSetDevice(0)); CK(cudaMemcpyAsync(hpin,d0,B,cudaMemcpyDeviceToHost,s0));
      CK(cudaStreamSynchronize(s0));
      CK(cudaSetDevice(1)); CK(cudaMemcpyAsync(d1,hpin,B,cudaMemcpyHostToDevice,s1));
      CK(cudaStreamSynchronize(s1));
    }
    CK(cudaSetDevice(0)); CK(cudaEventRecord(e1,s0)); CK(cudaEventSynchronize(e1));
    CK(cudaEventElapsedTime(&ms,e0,e1));
    double bounce = bw_gbs(B, ms/1e3, iters);
    printf("BW %zu %.3f %.3f %.3f\n", B, uni, bidir, bounce);
    cudaEventDestroy(bs); cudaEventDestroy(be);
  }

  // one-way latency at PP boundary sizes: enqueue+sync round for a single copy
  printf("--- latency: bytes, p2p_us, bounce_us (mean of 5000) ---\n");
  size_t lsz[] = {4*1024, 12*1024, 16*1024, 64*1024};
  for(int k=0;k<4;k++){
    size_t B=lsz[k]; int it=5000;
    CK(cudaSetDevice(0));
    for(int i=0;i<100;i++){ CK(cudaMemcpyPeerAsync(d1,1,d0,0,B,s0)); } CK(cudaStreamSynchronize(s0));
    CK(cudaEventRecord(e0,s0));
    for(int i=0;i<it;i++){ CK(cudaMemcpyPeerAsync(d1,1,d0,0,B,s0)); CK(cudaStreamSynchronize(s0)); }
    CK(cudaEventRecord(e1,s0)); CK(cudaEventSynchronize(e1));
    float ms=0; CK(cudaEventElapsedTime(&ms,e0,e1)); double p2pus = ms*1e3/it;
    CK(cudaEventRecord(e0,s0));
    for(int i=0;i<it;i++){
      CK(cudaSetDevice(0)); CK(cudaMemcpyAsync(hpin,d0,B,cudaMemcpyDeviceToHost,s0)); CK(cudaStreamSynchronize(s0));
      CK(cudaSetDevice(1)); CK(cudaMemcpyAsync(d1,hpin,B,cudaMemcpyHostToDevice,s1)); CK(cudaStreamSynchronize(s1));
    }
    CK(cudaSetDevice(0)); CK(cudaEventRecord(e1,s0)); CK(cudaEventSynchronize(e1));
    CK(cudaEventElapsedTime(&ms,e0,e1)); double bus = ms*1e3/it;
    printf("LAT %zu %.3f %.3f\n", B, p2pus, bus);
  }
  printf("DONE\n");
  return 0;
}
