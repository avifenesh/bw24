// m0-nccl Rust spike: prove cudarc 0.19.8 `nccl` feature drives libnccl.so.2 on this box.
// Single-process two-device NCCL ping-pong (GPU pair from argv), JSONL rows on stdout.
use cudarc::driver::CudaContext;
use cudarc::nccl::{result, Comm};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dev_a: usize = args.get(1).map(|s| s.parse().unwrap()).unwrap_or(0);
    let dev_b: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(3);

    let ctx_a = CudaContext::new(dev_a).expect("ctx A");
    let ctx_b = CudaContext::new(dev_b).expect("ctx B");
    let s_a = ctx_a.default_stream();
    let s_b = ctx_b.default_stream();
    let comms = Comm::from_devices(vec![s_a.clone(), s_b.clone()]).expect("nccl init");
    eprintln!("nccl comms up: ranks={} devs=[{},{}]", comms.len(), dev_a, dev_b);

    // element counts of f32: 4KB, 64KB, 1MB, 16MB
    for &n in &[1024usize, 16 * 1024, 256 * 1024, 4 * 1024 * 1024] {
        let sz = n * 4;
        let send_a = s_a.alloc_zeros::<f32>(n).expect("alloc a");
        let mut recv_a = s_a.alloc_zeros::<f32>(n).expect("alloc a2");
        let send_b = s_b.alloc_zeros::<f32>(n).expect("alloc b");
        let mut recv_b = s_b.alloc_zeros::<f32>(n).expect("alloc b2");

        let rt = |recv_a: &mut cudarc::driver::CudaSlice<f32>,
                  recv_b: &mut cudarc::driver::CudaSlice<f32>| {
            result::group_start().unwrap();
            comms[0].send(&send_a, 1).unwrap();
            comms[1].recv(recv_b, 0).unwrap();
            result::group_end().unwrap();
            result::group_start().unwrap();
            comms[1].send(&send_b, 0).unwrap();
            comms[0].recv(recv_a, 1).unwrap();
            result::group_end().unwrap();
        };

        for _ in 0..10 {
            rt(&mut recv_a, &mut recv_b);
        }
        s_a.synchronize().unwrap();
        s_b.synchronize().unwrap();

        let iters = 100;
        for rep in 0..5 {
            let t0 = Instant::now();
            for _ in 0..iters {
                rt(&mut recv_a, &mut recv_b);
            }
            s_a.synchronize().unwrap();
            s_b.synchronize().unwrap();
            let total_ms = t0.elapsed().as_secs_f64() * 1e3;
            let lat_us = total_ms * 1000.0 / (iters as f64 * 2.0);
            let bw = sz as f64 / (lat_us * 1e-6) / 1e9;
            println!(
                "{{\"test\":\"pingpong\",\"impl\":\"cudarc-nccl\",\"devA\":{},\"devB\":{},\"size_bytes\":{},\"round_trips\":{},\"transfers\":{},\"rep\":{},\"total_ms\":{:.3},\"lat_us_oneway\":{:.3},\"bw_GBps_oneway\":{:.2}}}",
                dev_a, dev_b, sz, iters, iters * 2, rep, total_ms, lat_us, bw
            );
        }
    }
}
