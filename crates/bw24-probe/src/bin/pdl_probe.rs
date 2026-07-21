//! PDL micro-probe (SOTA item 2 falsification, 2026-07-13): measures the per-pair cost of a
//! dependent kernel chain (spin -> consume) x N on ONE stream, four arms:
//!   eager-plain   : cuLaunchKernel
//!   eager-pdl     : cuLaunchKernelEx + PROGRAMMATIC_STREAM_SERIALIZATION on every launch
//!   graph-plain   : the plain chain captured once, replayed
//!   graph-pdl     : the PDL chain captured once (programmatic edges), replayed
//! The engine's decode rides captured graphs, so graph-pdl vs graph-plain is THE number.
//! Correctness oracle: out[0] must equal pairs*reps in every arm (the chain really ran, in
//! order). Spin sized to the glue-kernel class (~2us) plus an empty-kernel arm.
use cudarc::driver::sys as cu;
use std::ffi::c_void;

unsafe fn ck(r: cu::CUresult, what: &str) {
    assert!(r == cu::CUresult::CUDA_SUCCESS, "{what}: {r:?}");
}

struct Chain {
    f_spin: cu::CUfunction,
    f_cons: cu::CUfunction,
    stream: cu::CUstream,
    dbuf: cu::CUdeviceptr,
    spin: i64,
}

impl Chain {
    unsafe fn launch_pair(&mut self, pdl: bool) { unsafe {
        let mut p_spin: [*mut c_void; 2] = [
            (&mut self.dbuf) as *mut _ as *mut c_void,
            (&mut self.spin) as *mut _ as *mut c_void,
        ];
        let mut p_cons: [*mut c_void; 1] = [(&mut self.dbuf) as *mut _ as *mut c_void];
        if pdl {
            let mut attr = cu::CUlaunchAttribute {
                id: cu::CUlaunchAttributeID::CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION,
                pad: [0; 4],
                value: cu::CUlaunchAttributeValue { programmaticStreamSerializationAllowed: 1 },
            };
            let cfg = cu::CUlaunchConfig {
                gridDimX: 8, gridDimY: 1, gridDimZ: 1,
                blockDimX: 128, blockDimY: 1, blockDimZ: 1,
                sharedMemBytes: 0, hStream: self.stream,
                attrs: &mut attr, numAttrs: 1,
            };
            ck(cu::cuLaunchKernelEx(&cfg, self.f_spin, p_spin.as_mut_ptr(),
                                    std::ptr::null_mut()), "launchEx spin");
            ck(cu::cuLaunchKernelEx(&cfg, self.f_cons, p_cons.as_mut_ptr(),
                                    std::ptr::null_mut()), "launchEx consume");
        } else {
            ck(cu::cuLaunchKernel(self.f_spin, 8, 1, 1, 128, 1, 1, 0, self.stream,
                                  p_spin.as_mut_ptr(), std::ptr::null_mut()), "launch spin");
            ck(cu::cuLaunchKernel(self.f_cons, 8, 1, 1, 128, 1, 1, 0, self.stream,
                                  p_cons.as_mut_ptr(), std::ptr::null_mut()), "launch consume");
        }
    }}
}

fn main() {
    let pairs: usize = std::env::var("PAIRS").ok().and_then(|s| s.parse().ok()).unwrap_or(2048);
    let spin: i64 = std::env::var("SPIN").ok().and_then(|s| s.parse().ok()).unwrap_or(4000);
    unsafe {
        ck(cu::cuInit(0), "cuInit");
        let mut dev = 0i32;
        ck(cu::cuDeviceGet(&mut dev, 0), "cuDeviceGet");
        let mut ctx: cu::CUcontext = std::ptr::null_mut();
        ck(cu::cuDevicePrimaryCtxRetain(&mut ctx, dev), "primaryCtxRetain");
        ck(cu::cuCtxSetCurrent(ctx), "ctxSetCurrent");

        let fatbin = std::fs::read(env!("BW24_FATBIN")).expect("fatbin read");
        let mut module: cu::CUmodule = std::ptr::null_mut();
        ck(cu::cuModuleLoadData(&mut module, fatbin.as_ptr() as *const c_void), "moduleLoad");
        let mut f_spin: cu::CUfunction = std::ptr::null_mut();
        let mut f_cons: cu::CUfunction = std::ptr::null_mut();
        ck(cu::cuModuleGetFunction(&mut f_spin, module, c"pdl_spin".as_ptr()), "getF spin");
        ck(cu::cuModuleGetFunction(&mut f_cons, module, c"pdl_consume".as_ptr()), "getF consume");

        let mut stream: cu::CUstream = std::ptr::null_mut();
        ck(cu::cuStreamCreate(&mut stream, 0x1 /* NON_BLOCKING */), "streamCreate");
        let mut dbuf: cu::CUdeviceptr = 0;
        ck(cu::cuMemAlloc_v2(&mut dbuf, 8), "memAlloc");

        let mut chain = Chain { f_spin, f_cons, stream, dbuf, spin };
        let zero = [0f32; 2];

        let reset = |c: &Chain| {
            ck(cu::cuMemcpyHtoD_v2(c.dbuf, zero.as_ptr() as *const c_void, 8), "reset");
        };
        let read0 = |c: &Chain| -> f32 {
            let mut h = [0f32; 2];
            ck(cu::cuMemcpyDtoH_v2(h.as_mut_ptr() as *mut c_void, c.dbuf, 8), "read");
            h[0]
        };

        // ---- eager arms ----
        for (label, pdl) in [("eager-plain", false), ("eager-pdl", true)] {
            reset(&chain);
            // warmup
            for _ in 0..64 { chain.launch_pair(pdl); }
            ck(cu::cuStreamSynchronize(stream), "sync");
            reset(&chain);
            let t0 = std::time::Instant::now();
            for _ in 0..pairs { chain.launch_pair(pdl); }
            ck(cu::cuStreamSynchronize(stream), "sync");
            let dt = t0.elapsed().as_secs_f64();
            let got = read0(&chain);
            println!("{label:12} {pairs} pairs spin={spin}: {:.3} ms = {:.0} ns/pair  (oracle {got} == {pairs}: {})",
                     dt * 1e3, dt * 1e9 / pairs as f64,
                     if got == pairs as f32 { "OK" } else { "FAIL" });
        }

        // ---- graph arms: capture `cap` pairs, replay pairs/cap times ----
        // graph-rewrite = captured PLAIN, then every kernel->kernel edge rewritten to the
        // programmatic encoding (1,0,1) post-capture — the engine-side wiring plan.
        let cap = 64usize;
        let reps = pairs / cap;
        for (label, pdl, rewrite) in [("graph-plain", false, false),
                                      ("graph-pdl", true, false),
                                      ("graph-rewrite", false, true)] {
            reset(&chain);
            ck(cu::cuStreamBeginCapture_v2(
                stream, cu::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL),
               "beginCapture");
            for _ in 0..cap { chain.launch_pair(pdl); }
            let mut graph: cu::CUgraph = std::ptr::null_mut();
            ck(cu::cuStreamEndCapture(stream, &mut graph), "endCapture");
            // PRE_INSTANTIATE=1: instantiate (and upload) BEFORE the rewrite — reproduces
            // the engine's capture flow (cudarc end_capture instantiates internally).
            let mut pre_exec: cu::CUgraphExec = std::ptr::null_mut();
            if rewrite && std::env::var("PRE_INSTANTIATE").is_ok() {
                ck(cu::cuGraphInstantiateWithFlags(&mut pre_exec, graph, 0), "pre-instantiate");
                ck(cu::cuGraphUpload(pre_exec, stream), "pre-upload");
            }
            if rewrite {
                let mut ne: usize = 0;
                ck(cu::cuGraphGetEdges_v2(graph, std::ptr::null_mut(), std::ptr::null_mut(),
                                          std::ptr::null_mut(), &mut ne), "rw count");
                let mut from = vec![std::ptr::null_mut(); ne];
                let mut to = vec![std::ptr::null_mut(); ne];
                let mut ed: Vec<cu::CUgraphEdgeData> = vec![std::mem::zeroed(); ne];
                ck(cu::cuGraphGetEdges_v2(graph, from.as_mut_ptr(), to.as_mut_ptr(),
                                          ed.as_mut_ptr(), &mut ne), "rw edges");
                for i in 0..ne {
                    ck(cu::cuGraphRemoveDependencies_v2(graph, &from[i], &to[i], &ed[i], 1),
                       "rw remove");
                    let prog = cu::CUgraphEdgeData {
                        from_port: cu::CU_GRAPH_KERNEL_NODE_PORT_PROGRAMMATIC as u8,
                        to_port: 0, type_: 1, reserved: [0; 5],
                    };
                    ck(cu::cuGraphAddDependencies_v2(graph, &from[i], &to[i], &prog, 1),
                       "rw add");
                }
                println!("  rewrote {ne} edges to programmatic");
            }
            // Edge-encoding ground truth: what does capture-with-PDL write into edgeData?
            if std::env::var("DUMP_EDGES").is_ok() {
                let mut ne: usize = 0;
                ck(cu::cuGraphGetEdges_v2(graph, std::ptr::null_mut(), std::ptr::null_mut(),
                                          std::ptr::null_mut(), &mut ne), "edges count");
                let mut from = vec![std::ptr::null_mut(); ne];
                let mut to = vec![std::ptr::null_mut(); ne];
                let mut ed: Vec<cu::CUgraphEdgeData> = vec![std::mem::zeroed(); ne];
                ck(cu::cuGraphGetEdges_v2(graph, from.as_mut_ptr(), to.as_mut_ptr(),
                                          ed.as_mut_ptr(), &mut ne), "edges");
                let mut hist = std::collections::HashMap::new();
                for e in &ed {
                    *hist.entry((e.from_port, e.to_port, e.type_)).or_insert(0usize) += 1;
                }
                println!("  {label} edges={ne} (from_port,to_port,type)->count: {hist:?}");
            }
            let mut gexec: cu::CUgraphExec = std::ptr::null_mut();
            ck(cu::cuGraphInstantiateWithFlags(&mut gexec, graph, 0), "instantiate");
            // warmup replay
            ck(cu::cuGraphLaunch(gexec, stream), "graphLaunch warm");
            ck(cu::cuStreamSynchronize(stream), "sync");
            reset(&chain);
            let t0 = std::time::Instant::now();
            for _ in 0..reps { ck(cu::cuGraphLaunch(gexec, stream), "graphLaunch"); }
            ck(cu::cuStreamSynchronize(stream), "sync");
            let dt = t0.elapsed().as_secs_f64();
            let got = read0(&chain);
            let want = (reps * cap) as f32;
            println!("{label:12} {} pairs spin={spin}: {:.3} ms = {:.0} ns/pair  (oracle {got} == {want}: {})",
                     reps * cap, dt * 1e3, dt * 1e9 / (reps * cap) as f64,
                     if got == want { "OK" } else { "FAIL" });
            ck(cu::cuGraphExecDestroy(gexec), "execDestroy");
            ck(cu::cuGraphDestroy(graph), "graphDestroy");
        }
    }
}
