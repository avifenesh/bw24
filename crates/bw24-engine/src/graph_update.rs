//! CUDA-graph exec-update (shared, model-agnostic): capture a decode step ONCE, then
//! re-tune individual kernel nodes' launch geometry per token via
//! `cuGraphExecKernelNodeSetParams` — the llama.cpp graph-serving mechanism (their decode
//! replays one instantiated graph per token with exact per-token grid shapes; nsys shows
//! zero launch gaps AND eager-exact grids, where a fixed-bucket replay wastes split blocks).
//!
//! Mechanism: `cuGraphKernelNodeGetParams_v2` returns the node's `CUDA_KERNEL_NODE_PARAMS`
//! whose `kernelParams` staging is DRIVER-OWNED and stays valid for the node's lifetime —
//! scalar args are updated by writing through those pointers, geometry by editing the
//! struct's gridDim fields, then `cuGraphExecKernelNodeSetParams` pushes the new params
//! into the instantiated exec (topology-preserving update; no re-instantiate).
//!
//! Safety model: every function here takes the raw handles from a live
//! [`cudarc::driver::CudaGraph`] (which owns destruction); callers must keep that graph
//! (and the capture keeper) alive while updating/launching.

use cudarc::driver::sys;

/// One kernel node of a captured graph: raw node handle, its full launch params
/// (grid/block/smem + driver-owned `kernelParams` staging), and the resolved symbol name.
pub struct KernelNode {
    pub node: sys::CUgraphNode,
    pub params: sys::CUDA_KERNEL_NODE_PARAMS,
    pub name: String,
}

// The raw CUgraphNode/param pointers are context-bound, not thread-bound; the Engine
// already serializes all graph work on its decode stream's thread.
unsafe impl Send for KernelNode {}

fn cu_try(r: sys::CUresult, what: &str) -> Result<(), Box<dyn std::error::Error>> {
    if r == sys::CUresult::CUDA_SUCCESS { Ok(()) } else { Err(format!("{what}: {r:?}").into()) }
}

/// Enumerate every KERNEL node of a captured graph with its launch params and symbol name.
/// Non-kernel nodes (memcpy/memset/empty) are skipped — geometry updates only apply to
/// kernel nodes; everything else replays as captured.
pub fn kernel_nodes(graph: &cudarc::driver::CudaGraph)
    -> Result<Vec<KernelNode>, Box<dyn std::error::Error>>
{
    let g = graph.cu_graph();
    let mut n: usize = 0;
    unsafe { cu_try(sys::cuGraphGetNodes(g, std::ptr::null_mut(), &mut n), "cuGraphGetNodes(count)")?; }
    let mut nodes: Vec<sys::CUgraphNode> = vec![std::ptr::null_mut(); n];
    unsafe { cu_try(sys::cuGraphGetNodes(g, nodes.as_mut_ptr(), &mut n), "cuGraphGetNodes")?; }
    nodes.truncate(n);
    let mut out = Vec::with_capacity(n);
    for node in nodes {
        let mut ty = sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_EMPTY;
        unsafe { cu_try(sys::cuGraphNodeGetType(node, &mut ty), "cuGraphNodeGetType")?; }
        if ty != sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL { continue; }
        let mut params: sys::CUDA_KERNEL_NODE_PARAMS = unsafe { std::mem::zeroed() };
        unsafe { cu_try(sys::cuGraphKernelNodeGetParams_v2(node, &mut params),
                        "cuGraphKernelNodeGetParams_v2")?; }
        let mut cname: *const std::ffi::c_char = std::ptr::null();
        let name = unsafe {
            if sys::cuFuncGetName(&mut cname, params.func) == sys::CUresult::CUDA_SUCCESS
                && !cname.is_null() {
                std::ffi::CStr::from_ptr(cname).to_string_lossy().into_owned()
            } else { String::from("<unknown>") }
        };
        out.push(KernelNode { node, params, name });
    }
    Ok(out)
}

/// Push updated launch params for one node into the instantiated exec. `params` is the
/// (edited) struct from [`kernel_nodes`] — same node topology, new geometry/arg values.
pub fn set_exec_params(graph: &cudarc::driver::CudaGraph, node: sys::CUgraphNode,
                       params: &sys::CUDA_KERNEL_NODE_PARAMS)
    -> Result<(), Box<dyn std::error::Error>>
{
    unsafe { cu_try(sys::cuGraphExecKernelNodeSetParams_v2(graph.cu_graph_exec(), node, params),
                    "cuGraphExecKernelNodeSetParams_v2") }
}

/// Overwrite one i32 scalar argument in the node's driver-owned kernelParams staging.
/// `idx` is the kernel's parameter position (launch_builder arg order). The write alone
/// does NOT reach the exec — call [`set_exec_params`] after editing to push the change.
///
/// # Safety
/// `idx` must be a valid parameter index for the node's kernel and that parameter must be
/// a 4-byte scalar; writing a wrong slot corrupts the launch.
pub unsafe fn write_i32_arg(params: &sys::CUDA_KERNEL_NODE_PARAMS, idx: usize, val: i32) {
    unsafe {
        let slot = *params.kernelParams.add(idx) as *mut i32;
        *slot = val;
    }
}

/// Read an i32 scalar argument from the node's kernelParams staging (see [`write_i32_arg`]).
///
/// # Safety
/// Same contract as [`write_i32_arg`] — `idx` must name a 4-byte scalar parameter.
pub unsafe fn read_i32_arg(params: &sys::CUDA_KERNEL_NODE_PARAMS, idx: usize) -> i32 {
    unsafe { *(*params.kernelParams.add(idx) as *const i32) }
}

/// Read a pointer-valued argument (device pointer as u64) from kernelParams staging.
///
/// # Safety
/// `idx` must name an 8-byte pointer parameter.
pub unsafe fn read_ptr_arg(params: &sys::CUDA_KERNEL_NODE_PARAMS, idx: usize) -> u64 {
    unsafe { *(*params.kernelParams.add(idx) as *const u64) }
}

/// One fa-decode main node with its paired combine — the per-token geometry-update unit.
///
/// Both classes get the FULL update (grid.y + n_splits arg + paired combine's n_splits):
/// the partial buffers are `zeros()` allocations whose memset is CAPTURED — every replay
/// re-zeroes them, so any split slot the main doesn't write holds m=0.0 (NOT the NEG_INF
/// empty the combine skips). The combine's merge count must therefore exactly equal the
/// main's written split count. `n_splits` is simultaneously the key partition and the
/// partial stride in every fa kernel, so main + combine move as one value:
/// - vec dc twins (`fa_decode_vec_q*_dc`): per = ceil(T_kv/n_splits), arg idx 11; the live
///   count comes from the caller's split ladder (eager lockstep).
/// - scalar unified (`fa_decode_f32`, ctr non-null): ns_eff = ceil(T_kv/split_keys) in-
///   kernel; setting n_splits (idx 12) = that same value keeps stride == partition.
pub struct FaMain {
    node: sys::CUgraphNode,
    params: sys::CUDA_KERNEL_NODE_PARAMS,
    /// gridDimX at capture = n_head_kv (vec) / n_head (scalar) — the split-ladder key.
    nkv: u32,
    /// captured grid.y — the bucket split count; live updates never exceed it (the partial
    /// buffers were sized for it).
    bucket_splits: u32,
    /// scalar-unified main: `split_keys` arg value (read at plan build) — grid-only shrink.
    self_split_keys: Option<i32>,
    combine: Option<(sys::CUgraphNode, sys::CUDA_KERNEL_NODE_PARAMS)>,
    /// last applied split count — updates are pushed only on change (splits step every
    /// `split_keys` tokens, so exec updates are rare, not per-token).
    cur: u32,
}

unsafe impl Send for FaMain {}

const VEC_NSP_IDX: usize = 11;      // Q,K,V,pO,pM,pL,hd,nh,nhkv,ctr,scale,[n_splits],ktb,vtb
const VEC_PARTO_IDX: usize = 3;
const SCALAR_NSP_IDX: usize = 12;   // ...,hd,nh,nhkv,tkv_host,ctr,scale,[n_splits],[split_keys],...
const SCALAR_SKI_IDX: usize = 13;
const COMBINE_NSP_IDX: usize = 6;   // pO,pM,pL,O,hd,nh,[n_splits]
const COMBINE_PARTO_IDX: usize = 0;

/// Classify a captured graph's fa-decode nodes into per-token-updatable [`FaMain`]s.
/// Pairing main->combine is by partO pointer identity (arg staging), not node order.
/// Nodes that aren't fa mains/combines are left untouched (they replay as captured).
pub fn fa_plan(graph: &cudarc::driver::CudaGraph)
    -> Result<Vec<FaMain>, Box<dyn std::error::Error>>
{
    let nodes = kernel_nodes(graph)?;
    // partO POINTERS ARE NOT UNIQUE: the partial buffers are pool transients, freed per
    // layer and reused by the next — pointer identity alone pairs many mains to one
    // combine (the token-2 corruption, 2026-07-12). Pair 1:1 in NODE ORDER: each main
    // takes the first unconsumed combine AFTER it whose partO pointer matches (single-
    // stream capture appends nodes in issue order, and the combine is always issued
    // right after its main within one fa_decode_* call).
    let mut mains: Vec<(usize, FaMain)> = Vec::new();
    let mut combines: Vec<Option<(usize, u64, sys::CUgraphNode, sys::CUDA_KERNEL_NODE_PARAMS)>> =
        Vec::new();
    for (i, n) in nodes.iter().enumerate() {
        match n.name.as_str() {
            "fa_decode_vec_q_v4_dc" | "fa_decode_vec_q_v3_dc" | "fa_decode_vec_q_v2_dc"
            | "fa_decode_vec_q_dc" | "fa_decode_vec_q_dpl16_dc" => {
                mains.push((i, FaMain {
                    nkv: n.params.gridDimX, bucket_splits: n.params.gridDimY,
                    self_split_keys: None, combine: None, cur: n.params.gridDimY,
                    node: n.node, params: n.params,
                }));
            }
            "fa_decode_f32" => {
                let ski = unsafe { read_i32_arg(&n.params, SCALAR_SKI_IDX) };
                mains.push((i, FaMain {
                    nkv: n.params.gridDimX, bucket_splits: n.params.gridDimY,
                    self_split_keys: Some(ski), combine: None, cur: n.params.gridDimY,
                    node: n.node, params: n.params,
                }));
            }
            "fa_decode_combine_f32" => {
                let po = unsafe { read_ptr_arg(&n.params, COMBINE_PARTO_IDX) };
                combines.push(Some((i, po, n.node, n.params)));
            }
            _ => {}
        }
    }
    let mut out = Vec::with_capacity(mains.len());
    for (mi, mut m) in mains {
        let po = unsafe { read_ptr_arg(&m.params, VEC_PARTO_IDX) };
        let slot = combines.iter_mut()
            .filter(|c| c.as_ref().is_some_and(|(ci, cpo, ..)| *ci > mi && *cpo == po))
            .min_by_key(|c| c.as_ref().unwrap().0);
        match slot {
            Some(c) => { let (_, _, cn, cp) = c.take().unwrap(); m.combine = Some((cn, cp)); }
            // a main without its combine cannot be updated consistently (stride vs merge
            // count would diverge) — refuse loudly rather than corrupt replays.
            None => return Err("fa_plan: fa main has no partO-paired combine node".into()),
        }
        out.push(m);
    }
    Ok(out)
}

/// Retune every fa main (and paired combine) in the instantiated exec to the live `t_kv`:
/// vec mains get the EAGER split count ns = ceil(t_kv/split_keys(t_kv, nkv)); scalar mains
/// shrink grid.y to their in-kernel ns_eff. No-op when the counts haven't stepped.
/// `split_keys` is the caller's ladder (fa_split_keys) so graph and eager stay in lockstep.
pub fn fa_apply(graph: &cudarc::driver::CudaGraph, plan: &mut [FaMain], t_kv: usize,
                split_keys: impl Fn(usize, usize) -> usize)
    -> Result<(), Box<dyn std::error::Error>>
{
    for m in plan.iter_mut() {
        let ns = match m.self_split_keys {
            Some(ski) => (t_kv + ski as usize - 1) / (ski as usize).max(1),
            None => { let sp = split_keys(t_kv, m.nkv as usize).max(1);
                      (t_kv + sp - 1) / sp }
        }.max(1) as u32;
        let ns = ns.min(m.bucket_splits);
        if ns == m.cur { continue; }
        m.params.gridDimY = ns;
        let nsp_idx = if m.self_split_keys.is_some() { SCALAR_NSP_IDX } else { VEC_NSP_IDX };
        unsafe { write_i32_arg(&m.params, nsp_idx, ns as i32); }
        set_exec_params(graph, m.node, &m.params)?;
        if let Some((cn, cp)) = &m.combine {
            unsafe { write_i32_arg(cp, COMBINE_NSP_IDX, ns as i32); }
            set_exec_params(graph, *cn, cp)?;
        }
        m.cur = ns;
    }
    Ok(())
}
