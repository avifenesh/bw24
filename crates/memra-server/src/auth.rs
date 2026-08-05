//! API-key management (lane/api-keys, 2026-08-05): multi-key bearer auth that maps
//! key -> tenant, so metering, QoS lane class, and prefix-cache isolation key off a real
//! tenant identity instead of one shared trust domain.
//!
//! DESIGN (launch-shaped, not enterprise-shaped):
//!   - Keyring source: `MEMRA_API_KEYS` — a TOML file path (`[[keys]]` entries, see
//!     `KeyEntry`) or an inline env list `tenant:sha256hex[:lane],...` for
//!     file-less deploys. Keys are stored as SHA-256 hex ONLY — the plaintext exists
//!     exactly once, on the `--gen-key` terminal.
//!   - Hot reload: mtime-poll (default 2s throttle) on every lookup — chosen over SIGHUP
//!     because it needs no signal thread and cannot be missed; a bad reload keeps the old
//!     ring and logs loudly (auth never degrades to open because of a typo).
//!   - Back-compat: `MEMRA_API_KEY` (the single static bearer — the owner's daily driver
//!     and every serve script) keeps working unchanged as tenant `"default"`, with or
//!     without a keyring configured. No keyring + no single key = open (dev behavior).
//!   - Tenant -> cache namespace: when a keyring is configured, every request's PC-ISO
//!     namespace is `t:<tenant>\x1f<cache_salt>` (see `scope_namespace`). Tenant ids are
//!     validated `[A-Za-z0-9_-]+`, so the `\x1f` separator cannot be forged from a
//!     client-controlled `cache_salt` — cross-tenant cache probing is structurally
//!     impossible. Keyring ABSENT keeps the raw-salt namespace, byte-identical to PC-ISO.
//!   - Lane class: a key is `interactive` (default) or `batch`. Batch-class keys default
//!     to the harvest QoS lane and are refused the protected interactive lane (403, loud
//!     — never a silent downgrade, per the honesty doctrine).
//!   - Per-key `rate_limit`: optional concurrency-slot override; the effective cap is
//!     min(override, global lane cap) — the global cap stays authoritative.
//!
//! CLI (`--gen-key` / `--revoke-key`, see `run_cli`): prints the plaintext once and
//! appends the hash entry; revoke flips `enabled = false` by key prefix. No web UI.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::RwLock;
use std::time::{Duration, Instant, SystemTime};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// The namespace separator between tenant id and client salt. Tenant ids are validated
/// `[A-Za-z0-9_-]+`, so no client-controlled string can produce a colliding namespace.
const NS_SEP: char = '\u{1f}';

/// SHA-256 of a plaintext key, lower-case hex — the only form a key is ever stored in.
pub fn sha256_hex(key: &str) -> String {
    let mut h = Sha256::new();
    h.update(key.as_bytes());
    let digest = h.finalize();
    let mut out = String::with_capacity(64);
    for b in digest {
        out.push_str(&format!("{b:02x}"));
    }
    out
}

/// A key's QoS lane class. `Interactive` keys behave exactly like pre-lane traffic
/// (default lane interactive, any `x-lane` honored). `Batch` keys default to the
/// harvest lane and may not claim the protected interactive lane.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LaneClass {
    #[default]
    Interactive,
    Batch,
}

impl LaneClass {
    pub fn parse(v: &str) -> Option<LaneClass> {
        match v {
            "interactive" => Some(LaneClass::Interactive),
            "batch" => Some(LaneClass::Batch),
            _ => None,
        }
    }
    pub fn as_str(&self) -> &'static str {
        match self {
            LaneClass::Interactive => "interactive",
            LaneClass::Batch => "batch",
        }
    }
}

/// One keyring entry — the on-disk TOML shape (`[[keys]]`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyEntry {
    /// Identification prefix of the plaintext key (safe to store/display; the revoke
    /// handle). `mk-<tenant>-<first 12 hex>` for generated keys.
    #[serde(default)]
    pub prefix: String,
    /// SHA-256 hex of the full plaintext key. Never the plaintext.
    pub sha256: String,
    pub tenant: String,
    /// "interactive" (default) | "batch".
    #[serde(default)]
    pub lane: Option<String>,
    /// enabled=false = revoked: the key authenticates as DISABLED (403), distinct from
    /// unknown (401) so a revoked caller gets an actionable error.
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Optional per-key concurrency-slot override; effective cap is
    /// min(rate_limit, global lane cap).
    #[serde(default)]
    pub rate_limit: Option<usize>,
    /// Unix seconds at generation (informational).
    #[serde(default)]
    pub created_unix: Option<u64>,
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Serialize, Deserialize, Default)]
struct KeyFile {
    #[serde(default)]
    keys: Vec<KeyEntry>,
}

/// The resolved identity a request acts as — what flows to cache scoping, lane
/// admission, rate-limit headers, and the usage/meter log line.
#[derive(Debug, Clone, PartialEq)]
pub struct TenantCtx {
    pub tenant: String,
    pub lane_class: LaneClass,
    pub rate_limit: Option<usize>,
}

impl TenantCtx {
    /// The single-key / open-server identity: tenant "default", interactive, no override.
    pub fn default_tenant() -> Self {
        TenantCtx {
            tenant: "default".into(),
            lane_class: LaneClass::Interactive,
            rate_limit: None,
        }
    }
}

/// Why a presented key was refused. `Unknown` -> 401, `Disabled` -> 403.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthDenied {
    Unknown,
    Disabled,
}

/// Parsed keyring: hash -> entry.
#[derive(Debug, Default)]
pub struct Keyring {
    by_hash: HashMap<String, KeyEntry>,
}

fn valid_tenant(t: &str) -> bool {
    !t.is_empty()
        && t.chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}

fn valid_sha256(s: &str) -> bool {
    s.len() == 64 && s.chars().all(|c| c.is_ascii_hexdigit())
}

impl Keyring {
    /// Build from entries, validating every field (a keyring with a malformed entry is
    /// refused whole — auth config errors must be loud, never partially applied).
    pub fn from_entries(entries: Vec<KeyEntry>) -> Result<Keyring, String> {
        let mut by_hash = HashMap::with_capacity(entries.len());
        for (i, mut e) in entries.into_iter().enumerate() {
            if !valid_tenant(&e.tenant) {
                return Err(format!(
                    "key entry {i}: bad tenant {:?} (want [A-Za-z0-9_-]+)", e.tenant));
            }
            e.sha256 = e.sha256.to_lowercase();
            if !valid_sha256(&e.sha256) {
                return Err(format!(
                    "key entry {i} (tenant {:?}): sha256 must be 64 hex chars", e.tenant));
            }
            if let Some(lane) = e.lane.as_deref() {
                if LaneClass::parse(lane).is_none() {
                    return Err(format!(
                        "key entry {i} (tenant {:?}): bad lane {lane:?} (interactive|batch)",
                        e.tenant));
                }
            }
            if e.rate_limit == Some(0) {
                return Err(format!(
                    "key entry {i} (tenant {:?}): rate_limit 0 would admit nothing — \
                     use enabled = false to revoke", e.tenant));
            }
            if by_hash.insert(e.sha256.clone(), e).is_some() {
                return Err(format!("key entry {i}: duplicate sha256 (same key listed twice)"));
            }
        }
        Ok(Keyring { by_hash })
    }

    /// Parse the TOML file form.
    pub fn from_toml(text: &str) -> Result<Keyring, String> {
        let f: KeyFile = toml::from_str(text).map_err(|e| format!("keys.toml parse: {e}"))?;
        Keyring::from_entries(f.keys)
    }

    /// Parse the inline env-list form: `tenant:sha256hex[:lane],...` — the file-less
    /// fallback. Revocation in this form = remove the entry (no disabled state).
    pub fn from_inline(spec: &str) -> Result<Keyring, String> {
        let mut entries = Vec::new();
        for part in spec.split(',').filter(|s| !s.trim().is_empty()) {
            let fields: Vec<&str> = part.trim().split(':').collect();
            if fields.len() < 2 || fields.len() > 3 {
                return Err(format!(
                    "bad MEMRA_API_KEYS inline entry {part:?} (want tenant:sha256hex[:lane])"));
            }
            entries.push(KeyEntry {
                prefix: String::new(),
                sha256: fields[1].to_string(),
                tenant: fields[0].to_string(),
                lane: fields.get(2).map(|s| s.to_string()),
                enabled: true,
                rate_limit: None,
                created_unix: None,
            });
        }
        if entries.is_empty() {
            return Err("MEMRA_API_KEYS inline list is empty".into());
        }
        Keyring::from_entries(entries)
    }

    pub fn len(&self) -> usize {
        self.by_hash.len()
    }

    /// Authenticate a plaintext bearer key against the ring.
    pub fn lookup(&self, key: &str) -> Result<TenantCtx, AuthDenied> {
        match self.by_hash.get(&sha256_hex(key)) {
            None => Err(AuthDenied::Unknown),
            Some(e) if !e.enabled => Err(AuthDenied::Disabled),
            Some(e) => Ok(TenantCtx {
                tenant: e.tenant.clone(),
                lane_class: e
                    .lane
                    .as_deref()
                    .and_then(LaneClass::parse)
                    .unwrap_or_default(),
                rate_limit: e.rate_limit,
            }),
        }
    }
}

/// The live keyring: source + hot-reload state. File-backed rings re-stat on lookup
/// (throttled to `poll`) and swap in the new ring when mtime moves; a reload that fails
/// to parse KEEPS the old ring and logs the error (never fail-open, never flap).
pub struct KeyStore {
    source: Source,
    poll: Duration,
    state: RwLock<State>,
}

enum Source {
    File(PathBuf),
    Inline,
}

struct State {
    ring: Keyring,
    mtime: Option<SystemTime>,
    checked: Instant,
}

fn file_mtime(p: &Path) -> Option<SystemTime> {
    std::fs::metadata(p).and_then(|m| m.modified()).ok()
}

impl KeyStore {
    /// Resolve the `MEMRA_API_KEYS` value: an existing file path loads as TOML;
    /// otherwise a value containing ':' parses as the inline list; anything else is a
    /// loud config error (a mistyped path must not silently become an empty ring).
    pub fn from_spec(spec: &str) -> Result<KeyStore, String> {
        let p = Path::new(spec);
        if p.is_file() {
            let text = std::fs::read_to_string(p)
                .map_err(|e| format!("MEMRA_API_KEYS {spec:?}: {e}"))?;
            let ring = Keyring::from_toml(&text).map_err(|e| format!("{spec}: {e}"))?;
            let n = ring.len();
            eprintln!("[auth] keyring loaded: {n} key(s) from {spec}");
            return Ok(KeyStore {
                source: Source::File(p.to_path_buf()),
                poll: Duration::from_secs(2),
                state: RwLock::new(State {
                    ring,
                    mtime: file_mtime(p),
                    checked: Instant::now(),
                }),
            });
        }
        if spec.contains(':') {
            let ring = Keyring::from_inline(spec)?;
            eprintln!("[auth] keyring loaded: {} inline key(s)", ring.len());
            return Ok(KeyStore {
                source: Source::Inline,
                poll: Duration::from_secs(2),
                state: RwLock::new(State {
                    ring,
                    mtime: None,
                    checked: Instant::now(),
                }),
            });
        }
        Err(format!(
            "MEMRA_API_KEYS={spec:?} is neither an existing keys.toml path nor an inline \
             tenant:sha256hex list"))
    }

    #[cfg(test)]
    fn with_poll(mut self, poll: Duration) -> KeyStore {
        self.poll = poll;
        self
    }

    /// Hot reload: if the file's mtime moved since the last (throttled) check, swap in
    /// the re-parsed ring. Parse failure keeps the old ring and logs.
    fn maybe_reload(&self) {
        let Source::File(path) = &self.source else { return };
        {
            let st = self.state.read().unwrap();
            if st.checked.elapsed() < self.poll {
                return;
            }
        }
        let mut st = self.state.write().unwrap();
        if st.checked.elapsed() < self.poll {
            return; // another thread just did the check
        }
        st.checked = Instant::now();
        let mtime = file_mtime(path);
        if mtime == st.mtime {
            return;
        }
        match std::fs::read_to_string(path).map_err(|e| e.to_string())
            .and_then(|t| Keyring::from_toml(&t))
        {
            Ok(ring) => {
                eprintln!("[auth] keyring reloaded: {} key(s) from {}", ring.len(),
                          path.display());
                st.ring = ring;
                st.mtime = mtime;
            }
            Err(e) => {
                eprintln!("[auth] keyring reload FAILED ({e}); keeping the previous ring");
                st.mtime = mtime; // don't re-log every poll until the file changes again
            }
        }
    }

    pub fn lookup(&self, key: &str) -> Result<TenantCtx, AuthDenied> {
        self.maybe_reload();
        self.state.read().unwrap().ring.lookup(key)
    }
}

/// Process-global keystore, initialized once from `MEMRA_API_KEYS` at startup
/// (`init_from_env` — a bad config is a startup FATAL, not a per-request surprise).
static KEYSTORE: std::sync::OnceLock<Option<KeyStore>> = std::sync::OnceLock::new();

/// Called once from main() before serving. Exits the process on a bad config.
pub fn init_from_env() {
    KEYSTORE.get_or_init(|| match std::env::var("MEMRA_API_KEYS") {
        Err(_) => None,
        Ok(spec) => match KeyStore::from_spec(&spec) {
            Ok(ks) => Some(ks),
            Err(e) => {
                eprintln!("[auth] FATAL: {e}");
                std::process::exit(1);
            }
        },
    });
}

/// The global keystore, if `MEMRA_API_KEYS` configured one.
pub fn global() -> Option<&'static KeyStore> {
    KEYSTORE.get().and_then(|o| o.as_ref())
}

/// The full auth law, pure over its inputs (unit-testable without env):
///   keyring key match      -> that key's tenant (or Disabled -> 403)
///   single static key match-> tenant "default" (the back-compat daily driver)
///   nothing configured     -> open, tenant "default"
///   anything else          -> Unknown -> 401
/// The keyring and the single key COMPOSE: setting MEMRA_API_KEYS does not break
/// MEMRA_API_KEY callers (the owner's serve scripts keep working unchanged).
pub fn authenticate_with(
    keyring: Option<&KeyStore>,
    single_key: Option<&str>,
    bearer: Option<&str>,
) -> Result<TenantCtx, AuthDenied> {
    if keyring.is_none() && single_key.is_none() {
        return Ok(TenantCtx::default_tenant()); // open server (dev behavior)
    }
    let Some(candidate) = bearer else {
        return Err(AuthDenied::Unknown);
    };
    if let Some(ks) = keyring {
        match ks.lookup(candidate) {
            Ok(ctx) => return Ok(ctx),
            Err(AuthDenied::Disabled) => return Err(AuthDenied::Disabled),
            Err(AuthDenied::Unknown) => {} // fall through to the single key
        }
    }
    if single_key.is_some_and(|k| k == candidate) {
        return Ok(TenantCtx::default_tenant());
    }
    Err(AuthDenied::Unknown)
}

/// PC-ISO namespace scoping: with a keyring configured, a request's cache namespace is
/// `t:<tenant>\x1f<salt>` — a tenant's keys share cache; different tenants never do; a
/// client-controlled salt cannot cross the `\x1f` boundary (tenant ids exclude it).
/// Without a keyring the raw salt passes through, byte-identical to PC-ISO behavior.
pub fn scope_namespace(tenant: &str, raw_salt: &str) -> String {
    format!("t:{tenant}{NS_SEP}{raw_salt}")
}

// ---- key lifecycle CLI (--gen-key / --revoke-key) ----

/// 24 bytes of /dev/urandom as 48 hex chars — the key's secret part.
fn random_hex48() -> Result<String, String> {
    use std::io::Read;
    let mut f = std::fs::File::open("/dev/urandom")
        .map_err(|e| format!("/dev/urandom: {e}"))?;
    let mut buf = [0u8; 24];
    f.read_exact(&mut buf).map_err(|e| format!("/dev/urandom read: {e}"))?;
    Ok(buf.iter().map(|b| format!("{b:02x}")).collect())
}

/// Generate a key for `tenant`, print the plaintext ONCE, append the hash entry to the
/// keys file (created if missing). Returns the plaintext (for tests).
pub fn gen_key(
    keys_path: &Path,
    tenant: &str,
    lane: LaneClass,
    rate_limit: Option<usize>,
) -> Result<String, String> {
    if !valid_tenant(tenant) {
        return Err(format!("bad tenant {tenant:?} (want [A-Za-z0-9_-]+)"));
    }
    if rate_limit == Some(0) {
        return Err("rate limit 0 would admit nothing".into());
    }
    let secret = random_hex48()?;
    let key = format!("mk-{tenant}-{secret}");
    let prefix = format!("mk-{tenant}-{}", &secret[..12]);
    let created = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    // Validate the existing file first (never append to a broken ring), and refuse a
    // prefix collision (the revoke handle must stay unambiguous).
    if keys_path.is_file() {
        let text = std::fs::read_to_string(keys_path)
            .map_err(|e| format!("{}: {e}", keys_path.display()))?;
        let f: KeyFile =
            toml::from_str(&text).map_err(|e| format!("{}: {e}", keys_path.display()))?;
        Keyring::from_entries(f.keys.clone())?;
        if f.keys.iter().any(|e| e.prefix == prefix) {
            return Err(format!("prefix {prefix} already exists (rerun to draw a new key)"));
        }
    }

    // Textual append preserves the file's comments; --revoke-key rewrites (see below).
    let mut fragment = String::new();
    if !keys_path.is_file() {
        fragment.push_str(
            "# memra API keyring (MEMRA_API_KEYS points here).\n\
             # Entries store SHA-256 of the key, never the plaintext. Managed by\n\
             # `memra-server --gen-key <tenant>` / `--revoke-key <prefix>` (revoke\n\
             # rewrites the file; comments outside this header are not preserved).\n");
    }
    fragment.push_str(&format!(
        "\n[[keys]]\nprefix = \"{prefix}\"\nsha256 = \"{}\"\ntenant = \"{tenant}\"\n\
         lane = \"{}\"\nenabled = true\ncreated_unix = {created}\n",
        sha256_hex(&key),
        lane.as_str()));
    if let Some(rl) = rate_limit {
        fragment.push_str(&format!("rate_limit = {rl}\n"));
    }
    use std::io::Write;
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(keys_path)
        .map_err(|e| format!("{}: {e}", keys_path.display()))?;
    f.write_all(fragment.as_bytes())
        .map_err(|e| format!("{}: {e}", keys_path.display()))?;
    Ok(key)
}

/// Disable every key whose `prefix` starts with `handle` (or whose sha256 matches the
/// handle's hash, if a full plaintext key was pasted). Exactly one match required —
/// ambiguity is an error, not a mass revoke. Rewrites the file (comments not preserved).
pub fn revoke_key(keys_path: &Path, handle: &str) -> Result<String, String> {
    let text = std::fs::read_to_string(keys_path)
        .map_err(|e| format!("{}: {e}", keys_path.display()))?;
    let mut f: KeyFile =
        toml::from_str(&text).map_err(|e| format!("{}: {e}", keys_path.display()))?;
    let full_hash = sha256_hex(handle);
    let matches: Vec<usize> = f
        .keys
        .iter()
        .enumerate()
        .filter(|(_, e)| {
            (!e.prefix.is_empty() && e.prefix.starts_with(handle)) || e.sha256 == full_hash
        })
        .map(|(i, _)| i)
        .collect();
    match matches.len() {
        0 => Err(format!("no key matches {handle:?}")),
        1 => {
            let i = matches[0];
            if !f.keys[i].enabled {
                return Err(format!("key {} is already revoked", f.keys[i].prefix));
            }
            f.keys[i].enabled = false;
            let revoked = f.keys[i].prefix.clone();
            let out = toml::to_string(&f).map_err(|e| e.to_string())?;
            std::fs::write(keys_path, out).map_err(|e| format!("{}: {e}", keys_path.display()))?;
            Ok(revoked)
        }
        n => Err(format!("{n} keys match {handle:?} — use a longer prefix")),
    }
}

/// CLI dispatch: handles `--gen-key` / `--revoke-key` if present, returning the exit
/// code; None = no key-management args, boot the server normally.
pub fn run_cli(args: &[String]) -> Option<i32> {
    let has = |flag: &str| args.iter().any(|a| a == flag);
    if !has("--gen-key") && !has("--revoke-key") {
        return None;
    }
    let value_of = |flag: &str| -> Option<String> {
        args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1).cloned())
    };
    let keys_path = value_of("--keys")
        .or_else(|| std::env::var("MEMRA_API_KEYS").ok())
        .map(PathBuf::from);
    let Some(keys_path) = keys_path else {
        eprintln!("error: no keys file — pass --keys /path/keys.toml or set MEMRA_API_KEYS");
        return Some(2);
    };
    if keys_path.exists() && !keys_path.is_file() {
        eprintln!("error: {} is not a file", keys_path.display());
        return Some(2);
    }

    if has("--gen-key") {
        let Some(tenant) = value_of("--gen-key") else {
            eprintln!("usage: memra-server --gen-key <tenant> [--lane interactive|batch] \
                       [--rate-limit N] [--keys /path/keys.toml]");
            return Some(2);
        };
        let lane = match value_of("--lane") {
            None => LaneClass::Interactive,
            Some(v) => match LaneClass::parse(&v) {
                Some(l) => l,
                None => {
                    eprintln!("error: bad --lane {v:?} (interactive|batch)");
                    return Some(2);
                }
            },
        };
        let rate_limit = match value_of("--rate-limit") {
            None => None,
            Some(v) => match v.parse::<usize>() {
                Ok(n) => Some(n),
                Err(_) => {
                    eprintln!("error: bad --rate-limit {v:?} (want a positive integer)");
                    return Some(2);
                }
            },
        };
        return Some(match gen_key(&keys_path, &tenant, lane, rate_limit) {
            Ok(key) => {
                println!("{key}");
                eprintln!("[gen-key] tenant {tenant:?} lane {} appended to {} — \
                           the plaintext above is shown ONCE and stored only as SHA-256",
                          lane.as_str(), keys_path.display());
                0
            }
            Err(e) => {
                eprintln!("error: {e}");
                1
            }
        });
    }

    // --revoke-key
    let Some(handle) = value_of("--revoke-key") else {
        eprintln!("usage: memra-server --revoke-key <prefix> [--keys /path/keys.toml]");
        return Some(2);
    };
    Some(match revoke_key(&keys_path, &handle) {
        Ok(prefix) => {
            eprintln!("[revoke-key] {prefix} disabled in {} (takes effect on the next \
                       keyring poll, <=2s on a running server)", keys_path.display());
            0
        }
        Err(e) => {
            eprintln!("error: {e}");
            1
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmpfile(name: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!("memra_auth_{}_{name}", std::process::id()));
        let _ = std::fs::remove_file(&p);
        p
    }

    const K_A1: &str = "mk-acme-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const K_A2: &str = "mk-acme-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const K_B1: &str = "mk-blue-cccccccccccccccccccccccccccccccccccccccccccccccc";
    const K_DIS: &str = "mk-dead-dddddddddddddddddddddddddddddddddddddddddddddddd";

    fn toml_ring() -> String {
        format!(
            "[[keys]]\nprefix = \"mk-acme-aaaa\"\nsha256 = \"{}\"\ntenant = \"acme\"\n\n\
             [[keys]]\nprefix = \"mk-acme-bbbb\"\nsha256 = \"{}\"\ntenant = \"acme\"\n\
             rate_limit = 2\n\n\
             [[keys]]\nprefix = \"mk-blue-cccc\"\nsha256 = \"{}\"\ntenant = \"blue\"\n\
             lane = \"batch\"\n\n\
             [[keys]]\nprefix = \"mk-dead-dddd\"\nsha256 = \"{}\"\ntenant = \"dead\"\n\
             enabled = false\n",
            sha256_hex(K_A1), sha256_hex(K_A2), sha256_hex(K_B1), sha256_hex(K_DIS))
    }

    #[test]
    fn sha256_hex_matches_known_vector() {
        // sha256("abc") — the FIPS 180-2 test vector.
        assert_eq!(
            sha256_hex("abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    }

    #[test]
    fn toml_ring_parses_and_looks_up_by_hash() {
        let ring = Keyring::from_toml(&toml_ring()).unwrap();
        assert_eq!(ring.len(), 4);
        // valid key -> tenant ctx with its lane class + override.
        let ctx = ring.lookup(K_A1).unwrap();
        assert_eq!(ctx.tenant, "acme");
        assert_eq!(ctx.lane_class, LaneClass::Interactive);
        assert_eq!(ctx.rate_limit, None);
        let ctx = ring.lookup(K_A2).unwrap();
        assert_eq!(ctx.tenant, "acme");
        assert_eq!(ctx.rate_limit, Some(2));
        let ctx = ring.lookup(K_B1).unwrap();
        assert_eq!(ctx.tenant, "blue");
        assert_eq!(ctx.lane_class, LaneClass::Batch);
        // disabled -> Disabled (403), unknown -> Unknown (401).
        assert_eq!(ring.lookup(K_DIS).unwrap_err(), AuthDenied::Disabled);
        assert_eq!(ring.lookup("mk-nope-x").unwrap_err(), AuthDenied::Unknown);
        // the PLAINTEXT never appears in the ring's source.
        assert!(!toml_ring().contains(K_A1));
    }

    #[test]
    fn malformed_rings_are_loud_errors() {
        // bad tenant chars (anything outside [A-Za-z0-9_-] would weaken the \x1f
        // namespace law; the \x1f char itself can't even be written in TOML).
        let bad = format!(
            "[[keys]]\nsha256 = \"{}\"\ntenant = \"a b\"\n", sha256_hex("k"));
        assert!(Keyring::from_toml(&bad).unwrap_err().contains("bad tenant"));
        assert!(Keyring::from_entries(vec![KeyEntry {
            prefix: String::new(), sha256: sha256_hex("k"),
            tenant: format!("a{}b", '\u{1f}'), lane: None, enabled: true,
            rate_limit: None, created_unix: None,
        }]).unwrap_err().contains("bad tenant"));
        // short hash.
        let bad = "[[keys]]\nsha256 = \"abc123\"\ntenant = \"t\"\n";
        assert!(Keyring::from_toml(bad).unwrap_err().contains("64 hex"));
        // bad lane.
        let bad = format!(
            "[[keys]]\nsha256 = \"{}\"\ntenant = \"t\"\nlane = \"turbo\"\n", sha256_hex("k"));
        assert!(Keyring::from_toml(&bad).unwrap_err().contains("bad lane"));
        // duplicate key.
        let dup = format!(
            "[[keys]]\nsha256 = \"{h}\"\ntenant = \"t\"\n\n\
             [[keys]]\nsha256 = \"{h}\"\ntenant = \"u\"\n", h = sha256_hex("k"));
        assert!(Keyring::from_toml(&dup).unwrap_err().contains("duplicate"));
        // rate_limit 0.
        let z = format!(
            "[[keys]]\nsha256 = \"{}\"\ntenant = \"t\"\nrate_limit = 0\n", sha256_hex("k"));
        assert!(Keyring::from_toml(&z).unwrap_err().contains("rate_limit 0"));
    }

    #[test]
    fn inline_env_list_parses() {
        let spec = format!("acme:{},blue:{}:batch", sha256_hex(K_A1), sha256_hex(K_B1));
        let ring = Keyring::from_inline(&spec).unwrap();
        assert_eq!(ring.lookup(K_A1).unwrap().tenant, "acme");
        assert_eq!(ring.lookup(K_B1).unwrap().lane_class, LaneClass::Batch);
        assert!(Keyring::from_inline("no-colon-here").is_err());
        assert!(Keyring::from_inline("").is_err());
    }

    #[test]
    fn keystore_hot_reloads_on_mtime_change() {
        let path = tmpfile("reload.toml");
        std::fs::write(&path, toml_ring()).unwrap();
        let ks = KeyStore::from_spec(path.to_str().unwrap()).unwrap()
            .with_poll(Duration::ZERO);
        assert_eq!(ks.lookup(K_A1).unwrap().tenant, "acme");
        // revoke A1 out-of-band (what --revoke-key does) with a bumped mtime.
        let revoked = toml_ring().replace(
            &format!("sha256 = \"{}\"\ntenant = \"acme\"\n", sha256_hex(K_A1)),
            &format!("sha256 = \"{}\"\ntenant = \"acme\"\nenabled = false\n", sha256_hex(K_A1)));
        std::fs::write(&path, revoked).unwrap();
        let new_mtime = SystemTime::now() + Duration::from_secs(2);
        let f = std::fs::File::options().write(true).open(&path).unwrap();
        f.set_modified(new_mtime).unwrap();
        drop(f);
        assert_eq!(ks.lookup(K_A1).unwrap_err(), AuthDenied::Disabled,
                   "mtime bump must reload the ring");
        // a BROKEN rewrite keeps the previous ring (never fail-open).
        std::fs::write(&path, "keys = \"not a ring\"").unwrap();
        let f = std::fs::File::options().write(true).open(&path).unwrap();
        f.set_modified(new_mtime + Duration::from_secs(2)).unwrap();
        drop(f);
        assert_eq!(ks.lookup(K_A1).unwrap_err(), AuthDenied::Disabled,
                   "broken reload must keep the previous ring");
        assert_eq!(ks.lookup(K_B1).unwrap().tenant, "blue");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn auth_law_composes_keyring_and_single_key() {
        let path = tmpfile("law.toml");
        std::fs::write(&path, toml_ring()).unwrap();
        let ks = KeyStore::from_spec(path.to_str().unwrap()).unwrap();
        // keyring key -> its tenant; single key -> "default"; both live at once.
        assert_eq!(authenticate_with(Some(&ks), Some("daily"), Some(K_A1)).unwrap().tenant,
                   "acme");
        assert_eq!(authenticate_with(Some(&ks), Some("daily"), Some("daily")).unwrap(),
                   TenantCtx::default_tenant());
        // wrong key -> Unknown; disabled -> Disabled; missing header -> Unknown.
        assert_eq!(authenticate_with(Some(&ks), Some("daily"), Some("nope")).unwrap_err(),
                   AuthDenied::Unknown);
        assert_eq!(authenticate_with(Some(&ks), Some("daily"), Some(K_DIS)).unwrap_err(),
                   AuthDenied::Disabled);
        assert_eq!(authenticate_with(Some(&ks), Some("daily"), None).unwrap_err(),
                   AuthDenied::Unknown);
        // single key only (the back-compat daily driver): unchanged.
        assert_eq!(authenticate_with(None, Some("daily"), Some("daily")).unwrap(),
                   TenantCtx::default_tenant());
        assert_eq!(authenticate_with(None, Some("daily"), Some("x")).unwrap_err(),
                   AuthDenied::Unknown);
        // nothing configured: open.
        assert_eq!(authenticate_with(None, None, None).unwrap(), TenantCtx::default_tenant());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn namespace_scoping_is_tenant_separated_and_unforgeable() {
        // same tenant, two keys, same salt -> SAME namespace (a tenant's keys share cache).
        assert_eq!(scope_namespace("acme", "s"), scope_namespace("acme", "s"));
        // different tenants never collide, salted or not.
        assert_ne!(scope_namespace("acme", ""), scope_namespace("blue", ""));
        assert_ne!(scope_namespace("acme", "s"), scope_namespace("blue", "s"));
        // a client salt cannot forge another tenant's namespace: the separator \x1f is
        // excluded from tenant ids, so "t:blue\x1f" can only be produced BY tenant blue.
        let forged_salt = format!("blue{}", '\u{1f}'); // attacker-controlled cache_salt
        assert_ne!(scope_namespace("acme", &forged_salt), scope_namespace("blue", ""));
        // salted vs unsalted stay distinct within a tenant.
        assert_ne!(scope_namespace("acme", "s"), scope_namespace("acme", ""));
    }

    #[test]
    fn gen_key_prints_once_and_stores_only_the_hash() {
        let path = tmpfile("gen.toml");
        let key = gen_key(&path, "acme", LaneClass::Interactive, None).unwrap();
        assert!(key.starts_with("mk-acme-"));
        assert_eq!(key.len(), "mk-acme-".len() + 48);
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(!text.contains(&key), "plaintext must never reach the file");
        assert!(text.contains(&sha256_hex(&key)));
        // the ring authenticates the printed key.
        let ring = Keyring::from_toml(&text).unwrap();
        assert_eq!(ring.lookup(&key).unwrap().tenant, "acme");
        // a second key appends; a batch-lane + rate-limit key carries both fields.
        let key2 = gen_key(&path, "blue", LaneClass::Batch, Some(4)).unwrap();
        let ring = Keyring::from_toml(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(ring.len(), 2);
        let ctx = ring.lookup(&key2).unwrap();
        assert_eq!(ctx.lane_class, LaneClass::Batch);
        assert_eq!(ctx.rate_limit, Some(4));
        // bad tenant is refused before touching the file.
        assert!(gen_key(&path, "bad tenant", LaneClass::Interactive, None).is_err());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn revoke_key_flips_enabled_by_prefix_exactly_once() {
        let path = tmpfile("revoke.toml");
        let key_a = gen_key(&path, "acme", LaneClass::Interactive, None).unwrap();
        let key_b = gen_key(&path, "acme", LaneClass::Interactive, None).unwrap();
        // ambiguous prefix (both start mk-acme-) -> error, nothing revoked.
        assert!(revoke_key(&path, "mk-acme-").unwrap_err().contains("2 keys"));
        // unique prefix -> revoked; the other key still works; re-revoke errors.
        let prefix_a = format!("mk-acme-{}", &key_a["mk-acme-".len().."mk-acme-".len() + 12]);
        revoke_key(&path, &prefix_a).unwrap();
        let ring = Keyring::from_toml(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(ring.lookup(&key_a).unwrap_err(), AuthDenied::Disabled);
        assert_eq!(ring.lookup(&key_b).unwrap().tenant, "acme");
        assert!(revoke_key(&path, &prefix_a).unwrap_err().contains("already revoked"));
        // full plaintext key also works as the handle (hash match).
        revoke_key(&path, &key_b).unwrap();
        let ring = Keyring::from_toml(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(ring.lookup(&key_b).unwrap_err(), AuthDenied::Disabled);
        // no match -> error.
        assert!(revoke_key(&path, "mk-zzz").unwrap_err().contains("no key"));
        let _ = std::fs::remove_file(&path);
    }
}
