use crate::bytes::{Bytes, Text};
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::fmt;

/// User verification policy for a credential key (chosen at creation).
///
/// Mapped to Secure Enclave access control on macOS. Soft backend stores the
/// value for honest UP/UV flags only.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum UvPolicy {
    /// No SE user interaction. UV must not be claimed. UP comes from an
    /// explicit non-biometric presence check (LocalAuthentication with
    /// passcode allowed, or a fixture script).
    None = 0,
    /// Touch ID or device password on every private key use (`kSecAccessControlUserPresence`).
    Presence = 1,
    /// Touch ID with the currently enrolled set only (`kSecAccessControlBiometryCurrentSet`).
    Biometry = 2,
}

impl UvPolicy {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::None),
            1 => Some(Self::Presence),
            2 => Some(Self::Biometry),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Presence => "presence",
            Self::Biometry => "biometry",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "none" => Some(Self::None),
            "presence" => Some(Self::Presence),
            "biometry" => Some(Self::Biometry),
            _ => None,
        }
    }

    /// Whether this policy can produce a UV bit after a successful use.
    pub fn supports_uv(self) -> bool {
        !matches!(self, Self::None)
    }
}

/// A stored credential. Every credential the authenticator creates is kept
/// in the store; `discoverable` marks resident ("rk") credentials.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Credential {
    /// Legacy IDs are [`CREDENTIAL_ID_LEN`] random bytes. Extended IDs add
    /// protection metadata before the same random bytes.
    pub id: Bytes,
    pub rp_id: Text,
    pub rp_name: Option<Text>,
    pub user_id: Bytes,
    pub user_name: Option<Text>,
    pub user_display_name: Option<Text>,
    /// Opaque backend reference to the private key.
    pub key_handle: Bytes,
    /// SEC1 uncompressed public key (65 bytes).
    pub public_key: Bytes,
    pub discoverable: bool,
    /// Creation order, larger is newer.
    pub created: u64,
    /// UV policy bound into the SE key (or recorded for the soft backend).
    pub uv_policy: UvPolicy,
}

impl Credential {
    /// Protection metadata is part of the stored opaque credential ID.
    ///
    /// Extended ID format: the 4 ASCII bytes `SFCP`, one level byte (1, 2 or
    /// 3), then the [`CREDENTIAL_ID_LEN`] random bytes, 37 bytes in total.
    /// The level is therefore visible to the relying party that holds the ID.
    /// Legacy IDs (exactly [`CREDENTIAL_ID_LEN`] bytes) have no metadata and
    /// map to level 1. An extended ID with an unknown level byte is treated
    /// as level 3, the most restrictive.
    pub fn cred_protect(&self) -> u8 {
        if self.id.len() == CREDENTIAL_ID_LEN + 5 && self.id.starts_with(b"SFCP") {
            match self.id[4] {
                1..=3 => self.id[4],
                _ => 3,
            }
        } else {
            1
        }
    }

    pub fn allows_assertion(&self, using_allow_list: bool, user_verified: bool) -> bool {
        match self.cred_protect() {
            1 => true,
            2 => using_allow_list || user_verified,
            _ => user_verified,
        }
    }
}

pub const CREDENTIAL_ID_LEN: usize = 32;

#[derive(Debug, Clone)]
pub struct NewKey {
    pub handle: Bytes,
    pub public_key: p256::PublicKey,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackendError {
    KeyNotFound,
    Cancelled,
    Failed(String),
}

impl fmt::Display for BackendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BackendError::KeyNotFound => f.write_str("key not found"),
            BackendError::Cancelled => f.write_str("cancelled by user"),
            BackendError::Failed(m) => write!(f, "backend failure: {m}"),
        }
    }
}

/// Creates and uses P-256 private keys; also the source of randomness.
pub trait Backend {
    fn random(&mut self, out: &mut [u8]);
    fn create_key(
        &mut self,
        rp_id: &str,
        user_id: &[u8],
        uv: UvPolicy,
    ) -> Result<NewKey, BackendError>;
    /// ECDSA P-256 SHA-256 over `message` (the backend hashes), DER encoded.
    fn sign(&mut self, handle: &[u8], message: &[u8]) -> Result<Vec<u8>, BackendError>;
    fn delete_key(&mut self, handle: &[u8]) -> Result<(), BackendError>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StoreError {
    Full,
    Io(String),
    Corrupt(String),
    CounterExhausted,
}

impl fmt::Display for StoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StoreError::Full => f.write_str("store full"),
            StoreError::Io(m) => write!(f, "store I/O: {m}"),
            StoreError::Corrupt(m) => write!(f, "store corrupt or wrong key: {m}"),
            StoreError::CounterExhausted => f.write_str("signature counter exhausted"),
        }
    }
}

/// Credential index plus the global signature counter.
pub trait CredentialStore {
    fn get(&self, id: &[u8]) -> Result<Option<Arc<Credential>>, StoreError>;
    /// All credentials (discoverable or not) for an RP ID.
    fn list_rp(&self, rp_id: &str) -> Result<Vec<Arc<Credential>>, StoreError>;
    /// Insert or replace (same ID).
    fn insert(&mut self, cred: Arc<Credential>) -> Result<(), StoreError>;
    fn remove(&mut self, id: &[u8]) -> Result<Option<Arc<Credential>>, StoreError>;
    /// Remove everything and return it. The counter is not reset.
    fn clear(&mut self) -> Result<Vec<Arc<Credential>>, StoreError>;
    /// Increment, persist, return the new value. Never repeats a value.
    fn next_sign_count(&mut self) -> Result<u32, StoreError>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PresenceKind {
    MakeCredential,
    GetAssertion,
    /// excludeList matched; presence is collected before reporting it.
    Excluded,
    Reset,
    Selection,
    /// Zero-length pinUvAuthParam probe (platform choosing a device).
    Probe,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PresenceRequest {
    pub kind: PresenceKind,
    pub rp_id: Option<Text>,
    pub user_name: Option<Text>,
    /// User verification (Touch ID) requested, not just presence.
    pub require_uv: bool,
    /// Assertion key policy, or None when no key use is known.
    pub key_uv_policy: Option<UvPolicy>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Presence {
    pub user_verified: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PresenceError {
    Denied,
    Cancelled,
    Timeout,
    UvFailed,
}

/// Test of user presence, optionally with built-in user verification.
pub trait UserPresence {
    fn request(&mut self, req: &PresenceRequest) -> Result<Presence, PresenceError>;
}

/// Async counterpart of [`Backend`].
///
/// Existing synchronous backends get this trait through a blanket adapter.
#[allow(async_fn_in_trait)]
pub trait AsyncBackend {
    async fn random(&mut self, out: &mut [u8]);
    async fn create_key(
        &mut self,
        rp_id: &str,
        user_id: &[u8],
        uv: UvPolicy,
    ) -> Result<NewKey, BackendError>;
    async fn sign(&mut self, handle: &[u8], message: &[u8]) -> Result<Vec<u8>, BackendError>;
    async fn delete_key(&mut self, handle: &[u8]) -> Result<(), BackendError>;
}

impl<T: Backend> AsyncBackend for T {
    async fn random(&mut self, out: &mut [u8]) {
        self.random(out)
    }

    async fn create_key(
        &mut self,
        rp_id: &str,
        user_id: &[u8],
        uv: UvPolicy,
    ) -> Result<NewKey, BackendError> {
        self.create_key(rp_id, user_id, uv)
    }

    async fn sign(&mut self, handle: &[u8], message: &[u8]) -> Result<Vec<u8>, BackendError> {
        self.sign(handle, message)
    }

    async fn delete_key(&mut self, handle: &[u8]) -> Result<(), BackendError> {
        self.delete_key(handle)
    }
}

/// Async counterpart of [`CredentialStore`].
///
/// Existing synchronous stores get this trait through a blanket adapter.
#[allow(async_fn_in_trait)]
pub trait AsyncCredentialStore {
    async fn get(&self, id: &[u8]) -> Result<Option<Arc<Credential>>, StoreError>;
    async fn list_rp(&self, rp_id: &str) -> Result<Vec<Arc<Credential>>, StoreError>;
    async fn insert(&mut self, cred: Arc<Credential>) -> Result<(), StoreError>;
    async fn remove(&mut self, id: &[u8]) -> Result<Option<Arc<Credential>>, StoreError>;
    async fn clear(&mut self) -> Result<Vec<Arc<Credential>>, StoreError>;
    async fn next_sign_count(&mut self) -> Result<u32, StoreError>;
}

impl<T: CredentialStore> AsyncCredentialStore for T {
    async fn get(&self, id: &[u8]) -> Result<Option<Arc<Credential>>, StoreError> {
        self.get(id)
    }

    async fn list_rp(&self, rp_id: &str) -> Result<Vec<Arc<Credential>>, StoreError> {
        self.list_rp(rp_id)
    }

    async fn insert(&mut self, cred: Arc<Credential>) -> Result<(), StoreError> {
        self.insert(cred)
    }

    async fn remove(&mut self, id: &[u8]) -> Result<Option<Arc<Credential>>, StoreError> {
        self.remove(id)
    }

    async fn clear(&mut self) -> Result<Vec<Arc<Credential>>, StoreError> {
        self.clear()
    }

    async fn next_sign_count(&mut self) -> Result<u32, StoreError> {
        self.next_sign_count()
    }
}

/// Async counterpart of [`UserPresence`].
///
/// Implementations that already have a synchronous presence source get this
/// trait through a blanket adapter. Async presence is useful when the answer
/// arrives through an event or a peripheral, but a transport must still send
/// KEEPALIVE responses while this future is pending.
#[allow(async_fn_in_trait)]
pub trait AsyncUserPresence {
    async fn request(&mut self, req: &PresenceRequest) -> Result<Presence, PresenceError>;
}

impl<T: UserPresence> AsyncUserPresence for T {
    async fn request(&mut self, req: &PresenceRequest) -> Result<Presence, PresenceError> {
        self.request(req)
    }
}
