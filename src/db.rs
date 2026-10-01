//! A database: a directory of named collections.
//!
//! ```text
//! <root>/collections/<name>/   one directory per collection
//! ```
//!
//! Creating a collection builds it in a temporary directory and renames it
//! into place, so a crash never leaves a half-created collection; dropping one
//! renames it out of the way before deleting it.

use crate::collection::{Collection, CollectionConfig, Options};
use crate::error::{Error, Result, invalid};
use crate::storage::fs::sync_dir;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

pub struct Database {
    _lock: crate::storage::fs::DirLock,
    root: PathBuf,
    opts: Options,
    collections: RwLock<BTreeMap<String, Arc<Collection>>>,
    seq: AtomicU64,
}

/// Collection names: 1-64 characters of `[A-Za-z0-9_-]`.
pub fn valid_name(name: &str) -> bool {
    !name.is_empty() && name.len() <= 64 && name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
}

impl Database {
    /// Opens (or creates) the database at `root` and every collection in it.
    pub fn open(root: &Path, opts: Options) -> Result<Database> {
        let cdir = root.join("collections");
        std::fs::create_dir_all(&cdir)?;
        let lock = crate::storage::fs::DirLock::acquire(root)?;
        let mut map = BTreeMap::new();
        for e in std::fs::read_dir(&cdir)? {
            let e = e?;
            let name = e.file_name().to_string_lossy().into_owned();
            if name.starts_with('.') {
                // An interrupted create or drop.
                std::fs::remove_dir_all(e.path())?;
                continue;
            }
            if !e.file_type()?.is_dir() || !valid_name(&name) {
                continue;
            }
            let c = Collection::open(&e.path(), &name, opts)?;
            map.insert(name, Arc::new(c));
        }
        Ok(Database {
            _lock: lock,
            root: root.to_path_buf(),
            opts,
            collections: RwLock::new(map),
            seq: AtomicU64::new(0),
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    fn tmp_name(&self, tag: &str, name: &str) -> PathBuf {
        let n = self.seq.fetch_add(1, Ordering::Relaxed);
        self.root.join("collections").join(format!(".{tag}-{name}-{}-{n}", std::process::id()))
    }

    pub fn create(&self, name: &str, config: CollectionConfig) -> Result<Arc<Collection>> {
        if !valid_name(name) {
            return invalid(format!("invalid collection name {name:?} (use 1-64 of A-Z a-z 0-9 _ -)"));
        }
        config.validate()?;
        let mut map = self.collections.write().unwrap_or_else(|e| e.into_inner());
        if map.contains_key(name) {
            return Err(Error::Exists(format!("collection {name:?}")));
        }
        let fin = self.root.join("collections").join(name);
        let tmp = self.tmp_name("create", name);
        drop(Collection::create(&tmp, name, config, self.opts)?);
        std::fs::rename(&tmp, &fin)?;
        sync_dir(&self.root.join("collections"), self.opts.sync)?;
        let c = Arc::new(Collection::open(&fin, name, self.opts)?);
        map.insert(name.to_string(), c.clone());
        Ok(c)
    }

    pub fn get(&self, name: &str) -> Result<Arc<Collection>> {
        self.collections
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .get(name)
            .cloned()
            .ok_or_else(|| Error::NotFound(format!("collection {name:?}")))
    }

    pub fn list(&self) -> Vec<Arc<Collection>> {
        self.collections.read().unwrap_or_else(|e| e.into_inner()).values().cloned().collect()
    }

    pub fn drop_collection(&self, name: &str) -> Result<()> {
        let mut map = self.collections.write().unwrap_or_else(|e| e.into_inner());
        let Some(c) = map.remove(name) else { return Err(Error::NotFound(format!("collection {name:?}"))) };
        c.close();
        let trash = self.tmp_name("drop", name);
        std::fs::rename(c.dir(), &trash)?;
        sync_dir(&self.root.join("collections"), self.opts.sync)?;
        drop(map);
        // Requests still holding the collection keep working on open files.
        std::fs::remove_dir_all(&trash)?;
        Ok(())
    }
}
