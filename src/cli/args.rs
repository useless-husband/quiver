//! A minimal `--key value` argument parser (no dependency needed for a
//! handful of subcommands).

use std::collections::HashMap;
use std::str::FromStr;

pub struct Args {
    pub positional: Vec<String>,
    opts: HashMap<String, String>,
    used: std::cell::RefCell<Vec<String>>,
}

impl Args {
    /// Parses `--key value`, `--key=value` and bare `--flag` (a flag is an
    /// option followed by another option or by nothing).
    pub fn parse(raw: &[String]) -> Args {
        let mut positional = Vec::new();
        let mut opts = HashMap::new();
        let mut i = 0;
        while i < raw.len() {
            let a = &raw[i];
            if let Some(key) = a.strip_prefix("--") {
                if let Some((k, v)) = key.split_once('=') {
                    opts.insert(k.to_string(), v.to_string());
                } else if i + 1 < raw.len() && !raw[i + 1].starts_with("--") {
                    opts.insert(key.to_string(), raw[i + 1].clone());
                    i += 1;
                } else {
                    opts.insert(key.to_string(), "true".to_string());
                }
            } else {
                positional.push(a.clone());
            }
            i += 1;
        }
        Args { positional, opts, used: Default::default() }
    }

    pub fn str(&self, key: &str) -> Option<String> {
        self.used.borrow_mut().push(key.to_string());
        self.opts.get(key).cloned()
    }

    pub fn req(&self, key: &str) -> Result<String, String> {
        self.str(key).ok_or_else(|| format!("missing required option --{key}"))
    }

    pub fn get<T: FromStr>(&self, key: &str, default: T) -> Result<T, String> {
        match self.str(key) {
            None => Ok(default),
            Some(v) => v.parse().map_err(|_| format!("invalid value for --{key}: {v}")),
        }
    }

    pub fn opt<T: FromStr>(&self, key: &str) -> Result<Option<T>, String> {
        match self.str(key) {
            None => Ok(None),
            Some(v) => v.parse().map(Some).map_err(|_| format!("invalid value for --{key}: {v}")),
        }
    }

    /// A comma-separated list.
    pub fn list<T: FromStr + Clone>(&self, key: &str, default: &[T]) -> Result<Vec<T>, String> {
        match self.str(key) {
            None => Ok(default.to_vec()),
            Some(v) => v
                .split(',')
                .filter(|s| !s.is_empty())
                .map(|s| s.trim().parse().map_err(|_| format!("invalid value in --{key}: {s}")))
                .collect(),
        }
    }

    /// Fails on options that no code path asked for (typos).
    pub fn finish(&self) -> Result<(), String> {
        let used = self.used.borrow();
        let mut unknown: Vec<&String> = self.opts.keys().filter(|k| !used.contains(k)).collect();
        unknown.sort();
        match unknown.first() {
            Some(k) => Err(format!("unknown option --{k}")),
            None => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_options_flags_and_lists() {
        let raw: Vec<String> =
            ["ann", "--m", "16", "--ef=10,20", "--verbose", "--k", "5", "x"].iter().map(|s| s.to_string()).collect();
        let a = Args::parse(&raw);
        assert_eq!(a.positional, vec!["ann", "x"]);
        assert_eq!(a.get("m", 0usize).unwrap(), 16);
        assert_eq!(a.list::<usize>("ef", &[]).unwrap(), vec![10, 20]);
        assert_eq!(a.str("verbose").as_deref(), Some("true"));
        assert!(a.finish().is_err()); // --k unused
        assert_eq!(a.get("k", 0usize).unwrap(), 5);
        assert!(a.finish().is_ok());
        assert!(a.get::<usize>("missing", 3).unwrap() == 3);
    }
}
