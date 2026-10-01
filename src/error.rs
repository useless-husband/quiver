use std::fmt;

/// Errors returned by the library and mapped to HTTP status codes by the
/// server.
#[derive(Debug)]
pub enum Error {
    /// The request is malformed (wrong dimension, bad parameter, ...). 400.
    Invalid(String),
    /// A collection or point does not exist. 404.
    NotFound(String),
    /// A collection with that name already exists. 409.
    Exists(String),
    /// A fixed-capacity index is full.
    Full,
    /// Data on disk failed a checksum or structural check.
    Corrupt(String),
    Io(std::io::Error),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Invalid(m) => write!(f, "invalid request: {m}"),
            Error::NotFound(m) => write!(f, "not found: {m}"),
            Error::Exists(m) => write!(f, "already exists: {m}"),
            Error::Full => write!(f, "index is full"),
            Error::Corrupt(m) => write!(f, "corrupt data: {m}"),
            Error::Io(e) => write!(f, "i/o error: {e}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

pub(crate) fn invalid<T>(msg: impl Into<String>) -> Result<T> {
    Err(Error::Invalid(msg.into()))
}
