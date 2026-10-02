pub mod args;
pub mod bench;
pub mod cmd;

use args::Args;

pub fn run(raw: &[String]) -> Result<(), String> {
    let Some(cmd) = raw.first() else { return Err("usage".into()) };
    let rest = &raw[1..];
    match cmd.as_str() {
        "version" | "--version" => {
            println!("quiver {} (kernels: {})", env!("CARGO_PKG_VERSION"), quiver::distance::kernels().name);
            Ok(())
        }
        "serve" => cmd::serve(&Args::parse(rest)),
        "list" | "create" | "drop" | "stats" | "get" | "delete" | "search" | "import" | "checkpoint" | "compact" => {
            cmd::client(cmd, &Args::parse(rest))
        }
        "bench" => {
            let a = Args::parse(rest.get(1..).unwrap_or(&[]));
            match rest.first().map(String::as_str) {
                Some("ann") => bench::ann(&a),
                Some("kernels") => bench::kernels(&a),
                Some("filter") => bench::filter(&a),
                _ => Err("usage".into()),
            }
        }
        "help" | "--help" | "-h" => Err("usage".into()),
        other => Err(format!("unknown command {other:?} (try `quiver help`)")),
    }
}
