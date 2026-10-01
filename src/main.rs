//! The `quiver` command: server, client and benchmarks.

mod cli;

fn usage() -> &'static str {
    "quiver - a compact vector database

USAGE:
  quiver bench ann --base B.fbin --queries Q.fbin --truth T.ibin --metric l2|ip|cosine
                   [--m 16] [--efc 200] [--ef 10,20,..] [--threads 1,4] [--build-threads 4]
                   [--k 10] [--dataset NAME] [--out results.csv]
  quiver version
"
}

fn main() {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let code = match cli::run(&raw) {
        Ok(()) => 0,
        Err(e) if e == "usage" => {
            eprint!("{}", usage());
            2
        }
        Err(e) => {
            eprintln!("quiver: {e}");
            1
        }
    };
    std::process::exit(code);
}
