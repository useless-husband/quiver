//! The `quiver` command: server, client and benchmarks.

mod cli;

fn usage() -> &'static str {
    "quiver - a compact vector database

SERVER
  quiver serve [--data DIR] [--listen 127.0.0.1:6333] [--threads 8]
               [--sync full|fsync|none] [--build-threads 4]
               [--compact-interval SECONDS (0 = off)] [--port-file PATH]

CLIENT (--url http://127.0.0.1:6333 or $QUIVER_URL)
  quiver list
  quiver create NAME --dim D --metric l2|ip|cosine [--m 16] [--ef-construction 200]
                [--ef-search 64] [--quantization none|int8|pqM] [--rerank 4] [--segment-rows N]
  quiver import NAME FILE.fbin|FILE.jsonl [--batch 1000] [--start-id 0] [--limit N]
  quiver search NAME --vector 0.1,0.2,.. [--k 10] [--ef 64] [--filter JSON] [--strategy auto|graph|brute]
  quiver get NAME ID
  quiver delete NAME ID[,ID..]
  quiver stats NAME | checkpoint NAME | compact NAME | drop NAME

BENCHMARKS
  quiver bench ann --base B.fbin --queries Q.fbin --truth T.ibin --metric l2|ip|cosine
                   [--m 16] [--efc 200] [--ef 10,20,..] [--threads 1,4] [--build-threads 4]
                   [--quant f32,int8,pqM] [--rerank 4] [--k 10] [--dataset NAME] [--out results.csv]

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
