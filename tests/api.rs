//! The HTTP API end to end against the real binary.

mod common;

use common::{Server, tmpdir};
use serde_json::json;

#[test]
fn api_round_trip_errors_and_restart() {
    let dir = tmpdir("api");
    let data = dir.join("data");
    let s = Server::start(&data, &["--sync", "fsync", "--compact-interval", "0"]);
    assert_eq!(s.ok("GET", "/healthz", None), json!({"status": "ok"}));
    s.ok("PUT", "/collections/items", Some(json!({"dim": 3, "metric": "l2", "segment_rows": 64})));
    assert_eq!(s.call("PUT", "/collections/items", Some(json!({"dim": 3, "metric": "l2"}))).status, 409);
    assert_eq!(s.call("PUT", "/collections/bad%20name", Some(json!({"dim": 3, "metric": "l2"}))).status, 400);
    assert_eq!(s.call("PUT", "/collections/x", Some(json!({"dim": 3, "metric": "nope"}))).status, 400);
    assert_eq!(s.call("GET", "/collections/missing", None).status, 404);

    // 200 points over several sealed segments.
    for chunk in 0..4 {
        let pts: Vec<_> = (0..50)
            .map(|i| {
                let id = chunk * 50 + i;
                json!({"id": id, "vector": [id as f32, 0.0, 1.0], "payload": {"even": id % 2 == 0, "n": id}})
            })
            .collect();
        let r = s.ok("PUT", "/collections/items/points", Some(json!({"points": pts})));
        assert_eq!(r["upserted"], 50);
    }
    let r = s.ok("POST", "/collections/items/search", Some(json!({"vector": [10.2, 0.0, 1.0], "k": 3})));
    let ids: Vec<u64> = r["hits"].as_array().unwrap().iter().map(|h| h["id"].as_u64().unwrap()).collect();
    assert_eq!(ids, vec![10, 11, 9]);
    let r = s.ok(
        "POST",
        "/collections/items/search",
        Some(json!({"vector": [10.2, 0.0, 1.0], "k": 3, "with_payload": true,
                    "filter": {"and": [{"field": "even", "eq": false}, {"field": "n", "gte": 100}]}})),
    );
    let hits = r["hits"].as_array().unwrap();
    assert_eq!(hits.iter().map(|h| h["id"].as_u64().unwrap()).collect::<Vec<_>>(), vec![101, 103, 105]);
    assert_eq!(hits[0]["payload"], json!({"even": false, "n": 101}));
    assert!(r["plan"].as_array().unwrap().len() >= 4, "several segments searched: {}", r["plan"]);

    // Errors.
    assert_eq!(s.call("POST", "/collections/items/search", Some(json!({"vector": [1.0], "k": 3}))).status, 400);
    assert_eq!(
        s.call("POST", "/collections/items/search", Some(json!({"vector": [1.0, 2.0, 3.0], "filter": {"field": 1}})))
            .status,
        400
    );
    assert_eq!(
        s.call("POST", "/collections/items/search", Some(json!({"vector": [1.0, 2.0, 3.0], "bogus": 1}))).status,
        400
    );
    assert_eq!(s.call("DELETE", "/collections/items/search", None).status, 405);
    assert_eq!(s.call("GET", "/nope", None).status, 404);
    let r = quiver::client::request(&s.addr, "POST", "/collections/items/points", None).unwrap();
    assert_eq!(r.status, 400, "empty body");

    // Delete, get, checkpoint, compact.
    assert_eq!(s.ok("POST", "/collections/items/points/delete", Some(json!({"ids": [10, 11, 999]})))["deleted"], 2);
    assert_eq!(s.call("GET", "/collections/items/points/10", None).status, 404);
    let p = s.ok("GET", "/collections/items/points/12", None);
    assert_eq!(p["vector"], json!([12.0, 0.0, 1.0]));
    let got = s.ok("POST", "/collections/items/points/get", Some(json!({"ids": [9, 10]})));
    assert!(got["points"][1].is_null() && got["points"][0]["id"] == 9);
    s.ok("POST", "/collections/items/checkpoint", None);
    let c = s.ok("POST", "/collections/items/compact", None);
    assert_eq!(c["rows_out"], 198, "{c}");
    let st = s.ok("GET", "/collections/items", None);
    assert_eq!(st["points"], 198);

    // Restart (graceful) and everything is still there.
    s.terminate();
    let s = Server::start(&data, &["--compact-interval", "0"]);
    let st = s.ok("GET", "/collections/items", None);
    assert_eq!(st["points"], 198);
    let r = s.ok("POST", "/collections/items/search", Some(json!({"vector": [10.2, 0.0, 1.0], "k": 2})));
    assert_eq!(r["hits"][0]["id"], 9);
    s.ok("DELETE", "/collections/items", None);
    assert_eq!(s.call("GET", "/collections/items", None).status, 404);
    drop(s);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn oversized_body_is_rejected() {
    let dir = tmpdir("big");
    let s = Server::start(&dir.join("data"), &["--compact-interval", "0"]);
    s.ok("PUT", "/collections/c", Some(json!({"dim": 2, "metric": "ip"})));
    // A 70 MB body exceeds the 64 MB limit.
    let big = "x".repeat(70 << 20);
    let body = json!({"points": [{"id": 1, "vector": [1.0, 2.0], "payload": {"s": big}}]});
    let r = quiver::client::request(&s.addr, "PUT", "/collections/c/points", Some(&body));
    match r {
        Ok(r) => assert_eq!(r.status, 413),
        Err(e) => assert!(
            e.kind() == std::io::ErrorKind::ConnectionReset || e.kind() == std::io::ErrorKind::BrokenPipe,
            "{e}"
        ),
    }
    assert_eq!(s.ok("GET", "/collections/c", None)["points"], 0);
    drop(s);
    std::fs::remove_dir_all(&dir).unwrap();
}
