//! Payload filters and the per-segment payload index.
//!
//! A payload is a JSON object stored with each vector. Filters select rows by
//! payload fields:
//!
//! ```json
//! {"field": "color", "eq": "red"}
//! {"field": "price", "gte": 10, "lt": 20}
//! {"field": "tags", "in": ["new", "sale"]}
//! {"and": [F, ...]}   {"or": [F, ...]}   {"not": F}
//! ```
//!
//! Field names are dotted paths into nested objects (`"meta.lang"`). Arrays
//! are multi-valued: `{"field": "tags", "eq": "new"}` matches
//! `{"tags": ["new", "old"]}`. Numbers compare as f64; strings and booleans
//! compare exactly. A row without the field matches no `eq`/`in`/range
//! condition (so `not` of one matches it).
//!
//! The index keeps posting lists for strings and booleans and a
//! `(value, row)` list per numeric field (sorted once a segment is sealed, so
//! ranges are two binary searches). [`PayloadIndex::eval`] turns a filter into
//! a bitset over the segment's rows; its population count is the exact
//! selectivity the search strategy is chosen by.

use crate::error::{Result, invalid};
use crate::index::bitset::Bitset;
use serde_json::Value;
use std::collections::HashMap;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Key {
    Str(String),
    Bool(bool),
}

#[derive(Clone, Debug, PartialEq)]
pub enum Scalar {
    Key(Key),
    Num(f64),
}

impl Scalar {
    fn from_json(v: &Value) -> Option<Scalar> {
        match v {
            Value::String(s) => Some(Scalar::Key(Key::Str(s.clone()))),
            Value::Bool(b) => Some(Scalar::Key(Key::Bool(*b))),
            Value::Number(n) => n.as_f64().map(Scalar::Num),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Filter {
    Eq(String, Scalar),
    In(String, Vec<Scalar>),
    /// Inclusive/exclusive bounds: `(value, inclusive)`.
    Range {
        field: String,
        lo: Option<(f64, bool)>,
        hi: Option<(f64, bool)>,
    },
    And(Vec<Filter>),
    Or(Vec<Filter>),
    Not(Box<Filter>),
}

const MAX_DEPTH: usize = 8;

impl Filter {
    /// Parses the JSON filter syntax described in the module documentation.
    pub fn parse(v: &Value) -> Result<Filter> {
        Self::parse_at(v, 0)
    }

    fn parse_at(v: &Value, depth: usize) -> Result<Filter> {
        if depth > 16 {
            return invalid("filter is nested too deeply");
        }
        let Some(o) = v.as_object() else { return invalid("a filter must be a JSON object") };
        let list = |k: &str| -> Result<Vec<Filter>> {
            let Some(a) = o[k].as_array() else { return invalid(format!("`{k}` takes an array of filters")) };
            a.iter().map(|x| Self::parse_at(x, depth + 1)).collect()
        };
        if o.contains_key("and") && o.len() == 1 {
            return Ok(Filter::And(list("and")?));
        }
        if o.contains_key("or") && o.len() == 1 {
            return Ok(Filter::Or(list("or")?));
        }
        if let Some(n) = o.get("not") {
            if o.len() != 1 {
                return invalid("`not` must be the only key of its object");
            }
            return Ok(Filter::Not(Box::new(Self::parse_at(n, depth + 1)?)));
        }
        let Some(field) = o.get("field").and_then(Value::as_str) else {
            return invalid("a condition needs a string `field` (or use and/or/not)");
        };
        if field.is_empty() || field.split('.').any(str::is_empty) {
            return invalid(format!("invalid field path {field:?}"));
        }
        let field = field.to_string();
        if let Some(x) = o.get("eq") {
            if o.len() != 2 {
                return invalid("`eq` cannot be combined with other operators");
            }
            let Some(s) = Scalar::from_json(x) else { return invalid("`eq` takes a string, number or boolean") };
            return Ok(Filter::Eq(field, s));
        }
        if let Some(x) = o.get("in") {
            if o.len() != 2 {
                return invalid("`in` cannot be combined with other operators");
            }
            let Some(a) = x.as_array() else { return invalid("`in` takes an array") };
            let vals: Option<Vec<Scalar>> = a.iter().map(Scalar::from_json).collect();
            let Some(vals) = vals else { return invalid("`in` values must be strings, numbers or booleans") };
            return Ok(Filter::In(field, vals));
        }
        let num = |k: &str| -> Result<Option<f64>> {
            match o.get(k) {
                None => Ok(None),
                Some(x) => match x.as_f64() {
                    Some(f) => Ok(Some(f)),
                    None => invalid(format!("`{k}` takes a number")),
                },
            }
        };
        let (gt, gte, lt, lte) = (num("gt")?, num("gte")?, num("lt")?, num("lte")?);
        let known = 1 + [gt, gte, lt, lte].iter().filter(|x| x.is_some()).count();
        if known == 1 || known != o.len() {
            return invalid(format!("condition on {field:?} needs one of eq, in, gt, gte, lt, lte"));
        }
        if gt.is_some() && gte.is_some() || lt.is_some() && lte.is_some() {
            return invalid("use at most one lower and one upper bound");
        }
        let lo = gt.map(|v| (v, false)).or(gte.map(|v| (v, true)));
        let hi = lt.map(|v| (v, false)).or(lte.map(|v| (v, true)));
        Ok(Filter::Range { field, lo, hi })
    }

    /// Evaluates the filter against one payload (the definition the index is
    /// tested against).
    pub fn matches(&self, payload: &Value) -> bool {
        match self {
            Filter::Eq(f, s) => values_at(payload, f).iter().any(|v| scalar_eq(v, s)),
            Filter::In(f, ss) => values_at(payload, f).iter().any(|v| ss.iter().any(|s| scalar_eq(v, s))),
            Filter::Range { field, lo, hi } => values_at(payload, field).iter().any(|v| match v {
                Scalar::Num(x) => in_range(*x, *lo, *hi),
                _ => false,
            }),
            Filter::And(fs) => fs.iter().all(|f| f.matches(payload)),
            Filter::Or(fs) => fs.iter().any(|f| f.matches(payload)),
            Filter::Not(f) => !f.matches(payload),
        }
    }
}

fn scalar_eq(a: &Scalar, b: &Scalar) -> bool {
    match (a, b) {
        (Scalar::Num(x), Scalar::Num(y)) => x == y,
        (Scalar::Key(x), Scalar::Key(y)) => x == y,
        _ => false,
    }
}

fn in_range(x: f64, lo: Option<(f64, bool)>, hi: Option<(f64, bool)>) -> bool {
    let lo_ok = match lo {
        None => true,
        Some((v, true)) => x >= v,
        Some((v, false)) => x > v,
    };
    let hi_ok = match hi {
        None => true,
        Some((v, true)) => x <= v,
        Some((v, false)) => x < v,
    };
    lo_ok && hi_ok
}

/// Calls `f(path, scalar)` for every scalar in a payload, flattening nested
/// objects into dotted paths and arrays into repeated values.
pub fn flatten(payload: &Value, f: &mut impl FnMut(&str, Scalar)) {
    fn walk(v: &Value, path: &mut String, depth: usize, f: &mut impl FnMut(&str, Scalar)) {
        if depth > MAX_DEPTH {
            return;
        }
        match v {
            Value::Object(o) => {
                for (k, x) in o {
                    let len = path.len();
                    if !path.is_empty() {
                        path.push('.');
                    }
                    path.push_str(k);
                    walk(x, path, depth + 1, f);
                    path.truncate(len);
                }
            }
            Value::Array(a) => {
                for x in a {
                    walk(x, path, depth + 1, f);
                }
            }
            other => {
                if !path.is_empty() {
                    if let Some(s) = Scalar::from_json(other) {
                        f(path, s);
                    }
                }
            }
        }
    }
    if payload.is_object() {
        walk(payload, &mut String::new(), 0, f);
    }
}

fn values_at(payload: &Value, field: &str) -> Vec<Scalar> {
    let mut out = Vec::new();
    flatten(payload, &mut |p, s| {
        if p == field {
            out.push(s);
        }
    });
    out
}

/// Posting lists and numeric columns for one segment.
#[derive(Default, Debug)]
pub struct PayloadIndex {
    keys: HashMap<String, HashMap<Key, Vec<u32>>>,
    nums: HashMap<String, Vec<(f64, u32)>>,
    sorted: bool,
}

impl PayloadIndex {
    pub fn new() -> PayloadIndex {
        PayloadIndex::default()
    }

    /// Indexes the payload of `row`. Rows may be added in any order.
    pub fn add(&mut self, row: u32, payload: &Value) {
        self.sorted = false;
        flatten(payload, &mut |path, s| match s {
            Scalar::Key(k) => {
                let list = self.keys.entry(path.to_string()).or_default().entry(k).or_default();
                if list.last() != Some(&row) {
                    list.push(row);
                }
            }
            Scalar::Num(x) => self.nums.entry(path.to_string()).or_default().push((x, row)),
        });
    }

    /// Sorts the numeric columns so ranges become binary searches (done once
    /// a segment stops changing).
    pub fn seal(&mut self) {
        for v in self.nums.values_mut() {
            v.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        }
        self.sorted = true;
    }

    fn range_into(&self, field: &str, lo: Option<(f64, bool)>, hi: Option<(f64, bool)>, out: &mut Bitset) {
        let Some(col) = self.nums.get(field) else { return };
        let mark = |out: &mut Bitset, row: u32| {
            if (row as usize) < out.len() {
                out.set(row as usize);
            }
        };
        if self.sorted {
            let start = match lo {
                None => 0,
                Some((v, true)) => col.partition_point(|e| e.0 < v),
                Some((v, false)) => col.partition_point(|e| e.0 <= v),
            };
            let end = match hi {
                None => col.len(),
                Some((v, true)) => col.partition_point(|e| e.0 <= v),
                Some((v, false)) => col.partition_point(|e| e.0 < v),
            };
            for e in &col[start..end.max(start)] {
                mark(out, e.1);
            }
        } else {
            for e in col {
                if in_range(e.0, lo, hi) {
                    mark(out, e.1);
                }
            }
        }
    }

    fn eq_into(&self, field: &str, s: &Scalar, out: &mut Bitset) {
        match s {
            Scalar::Key(k) => {
                if let Some(list) = self.keys.get(field).and_then(|m| m.get(k)) {
                    for &r in list {
                        if (r as usize) < out.len() {
                            out.set(r as usize);
                        }
                    }
                }
            }
            Scalar::Num(x) => self.range_into(field, Some((*x, true)), Some((*x, true)), out),
        }
    }

    /// The rows `0..rows` that match `f`.
    pub fn eval(&self, f: &Filter, rows: usize) -> Bitset {
        match f {
            Filter::Eq(field, s) => {
                let mut b = Bitset::new(rows);
                self.eq_into(field, s, &mut b);
                b
            }
            Filter::In(field, ss) => {
                let mut b = Bitset::new(rows);
                for s in ss {
                    self.eq_into(field, s, &mut b);
                }
                b
            }
            Filter::Range { field, lo, hi } => {
                let mut b = Bitset::new(rows);
                self.range_into(field, *lo, *hi, &mut b);
                b
            }
            Filter::And(fs) => {
                let mut b = Bitset::full(rows);
                for f in fs {
                    b.and_with(&self.eval(f, rows));
                }
                b
            }
            Filter::Or(fs) => {
                let mut b = Bitset::new(rows);
                for f in fs {
                    b.or_with(&self.eval(f, rows));
                }
                b
            }
            Filter::Not(f) => {
                let mut b = self.eval(f, rows);
                b.negate();
                b
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;
    use crate::testkit::property;
    use serde_json::json;

    #[test]
    fn parse_and_match() {
        let p = json!({"color": "red", "price": 12.5, "tags": ["a", "b"], "meta": {"lang": "en", "n": 3}, "ok": true});
        let cases = [
            (json!({"field": "color", "eq": "red"}), true),
            (json!({"field": "color", "eq": "blue"}), false),
            (json!({"field": "price", "gte": 10, "lt": 20}), true),
            (json!({"field": "price", "gt": 12.5}), false),
            (json!({"field": "tags", "eq": "b"}), true),
            (json!({"field": "tags", "in": ["x", "a"]}), true),
            (json!({"field": "meta.lang", "eq": "en"}), true),
            (json!({"field": "meta.n", "eq": 3}), true),
            (json!({"field": "ok", "eq": true}), true),
            (json!({"field": "missing", "eq": 1}), false),
            (json!({"not": {"field": "missing", "eq": 1}}), true),
            (json!({"and": [{"field": "color", "eq": "red"}, {"field": "price", "lt": 5}]}), false),
            (json!({"or": [{"field": "color", "eq": "red"}, {"field": "price", "lt": 5}]}), true),
        ];
        for (f, want) in cases {
            assert_eq!(Filter::parse(&f).unwrap().matches(&p), want, "{f}");
        }
        for bad in [
            json!(3),
            json!({"field": "x"}),
            json!({"field": "x", "eq": [1]}),
            json!({"field": "x", "gt": "a"}),
            json!({"field": "x", "gt": 1, "gte": 2}),
            json!({"field": "x", "eq": 1, "lt": 2}),
            json!({"field": "", "eq": 1}),
            json!({"and": 3}),
            json!({"not": {"field": "x", "eq": 1}, "and": []}),
        ] {
            assert!(Filter::parse(&bad).is_err(), "{bad} should be rejected");
        }
    }

    fn random_payload(rng: &mut Rng) -> Value {
        let mut o = serde_json::Map::new();
        if rng.bool(0.8) {
            let c = ["red", "green", "blue"][rng.below(3)];
            o.insert("c".into(), json!(c));
        }
        if rng.bool(0.8) {
            o.insert("n".into(), json!(rng.below(20)));
        }
        if rng.bool(0.5) {
            o.insert("f".into(), json!(rng.below(100) as f64 / 4.0));
        }
        if rng.bool(0.5) {
            let tags: Vec<&str> = (0..rng.below(3)).map(|_| ["x", "y", "z"][rng.below(3)]).collect();
            o.insert("t".into(), json!(tags));
        }
        if rng.bool(0.3) {
            o.insert("m".into(), json!({"k": rng.bool(0.5)}));
        }
        Value::Object(o)
    }

    fn random_filter(rng: &mut Rng, depth: usize) -> Value {
        let colour = ["red", "green", "blue"][rng.below(3)];
        let field = ["t", "c"][rng.below(2)];
        let member = ["x", "y", "z", "red"][rng.below(4)];
        match if depth > 2 { rng.below(4) } else { rng.below(7) } {
            0 => json!({"field": "c", "eq": colour}),
            1 => json!({"field": "n", "gte": rng.below(20), "lt": rng.below(25)}),
            2 => json!({"field": field, "in": [member, "z"]}),
            3 => match rng.below(3) {
                0 => json!({"field": "f", "lte": rng.below(100) as f64 / 4.0}),
                1 => json!({"field": "n", "eq": rng.below(20)}),
                _ => json!({"field": "m.k", "eq": rng.bool(0.5)}),
            },
            4 => json!({"and": [random_filter(rng, depth + 1), random_filter(rng, depth + 1)]}),
            5 => json!({"or": [random_filter(rng, depth + 1), random_filter(rng, depth + 1)]}),
            _ => json!({"not": random_filter(rng, depth + 1)}),
        }
    }

    /// The index (sorted and unsorted) agrees with `matches` on every row.
    #[test]
    fn index_agrees_with_direct_evaluation() {
        property("payload index vs matches", 300, |rng| {
            let n = rng.range(1, 300);
            let payloads: Vec<Value> = (0..n).map(|_| random_payload(rng)).collect();
            let mut idx = PayloadIndex::new();
            let mut order: Vec<usize> = (0..n).collect();
            rng.shuffle(&mut order);
            for &i in &order {
                idx.add(i as u32, &payloads[i]);
            }
            let fj = random_filter(rng, 0);
            let f = Filter::parse(&fj).map_err(|e| e.to_string())?;
            for sealed in [false, true] {
                if sealed {
                    idx.seal();
                }
                let bits = idx.eval(&f, n);
                for (i, p) in payloads.iter().enumerate() {
                    if bits.get(i) != f.matches(p) {
                        return Err(format!("row {i} payload {p} filter {fj} sealed={sealed}"));
                    }
                }
            }
            Ok(())
        });
    }
}
