//! Lloyd's k-means, used to train product-quantization codebooks.

use crate::distance::kernels;
use crate::rng::Rng;

/// Clusters `n = data.len() / dim` points into `k` centroids with `iters`
/// Lloyd iterations. Initial centroids are distinct random points; an empty
/// cluster is re-seeded by splitting the largest one (the FAISS approach),
/// so every centroid stays in use. Returns `k * dim` floats.
pub fn kmeans(data: &[f32], dim: usize, k: usize, iters: usize, seed: u64, threads: usize) -> Vec<f32> {
    let n = data.len() / dim;
    assert!(n > 0 && k > 0);
    let kern = kernels();
    let mut rng = Rng::new(seed);
    let mut order: Vec<usize> = (0..n).collect();
    rng.shuffle(&mut order);
    let mut cent = vec![0.0f32; k * dim];
    for c in 0..k {
        let src = order[c % n];
        cent[c * dim..(c + 1) * dim].copy_from_slice(&data[src * dim..(src + 1) * dim]);
    }
    if n <= k {
        return cent;
    }
    let mut assign = vec![0u32; n];
    let threads = threads.max(1);
    for _ in 0..iters {
        // Assignment step, in parallel over contiguous chunks.
        let chunk = n.div_ceil(threads);
        std::thread::scope(|s| {
            for (ci, slot) in assign.chunks_mut(chunk).enumerate() {
                let cent = &cent;
                s.spawn(move || {
                    for (j, a) in slot.iter_mut().enumerate() {
                        let i = ci * chunk + j;
                        let x = &data[i * dim..(i + 1) * dim];
                        let mut best = (f32::INFINITY, 0u32);
                        for c in 0..k {
                            let d = (kern.l2sq_f32)(x, &cent[c * dim..(c + 1) * dim]);
                            if d < best.0 {
                                best = (d, c as u32);
                            }
                        }
                        *a = best.1;
                    }
                });
            }
        });
        // Update step.
        let mut sum = vec![0.0f64; k * dim];
        let mut count = vec![0usize; k];
        for i in 0..n {
            let c = assign[i] as usize;
            count[c] += 1;
            for d in 0..dim {
                sum[c * dim + d] += data[i * dim + d] as f64;
            }
        }
        for c in 0..k {
            if count[c] > 0 {
                for d in 0..dim {
                    cent[c * dim + d] = (sum[c * dim + d] / count[c] as f64) as f32;
                }
            }
        }
        // Re-seed empty clusters from the largest cluster, nudged apart.
        for c in 0..k {
            if count[c] == 0 {
                let big = (0..k).max_by_key(|&j| count[j]).unwrap();
                for d in 0..dim {
                    let v = cent[big * dim + d];
                    let eps = 1e-4 * (v.abs() + 1e-3);
                    cent[c * dim + d] = v + eps;
                    cent[big * dim + d] = v - eps;
                }
                count[c] = count[big] / 2;
                count[big] -= count[c];
            }
        }
    }
    cent
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recovers_well_separated_clusters() {
        let mut rng = Rng::new(4);
        let truth = [[0.0f32, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]];
        let mut data = Vec::new();
        for i in 0..4000 {
            let c = truth[i % 4];
            data.push(c[0] + rng.normal() * 0.5);
            data.push(c[1] + rng.normal() * 0.5);
        }
        let cent = kmeans(&data, 2, 4, 25, 9, 2);
        for t in truth {
            let best = (0..4)
                .map(|c| (cent[c * 2] - t[0]).powi(2) + (cent[c * 2 + 1] - t[1]).powi(2))
                .fold(f32::INFINITY, f32::min);
            assert!(best < 0.1, "no centroid near {t:?}: {cent:?}");
        }
        // More centroids than points: every point becomes a centroid.
        let few = kmeans(&data[..6], 2, 8, 5, 1, 1);
        assert_eq!(few.len(), 16);
    }
}
