use anyhow::{Context, Result};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Deserialize, Serialize)]
pub struct MLPParams {
    weights: Vec<Vec<Vec<f64>>>, // Deserialize as Vec<Vec<f64>> first
    biases: Vec<Vec<f64>>,       // Deserialize as Vec<f64> first
    activation: String,          // ReLU, Common activation function for all layers
    output_activation: String,   // logistic, Activation function for the output layer
}
pub struct MLPParamsND {
    weights: Vec<Array2<f64>>, // Converted to ndarray types
    biases: Vec<Array1<f64>>,  // Converted to ndarray types
    activation: String,
    output_activation: String,
}

impl MLPParams {
    fn to_ndarray(self) -> Result<MLPParamsND> {
        let weights = self
            .weights
            .into_iter()
            .map(|w| {
                Array2::from_shape_vec((w.len(), w[0].len()), w.into_iter().flatten().collect())
                    .with_context(|| "Failed to convert weights to ndarray")
            })
            .collect::<Result<Vec<Array2<f64>>>>()?;

        let biases = self
            .biases
            .into_iter()
            .map(|b| {
                Array1::from_shape_vec(b.len(), b)
                    .with_context(|| "Failed to convert biases to ndarray")
            })
            .collect::<Result<Vec<Array1<f64>>>>()?;

        Ok(MLPParamsND {
            weights,
            biases,
            activation: self.activation,
            output_activation: self.output_activation,
        })
    }
}

pub fn load_mlp_params(file_path: &str) -> Result<MLPParamsND> {
    let file = std::fs::File::open(file_path)
        .with_context(|| format!("Failed to open file at path: {}", file_path))?;
    let params: MLPParams = serde_json::from_reader(file)
        .with_context(|| "Failed to parse MLP parameters from JSON")?;
    params.to_ndarray()
}

/// Same as [`load_mlp_params`], but reads the JSON from memory. Used with the
/// model shipped in `resources/` via `include_str!`, so a build needs no
/// external data files at run time.
pub fn load_mlp_params_from_str(json: &str) -> Result<MLPParamsND> {
    let params: MLPParams = serde_json::from_str(json)
        .with_context(|| "Failed to parse MLP parameters from JSON")?;
    params.to_ndarray()
}

/// Nucleotide -> one-hot column within a position block (A,C,G,T,N). Anything
/// else (lower case, IUPAC codes) is treated as N, matching the training encoding.
#[inline]
pub fn base_code(b: u8) -> usize {
    match b {
        b'A' => 0,
        b'C' => 1,
        b'G' => 2,
        b'T' => 3,
        _ => 4,
    }
}

/// The Forseti binding-affinity MLP evaluated natively: one-hot(k-mer, 5 symbols)
/// -> hidden (ReLU) -> 1 (sigmoid).
///
/// The input is one-hot, so the first layer is not a matrix-vector product: for
/// each of the k positions exactly one input is 1, and `W1 . x` is the sum of k
/// columns of `W1`. We store `W1` column-major (`w1[input][hidden]`) so each
/// position contributes one contiguous `hidden`-length add. That is ~k*hidden
/// adds per k-mer (3,000 for k=30, hidden=100) instead of the 15,000 multiply-adds
/// a dense matmul performs, and it needs no tensor library: this replaced the
/// libtorch (`tch`) backend, which cost ~10-50 us of dispatch per call around a
/// ~5 us computation and spawned an OpenMP pool per worker thread.
///
/// Arithmetic is f32 like the libtorch path was (weights were loaded with
/// `Kind::Float`), so outputs agree with it to summation-order rounding (~1e-7).
pub struct NativeMlp {
    k: usize,
    hidden: usize,
    /// `w1[(j*5 + code) * hidden + h]`
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: f32,
}

impl NativeMlp {
    pub fn from_params(params: &MLPParamsND) -> Result<Self> {
        anyhow::ensure!(
            params.weights.len() == 2 && params.biases.len() == 2,
            "NativeMlp expects exactly two layers, got {}",
            params.weights.len()
        );
        anyhow::ensure!(
            params.activation.eq_ignore_ascii_case("relu")
                && params.output_activation.eq_ignore_ascii_case("logistic"),
            "NativeMlp expects relu + logistic, got {} + {}",
            params.activation,
            params.output_activation
        );
        let w1_t = &params.weights[0]; // (hidden, inputs) -- already transposed in JSON
        let w2_t = &params.weights[1]; // (1, hidden)
        let (hidden, inputs) = (w1_t.shape()[0], w1_t.shape()[1]);
        anyhow::ensure!(inputs % 5 == 0, "input width {} is not a multiple of 5", inputs);
        anyhow::ensure!(
            w2_t.shape() == [1, hidden] && params.biases[0].len() == hidden && params.biases[1].len() == 1,
            "MLP shape mismatch: W1 {:?}, W2 {:?}, b1 {}, b2 {}",
            w1_t.shape(),
            w2_t.shape(),
            params.biases[0].len(),
            params.biases[1].len()
        );
        let mut w1 = vec![0f32; inputs * hidden];
        for h in 0..hidden {
            for i in 0..inputs {
                w1[i * hidden + h] = w1_t[(h, i)] as f32;
            }
        }
        Ok(NativeMlp {
            k: inputs / 5,
            hidden,
            w1,
            b1: params.biases[0].iter().map(|&v| v as f32).collect(),
            w2: w2_t.iter().map(|&v| v as f32).collect(),
            b2: params.biases[1][0] as f32,
        })
    }

    #[inline]
    pub fn k(&self) -> usize {
        self.k
    }

    #[inline]
    pub fn hidden(&self) -> usize {
        self.hidden
    }

    /// Affinity of the k-mer `bytes[start..start+k]`, using `hbuf` (len = hidden)
    /// as scratch so the hot loop allocates nothing.
    #[inline]
    pub fn predict_at(&self, bytes: &[u8], start: usize, hbuf: &mut [f32]) -> f64 {
        debug_assert_eq!(hbuf.len(), self.hidden);
        hbuf.copy_from_slice(&self.b1);
        for j in 0..self.k {
            let col = (j * 5 + base_code(bytes[start + j])) * self.hidden;
            let w = &self.w1[col..col + self.hidden];
            for (acc, &wv) in hbuf.iter_mut().zip(w) {
                *acc += wv;
            }
        }
        let mut z = self.b2;
        for (&hv, &wv) in hbuf.iter().zip(&self.w2) {
            if hv > 0.0 {
                z += hv * wv;
            }
        }
        (1.0 / (1.0 + (-z).exp())) as f64
    }

    /// Affinities for the k-mers starting at `starts`; replaces the batched
    /// libtorch forward.
    pub fn predict_starts(&self, bytes: &[u8], starts: &[usize]) -> Array1<f64> {
        let mut hbuf = vec![0f32; self.hidden];
        Array1::from_iter(starts.iter().map(|&s| self.predict_at(bytes, s, &mut hbuf)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference values produced by PyTorch (f32) from the same JSON, via
    /// scripts/gen_mlp_reference.py. Each line: <30-mer>\t<sigmoid output>.
    const REFERENCE: &str = include_str!("../resources/mlp_reference_kmers.tsv");
    const PARAMS: &str = include_str!("../resources/mlp_params_Transpose.json");

    #[test]
    fn native_mlp_matches_pytorch_reference() {
        let mlp = NativeMlp::from_params(&load_mlp_params_from_str(PARAMS).unwrap()).unwrap();
        let mut hbuf = vec![0f32; mlp.hidden];
        let mut n = 0usize;
        let mut max_abs = 0f64;
        for line in REFERENCE.lines().filter(|l| !l.is_empty()) {
            let (kmer, val) = line.split_once('\t').unwrap();
            let expected: f64 = val.parse().unwrap();
            let got = mlp.predict_at(kmer.as_bytes(), 0, &mut hbuf);
            max_abs = max_abs.max((got - expected).abs());
            n += 1;
        }
        eprintln!("native vs pytorch: {} k-mers, max |diff| = {:e}", n, max_abs);
        assert!(n >= 1000, "reference set too small: {}", n);
        assert!(max_abs < 1e-5, "max |native - torch| = {:e} over {} k-mers", max_abs, n);
    }
}

/// Same as [`load_spline_lookup_table`], but reads the JSON from memory
/// (see [`load_mlp_params_from_str`]).
pub fn load_spline_lookup_table_from_str(json: &str) -> anyhow::Result<Array1<f64>> {
    let json_data: Value = serde_json::from_str(json)?;
    parse_spline_y(&json_data)
}

fn parse_spline_y(json_data: &Value) -> anyhow::Result<Array1<f64>> {
    // Extract the "y" field as an array
    if let Some(y_values) = json_data["y"].as_array() {
        // Convert JSON array to Vec<f64>.
        //
        // The table is a fragment-length density (it sums to 1 and peaks around
        // index 216), but the spline fit undershoots below zero in the short-
        // fragment tail where the true density is ~0: 29 of the 1011 entries are
        // negative (indices 0-6 and 37-58, smallest -1.6e-5). Those are fitting
        // artefacts, and feeding one to `(affinity * frag_prob + EPS).ln()`
        // yields NaN, which then poisons that position's running sum. Clamp them
        // back to 0 here, once, so every consumer sees a valid density.
        let y_vec: Vec<f64> = y_values
            .iter()
            .map(|v| v.as_f64().unwrap_or(0.0).max(0.0)) // Ensure conversion; no negative density
            .collect();

        // Convert Vec<f64> to ndarray::Array1
        Ok(Array1::from(y_vec))
    } else {
        Err(anyhow::anyhow!("Missing 'y' field in JSON"))
    }
}