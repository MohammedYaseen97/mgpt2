use std::collections::HashMap;
use pyo3::prelude::*;
use rayon::prelude::*;

/// Calculate stats for multiple chunks in parallel.
#[pyfunction]
fn get_stats_parallel(chunks: Vec<Vec<u32>>) -> PyResult<HashMap<(u32, u32), u32>> {
    let stats: HashMap<(u32, u32), u32> = chunks
        .par_iter()  // Parallel iterator over chunks
        .map(|chunk_ids| {
            let mut local_stats = HashMap::new();
            for window in chunk_ids.windows(2) {
                if let [a, b] = window {
                    let pair = (*a, *b);
                    *local_stats.entry(pair).or_insert(0) += 1;
                }
            }
            local_stats
        })
        .reduce(
            || HashMap::new(),
            |mut acc, local_stats| {
                for (key, value) in local_stats {
                    *acc.entry(key).or_insert(0) += value;
                }
                acc
            },
        );

    Ok(stats)
}

/// Perform merge operations on multiple chunks in parallel.
#[pyfunction]
fn merge_parallel(
    chunks: Vec<Vec<u32>>,
    pair: (u32, u32),
    idx: u32,
) -> PyResult<Vec<Vec<u32>>> {
    let merged_chunks: Vec<Vec<u32>> = chunks
        .par_iter()
        .map(|chunk_ids| {
            let mut merged_ids = Vec::new();
            let mut skip_next = false;

            for window in chunk_ids.windows(2) {
                if skip_next {
                    skip_next = false;
                    continue;
                }
                if let [a, b] = window {
                    if (*a, *b) == pair {
                        merged_ids.push(idx);
                        skip_next = true;
                    } else {
                        merged_ids.push(*a);
                    }
                }
            }

            // Push the last ID if not merged
            if let Some(&last) = chunk_ids.last() {
                if !skip_next {
                    merged_ids.push(last);
                }
            }

            merged_ids
        })
        .collect();

    Ok(merged_chunks)
}

/// Python module definition
#[pymodule]
fn tokenizer(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_stats_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(merge_parallel, m)?)?;
    Ok(())
}