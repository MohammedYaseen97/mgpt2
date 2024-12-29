use std::collections::HashMap;
use pyo3::prelude::*;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use std::sync::{Arc, Mutex};

/// Calculate stats for multiple chunks in parallel with custom thread pool
#[pyfunction]
fn get_stats_parallel(chunks: Vec<Vec<u32>>) -> PyResult<HashMap<(u32, u32), u32>> {
    // Set the custom thread pool size (e.g., use 8 threads)
    ThreadPoolBuilder::new()
        .num_threads(8)  // Set custom number of threads
        .build_global()
        .unwrap();

    let stats: Arc<Mutex<HashMap<(u32, u32), u32>>> = Arc::new(Mutex::new(HashMap::new()));

    chunks.par_iter().for_each(|chunk_ids| {
        let mut local_stats = HashMap::new();
        for window in chunk_ids.windows(2) {
            if let [a, b] = window {
                let pair = (*a, *b);
                *local_stats.entry(pair).or_insert(0) += 1;
            }
        }

        let mut stats_lock = stats.lock().unwrap();
        for (key, value) in local_stats {
            *stats_lock.entry(key).or_insert(0) += value;
        }
    });

    Ok(Arc::try_unwrap(stats).unwrap().into_inner().unwrap())
}

/// Perform merge operations on multiple chunks in parallel with custom thread pool
#[pyfunction]
fn merge_parallel(
    chunks: Vec<Vec<u32>>,
    pair: (u32, u32),
    idx: u32,
) -> PyResult<Vec<Vec<u32>>> {
    // Set the custom thread pool size (e.g., use 8 threads)
    ThreadPoolBuilder::new()
        .num_threads(8)  // Set custom number of threads
        .build_global()
        .unwrap();

    let merged_chunks: Vec<Vec<u32>> = chunks
        .par_iter()
        .map(|chunk_ids| {
            let mut merged_ids = Vec::with_capacity(chunk_ids.len());
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
