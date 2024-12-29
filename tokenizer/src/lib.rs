use std::collections::HashMap;
use pyo3::prelude::*;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use std::sync::{Arc, Mutex};
use dashmap::DashMap;

/// Calculate frequency statistics for pairs of consecutive numbers in chunks
#[pyfunction]
fn get_stats_parallel(chunks: Vec<Vec<u32>>) -> PyResult<HashMap<(u32, u32), u32>> {
    // Initialize thread pool with 8 threads
    ThreadPoolBuilder::new()
        .num_threads(8)
        .build_global()
        .expect("Failed to initialize thread pool");

    // Create a thread-safe HashMap to store the statistics
    let stats = Arc::new(Mutex::new(HashMap::new()));

    // Process each chunk in parallel
    chunks.par_iter().for_each(|chunk_ids| {
        // Create a DashMap for concurrent access
        let local_stats = DashMap::new();

        // Look at each consecutive pair of numbers in parallel
        chunk_ids.par_windows(2).for_each(|pair| {
            // Make sure we have exactly 2 numbers
            if let [first_num, second_num] = pair {
                // Create a pair from the two numbers
                let number_pair = (*first_num, *second_num);
                
                // Increment the count for this pair
                let mut count = local_stats.entry(number_pair).or_insert(0);
                *count += 1;
            }
        });

        // Get exclusive access to the shared stats
        let mut shared_stats = stats.lock().unwrap();
        
        // Add the local counts to the shared stats
        for (pair, count) in local_stats {
            let shared_count = shared_stats.entry(pair).or_insert(0);
            *shared_count += count;
        }
    });

    // Convert the thread-safe HashMap back to a regular HashMap
    let final_stats = Arc::try_unwrap(stats)
        .unwrap()
        .into_inner()
        .unwrap();

    Ok(final_stats)
}

/// Merge consecutive numbers that match a specific pair into a single new number
#[pyfunction]
fn merge_parallel(
    chunks: Vec<Vec<u32>>,
    pair_to_merge: (u32, u32),
    new_number: u32,
) -> PyResult<Vec<Vec<u32>>> {
    // Initialize thread pool with 8 threads
    ThreadPoolBuilder::new()
        .num_threads(8)
        .build_global()
        .expect("Failed to initialize thread pool");

    let merged_chunks: Vec<Vec<u32>> = chunks
        .par_iter()
        .map(|chunk_ids| {
            let mut result = Vec::with_capacity(chunk_ids.len());
            let mut skip_next_number = false;

            // Look at each consecutive pair of numbers
            for pair in chunk_ids.windows(2) {
                if skip_next_number {
                    skip_next_number = false;
                    continue;
                }

                if let [first_num, second_num] = pair {
                    let current_pair = (*first_num, *second_num);
                    
                    if current_pair == pair_to_merge {
                        // Replace the pair with the new number
                        result.push(new_number);
                        skip_next_number = true;
                    } else {
                        // Keep the first number as is
                        result.push(*first_num);
                    }
                }
            }

            // Handle the last number in the chunk
            if let Some(&last_number) = chunk_ids.last() {
                if !skip_next_number {
                    result.push(last_number);
                }
            }

            result
        })
        .collect();

    Ok(merged_chunks)
}

/// Register the Python module functions
#[pymodule]
fn tokenizer(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_stats_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(merge_parallel, m)?)?;
    Ok(())
}
