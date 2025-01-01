#include <omp.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    uint32_t first;
    uint32_t second;
} Pair;

typedef struct {
    Pair pair;
    uint32_t count;
} PairCount;

// Function to calculate frequency statistics for pairs of consecutive numbers
PairCount* get_stats_parallel(uint32_t** chunks, size_t* chunk_sizes, size_t num_chunks, size_t* num_pairs) {
    // Allocate memory for storing pair counts
    PairCount* pair_counts = malloc(1000 * sizeof(PairCount)); // Adjust size as needed
    size_t pair_count_size = 0;

    #pragma omp parallel
    {
        PairCount* local_counts = malloc(1000 * sizeof(PairCount)); // Adjust size as needed
        size_t local_count_size = 0;

        #pragma omp for
        for (size_t i = 0; i < num_chunks; i++) {
            for (size_t j = 0; j < chunk_sizes[i] - 1; j++) {
                Pair current_pair = {chunks[i][j], chunks[i][j + 1]};
                int found = 0;
                for (size_t k = 0; k < local_count_size; k++) {
                    if (local_counts[k].pair.first == current_pair.first && local_counts[k].pair.second == current_pair.second) {
                        local_counts[k].count++;
                        found = 1;
                        break;
                    }
                }
                if (!found) {
                    local_counts[local_count_size].pair = current_pair;
                    local_counts[local_count_size].count = 1;
                    local_count_size++;
                }
            }
        }

        #pragma omp critical
        {
            for (size_t i = 0; i < local_count_size; i++) {
                int found = 0;
                for (size_t j = 0; j < pair_count_size; j++) {
                    if (pair_counts[j].pair.first == local_counts[i].pair.first && pair_counts[j].pair.second == local_counts[i].pair.second) {
                        pair_counts[j].count += local_counts[i].count;
                        found = 1;
                        break;
                    }
                }
                if (!found) {
                    pair_counts[pair_count_size] = local_counts[i];
                    pair_count_size++;
                }
            }
        }

        free(local_counts);
    }

    *num_pairs = pair_count_size;
    return pair_counts;
}

// Function to merge consecutive numbers that match a specific pair into a single new number
void merge_parallel(uint32_t** chunks, size_t* chunk_sizes, size_t num_chunks, Pair pair_to_merge, uint32_t new_number) {
    #pragma omp parallel for
    for (size_t i = 0; i < num_chunks; i++) {
        size_t write_index = 0;
        for (size_t j = 0; j < chunk_sizes[i] - 1; j++) {
            if (chunks[i][j] == pair_to_merge.first && chunks[i][j + 1] == pair_to_merge.second) {
                chunks[i][write_index++] = new_number;
                j++; // Skip the next number
            } else {
                chunks[i][write_index++] = chunks[i][j];
            }
        }
        if (chunk_sizes[i] > 0 && (chunks[i][chunk_sizes[i] - 2] != pair_to_merge.first || chunks[i][chunk_sizes[i] - 1] != pair_to_merge.second)) {
            chunks[i][write_index++] = chunks[i][chunk_sizes[i] - 1];
        }
        chunk_sizes[i] = write_index;
    }
}