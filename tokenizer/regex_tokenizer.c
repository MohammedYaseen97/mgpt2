#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <stdint.h>
#include <stdbool.h>
#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>

// Structures
typedef struct {
    int first;
    int second;
} Pair;

typedef struct {
    Pair pair;
    int count;
} StatEntry;

typedef struct {
    StatEntry* entries;
    int size;
    int capacity;
} Stats;

typedef struct {
    int* data;
    int size;
    int capacity;
} IntArray;

typedef struct {
    IntArray** chunks;  // Array of chunks, each chunk is an IntArray
    int num_chunks;
} TokenChunks;

typedef struct {
    uint8_t* bytes;
    size_t length;
} ByteArray;

// Forward declarations
Stats* create_stats();
void stats_add(Stats* stats, Pair pair, int count);
void get_stats_chunk(IntArray* chunk, Stats* stats);
void get_stats_parallel(TokenChunks* chunks, Stats* stats);
Pair find_max_pair(Stats* stats);
void merge_chunk(IntArray* chunk, Pair pair, int new_id);
void merge_parallel(TokenChunks* chunks, Pair pair, int new_id);
ByteArray* string_to_bytes(const char* text);
TokenChunks* regex_split_text(const char* pattern, const char* text);
void save_tokenizer(const char* vocab_file, const char* merges_file, Stats* final_stats, int vocab_size);
void train_tokenizer(const char* text, const char* regex_pattern, int vocab_size, const char* vocab_file, const char* merges_file);
TokenChunks* create_token_chunks(int initial_capacity);
void add_chunk(TokenChunks* chunks, int* data, int size);
void free_token_chunks(TokenChunks* chunks);
char* read_file(const char* filename, size_t* length);

// Helper functions for Stats
Stats* create_stats() {
    Stats* stats = (Stats*)malloc(sizeof(Stats));
    stats->capacity = 1024;
    stats->size = 0;
    stats->entries = (StatEntry*)malloc(sizeof(StatEntry) * stats->capacity);
    return stats;
}

void stats_add(Stats* stats, Pair pair, int count) {
    bool found = false;
    int found_idx = -1;
    
    #pragma omp critical
    {
        // Check if pair exists
        for (int i = 0; i < stats->size; i++) {
            if (stats->entries[i].pair.first == pair.first && 
                stats->entries[i].pair.second == pair.second) {
                found = true;
                found_idx = i;
                break;
            }
        }

        if (found) {
            stats->entries[found_idx].count += count;
        } else {
            // Add new pair
            if (stats->size >= stats->capacity) {
                stats->capacity *= 2;
                stats->entries = (StatEntry*)realloc(stats->entries, 
                                                   sizeof(StatEntry) * stats->capacity);
            }
            stats->entries[stats->size].pair = pair;
            stats->entries[stats->size].count = count;
            stats->size++;
        }
    }
}

// Get stats for a single chunk
void get_stats_chunk(IntArray* chunk, Stats* stats) {
    for (int i = 0; i < chunk->size - 1; i++) {
        Pair pair = {chunk->data[i], chunk->data[i + 1]};
        stats_add(stats, pair, 1);
    }
}

// Parallel get_stats for all chunks
void get_stats_parallel(TokenChunks* chunks, Stats* stats) {
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < chunks->num_chunks; i++) {
        get_stats_chunk(chunks->chunks[i], stats);
    }
}

// Find max pair in stats
Pair find_max_pair(Stats* stats) {
    int max_count = -1;
    Pair max_pair = {-1, -1};
    
    #pragma omp parallel
    {
        int local_max_count = -1;
        Pair local_max_pair = {-1, -1};
        
        #pragma omp for schedule(static)
        for (int i = 0; i < stats->size; i++) {
            if (stats->entries[i].count > local_max_count) {
                local_max_count = stats->entries[i].count;
                local_max_pair = stats->entries[i].pair;
            }
        }
        
        #pragma omp critical
        {
            if (local_max_count > max_count) {
                max_count = local_max_count;
                max_pair = local_max_pair;
            }
        }
    }
    
    return max_pair;
}

// Merge function for a single chunk
void merge_chunk(IntArray* chunk, Pair pair, int new_id) {
    IntArray* new_chunk = (IntArray*)malloc(sizeof(IntArray));
    new_chunk->capacity = chunk->capacity;
    new_chunk->data = (int*)malloc(sizeof(int) * chunk->capacity);
    new_chunk->size = 0;
    
    for (int i = 0; i < chunk->size; i++) {
        if (i < chunk->size - 1 && 
            chunk->data[i] == pair.first && 
            chunk->data[i + 1] == pair.second) {
            new_chunk->data[new_chunk->size++] = new_id;
            i++;
        } else {
            new_chunk->data[new_chunk->size++] = chunk->data[i];
        }
    }
    
    // Copy back to original chunk
    memcpy(chunk->data, new_chunk->data, sizeof(int) * new_chunk->size);
    chunk->size = new_chunk->size;
    
    // Cleanup
    free(new_chunk->data);
    free(new_chunk);
}

// Parallel merge for all chunks
void merge_parallel(TokenChunks* chunks, Pair pair, int new_id) {
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < chunks->num_chunks; i++) {
        merge_chunk(chunks->chunks[i], pair, new_id);
    }
}

// Function to convert UTF-8 string to bytes
ByteArray* string_to_bytes(const char* text) {
    ByteArray* arr = (ByteArray*)malloc(sizeof(ByteArray));
    arr->length = strlen(text);
    arr->bytes = (uint8_t*)malloc(arr->length);
    memcpy(arr->bytes, text, arr->length);
    return arr;
}

// Function to split text using regex
TokenChunks* regex_split_text(const char* pattern, const char* text) {
    printf("Starting regex split with text length: %zu\n", strlen(text));
    
    int errorcode;
    PCRE2_SIZE erroroffset;
    
    // Compile the regex pattern
    pcre2_code* re = pcre2_compile(
        (PCRE2_SPTR)pattern,
        PCRE2_ZERO_TERMINATED,
        PCRE2_UTF,
        &errorcode,
        &erroroffset,
        NULL
    );
    
    if (re == NULL) {
        PCRE2_UCHAR buffer[256];
        pcre2_get_error_message(errorcode, buffer, sizeof(buffer));
        printf("Regex compilation failed at offset %zu: %s\n", erroroffset, buffer);
        return NULL;
    }
    
    // Create match data block
    pcre2_match_data* match_data = pcre2_match_data_create_from_pattern(re, NULL);
    
    // Initialize chunks with larger capacity
    TokenChunks* chunks = create_token_chunks(1024 * 1024);  // Start with space for 1M chunks
    if (!chunks) {
        printf("Failed to create token chunks\n");
        return NULL;
    }
    
    // Find all matches
    PCRE2_SIZE* ovector;
    size_t text_len = strlen(text);
    size_t start_offset = 0;
    
    while (start_offset < text_len) {
        int rc = pcre2_match(
            re,
            (PCRE2_SPTR)text,
            text_len,
            start_offset,
            0,
            match_data,
            NULL
        );
        
        if (rc < 0) {
            if (rc != PCRE2_ERROR_NOMATCH) {
                printf("PCRE2 matching error: %d\n", rc);
            }
            break;
        }
        
        // Add size checks
        if (chunks->num_chunks >= 1024 * 1024) {
            printf("Too many chunks created, stopping at %d chunks\n", chunks->num_chunks);
            break;
        }
        
        ovector = pcre2_get_ovector_pointer(match_data);
        size_t match_length = ovector[1] - ovector[0];
        
        // Extract the matched text and convert to bytes
        char* matched_text = (char*)malloc(match_length + 1);
        memcpy(matched_text, text + ovector[0], match_length);
        matched_text[match_length] = '\0';
        
        // Convert to UTF-8 bytes
        ByteArray* bytes = string_to_bytes(matched_text);
        
        // Add as a new chunk
        add_chunk(chunks, (int*)bytes->bytes, bytes->length);
        
        // Cleanup
        free(matched_text);
        free(bytes->bytes);
        free(bytes);
        
        start_offset = ovector[1];
        
        if (chunks->num_chunks % 10000 == 0) {
            printf("Processed %d chunks\n", chunks->num_chunks);
        }
    }
    
    // Cleanup regex
    pcre2_match_data_free(match_data);
    pcre2_code_free(re);
    
    printf("Finished regex split, created %d chunks\n", chunks->num_chunks);
    return chunks;
}

// Function to save vocabulary and merges in Python-compatible format
void save_tokenizer(const char* vocab_file, const char* merges_file, 
                   Stats* final_stats, int vocab_size) {
    // Save vocabulary
    FILE* vf = fopen(vocab_file, "wb");
    if (!vf) {
        printf("Error opening vocabulary file\n");
        return;
    }
    
    // Write vocabulary size
    fwrite(&vocab_size, sizeof(int), 1, vf);
    
    // Write base bytes (0-255)
    for (int i = 0; i < 256; i++) {
        uint8_t byte = (uint8_t)i;
        fwrite(&byte, sizeof(uint8_t), 1, vf);
    }
    
    fclose(vf);
    
    // Save merges
    FILE* mf = fopen(merges_file, "wb");
    if (!mf) {
        printf("Error opening merges file\n");
        return;
    }
    
    // Write number of merges
    int num_merges = vocab_size - 256;
    fwrite(&num_merges, sizeof(int), 1, mf);
    
    // Write each merge operation
    for (int i = 0; i < final_stats->size && i < num_merges; i++) {
        Pair pair = final_stats->entries[i].pair;
        int new_id = 256 + i;
        
        // Write merge operation (first token, second token, new token id)
        fwrite(&pair.first, sizeof(int), 1, mf);
        fwrite(&pair.second, sizeof(int), 1, mf);
        fwrite(&new_id, sizeof(int), 1, mf);
    }
    
    fclose(mf);
}

// Modified train_tokenizer function
void train_tokenizer(const char* text, const char* regex_pattern, 
                    int vocab_size, const char* vocab_file, 
                    const char* merges_file) {
    printf("Starting tokenizer training with vocab size %d\n", vocab_size);
    
    TokenChunks* chunks = regex_split_text(regex_pattern, text);
    if (!chunks) {
        printf("Error: Failed to split text into chunks\n");
        return;
    }
    if (chunks->num_chunks == 0) {
        printf("Error: No chunks were created\n");
        free_token_chunks(chunks);
        return;
    }
    
    printf("Successfully created %d chunks\n", chunks->num_chunks);
    
    int num_merges = vocab_size - 256;
    Stats* final_stats = create_stats();
    
    for (int i = 0; i < num_merges; i++) {
        Stats* stats = create_stats();
        get_stats_parallel(chunks, stats);
        
        if (stats->size == 0) {
            printf("Warning: No pairs found at iteration %d\n", i);
            free(stats->entries);
            free(stats);
            break;
        }
        
        // Find best pair
        Pair best_pair = find_max_pair(stats);
        int new_id = 256 + i;
        
        // Store merge in final stats
        stats_add(final_stats, best_pair, 1);
        
        // Perform merges
        merge_parallel(chunks, best_pair, new_id);
        
        // Cleanup
        free(stats->entries);
        free(stats);
        
        if (i % 100 == 0) {
            printf("Processed merge %d/%d (stats size: %d)\n", 
                   i + 1, num_merges, stats->size);
        }
    }
    
    // Save the tokenizer
    save_tokenizer(vocab_file, merges_file, final_stats, vocab_size);
    
    // Cleanup
    free_token_chunks(chunks);
    free(final_stats->entries);
    free(final_stats);
}

// Helper functions for chunk management
TokenChunks* create_token_chunks(int initial_capacity) {
    TokenChunks* chunks = (TokenChunks*)malloc(sizeof(TokenChunks));
    chunks->chunks = (IntArray**)malloc(sizeof(IntArray*) * initial_capacity);
    chunks->num_chunks = 0;
    return chunks;
}

void add_chunk(TokenChunks* chunks, int* data, int size) {
    if (chunks->num_chunks >= 1024 * 1024) {
        printf("Error: Maximum chunk capacity reached\n");
        return;
    }
    
    IntArray* chunk = (IntArray*)malloc(sizeof(IntArray));
    if (!chunk) {
        printf("Error: Failed to allocate chunk\n");
        return;
    }
    
    chunk->data = (int*)malloc(sizeof(int) * size);
    if (!chunk->data) {
        printf("Error: Failed to allocate chunk data of size %d\n", size);
        free(chunk);
        return;
    }
    
    memcpy(chunk->data, data, sizeof(int) * size);
    chunk->size = size;
    chunk->capacity = size;
    chunks->chunks[chunks->num_chunks++] = chunk;
}

void free_token_chunks(TokenChunks* chunks) {
    for (int i = 0; i < chunks->num_chunks; i++) {
        free(chunks->chunks[i]->data);
        free(chunks->chunks[i]);
    }
    free(chunks->chunks);
    free(chunks);
}

// Add this function before main
char* read_file(const char* filename, size_t* length) {
    FILE* f = fopen(filename, "rb");
    if (!f) {
        printf("Error: Could not open file %s\n", filename);
        return NULL;
    }
    
    // Get file size
    fseek(f, 0, SEEK_END);
    *length = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    // Allocate buffer and read file
    char* buffer = (char*)malloc(*length + 1);
    if (!buffer) {
        printf("Error: Could not allocate memory for file contents\n");
        fclose(f);
        return NULL;
    }
    
    size_t bytes_read = fread(buffer, 1, *length, f);
    if (bytes_read != *length) {
        printf("Error: Could not read entire file\n");
        free(buffer);
        fclose(f);
        return NULL;
    }
    
    buffer[*length] = '\0';  // Null terminate the string
    fclose(f);
    return buffer;
}

int main() {
    const char* regex_pattern = "(?i)('s|'t|'re|'ve|'m|'ll|'d)|"  // contractions
        " ?\\b[[:alpha:]\\x{0900}-\\x{0963}\\x{0966}-\\x{097F}]+\\b|"     // words + devanagari
        " ?\\b[[:alpha:]\\x{0C80}-\\x{0C9E}\\x{0CA0}-\\x{0CFF}]+\\b|"     // words + kannada
        " ?[[:digit:]]+|"                                                   // numbers
        " ?[.,!?;:'\"-]|"                                                  // punctuation
        " ?[\\x{0964}-\\x{0965}]|"                                         // devanagari punctuation
        " ?[\\x{0C9E}-\\x{0C9F}]|"                                        // kannada punctuation
        " ?[^\\s[:alpha:][:digit:]\\x{0900}-\\x{097F}\\x{0C80}-\\x{0CFF}]+|" // other symbols
        "\\s+(?!\\S)|"                                                     // trailing whitespace
        "\\s+";                                                           // other whitespace
    
    size_t text_length;
    char* text = read_file("tok_corpus.txt", &text_length);
    if (!text) {
        return 1;
    }
    
    printf("Read file successfully, size: %zu bytes\n", text_length);
    if (text_length == 0) {
        printf("Error: Empty input file\n");
        free(text);
        return 1;
    }
    
    // Train the tokenizer
    train_tokenizer(text, regex_pattern, 50256, 
                   "tokenizer_vocab.bin", "tokenizer_merges.bin");
    
    free(text);
    return 0;
} 