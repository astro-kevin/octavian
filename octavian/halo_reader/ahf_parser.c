// REVIEW: 
// It took a long time to write this, and I needed extensive help from Claude.
// I would not be able to debug it but hopefully someone who understands C can take a look.
// This should be regarded as entirely experimental, please do not use it for data analysis!

// libraries
#include <stdio.h>   // file operations: fopen, fgets, fclose
#include <stdlib.h>  // general utilities
#include <stdint.h>  // fixed-width integer types

// NOTE: this is a C implementation of the state machine

long parse_ahf_particles(
    const char *filename,   // string (path to file)
    long *out_hids,         // pointer to array of halo IDs (pre-allocated from Python)
    long *out_pids,         // pointer to array of particle IDs
    char *out_ptypes,       // pointer to array of ptype codes
    int *valid_ptypes,      // pointer to array of valid ptype codes {0, 1, 4, 5}
    int n_valid             // how many valid ptypes
) 
{
    FILE *f = fopen(filename, "r");  // "r" = read mode
    if (f == NULL) return -1;        // if file doesn't exist, return error
    char line[256];          // buffer to hold a line as we read in (256 characters, should be enough)
    long remaining = 0;      // for the state machine: particles in halo
    long current_hid = 0;    // current halo ID
    long idx = 0;            // index position in output array
    long a, b;               // a: nparticles, b: hid
    int i, valid;            // loop counter and flag

    while (fgets(line, sizeof(line), f)) {         // fgets reads one line from the file into the line buffer (like a for loop)
        if (sscanf(line, "%ld %ld", &a, &b) != 2) // sscanf scans the string line to extract values from it according to the specified pattern
            continue; // so if there are not two entries on the line, continue
        if (remaining == 0) { // if remaining is 0 we have hit a header line
        // header line: (nparticles | haloID)
        remaining = a; 
        current_hid = b;
        continue; // move on to next line
        }

        remaining--; // C equivalent of python remaining -= 1
        valid = 0;
        for (i = 0; i < n_valid; i++) { // check the four possible ptypes
            if ((int)b == valid_ptypes[i]) {
                valid = 1;
                break; // particle is one found in the ptype map
            }
        }
        if (!valid) continue; // if valid is 0, this particle is not Octavian-friendly and is skipped
        out_hids[idx] = current_hid;
        out_pids[idx] = a;
        out_ptypes[idx] = (char)b;
        idx++; // increment our write position (move on to next one when writing in)
    }
    fclose(f); // must explicitly close the file 
    return idx; // number of particles we wrote into the array
}

#define MISSING_ROW UINT32_MAX

static long find_halo_id(int64_t hid, const int64_t *raw_halo_ids, long n_halos)
{
    long lo = 0;
    long hi = n_halos;
    while (lo < hi) {
        long mid = lo + (hi - lo) / 2;
        int64_t val = raw_halo_ids[mid];
        if (val < hid) lo = mid + 1;
        else hi = mid;
    }
    if (lo < n_halos && raw_halo_ids[lo] == hid) return lo;
    return -1;
}

static int slot_from_raw_ptype(int64_t raw_ptype)
{
    if (raw_ptype == 0) return 0;  // gas
    if (raw_ptype == 1) return 1;  // dm
    if (raw_ptype == 4) return 2;  // star
    if (raw_ptype == 5) return 3;  // bh
    return -1;
}

static int membership_depth(const int32_t *values, int width)
{
    int depth = -1;
    for (int col = 0; col < width; col++) {
        if (values[col] >= 0) depth = col;
    }
    return depth;
}

static void clear_after(int32_t *values, int width, int depth)
{
    for (int col = depth + 1; col < width; col++) {
        values[col] = -1;
    }
}

static void copy_candidate(int32_t *membership_array, const int32_t *candidate, int width, int depth)
{
    for (int col = 0; col <= depth; col++) {
        membership_array[col] = candidate[col];
    }
    clear_after(membership_array, width, depth);
}

static int is_prefix(const int32_t *prefix, int prefix_depth, const int32_t *membership_array, int array_depth)
{
    if (prefix_depth > array_depth) return 0;
    for (int col = 0; col <= prefix_depth; col++) {
        if (prefix[col] != membership_array[col]) return 0;
    }
    return 1;
}

static void choose_membership_value(
    int32_t *membership_array,
    uint32_t row,
    int width,
    const int32_t *candidate,
    uint64_t *counts
)
{
    int candidate_depth = membership_depth(candidate, width);
    if (candidate_depth < 0) return;

    int32_t *current = membership_array + (uint64_t)row * (uint64_t)width;
    int current_depth = membership_depth(current, width);
    if (current_depth < 0) {
        copy_candidate(current, candidate, width, candidate_depth);
        return;
    }

    if (is_prefix(current, current_depth, candidate, candidate_depth)) {
        copy_candidate(current, candidate, width, candidate_depth);
        return;
    } else if (is_prefix(candidate, candidate_depth, current, current_depth)) {
        return;
    }

    counts[7]++;
    if (
        candidate_depth > current_depth ||
        (candidate_depth == current_depth && candidate[candidate_depth] < current[current_depth])
    ) {
        copy_candidate(current, candidate, width, candidate_depth);
    }
}

long fill_ahf_membership_arrays(
    const char *filename,
    const int64_t *raw_halo_ids,
    const int32_t *ancestor_arrays,
    long n_halos,
    const uint32_t *row_lookup,
    const int8_t *slot_lookup,
    int64_t max_pid,
    int width,
    int32_t *array_gas,
    int32_t *array_dm,
    int32_t *array_star,
    int32_t *array_bh,
    uint64_t *counts
)
{
    FILE *f = fopen(filename, "r");
    if (f == NULL) return -1;

    char line[256];
    int64_t remaining = 0;
    long current_halo_id = -1;
    long long a_ll, b_ll;
    long written = 0;

    while (fgets(line, sizeof(line), f)) {
        if (sscanf(line, "%lld %lld", &a_ll, &b_ll) != 2) continue;
        int64_t a = (int64_t)a_ll;
        int64_t b = (int64_t)b_ll;

        if (remaining == 0) {
            remaining = a;
            current_halo_id = find_halo_id(b, raw_halo_ids, n_halos);
            continue;
        }

        remaining--;
        int slot = slot_from_raw_ptype(b);
        if (slot < 0) {
            counts[4]++;
            continue;
        }
        if (current_halo_id < 0 || a < 0 || a > max_pid) {
            counts[5]++;
            continue;
        }

        uint32_t row = row_lookup[a];
        int lookup_slot = (int)slot_lookup[a];
        if (row == MISSING_ROW || lookup_slot != slot) {
            counts[6]++;
            continue;
        }

        const int32_t *candidate = ancestor_arrays + (uint64_t)current_halo_id * (uint64_t)width;
        if (slot == 0) choose_membership_value(array_gas, row, width, candidate, counts);
        else if (slot == 1) choose_membership_value(array_dm, row, width, candidate, counts);
        else if (slot == 2) choose_membership_value(array_star, row, width, candidate, counts);
        else choose_membership_value(array_bh, row, width, candidate, counts);

        counts[slot]++;
        written++;
    }

    fclose(f);
    return written;
}


static void choose_scalar_membership_value(
    int32_t *memberships,
    uint32_t row,
    int32_t candidate_hid,
    const int32_t *ancestor_arrays,
    const int32_t *depth_by_halo,
    int width,
    uint64_t *counts
)
{
    int candidate_depth = depth_by_halo[candidate_hid];
    if (candidate_depth < 0) return;

    int32_t encoded_current = memberships[row];
    if (encoded_current <= 0) {
        memberships[row] = candidate_hid + 1;
        return;
    }

    int32_t current_hid = encoded_current - 1;
    int current_depth = depth_by_halo[current_hid];
    if (current_depth < 0) {
        memberships[row] = candidate_hid + 1;
        return;
    }

    const int32_t *candidate_ancestors = ancestor_arrays + (uint64_t)candidate_hid * (uint64_t)width;
    const int32_t *current_ancestors = ancestor_arrays + (uint64_t)current_hid * (uint64_t)width;

    if (current_depth <= candidate_depth && candidate_ancestors[current_depth] == current_hid) {
        memberships[row] = candidate_hid + 1;
        return;
    }

    if (candidate_depth <= current_depth && current_ancestors[candidate_depth] == candidate_hid) {
        return;
    }

    counts[7]++;
    if (
        candidate_depth > current_depth ||
        (candidate_depth == current_depth && candidate_hid < current_hid)
    ) {
        memberships[row] = candidate_hid + 1;
    }
}

long fill_ahf_scalar_membership_arrays(
    const char *filename,
    const int64_t *raw_halo_ids,
    const int32_t *ancestor_arrays,
    long n_halos,
    const uint32_t *row_lookup,
    const int8_t *slot_lookup,
    int64_t max_pid,
    int width,
    const int32_t *depth_by_halo,
    int32_t *membership_gas,
    int32_t *membership_dm,
    int32_t *membership_star,
    int32_t *membership_bh,
    uint64_t *counts
)
{
    FILE *f = fopen(filename, "r");
    if (f == NULL) return -1;

    char line[256];
    int64_t remaining = 0;
    long current_halo_id = -1;
    long long a_ll, b_ll;
    long written = 0;

    while (fgets(line, sizeof(line), f)) {
        if (sscanf(line, "%lld %lld", &a_ll, &b_ll) != 2) continue;
        int64_t a = (int64_t)a_ll;
        int64_t b = (int64_t)b_ll;

        if (remaining == 0) {
            remaining = a;
            current_halo_id = find_halo_id(b, raw_halo_ids, n_halos);
            continue;
        }

        remaining--;
        int slot = slot_from_raw_ptype(b);
        if (slot < 0) {
            counts[4]++;
            continue;
        }
        if (current_halo_id < 0 || a < 0 || a > max_pid) {
            counts[5]++;
            continue;
        }

        uint32_t encoded_row = row_lookup[a];
        int lookup_slot = (int)slot_lookup[a];
        if (encoded_row == 0 || lookup_slot != slot + 1) {
            counts[6]++;
            continue;
        }
        uint32_t row = encoded_row - 1;

        if (slot == 0) choose_scalar_membership_value(membership_gas, row, (int32_t)current_halo_id, ancestor_arrays, depth_by_halo, width, counts);
        else if (slot == 1) choose_scalar_membership_value(membership_dm, row, (int32_t)current_halo_id, ancestor_arrays, depth_by_halo, width, counts);
        else if (slot == 2) choose_scalar_membership_value(membership_star, row, (int32_t)current_halo_id, ancestor_arrays, depth_by_halo, width, counts);
        else choose_scalar_membership_value(membership_bh, row, (int32_t)current_halo_id, ancestor_arrays, depth_by_halo, width, counts);

        counts[slot]++;
        written++;
    }

    fclose(f);
    return written;
}
