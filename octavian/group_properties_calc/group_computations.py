"""

This file contains numba functions used to compute physical quantities in bulk.

Numba requires quite basic syntax meaning these functions are inherently quite readable.

"""

import numpy as np
from numba import njit, prange, boolean


"""

Physical Quantities

"""

@njit(parallel=True)
def compute_centre_of_mass(pos, vel, mass, group_idx, n_groups):
    com_pos = np.zeros((n_groups, 3))
    com_vel = np.zeros((n_groups, 3))
    total_mass = np.zeros(n_groups)

    for i in prange(len(mass)):
        g = group_idx[i]
        m = mass[i]
        total_mass[g] += m
        for d in range(3):
            com_pos[g, d] += pos[i, d] * m
            com_vel[g, d] += vel[i, d] * m

    for g in range(n_groups):
        if total_mass[g] > 0:
            for d in range(3):
                com_pos[g, d] /= total_mass[g]
                com_vel[g, d] /= total_mass[g]

    return com_pos, com_vel, total_mass

@njit(parallel=True)
def compute_angular_momentum(pos_rel, vel_rel, mass, group_idx, n_groups):
    L = np.zeros((n_groups, 3))
    ktot_sum = np.zeros(n_groups)

    for i in prange(len(mass)):
        g = group_idx[i]
        rx, ry, rz = pos_rel[i, 0], pos_rel[i, 1], pos_rel[i, 2]
        vx, vy, vz = vel_rel[i, 0], vel_rel[i, 1], vel_rel[i, 2]
        m = mass[i]

        px, py, pz = m * vx, m * vy, m * vz
        L[g, 0] += ry * pz - rz * py
        L[g, 1] += rz * px - rx * pz
        L[g, 2] += rx * py - ry * px

        ktot_sum[g] += 0.5 * m * (vx**2 + vy**2 + vz**2)

    return L, ktot_sum

@njit(parallel=True)
def compute_rotation_quantities(pos_rel, vel_rel, mass, group_idx, L_group, n_groups):
    counter_rotating_mass = np.zeros(n_groups)
    krot_sum = np.zeros(n_groups)
    ktot_sum = np.zeros(n_groups)

    for i in prange(len(mass)):
        g = group_idx[i]
        rx, ry, rz = pos_rel[i, 0], pos_rel[i, 1], pos_rel[i, 2]
        vx, vy, vz = vel_rel[i, 0], vel_rel[i, 1], vel_rel[i, 2]
        m = mass[i]

        px, py, pz = m * vx, m * vy, m * vz
        Lx = ry * pz - rz * py
        Ly = rz * px - rx * pz
        Lz = rx * py - ry * px

        Lgx, Lgy, Lgz = L_group[g, 0], L_group[g, 1], L_group[g, 2]
        L_dot = Lx * Lgx + Ly * Lgy + Lz * Lgz

        if L_dot < 0:
            counter_rotating_mass[g] += m

        cx = ry * Lgz - rz * Lgy
        cy = rz * Lgx - rx * Lgz
        cz = rx * Lgy - ry * Lgx
        rz_cyl = np.sqrt(cx**2 + cy**2 + cz**2)

        ktot = 0.5 * m * (vx**2 + vy**2 + vz**2)
        ktot_sum[g] += ktot

        if rz_cyl > 0.0:
            krot_sum[g] += 0.5 * (L_dot / rz_cyl)**2 / m

    return counter_rotating_mass, krot_sum, ktot_sum

@njit
def compute_radial_quantiles(radius, mass, group_idx, n_groups, quantiles):

    counts = np.zeros(n_groups, dtype=np.int64)
    for i in range(len(mass)):
        counts[group_idx[i]] += 1

    start = np.zeros(n_groups, dtype=np.int64)
    for g in range(1, n_groups):
        start[g] = start[g-1] + counts[g-1]

    idx_sorted = np.empty(len(mass), dtype=np.int64)
    pos = np.zeros(n_groups, dtype=np.int64)
    for i in range(len(mass)):
        g = group_idx[i]
        idx_sorted[start[g] + pos[g]] = i
        pos[g] += 1

    result = np.full((n_groups, len(quantiles)), np.nan)
    for g in range(n_groups):
        s = start[g]
        e = s + counts[g]
        if s == e:
            continue
        indices = idx_sorted[s:e]
        r = radius[indices]
        m = mass[indices]
        order = np.argsort(r)
        r = r[order]
        m = m[order]
        cum = np.cumsum(m)
        frac = cum / cum[-1]
        for q in range(len(quantiles)):
            idx = np.searchsorted(frac, quantiles[q])
            if idx < len(r):
                result[g, q] = r[idx]

    return result


@njit(parallel=True)
def compute_radial_quantiles_and_rmax(radius, mass, group_idx, n_groups, quantiles):
    counts = np.zeros(n_groups, dtype=np.int64)
    for i in range(len(mass)):
        counts[group_idx[i]] += 1

    start = np.zeros(n_groups, dtype=np.int64)
    for g in range(1, n_groups):
        start[g] = start[g-1] + counts[g-1]

    idx_sorted = np.empty(len(mass), dtype=np.int64)
    pos = np.zeros(n_groups, dtype=np.int64)
    for i in range(len(mass)):
        g = group_idx[i]
        idx_sorted[start[g] + pos[g]] = i
        pos[g] += 1

    quantile_result = np.full((n_groups, len(quantiles)), np.nan)
    rmax = np.full(n_groups, np.nan)

    for g in prange(n_groups):
        s = start[g]
        e = s + counts[g]
        if s == e:
            continue
        indices = idx_sorted[s:e]
        r = radius[indices]
        m = mass[indices]
        order = np.argsort(r)
        r = r[order]
        m = m[order]
        cum = np.cumsum(m)
        total_mass = cum[-1]
        if total_mass <= 0:
            continue
        frac = cum / total_mass
        for q in range(len(quantiles)):
            idx = np.searchsorted(frac, quantiles[q])
            if idx < len(r):
                quantile_result[g, q] = r[idx]
        rmax[g] = r[-1]

    return quantile_result, rmax


@njit(parallel=True)
def compute_gas_scalar_sums(
    unique_groups,
    start,
    end,
    masses,
    metallicities,
    sfrs,
    temperatures,
    rhos,
    mass_HI,
    mass_H2,
    mass_HI_ism,
    mass_H2_ism,
    dust_masses,
    n_groups,
    nhlim,
):
    group_mass = np.zeros(n_groups)
    gas_HI = np.zeros(n_groups)
    gas_H2 = np.zeros(n_groups)
    gas_HI_ism = np.zeros(n_groups)
    gas_H2_ism = np.zeros(n_groups)
    dust = np.zeros(n_groups)
    ndust = np.zeros(n_groups, dtype=np.int64)
    sfr = np.zeros(n_groups)
    metal_mass = np.zeros(n_groups)
    metal_sfr = np.zeros(n_groups)
    temp_mass = np.zeros(n_groups)
    cgm_mass = np.zeros(n_groups)
    cgm_temp_mass = np.zeros(n_groups)
    cgm_temp_metal = np.zeros(n_groups)
    cgm_metal_mass = np.zeros(n_groups)

    for gi in prange(len(unique_groups)):
        group = unique_groups[gi]
        s = start[gi]
        e = end[gi]
        for i in range(s, e):
            m = masses[i]
            z = metallicities[i]
            sf = sfrs[i]
            temp = temperatures[i]
            rho = rhos[i]

            group_mass[group] += m
            gas_HI[group] += mass_HI[i]
            gas_H2[group] += mass_H2[i]
            gas_HI_ism[group] += mass_HI_ism[i]
            gas_H2_ism[group] += mass_H2_ism[i]
            sfr[group] += sf
            metal_mass[group] += z * m
            metal_sfr[group] += z * sf
            temp_mass[group] += temp * m

            if rho >= nhlim:
                dust[group] += dust_masses[i]
                if dust_masses[i] > 0:
                    ndust[group] += 1
            else:
                cgm_mass[group] += m
                cgm_temp_mass[group] += temp * m
                cgm_temp_metal[group] += temp * m * z
                cgm_metal_mass[group] += z * m

    return (
        group_mass,
        gas_HI,
        gas_H2,
        gas_HI_ism,
        gas_H2_ism,
        dust,
        ndust,
        sfr,
        metal_mass,
        metal_sfr,
        temp_mass,
        cgm_mass,
        cgm_temp_mass,
        cgm_temp_metal,
        cgm_metal_mass,
    )


@njit(parallel=True)
def compute_star_scalar_sums(
    unique_groups,
    start,
    end,
    masses,
    metallicities,
    ages,
    n_groups,
):
    total_mass = np.zeros(n_groups)
    metal_mass = np.zeros(n_groups)
    age_mass = np.zeros(n_groups)
    age_metal = np.zeros(n_groups)
    young_mass = np.zeros(n_groups)

    for gi in prange(len(unique_groups)):
        group = unique_groups[gi]
        s = start[gi]
        e = end[gi]
        for i in range(s, e):
            m = masses[i]
            z = metallicities[i]
            age = ages[i]
            total_mass[group] += m
            metal_mass[group] += z * m
            age_mass[group] += age * m
            age_metal[group] += age * m * z
            if age < 0.1:
                young_mass[group] += m

    return total_mass, metal_mass, age_mass, age_metal, young_mass


@njit
def compute_central_galaxy_flags(halo_positions, stellar_mass, galaxy_ids, n_halos):
    central = np.zeros(len(galaxy_ids), dtype=boolean)
    central_by_halo = np.full(n_halos, -1, dtype=np.int64)
    central_index = np.full(n_halos, -1, dtype=np.int64)
    max_mass = np.full(n_halos, -np.inf)

    for i in range(len(galaxy_ids)):
        h = halo_positions[i]
        if h < 0 or h >= n_halos:
            continue
        if stellar_mass[i] > max_mass[h]:
            max_mass[h] = stellar_mass[i]
            central_index[h] = i
            central_by_halo[h] = galaxy_ids[i]

    for h in range(n_halos):
        i = central_index[h]
        if i >= 0:
            central[i] = True

    return central, central_by_halo


@njit(parallel=True)
def compute_aperture_component_properties(
    neighbor_offsets,
    neighbor_indices,
    particle_masses,
    particle_codes,
    particle_velocities,
    include_matrix,
):
    """Reduce aperture neighbor lists into component masses and velocity dispersions.

    include_matrix maps particle-type codes to each requested output component, allowing
    one neighbor pass to fill gas, star, HI, H2, dust, total, and baryon outputs."""
    n_galaxies = len(neighbor_offsets) - 1
    n_outputs = include_matrix.shape[0]

    output_mass = np.zeros((n_galaxies, n_outputs))
    output_count = np.zeros((n_galaxies, n_outputs), dtype=np.int64)
    mean_momentum = np.zeros((n_galaxies, n_outputs, 3))
    momentum_var = np.zeros((n_galaxies, n_outputs, 3))
    sigma = np.zeros((n_galaxies, n_outputs))

    for g in prange(n_galaxies):
        s = neighbor_offsets[g]
        e = neighbor_offsets[g + 1]

        for ni in range(s, e):
            p = neighbor_indices[ni]
            code = particle_codes[p]
            mass = particle_masses[p]
            if mass <= 0:
                continue
            for out in range(n_outputs):
                if include_matrix[out, code]:
                    output_mass[g, out] += mass
                    output_count[g, out] += 1
                    for d in range(3):
                        mean_momentum[g, out, d] += mass * particle_velocities[p, d]

        for out in range(n_outputs):
            count = output_count[g, out]
            if count > 0:
                for d in range(3):
                    mean_momentum[g, out, d] /= count

        for ni in range(s, e):
            p = neighbor_indices[ni]
            code = particle_codes[p]
            mass = particle_masses[p]
            if mass <= 0:
                continue
            for out in range(n_outputs):
                if include_matrix[out, code]:
                    for d in range(3):
                        delta = mass * particle_velocities[p, d] - mean_momentum[g, out, d]
                        momentum_var[g, out, d] += delta * delta

        for out in range(n_outputs):
            count = output_count[g, out]
            if count > 0 and output_mass[g, out] > 0:
                mean_mass = output_mass[g, out] / count
                sig2 = 0.0
                for d in range(3):
                    axis_sigma = np.sqrt(momentum_var[g, out, d] / count) / mean_mass
                    sig2 += axis_sigma * axis_sigma
                sigma[g, out] = np.sqrt(sig2 / 3.0)

    return output_mass, sigma


@njit
def compute_galaxy_hydrogen_assignment(
    gas_unique_halos,
    gas_start,
    gas_end,
    gal_unique_halos,
    gal_start,
    gal_end,
    gas_pos,
    mass_HI,
    mass_H2,
    galaxy_pos,
    galaxy_mass,
    galaxy_indices,
    n_galaxies,
    boxsize,
):
    """Assign gas HI and H2 masses to galaxies inside matching host-halo blocks.

    Within a host halo, each gas particle chooses the galaxy maximizing stellar mass over
    periodic distance squared while keeping the sweep local to one halo at a time."""
    galaxy_HI = np.zeros(n_galaxies)
    galaxy_H2 = np.zeros(n_galaxies)
    halfbox = 0.5 * boxsize
    gh = 0

    for h in range(len(gas_unique_halos)):
        halo_id = gas_unique_halos[h]
        while gh < len(gal_unique_halos) and gal_unique_halos[gh] < halo_id:
            gh += 1
        if gh >= len(gal_unique_halos):
            break
        if gal_unique_halos[gh] != halo_id:
            continue

        gs = gas_start[h]
        ge = gas_end[h]
        js = gal_start[gh]
        je = gal_end[gh]
        for i in range(gs, ge):
            hi = mass_HI[i]
            h2 = mass_H2[i]
            if hi == 0.0 and h2 == 0.0:
                continue

            best = galaxy_indices[js]
            best_mwd = -1.0
            for j in range(js, je):
                dx = abs(gas_pos[i, 0] - galaxy_pos[j, 0])
                dy = abs(gas_pos[i, 1] - galaxy_pos[j, 1])
                dz = abs(gas_pos[i, 2] - galaxy_pos[j, 2])
                if dx > halfbox:
                    dx = boxsize - dx
                if dy > halfbox:
                    dy = boxsize - dy
                if dz > halfbox:
                    dz = boxsize - dz
                d2 = dx * dx + dy * dy + dz * dz
                if d2 == 0.0:
                    best = galaxy_indices[j]
                    break
                mwd = galaxy_mass[j] / d2
                if mwd > best_mwd:
                    best_mwd = mwd
                    best = galaxy_indices[j]

            galaxy_HI[best] += hi
            galaxy_H2[best] += h2

    return galaxy_HI, galaxy_H2


@njit
def compute_virial_quantities(radius, mass, group_idx, n_groups, rhocrit, factors):

    volume_factor = 4.0 / 3.0 * np.pi

    counts = np.zeros(n_groups, dtype=np.int64)
    for i in range(len(mass)):
        counts[group_idx[i]] += 1

    start = np.zeros(n_groups, dtype=np.int64)
    for g in range(1, n_groups):
        start[g] = start[g-1] + counts[g-1]

    idx_sorted = np.empty(len(mass), dtype=np.int64)
    pos = np.zeros(n_groups, dtype=np.int64)
    for i in range(len(mass)):
        g = group_idx[i]
        idx_sorted[start[g] + pos[g]] = i
        pos[g] += 1

    result_r = np.full((n_groups, len(factors)), np.nan)
    result_m = np.full((n_groups, len(factors)), np.nan)

    for g in range(n_groups):
        s = start[g]
        e = s + counts[g]
        if s == e:
            continue
        indices = idx_sorted[s:e]
        r = radius[indices]
        m = mass[indices]
        order = np.argsort(r)
        r = r[order]
        m = m[order]
        cumulative = np.cumsum(m)

        for i in range(len(r)):
            if r[i] > 0:
                overdensity = cumulative[i] / (volume_factor * r[i]**3) / rhocrit
                for f in range(len(factors)):
                    if overdensity >= factors[f]:
                        result_r[g, f] = r[i]
                        result_m[g, f] = cumulative[i]

    return result_r, result_m


@njit
def accumulate_membership_array_common_first(
    halo_id_array,
    group_index_map,
    positions,
    velocities,
    masses,
    potentials,
    counts,
    group_mass,
    pos_mass_sum,
    vel_mass_sum,
    min_potential,
    minpot_position,
    minpot_velocity,
    mass_mode,
    do_minpot,
):
    """First pass for common properties over particle ancestry arrays.

    Every valid ancestry entry contributes to its corresponding output halo. For black
    holes, mass_mode keeps the maximum mass instead of summing particle masses."""
    width = halo_id_array.shape[1]
    for i in range(halo_id_array.shape[0]):
        m = masses[i]
        for col in range(width):
            halo_id = halo_id_array[i, col]
            if halo_id < 0 or halo_id >= len(group_index_map):
                continue
            g = group_index_map[halo_id]
            if g < 0:
                continue

            counts[g] += 1
            if mass_mode == 1:
                if m > group_mass[g]:
                    group_mass[g] = m
            else:
                group_mass[g] += m

            for d in range(3):
                pos_mass_sum[g, d] += positions[i, d] * m
                vel_mass_sum[g, d] += velocities[i, d] * m

            if do_minpot and potentials[i] < min_potential[g]:
                min_potential[g] = potentials[i]
                for d in range(3):
                    minpot_position[g, d] = positions[i, d]
                    minpot_velocity[g, d] = velocities[i, d]


@njit
def accumulate_membership_array_common_second(
    halo_id_array,
    group_index_map,
    positions,
    velocities,
    masses,
    ref_positions,
    ref_velocities,
    com_velocities,
    disp_sums,
    angular_momentum,
):
    width = halo_id_array.shape[1]
    for i in range(halo_id_array.shape[0]):
        m = masses[i]
        for col in range(width):
            halo_id = halo_id_array[i, col]
            if halo_id < 0 or halo_id >= len(group_index_map):
                continue
            g = group_index_map[halo_id]
            if g < 0:
                continue

            rx = positions[i, 0] - ref_positions[g, 0]
            ry = positions[i, 1] - ref_positions[g, 1]
            rz = positions[i, 2] - ref_positions[g, 2]
            vx_ref = velocities[i, 0] - ref_velocities[g, 0]
            vy_ref = velocities[i, 1] - ref_velocities[g, 1]
            vz_ref = velocities[i, 2] - ref_velocities[g, 2]
            vx_com = velocities[i, 0] - com_velocities[g, 0]
            vy_com = velocities[i, 1] - com_velocities[g, 1]
            vz_com = velocities[i, 2] - com_velocities[g, 2]

            disp_sums[g] += vx_com * vx_com + vy_com * vy_com + vz_com * vz_com

            px = m * vx_ref
            py = m * vy_ref
            pz = m * vz_ref
            angular_momentum[g, 0] += ry * pz - rz * py
            angular_momentum[g, 1] += rz * px - rx * pz
            angular_momentum[g, 2] += rx * py - ry * px


@njit
def accumulate_membership_array_rotation(
    halo_id_array,
    group_index_map,
    positions,
    velocities,
    masses,
    ref_positions,
    ref_velocities,
    angular_momentum,
    counter_rotating_mass,
    krot_sum,
    ktot_sum,
):
    width = halo_id_array.shape[1]
    for i in range(halo_id_array.shape[0]):
        m = masses[i]
        for col in range(width):
            halo_id = halo_id_array[i, col]
            if halo_id < 0 or halo_id >= len(group_index_map):
                continue
            g = group_index_map[halo_id]
            if g < 0:
                continue

            rx = positions[i, 0] - ref_positions[g, 0]
            ry = positions[i, 1] - ref_positions[g, 1]
            rz = positions[i, 2] - ref_positions[g, 2]
            vx = velocities[i, 0] - ref_velocities[g, 0]
            vy = velocities[i, 1] - ref_velocities[g, 1]
            vz = velocities[i, 2] - ref_velocities[g, 2]

            px = m * vx
            py = m * vy
            pz = m * vz
            lx = ry * pz - rz * py
            ly = rz * px - rx * pz
            lz = rx * py - ry * px

            lgx = angular_momentum[g, 0]
            lgy = angular_momentum[g, 1]
            lgz = angular_momentum[g, 2]
            ldot = lx * lgx + ly * lgy + lz * lgz
            if ldot < 0.0:
                counter_rotating_mass[g] += m

            cx = ry * lgz - rz * lgy
            cy = rz * lgx - rx * lgz
            cz = rx * lgy - ry * lgx
            rz_cyl = np.sqrt(cx * cx + cy * cy + cz * cz)

            ktot = 0.5 * m * (vx * vx + vy * vy + vz * vz)
            ktot_sum[g] += ktot
            if rz_cyl > 0.0:
                krot_sum[g] += 0.5 * (ldot / rz_cyl) ** 2 / m


@njit
def flatten_membership_array_radius_mass(
    halo_id_array,
    group_index_map,
    positions,
    masses,
    ref_positions,
):
    width = halo_id_array.shape[1]
    n = 0
    for i in range(halo_id_array.shape[0]):
        for col in range(width):
            halo_id = halo_id_array[i, col]
            if halo_id >= 0 and halo_id < len(group_index_map) and group_index_map[halo_id] >= 0:
                n += 1

    group_idx = np.empty(n, dtype=np.int64)
    radii = np.empty(n)
    flat_mass = np.empty(n)

    out = 0
    for i in range(halo_id_array.shape[0]):
        for col in range(width):
            halo_id = halo_id_array[i, col]
            if halo_id < 0 or halo_id >= len(group_index_map):
                continue
            g = group_index_map[halo_id]
            if g < 0:
                continue

            dx = positions[i, 0] - ref_positions[g, 0]
            dy = positions[i, 1] - ref_positions[g, 1]
            dz = positions[i, 2] - ref_positions[g, 2]
            group_idx[out] = g
            radii[out] = np.sqrt(dx * dx + dy * dy + dz * dz)
            flat_mass[out] = masses[i]
            out += 1

    return group_idx, radii, flat_mass


@njit
def compute_membership_array_gas_scalar_sums(
    halo_id_array,
    group_index_map,
    masses,
    metallicities,
    sfrs,
    temperatures,
    rhos,
    mass_HI,
    mass_H2,
    mass_HI_ism,
    mass_H2_ism,
    dust_masses,
    n_groups,
    nhlim,
):
    """Accumulate gas scalar outputs over every valid halo ancestry entry.

    This mirrors compute_gas_scalar_sums but uses particle by ancestry arrays instead of
    a single group index, so subhalo and ancestor rows receive their own gas totals."""
    group_mass = np.zeros(n_groups)
    gas_HI = np.zeros(n_groups)
    gas_H2 = np.zeros(n_groups)
    gas_HI_ism = np.zeros(n_groups)
    gas_H2_ism = np.zeros(n_groups)
    dust = np.zeros(n_groups)
    ndust = np.zeros(n_groups, dtype=np.int64)
    sfr = np.zeros(n_groups)
    metal_mass = np.zeros(n_groups)
    metal_sfr = np.zeros(n_groups)
    temp_mass = np.zeros(n_groups)
    cgm_mass = np.zeros(n_groups)
    cgm_temp_mass = np.zeros(n_groups)
    cgm_temp_metal = np.zeros(n_groups)
    cgm_metal_mass = np.zeros(n_groups)

    width = halo_id_array.shape[1]
    for i in range(halo_id_array.shape[0]):
        m = masses[i]
        z = metallicities[i]
        sf = sfrs[i]
        temp = temperatures[i]
        rho = rhos[i]
        for col in range(width):
            halo_id = halo_id_array[i, col]
            if halo_id < 0 or halo_id >= len(group_index_map):
                continue
            g = group_index_map[halo_id]
            if g < 0:
                continue

            group_mass[g] += m
            gas_HI[g] += mass_HI[i]
            gas_H2[g] += mass_H2[i]
            gas_HI_ism[g] += mass_HI_ism[i]
            gas_H2_ism[g] += mass_H2_ism[i]
            sfr[g] += sf
            metal_mass[g] += z * m
            metal_sfr[g] += z * sf
            temp_mass[g] += temp * m

            if rho >= nhlim:
                dust[g] += dust_masses[i]
                if dust_masses[i] > 0.0:
                    ndust[g] += 1
            else:
                cgm_mass[g] += m
                cgm_temp_mass[g] += temp * m
                cgm_temp_metal[g] += temp * m * z
                cgm_metal_mass[g] += z * m

    return (
        group_mass,
        gas_HI,
        gas_H2,
        gas_HI_ism,
        gas_H2_ism,
        dust,
        ndust,
        sfr,
        metal_mass,
        metal_sfr,
        temp_mass,
        cgm_mass,
        cgm_temp_mass,
        cgm_temp_metal,
        cgm_metal_mass,
    )


@njit
def compute_membership_array_star_scalar_sums(
    halo_id_array,
    group_index_map,
    masses,
    metallicities,
    ages,
    n_groups,
):
    total_mass = np.zeros(n_groups)
    metal_mass = np.zeros(n_groups)
    age_mass = np.zeros(n_groups)
    age_metal = np.zeros(n_groups)
    young_mass = np.zeros(n_groups)

    width = halo_id_array.shape[1]
    for i in range(halo_id_array.shape[0]):
        m = masses[i]
        z = metallicities[i]
        age = ages[i]
        for col in range(width):
            halo_id = halo_id_array[i, col]
            if halo_id < 0 or halo_id >= len(group_index_map):
                continue
            g = group_index_map[halo_id]
            if g < 0:
                continue

            total_mass[g] += m
            metal_mass[g] += z * m
            age_mass[g] += age * m
            age_metal[g] += age * m * z
            if age < 0.1:
                young_mass[g] += m

    return total_mass, metal_mass, age_mass, age_metal, young_mass


@njit
def compute_membership_array_bh_max(
    halo_id_array,
    group_index_map,
    masses,
    bhmdots,
    n_groups,
):
    max_mass = np.full(n_groups, -np.inf)
    bhmdot = np.full(n_groups, np.nan)

    width = halo_id_array.shape[1]
    for i in range(halo_id_array.shape[0]):
        m = masses[i]
        for col in range(width):
            halo_id = halo_id_array[i, col]
            if halo_id < 0 or halo_id >= len(group_index_map):
                continue
            g = group_index_map[halo_id]
            if g < 0:
                continue
            if m > max_mass[g]:
                max_mass[g] = m
                bhmdot[g] = bhmdots[i]

    return max_mass, bhmdot
