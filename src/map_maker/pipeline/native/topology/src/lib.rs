use std::f64::consts::FRAC_PI_2;
use std::mem::size_of;

const FACE_COUNT: usize = 6;
const D4_NEIGHBORS: usize = 4;

#[no_mangle]
pub extern "C" fn topology_native_abi_version() -> u32 {
    2
}

#[no_mangle]
pub extern "C" fn cubed_sphere_hierarchy_abi_version() -> u32 {
    1
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
enum Face {
    PositiveX = 0,
    NegativeX = 1,
    PositiveY = 2,
    NegativeY = 3,
    PositiveZ = 4,
    NegativeZ = 5,
}

impl Face {
    fn from_index(index: usize) -> Self {
        match index {
            0 => Self::PositiveX,
            1 => Self::NegativeX,
            2 => Self::PositiveY,
            3 => Self::NegativeY,
            4 => Self::PositiveZ,
            5 => Self::NegativeZ,
            _ => panic!("invalid cubed-sphere face {index}"),
        }
    }

    fn basis(self) -> ([f64; 3], [f64; 3], [f64; 3]) {
        match self {
            Self::PositiveX => ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]),
            Self::NegativeX => ([-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]),
            Self::PositiveY => ([0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]),
            Self::NegativeY => ([0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]),
            Self::PositiveZ => ([0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]),
            Self::NegativeZ => ([0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Vec3 {
    x: f64,
    y: f64,
    z: f64,
}

impl Vec3 {
    fn from_array(value: [f64; 3]) -> Self {
        Self {
            x: value[0],
            y: value[1],
            z: value[2],
        }
    }

    fn dot(self, other: Self) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    fn cross(self, other: Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    fn norm(self) -> f64 {
        self.dot(self).sqrt()
    }

    fn normalized(self) -> Self {
        let norm = self.norm().max(f64::EPSILON);
        Self {
            x: self.x / norm,
            y: self.y / norm,
            z: self.z / norm,
        }
    }
}

fn face_vector(face: Face, u: f64, v: f64) -> Vec3 {
    let (normal, right, down) = face.basis();
    Vec3 {
        x: normal[0] + u * right[0] + v * down[0],
        y: normal[1] + u * right[1] + v * down[1],
        z: normal[2] + u * right[2] + v * down[2],
    }
    .normalized()
}

fn dominant_face(direction: Vec3) -> Face {
    let abs_x = direction.x.abs();
    let abs_y = direction.y.abs();
    let abs_z = direction.z.abs();
    if abs_x >= abs_y && abs_x >= abs_z {
        if direction.x >= 0.0 {
            Face::PositiveX
        } else {
            Face::NegativeX
        }
    } else if abs_y >= abs_z {
        if direction.y >= 0.0 {
            Face::PositiveY
        } else {
            Face::NegativeY
        }
    } else if direction.z >= 0.0 {
        Face::PositiveZ
    } else {
        Face::NegativeZ
    }
}

fn direction_to_face_uv(direction: Vec3) -> (Face, f64, f64) {
    let face = dominant_face(direction);
    let (normal, right, down) = face.basis();
    let normal = Vec3::from_array(normal);
    let denominator = direction.dot(normal).max(f64::EPSILON);
    let u = direction.dot(Vec3::from_array(right)) / denominator;
    let v = direction.dot(Vec3::from_array(down)) / denominator;
    (face, u, v)
}

fn angular_coordinate(index: usize, resolution: usize) -> f64 {
    ((index as f64 + 0.5) / resolution as f64 - 0.5) * FRAC_PI_2
}

fn angular_edge(index: usize, resolution: usize) -> f64 {
    (index as f64 / resolution as f64 - 0.5) * FRAC_PI_2
}

fn cell_direction(face: Face, row: usize, col: usize, resolution: usize) -> Vec3 {
    let alpha = angular_coordinate(col, resolution);
    let beta = angular_coordinate(row, resolution);
    face_vector(face, alpha.tan(), beta.tan())
}

fn cell_corner(face: Face, row_edge: usize, col_edge: usize, resolution: usize) -> Vec3 {
    let alpha = angular_edge(col_edge, resolution);
    let beta = angular_edge(row_edge, resolution);
    face_vector(face, alpha.tan(), beta.tan())
}

fn spherical_triangle_area(a: Vec3, b: Vec3, c: Vec3) -> f64 {
    let numerator = a.dot(b.cross(c)).abs();
    let denominator = 1.0 + a.dot(b) + b.dot(c) + c.dot(a);
    2.0 * numerator.atan2(denominator.max(0.0))
}

fn cell_area(face: Face, row: usize, col: usize, resolution: usize) -> f64 {
    let top_left = cell_corner(face, row, col, resolution);
    let top_right = cell_corner(face, row, col + 1, resolution);
    let bottom_right = cell_corner(face, row + 1, col + 1, resolution);
    let bottom_left = cell_corner(face, row + 1, col, resolution);
    spherical_triangle_area(top_left, top_right, bottom_right)
        + spherical_triangle_area(top_left, bottom_right, bottom_left)
}

fn coordinate_to_index(value: f64, resolution: usize) -> usize {
    let normalized = value.atan() / FRAC_PI_2 + 0.5;
    ((normalized * resolution as f64).floor() as isize).clamp(0, resolution as isize - 1) as usize
}

fn global_index(face: Face, row: usize, col: usize, resolution: usize) -> usize {
    face as usize * resolution * resolution + row * resolution + col
}

fn checked_cell_count(resolution: usize) -> Option<usize> {
    resolution
        .checked_mul(resolution)
        .and_then(|face_cells| face_cells.checked_mul(FACE_COUNT))
        .filter(|cells| *cells <= i32::MAX as usize)
}

fn checked_slice_len<T>(len: usize) -> Option<usize> {
    len.checked_mul(size_of::<T>())
        .filter(|bytes| *bytes <= isize::MAX as usize)
        .map(|_| len)
}

fn hierarchy_resolutions(fine_resolution: i32, factor: i32) -> Option<(usize, usize)> {
    if fine_resolution <= 0 || factor <= 1 || fine_resolution % factor != 0 {
        return None;
    }
    let fine = fine_resolution as usize;
    let coarse = (fine_resolution / factor) as usize;
    checked_cell_count(fine)?;
    checked_cell_count(coarse)?;
    Some((fine, coarse))
}

fn neighbor_index(
    face: Face,
    row: usize,
    col: usize,
    row_offset: isize,
    col_offset: isize,
    resolution: usize,
) -> usize {
    let step = FRAC_PI_2 / resolution as f64;
    let alpha = angular_coordinate(col, resolution) + col_offset as f64 * step;
    let beta = angular_coordinate(row, resolution) + row_offset as f64 * step;
    let direction = face_vector(face, alpha.tan(), beta.tan());
    let (neighbor_face, u, v) = direction_to_face_uv(direction);
    let neighbor_col = coordinate_to_index(u, resolution);
    let neighbor_row = coordinate_to_index(v, resolution);
    global_index(neighbor_face, neighbor_row, neighbor_col, resolution)
}

fn generate(
    resolution: usize,
    xyz: &mut [f32],
    longitude: &mut [f64],
    latitude: &mut [f64],
    areas: &mut [f64],
    neighbors: &mut [i32],
) {
    let offsets = [(-1isize, 0isize), (1, 0), (0, -1), (0, 1)];
    for face_index in 0..FACE_COUNT {
        let face = Face::from_index(face_index);
        for row in 0..resolution {
            for col in 0..resolution {
                let index = global_index(face, row, col, resolution);
                let direction = cell_direction(face, row, col, resolution);
                let xyz_offset = index * 3;
                xyz[xyz_offset] = direction.x as f32;
                xyz[xyz_offset + 1] = direction.y as f32;
                xyz[xyz_offset + 2] = direction.z as f32;
                longitude[index] = direction.y.atan2(direction.x);
                latitude[index] = direction.z.asin();
                areas[index] = cell_area(face, row, col, resolution);
                for (slot, (row_offset, col_offset)) in offsets.iter().enumerate() {
                    neighbors[index * D4_NEIGHBORS + slot] =
                        neighbor_index(face, row, col, *row_offset, *col_offset, resolution) as i32;
                }
            }
        }
    }
}

fn build_parent_map(fine_resolution: usize, factor: usize, output: &mut [i32]) {
    let coarse_resolution = fine_resolution / factor;
    for face_index in 0..FACE_COUNT {
        let face = Face::from_index(face_index);
        for row in 0..fine_resolution {
            for col in 0..fine_resolution {
                let fine_index = global_index(face, row, col, fine_resolution);
                output[fine_index] =
                    global_index(face, row / factor, col / factor, coarse_resolution) as i32;
            }
        }
    }
}

fn build_children_map(coarse_resolution: usize, factor: usize, output: &mut [i32]) {
    let fine_resolution = coarse_resolution * factor;
    let children_per_parent = factor * factor;
    for face_index in 0..FACE_COUNT {
        let face = Face::from_index(face_index);
        for row in 0..coarse_resolution {
            for col in 0..coarse_resolution {
                let parent = global_index(face, row, col, coarse_resolution);
                for child_row in 0..factor {
                    for child_col in 0..factor {
                        let slot = child_row * factor + child_col;
                        output[parent * children_per_parent + slot] = global_index(
                            face,
                            row * factor + child_row,
                            col * factor + child_col,
                            fine_resolution,
                        )
                            as i32;
                    }
                }
            }
        }
    }
}

fn fill_d4_halo<T: Copy>(values: &[T], resolution: usize, halo: &mut [T], corner: T) {
    let halo_resolution = resolution + 2;
    halo.fill(corner);
    for face_index in 0..FACE_COUNT {
        let face = Face::from_index(face_index);
        let halo_face_offset = face_index * halo_resolution * halo_resolution;
        for row in 0..resolution {
            for col in 0..resolution {
                let source = global_index(face, row, col, resolution);
                let target = halo_face_offset + (row + 1) * halo_resolution + col + 1;
                halo[target] = values[source];
            }
        }
        for offset in 0..resolution {
            let north = neighbor_index(face, 0, offset, -1, 0, resolution);
            let south = neighbor_index(face, resolution - 1, offset, 1, 0, resolution);
            let west = neighbor_index(face, offset, 0, 0, -1, resolution);
            let east = neighbor_index(face, offset, resolution - 1, 0, 1, resolution);
            halo[halo_face_offset + offset + 1] = values[north];
            halo[halo_face_offset + (halo_resolution - 1) * halo_resolution + offset + 1] =
                values[south];
            halo[halo_face_offset + (offset + 1) * halo_resolution] = values[west];
            halo[halo_face_offset + (offset + 1) * halo_resolution + halo_resolution - 1] =
                values[east];
        }
    }
}

#[no_mangle]
/// Populate cubed-sphere coordinate, area, and D4-neighbor buffers.
///
/// # Safety
///
/// Every output pointer must reference a writable, non-overlapping buffer with
/// the length implied by `face_resolution`: `6*n*n*3` for `xyz`, `6*n*n` for
/// longitude, latitude, and area, and `6*n*n*4` for neighbors.
pub unsafe extern "C" fn cubed_sphere_generate(
    face_resolution: i32,
    xyz_ptr: *mut f32,
    longitude_ptr: *mut f64,
    latitude_ptr: *mut f64,
    area_ptr: *mut f64,
    neighbors_ptr: *mut i32,
) -> i32 {
    if face_resolution <= 0
        || xyz_ptr.is_null()
        || longitude_ptr.is_null()
        || latitude_ptr.is_null()
        || area_ptr.is_null()
        || neighbors_ptr.is_null()
    {
        return 1;
    }
    let resolution = face_resolution as usize;
    let Some(cells) = checked_cell_count(resolution) else {
        return 2;
    };
    let Some(xyz_len) = cells.checked_mul(3).and_then(checked_slice_len::<f32>) else {
        return 2;
    };
    let Some(neighbor_len) = cells
        .checked_mul(D4_NEIGHBORS)
        .and_then(checked_slice_len::<i32>)
    else {
        return 2;
    };
    if checked_slice_len::<f64>(cells).is_none() {
        return 2;
    }
    let xyz = std::slice::from_raw_parts_mut(xyz_ptr, xyz_len);
    let longitude = std::slice::from_raw_parts_mut(longitude_ptr, cells);
    let latitude = std::slice::from_raw_parts_mut(latitude_ptr, cells);
    let areas = std::slice::from_raw_parts_mut(area_ptr, cells);
    let neighbors = std::slice::from_raw_parts_mut(neighbors_ptr, neighbor_len);
    generate(resolution, xyz, longitude, latitude, areas, neighbors);
    0
}

#[no_mangle]
/// Map every fine cell to a same-face parent global ID.
///
/// # Safety
///
/// `parent_ptr` must reference `6*fine_resolution*fine_resolution` writable
/// `i32` values and must not alias another active buffer.
pub unsafe extern "C" fn cubed_sphere_parent_map(
    fine_resolution: i32,
    factor: i32,
    parent_ptr: *mut i32,
) -> i32 {
    if parent_ptr.is_null() {
        return 1;
    }
    let Some((fine, _coarse)) = hierarchy_resolutions(fine_resolution, factor) else {
        return 2;
    };
    let cells = checked_cell_count(fine).expect("validated fine resolution");
    if checked_slice_len::<i32>(cells).is_none() {
        return 2;
    }
    let output = std::slice::from_raw_parts_mut(parent_ptr, cells);
    build_parent_map(fine, factor as usize, output);
    0
}

#[no_mangle]
/// Map every coarse cell to its row-major same-face child global IDs.
///
/// # Safety
///
/// `children_ptr` must reference `6*coarse_resolution*coarse_resolution*factor^2`
/// writable `i32` values and must not alias another active buffer.
pub unsafe extern "C" fn cubed_sphere_children_map(
    coarse_resolution: i32,
    factor: i32,
    children_ptr: *mut i32,
) -> i32 {
    if coarse_resolution <= 0 || factor <= 1 || children_ptr.is_null() {
        return 1;
    }
    let Some(fine_resolution) = coarse_resolution.checked_mul(factor) else {
        return 2;
    };
    let Some((fine, coarse)) = hierarchy_resolutions(fine_resolution, factor) else {
        return 2;
    };
    let output_len = checked_cell_count(fine).expect("validated fine resolution");
    if checked_slice_len::<i32>(output_len).is_none() {
        return 2;
    }
    let output = std::slice::from_raw_parts_mut(children_ptr, output_len);
    build_children_map(coarse, factor as usize, output);
    0
}

#[no_mangle]
/// Restrict an extensive field by summing same-parent children.
///
/// # Safety
///
/// `fine_values_ptr` must reference one `f64` per fine cell and `coarse_values_ptr`
/// one writable `f64` per coarse cell. The buffers must not overlap.
pub unsafe extern "C" fn cubed_sphere_restrict_extensive_f64(
    fine_resolution: i32,
    factor: i32,
    fine_values_ptr: *const f64,
    coarse_values_ptr: *mut f64,
) -> i32 {
    if fine_values_ptr.is_null() || coarse_values_ptr.is_null() {
        return 1;
    }
    let Some((fine, coarse)) = hierarchy_resolutions(fine_resolution, factor) else {
        return 2;
    };
    let fine_cells = checked_cell_count(fine).expect("validated fine resolution");
    let coarse_cells = checked_cell_count(coarse).expect("validated coarse resolution");
    if checked_slice_len::<f64>(fine_cells).is_none()
        || checked_slice_len::<f64>(coarse_cells).is_none()
    {
        return 2;
    }
    let fine_values = std::slice::from_raw_parts(fine_values_ptr, fine_cells);
    let coarse_values = std::slice::from_raw_parts_mut(coarse_values_ptr, coarse_cells);
    coarse_values.fill(0.0);
    for face_index in 0..FACE_COUNT {
        let face = Face::from_index(face_index);
        for row in 0..fine {
            for col in 0..fine {
                let fine_index = global_index(face, row, col, fine);
                let value = fine_values[fine_index];
                if !value.is_finite() {
                    return 3;
                }
                let coarse_index =
                    global_index(face, row / factor as usize, col / factor as usize, coarse);
                coarse_values[coarse_index] += value;
                if !coarse_values[coarse_index].is_finite() {
                    return 3;
                }
            }
        }
    }
    0
}

#[no_mangle]
/// Restrict an intensive field by spherical-area-weighted averaging.
///
/// # Safety
///
/// Fine value and area pointers must each reference one `f64` per fine cell;
/// the coarse output must reference one writable `f64` per coarse cell. Buffers
/// must not overlap.
pub unsafe extern "C" fn cubed_sphere_restrict_intensive_f64(
    fine_resolution: i32,
    factor: i32,
    fine_values_ptr: *const f64,
    fine_areas_ptr: *const f64,
    coarse_values_ptr: *mut f64,
) -> i32 {
    if fine_values_ptr.is_null() || fine_areas_ptr.is_null() || coarse_values_ptr.is_null() {
        return 1;
    }
    let Some((fine, coarse)) = hierarchy_resolutions(fine_resolution, factor) else {
        return 2;
    };
    let fine_cells = checked_cell_count(fine).expect("validated fine resolution");
    let coarse_cells = checked_cell_count(coarse).expect("validated coarse resolution");
    if checked_slice_len::<f64>(fine_cells).is_none()
        || checked_slice_len::<f64>(coarse_cells).is_none()
    {
        return 2;
    }
    let fine_values = std::slice::from_raw_parts(fine_values_ptr, fine_cells);
    let fine_areas = std::slice::from_raw_parts(fine_areas_ptr, fine_cells);
    let coarse_values = std::slice::from_raw_parts_mut(coarse_values_ptr, coarse_cells);
    coarse_values.fill(0.0);
    let mut coarse_areas = vec![0.0f64; coarse_cells];
    for face_index in 0..FACE_COUNT {
        let face = Face::from_index(face_index);
        for row in 0..fine {
            for col in 0..fine {
                let fine_index = global_index(face, row, col, fine);
                let area = fine_areas[fine_index];
                if !area.is_finite() || area <= 0.0 || !fine_values[fine_index].is_finite() {
                    return 3;
                }
                let coarse_index =
                    global_index(face, row / factor as usize, col / factor as usize, coarse);
                let previous_area = coarse_areas[coarse_index];
                let combined_area = previous_area + area;
                if !combined_area.is_finite() || combined_area <= previous_area {
                    return 3;
                }
                if previous_area == 0.0 {
                    coarse_values[coarse_index] = fine_values[fine_index];
                } else if coarse_values[coarse_index] != fine_values[fine_index] {
                    let previous_weight = previous_area / combined_area;
                    let new_weight = area / combined_area;
                    let previous_value = coarse_values[coarse_index];
                    let new_value = fine_values[fine_index];
                    let combined_value =
                        if previous_value.is_sign_positive() == new_value.is_sign_positive() {
                            previous_value + (new_value - previous_value) * new_weight
                        } else {
                            previous_value * previous_weight + new_value * new_weight
                        };
                    if !combined_value.is_finite() {
                        return 3;
                    }
                    coarse_values[coarse_index] = combined_value;
                }
                coarse_areas[coarse_index] = combined_area;
            }
        }
    }
    0
}

#[no_mangle]
/// Prolongate a coarse prior by copying each parent value to all children.
///
/// # Safety
///
/// The coarse pointer must reference one `f64` per coarse cell and the fine
/// output one writable `f64` per fine cell. Buffers must not overlap.
pub unsafe extern "C" fn cubed_sphere_prolongate_constant_f64(
    coarse_resolution: i32,
    factor: i32,
    coarse_values_ptr: *const f64,
    fine_values_ptr: *mut f64,
) -> i32 {
    if coarse_resolution <= 0
        || factor <= 1
        || coarse_values_ptr.is_null()
        || fine_values_ptr.is_null()
    {
        return 1;
    }
    let Some(fine_resolution) = coarse_resolution.checked_mul(factor) else {
        return 2;
    };
    let Some((fine, coarse)) = hierarchy_resolutions(fine_resolution, factor) else {
        return 2;
    };
    let coarse_cells = checked_cell_count(coarse).expect("validated coarse resolution");
    let fine_cells = checked_cell_count(fine).expect("validated fine resolution");
    if checked_slice_len::<f64>(coarse_cells).is_none()
        || checked_slice_len::<f64>(fine_cells).is_none()
    {
        return 2;
    }
    let coarse_values = std::slice::from_raw_parts(coarse_values_ptr, coarse_cells);
    let fine_values = std::slice::from_raw_parts_mut(fine_values_ptr, fine_cells);
    for face_index in 0..FACE_COUNT {
        let face = Face::from_index(face_index);
        for row in 0..fine {
            for col in 0..fine {
                let fine_index = global_index(face, row, col, fine);
                let coarse_index =
                    global_index(face, row / factor as usize, col / factor as usize, coarse);
                fine_values[fine_index] = coarse_values[coarse_index];
            }
        }
    }
    0
}

#[no_mangle]
/// Fill a width-one D4 face halo; the four ambiguous corners remain NaN.
///
/// # Safety
///
/// `values_ptr` must reference one `f32` per canonical cell and `halo_ptr` must
/// reference `6*(resolution+2)^2` writable `f32` values. Buffers must not overlap.
pub unsafe extern "C" fn cubed_sphere_fill_d4_halo_f32(
    face_resolution: i32,
    values_ptr: *const f32,
    halo_ptr: *mut f32,
) -> i32 {
    if face_resolution <= 0 || values_ptr.is_null() || halo_ptr.is_null() {
        return 1;
    }
    let resolution = face_resolution as usize;
    let Some(cells) = checked_cell_count(resolution) else {
        return 2;
    };
    let Some(halo_resolution) = resolution.checked_add(2) else {
        return 2;
    };
    let Some(halo_cells) = halo_resolution
        .checked_mul(halo_resolution)
        .and_then(|face_cells| face_cells.checked_mul(FACE_COUNT))
    else {
        return 2;
    };
    if checked_slice_len::<f32>(cells).is_none() || checked_slice_len::<f32>(halo_cells).is_none() {
        return 2;
    }
    let values = std::slice::from_raw_parts(values_ptr, cells);
    let halo = std::slice::from_raw_parts_mut(halo_ptr, halo_cells);
    fill_d4_halo(values, resolution, halo, f32::NAN);
    0
}

#[no_mangle]
/// Fill a width-one D4 `f64` face halo; the four ambiguous corners remain NaN.
///
/// # Safety
///
/// `values_ptr` must reference one `f64` per canonical cell and `halo_ptr` must
/// reference `6*(resolution+2)^2` writable `f64` values. Buffers must not overlap.
pub unsafe extern "C" fn cubed_sphere_fill_d4_halo_f64(
    face_resolution: i32,
    values_ptr: *const f64,
    halo_ptr: *mut f64,
) -> i32 {
    if face_resolution <= 0 || values_ptr.is_null() || halo_ptr.is_null() {
        return 1;
    }
    let resolution = face_resolution as usize;
    let Some(cells) = checked_cell_count(resolution) else {
        return 2;
    };
    let Some(halo_resolution) = resolution.checked_add(2) else {
        return 2;
    };
    let Some(halo_cells) = halo_resolution
        .checked_mul(halo_resolution)
        .and_then(|face_cells| face_cells.checked_mul(FACE_COUNT))
    else {
        return 2;
    };
    if checked_slice_len::<f64>(cells).is_none() || checked_slice_len::<f64>(halo_cells).is_none() {
        return 2;
    }
    let values = std::slice::from_raw_parts(values_ptr, cells);
    let halo = std::slice::from_raw_parts_mut(halo_ptr, halo_cells);
    fill_d4_halo(values, resolution, halo, f64::NAN);
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generated(resolution: usize) -> (Vec<f32>, Vec<f64>, Vec<i32>) {
        let cells = FACE_COUNT * resolution * resolution;
        let mut xyz = vec![0.0f32; cells * 3];
        let mut longitude = vec![0.0f64; cells];
        let mut latitude = vec![0.0f64; cells];
        let mut areas = vec![0.0f64; cells];
        let mut neighbors = vec![-1i32; cells * D4_NEIGHBORS];
        generate(
            resolution,
            &mut xyz,
            &mut longitude,
            &mut latitude,
            &mut areas,
            &mut neighbors,
        );
        (xyz, areas, neighbors)
    }

    #[test]
    fn areas_sum_to_unit_sphere() {
        for resolution in [1usize, 2, 8, 32] {
            let (_, areas, _) = generated(resolution);
            let total: f64 = areas.iter().sum();
            assert!(
                (total - 4.0 * std::f64::consts::PI).abs() < 1e-10,
                "resolution {resolution}: {total}"
            );
            assert!(areas.iter().all(|area| *area > 0.0));
        }
    }

    #[test]
    fn cell_centers_are_unit_vectors() {
        let (xyz, _, _) = generated(16);
        for vector in xyz.chunks_exact(3) {
            let norm =
                (vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]).sqrt();
            assert!((norm - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn d4_neighbors_are_valid_unique_and_reciprocal() {
        for resolution in [1usize, 2, 8, 17] {
            let cells = FACE_COUNT * resolution * resolution;
            let (_, _, neighbors) = generated(resolution);
            for index in 0..cells {
                let adjacent = &neighbors[index * D4_NEIGHBORS..(index + 1) * D4_NEIGHBORS];
                assert!(adjacent
                    .iter()
                    .all(|neighbor| (0..cells as i32).contains(neighbor)));
                let mut unique = adjacent.to_vec();
                unique.sort_unstable();
                unique.dedup();
                assert_eq!(
                    unique.len(),
                    D4_NEIGHBORS,
                    "cell {index} at resolution {resolution}"
                );
                for &neighbor in adjacent {
                    let reverse = &neighbors
                        [neighbor as usize * D4_NEIGHBORS..(neighbor as usize + 1) * D4_NEIGHBORS];
                    assert!(
                        reverse.contains(&(index as i32)),
                        "{index} -> {neighbor} is not reciprocal"
                    );
                }
            }
        }
    }

    #[test]
    fn face_center_round_trips() {
        for face_index in 0..FACE_COUNT {
            let face = Face::from_index(face_index);
            let direction = face_vector(face, 0.0, 0.0);
            let (round_trip, u, v) = direction_to_face_uv(direction);
            assert_eq!(round_trip, face);
            assert!(u.abs() < 1e-12);
            assert!(v.abs() < 1e-12);
        }
    }

    #[test]
    fn ffi_rejects_resolution_beyond_global_index_capacity() {
        let mut xyz = 0.0f32;
        let mut longitude = 0.0f64;
        let mut latitude = 0.0f64;
        let mut area = 0.0f64;
        let mut neighbor = 0i32;
        let status = unsafe {
            cubed_sphere_generate(
                i32::MAX,
                &mut xyz,
                &mut longitude,
                &mut latitude,
                &mut area,
                &mut neighbor,
            )
        };
        assert_eq!(status, 2);
    }

    #[test]
    fn parent_and_children_maps_are_inverse_and_row_major() {
        let fine = 6usize;
        let coarse = 3usize;
        let factor = 2usize;
        let fine_cells = FACE_COUNT * fine * fine;
        let coarse_cells = FACE_COUNT * coarse * coarse;
        let mut parents = vec![-1i32; fine_cells];
        let mut children = vec![-1i32; fine_cells];
        build_parent_map(fine, factor, &mut parents);
        build_children_map(coarse, factor, &mut children);

        for parent in 0..coarse_cells {
            let child_ids = &children[parent * factor * factor..(parent + 1) * factor * factor];
            assert_eq!(child_ids.len(), factor * factor);
            for &child in child_ids {
                assert_eq!(parents[child as usize], parent as i32);
            }
        }

        let parent = global_index(Face::PositiveY, 1, 2, coarse);
        assert_eq!(
            &children[parent * 4..parent * 4 + 4],
            &[
                global_index(Face::PositiveY, 2, 4, fine) as i32,
                global_index(Face::PositiveY, 2, 5, fine) as i32,
                global_index(Face::PositiveY, 3, 4, fine) as i32,
                global_index(Face::PositiveY, 3, 5, fine) as i32,
            ]
        );
    }

    #[test]
    fn restriction_conserves_extensive_and_area_weighted_fields() {
        let fine = 8usize;
        let factor = 2usize;
        let coarse = fine / factor;
        let fine_cells = FACE_COUNT * fine * fine;
        let coarse_cells = FACE_COUNT * coarse * coarse;
        let (_, areas, _) = generated(fine);
        let values: Vec<f64> = (0..fine_cells).map(|index| index as f64 + 1.0).collect();

        let mut extensive = vec![0.0f64; coarse_cells];
        let status = unsafe {
            cubed_sphere_restrict_extensive_f64(
                fine as i32,
                factor as i32,
                values.as_ptr(),
                extensive.as_mut_ptr(),
            )
        };
        assert_eq!(status, 0);
        assert_eq!(values.iter().sum::<f64>(), extensive.iter().sum::<f64>());

        let mut intensive = vec![0.0f64; coarse_cells];
        let status = unsafe {
            cubed_sphere_restrict_intensive_f64(
                fine as i32,
                factor as i32,
                values.as_ptr(),
                areas.as_ptr(),
                intensive.as_mut_ptr(),
            )
        };
        assert_eq!(status, 0);
        let fine_integral: f64 = values
            .iter()
            .zip(areas.iter())
            .map(|(value, area)| value * area)
            .sum();
        let mut coarse_areas = vec![0.0f64; coarse_cells];
        let area_status = unsafe {
            cubed_sphere_restrict_extensive_f64(
                fine as i32,
                factor as i32,
                areas.as_ptr(),
                coarse_areas.as_mut_ptr(),
            )
        };
        assert_eq!(area_status, 0);
        let coarse_integral: f64 = intensive
            .iter()
            .zip(coarse_areas.iter())
            .map(|(value, area)| value * area)
            .sum();
        assert!((fine_integral - coarse_integral).abs() < 1e-10);
        assert!((coarse_areas.iter().sum::<f64>() - 4.0 * std::f64::consts::PI).abs() < 1e-12);
    }

    #[test]
    fn prolongation_copies_each_parent_to_all_children() {
        let coarse = 3usize;
        let factor = 2usize;
        let fine = coarse * factor;
        let coarse_cells = FACE_COUNT * coarse * coarse;
        let fine_cells = FACE_COUNT * fine * fine;
        let coarse_values: Vec<f64> = (0..coarse_cells).map(|index| index as f64).collect();
        let mut fine_values = vec![f64::NAN; fine_cells];
        let status = unsafe {
            cubed_sphere_prolongate_constant_f64(
                coarse as i32,
                factor as i32,
                coarse_values.as_ptr(),
                fine_values.as_mut_ptr(),
            )
        };
        assert_eq!(status, 0);
        let mut parents = vec![-1i32; fine_cells];
        build_parent_map(fine, factor, &mut parents);
        for (child, &parent) in parents.iter().enumerate() {
            assert_eq!(fine_values[child], coarse_values[parent as usize]);
        }
    }

    #[test]
    fn d4_halo_uses_topology_neighbors_and_leaves_corners_undefined() {
        let resolution = 7usize;
        let cells = FACE_COUNT * resolution * resolution;
        let values: Vec<f32> = (0..cells).map(|index| index as f32).collect();
        let halo_resolution = resolution + 2;
        let mut halo = vec![0.0f32; FACE_COUNT * halo_resolution * halo_resolution];
        fill_d4_halo(&values, resolution, &mut halo, f32::NAN);

        for face_index in 0..FACE_COUNT {
            let face = Face::from_index(face_index);
            let offset = face_index * halo_resolution * halo_resolution;
            for row in 0..resolution {
                for col in 0..resolution {
                    assert_eq!(
                        halo[offset + (row + 1) * halo_resolution + col + 1],
                        values[global_index(face, row, col, resolution)]
                    );
                }
            }
            for edge_offset in 0..resolution {
                assert_eq!(
                    halo[offset + edge_offset + 1],
                    values[neighbor_index(face, 0, edge_offset, -1, 0, resolution)]
                );
                assert_eq!(
                    halo[offset + (halo_resolution - 1) * halo_resolution + edge_offset + 1],
                    values[neighbor_index(face, resolution - 1, edge_offset, 1, 0, resolution)]
                );
                assert_eq!(
                    halo[offset + (edge_offset + 1) * halo_resolution],
                    values[neighbor_index(face, edge_offset, 0, 0, -1, resolution)]
                );
                assert_eq!(
                    halo[offset + (edge_offset + 1) * halo_resolution + halo_resolution - 1],
                    values[neighbor_index(face, edge_offset, resolution - 1, 0, 1, resolution)]
                );
            }
            for corner in [
                offset,
                offset + halo_resolution - 1,
                offset + (halo_resolution - 1) * halo_resolution,
                offset + halo_resolution * halo_resolution - 1,
            ] {
                assert!(halo[corner].is_nan());
            }
        }
    }

    #[test]
    fn hierarchy_ffi_rejects_invalid_factors_and_non_finite_fields() {
        let values = vec![1.0f64; FACE_COUNT * 4 * 4];
        let mut coarse = vec![0.0f64; FACE_COUNT * 2 * 2];
        let invalid_factor = unsafe {
            cubed_sphere_restrict_extensive_f64(4, 3, values.as_ptr(), coarse.as_mut_ptr())
        };
        assert_eq!(invalid_factor, 2);

        let mut invalid_values = values;
        invalid_values[3] = f64::NAN;
        let non_finite = unsafe {
            cubed_sphere_restrict_extensive_f64(4, 2, invalid_values.as_ptr(), coarse.as_mut_ptr())
        };
        assert_eq!(non_finite, 3);

        let maximum_values = vec![f64::MAX; FACE_COUNT * 4 * 4];
        let overflow = unsafe {
            cubed_sphere_restrict_extensive_f64(4, 2, maximum_values.as_ptr(), coarse.as_mut_ptr())
        };
        assert_eq!(overflow, 3);
    }

    #[test]
    fn intensive_restriction_handles_extreme_finite_constant_values() {
        let values = vec![f64::MAX; FACE_COUNT * 4 * 4];
        let areas = vec![f64::MAX / 4.0; FACE_COUNT * 4 * 4];
        let mut coarse = vec![0.0f64; FACE_COUNT * 2 * 2];
        let status = unsafe {
            cubed_sphere_restrict_intensive_f64(
                4,
                2,
                values.as_ptr(),
                areas.as_ptr(),
                coarse.as_mut_ptr(),
            )
        };
        assert_eq!(status, 0);
        assert!(coarse.iter().all(|value| *value == f64::MAX));
    }
}
