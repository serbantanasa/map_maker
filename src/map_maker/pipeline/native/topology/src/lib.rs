use std::f64::consts::FRAC_PI_2;

const FACE_COUNT: usize = 6;
const D4_NEIGHBORS: usize = 4;

#[no_mangle]
pub extern "C" fn topology_native_abi_version() -> u32 {
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
    let Some(cells) = resolution
        .checked_mul(resolution)
        .and_then(|face_cells| face_cells.checked_mul(FACE_COUNT))
    else {
        return 2;
    };
    if cells > i32::MAX as usize {
        return 2;
    }
    let xyz = std::slice::from_raw_parts_mut(xyz_ptr, cells * 3);
    let longitude = std::slice::from_raw_parts_mut(longitude_ptr, cells);
    let latitude = std::slice::from_raw_parts_mut(latitude_ptr, cells);
    let areas = std::slice::from_raw_parts_mut(area_ptr, cells);
    let neighbors = std::slice::from_raw_parts_mut(neighbors_ptr, cells * D4_NEIGHBORS);
    generate(resolution, xyz, longitude, latitude, areas, neighbors);
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
}
