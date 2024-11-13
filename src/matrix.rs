#![allow(dead_code)]

#[derive(Clone, Debug)]
pub struct Matrix {
    pub rows: usize,
    pub columns: usize,
    pub entries: Vec<f64>,
}

impl Matrix {
    pub fn empty(rows: usize, columns: usize) -> Self {
        Self {
            rows,
            columns,
            entries: vec![f64::default(); rows * columns],
        }
    }

    pub fn random<F>(rows: usize, columns: usize, entry_generator_fn: F) -> Self
    where
        F: Fn() -> f64,
    {
        Self {
            rows,
            columns,
            entries: std::iter::repeat_with(entry_generator_fn)
                .take(rows * columns)
                .collect(),
        }
    }

    pub fn power(&mut self, power: f64) {
        self.entries
            .iter_mut()
            .for_each(|entry| *entry = f64::powf(*entry, power));
    }

    pub fn product(&self, other: &Self) -> Self {
        assert_eq!(
            self.columns,
            other.rows,
            "number of columns in the first matrix must equal the number of rows in the second matrix"
        );

        let mut product = vec![0_f64; self.rows * other.columns];

        for row in 0..self.rows {
            for column in 0..other.columns {
                let entry = (0..self.columns)
                    .map(|index| self[(row, index)] * other[(index, column)])
                    .sum();

                product[row * other.columns + column] = entry;
            }
        }

        Self {
            rows: self.rows,
            columns: other.columns,
            entries: product,
        }
    }

    pub fn multiply(&self, other: &Self) -> Self {
        assert!(
            self.rows.eq(&other.rows) && self.columns.eq(&other.columns),
            "matrix dimensions do not match"
        );

        Self {
            rows: self.rows,
            columns: self.columns,
            entries: self
                .entries
                .to_owned()
                .into_iter()
                .zip(other.entries.iter())
                .map(|(entry_1, entry_2)| entry_1 * *entry_2)
                .collect(),
        }
    }

    pub fn transpose(&self) -> Self {
        let mut transpose = Self::empty(self.columns, self.rows);

        for row in 0..self.rows {
            for column in 0..self.columns {
                transpose[(column, row)] = self[(row, column)];
            }
        }

        transpose
    }
}

impl core::ops::Index<(usize, usize)> for Matrix {
    type Output = f64;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.entries[index.0 * self.columns + index.1]
    }
}

impl core::ops::IndexMut<(usize, usize)> for Matrix {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.entries[index.0 * self.columns + index.1]
    }
}

impl core::cmp::PartialEq for Matrix {
    fn eq(&self, other: &Self) -> bool {
        self.rows == other.rows
            && self.columns == other.columns
            && self.entries.iter().eq(other.entries.iter())
    }
}

impl core::ops::Add for Matrix {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        assert!(
            self.rows.eq(&rhs.rows) && self.columns.eq(&rhs.columns),
            "matrix dimensions do not match"
        );

        Matrix {
            rows: self.rows,
            columns: self.columns,
            entries: self
                .entries
                .iter()
                .zip(rhs.entries)
                .map(|(left, right)| left + right)
                .collect(),
        }
    }
}

impl core::ops::AddAssign for Matrix {
    fn add_assign(&mut self, rhs: Self) {
        assert!(
            self.rows.eq(&rhs.rows) && self.columns.eq(&rhs.columns),
            "matrix dimensions do not match"
        );

        self.entries
            .iter_mut()
            .zip(rhs.entries)
            .for_each(|(left, right)| *left += right);
    }
}

impl core::ops::Sub for Matrix {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        assert!(
            self.rows.eq(&rhs.rows) && self.columns.eq(&rhs.columns),
            "matrix dimensions do not match"
        );

        Matrix {
            rows: self.rows,
            columns: self.columns,
            entries: self
                .entries
                .iter()
                .zip(rhs.entries)
                .map(|(left, right)| left - right)
                .collect(),
        }
    }
}

impl core::ops::SubAssign for Matrix {
    fn sub_assign(&mut self, rhs: Self) {
        assert!(
            self.rows.eq(&rhs.rows) && self.columns.eq(&rhs.columns),
            "matrix dimensions do not match"
        );

        self.entries
            .iter_mut()
            .zip(rhs.entries)
            .for_each(|(left, right)| *left -= right);
    }
}

impl core::ops::Mul for Matrix {
    type Output = Matrix;

    fn mul(self, rhs: Self) -> Self::Output {
        self.product(&rhs)
    }
}

impl<N: num_traits::Num> core::ops::Mul<N> for Matrix
where
    N: Copy + Into<f64>,
{
    type Output = Matrix;

    fn mul(self, rhs: N) -> Self::Output {
        Matrix {
            rows: self.rows,
            columns: self.columns,
            entries: self
                .entries
                .to_owned()
                .into_iter()
                .map(|entry| entry * rhs.into())
                .collect(),
        }
    }
}

impl<N: num_traits::Num> core::ops::MulAssign<N> for Matrix
where
    N: Copy + Into<f64>,
{
    fn mul_assign(&mut self, rhs: N) {
        self.entries
            .iter_mut()
            .for_each(|entry| *entry *= rhs.into());
    }
}

#[cfg(test)]
mod product_tests {
    use super::Matrix;

    #[test]
    fn test_product_square_matrices() {
        let matrix_a = Matrix {
            rows: 2,
            columns: 2,
            entries: vec![1.0, 2.0, 3.0, 4.0], // 2x2 matrix
        };
        let matrix_b = Matrix {
            rows: 2,
            columns: 2,
            entries: vec![2.0, 0.0, 1.0, 3.0], // 2x2 matrix
        };

        let product = matrix_a.product(&matrix_b);

        assert_eq!(product.rows, 2);
        assert_eq!(product.columns, 2);
        assert_eq!(product.entries, vec![4.0, 6.0, 10.0, 12.0]);
    }

    #[test]
    fn test_product_rectangular_matrices() {
        let matrix_a = Matrix {
            rows: 2,
            columns: 3,
            entries: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], // 2x3 matrix
        };
        let matrix_b = Matrix {
            rows: 3,
            columns: 2,
            entries: vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], // 3x2 matrix
        };

        let product = matrix_a.product(&matrix_b);

        assert_eq!(product.rows, 2);
        assert_eq!(product.columns, 2);
        assert_eq!(product.entries, vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn test_product_with_identity_matrix() {
        let matrix = Matrix {
            rows: 3,
            columns: 3,
            entries: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        };
        let identity = Matrix {
            rows: 3,
            columns: 3,
            entries: vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        };

        let product = matrix.product(&identity);

        assert_eq!(product.rows, 3);
        assert_eq!(product.columns, 3);
        assert_eq!(
            product.entries,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        );
    }

    #[test]
    fn test_product_single_row_and_column() {
        let matrix_a = Matrix {
            rows: 1,
            columns: 3,
            entries: vec![1.0, 2.0, 3.0], // 1x3 matrix
        };
        let matrix_b = Matrix {
            rows: 3,
            columns: 1,
            entries: vec![4.0, 5.0, 6.0], // 3x1 matrix
        };

        let product = matrix_a.product(&matrix_b);

        assert_eq!(product.rows, 1);
        assert_eq!(product.columns, 1);
        assert_eq!(product.entries, vec![32.0]);
    }

    #[test]
    fn test_product_with_zero_matrix() {
        let matrix_a = Matrix {
            rows: 2,
            columns: 3,
            entries: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], // 2x3 matrix
        };
        let zero_matrix = Matrix {
            rows: 3,
            columns: 2,
            entries: vec![0.0; 6], // 3x2 zero matrix
        };

        let product = matrix_a.product(&zero_matrix);

        assert_eq!(product.rows, 2);
        assert_eq!(product.columns, 2);
        assert_eq!(product.entries, vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    #[should_panic(
        expected = "number of columns in the first matrix must equal the number of rows in the second matrix"
    )]
    fn test_product_incompatible_matrices() {
        let matrix_a = Matrix {
            rows: 2,
            columns: 3,
            entries: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], // 2x3 matrix
        };
        let matrix_b = Matrix {
            rows: 2,
            columns: 2,
            entries: vec![7.0, 8.0, 9.0, 10.0], // 2x2 matrix
        };

        // This should panic because matrix_a.columns != matrix_b.rows
        let _ = matrix_a.product(&matrix_b);
    }
}

#[cfg(test)]
mod transpose_tests {
    use super::Matrix;

    #[test]
    fn test_transpose_square_matrix() {
        let matrix = Matrix {
            rows: 2,
            columns: 2,
            entries: vec![1.0, 2.0, 3.0, 4.0], // 2x2 matrix
        };

        let transposed = matrix.transpose();

        assert_eq!(transposed.rows, 2);
        assert_eq!(transposed.columns, 2);
        assert_eq!(transposed.entries, vec![1.0, 3.0, 2.0, 4.0]);
    }

    #[test]
    fn test_transpose_rectangular_matrix() {
        let matrix = Matrix {
            rows: 2,
            columns: 3,
            entries: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], // 2x3 matrix
        };

        let transposed = matrix.transpose();

        assert_eq!(transposed.rows, 3);
        assert_eq!(transposed.columns, 2);
        assert_eq!(transposed.entries, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_transpose_single_row_matrix() {
        let matrix = Matrix {
            rows: 1,
            columns: 4,
            entries: vec![1.0, 2.0, 3.0, 4.0], // 1x4 matrix
        };

        let transposed = matrix.transpose();

        assert_eq!(transposed.rows, 4);
        assert_eq!(transposed.columns, 1);
        assert_eq!(transposed.entries, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_transpose_single_column_matrix() {
        let matrix = Matrix {
            rows: 4,
            columns: 1,
            entries: vec![1.0, 2.0, 3.0, 4.0], // 4x1 matrix
        };

        let transposed = matrix.transpose();

        assert_eq!(transposed.rows, 1);
        assert_eq!(transposed.columns, 4);
        assert_eq!(transposed.entries, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_transpose_empty_matrix() {
        let matrix = Matrix {
            rows: 0,
            columns: 0,
            entries: vec![], // empty matrix
        };

        let transposed = matrix.transpose();

        assert_eq!(transposed.rows, 0);
        assert_eq!(transposed.columns, 0);
        assert_eq!(transposed.entries, vec![]);
    }
}

#[cfg(test)]
mod multiply_tests {
    use super::Matrix;

    #[test]
    fn test_elementwise_multiplication_basic() {
        let matrix_a = Matrix {
            rows: 2,
            columns: 2,
            entries: vec![1.0, 2.0, 3.0, 4.0],
        };
        let matrix_b = Matrix {
            rows: 2,
            columns: 2,
            entries: vec![5.0, 6.0, 7.0, 8.0],
        };

        let result = matrix_a.multiply(&matrix_b);

        let expected = Matrix {
            rows: 2,
            columns: 2,
            entries: vec![5.0, 12.0, 21.0, 32.0],
        };

        assert_eq!(result.entries, expected.entries);
    }

    #[test]
    fn test_elementwise_multiplication_single_element() {
        let matrix_a = Matrix {
            rows: 1,
            columns: 1,
            entries: vec![3.0],
        };
        let matrix_b = Matrix {
            rows: 1,
            columns: 1,
            entries: vec![4.0],
        };

        let result = matrix_a.multiply(&matrix_b);

        let expected = Matrix {
            rows: 1,
            columns: 1,
            entries: vec![12.0],
        };

        assert_eq!(result.entries, expected.entries);
    }

    #[test]
    #[should_panic(expected = "matrix dimensions do not match")]
    fn test_elementwise_multiplication_mismatched_dimensions() {
        let matrix_a = Matrix {
            rows: 2,
            columns: 3,
            entries: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        };
        let matrix_b = Matrix {
            rows: 2,
            columns: 2,
            entries: vec![7.0, 8.0, 9.0, 10.0],
        };

        // This should panic due to dimension mismatch
        let _result = matrix_a.multiply(&matrix_b);
    }

    #[test]
    fn test_elementwise_multiplication_empty_matrix() {
        let matrix_a = Matrix {
            rows: 0,
            columns: 0,
            entries: vec![],
        };
        let matrix_b = Matrix {
            rows: 0,
            columns: 0,
            entries: vec![],
        };

        let result = matrix_a.multiply(&matrix_b);

        let expected = Matrix {
            rows: 0,
            columns: 0,
            entries: vec![],
        };

        assert_eq!(result.entries, expected.entries);
    }
}
