use crate::matrix::Matrix;

#[derive(Debug)]
pub struct NeuralNetwork {
    input: Matrix,
    hidden: Vec<Matrix>,
    output: Matrix,
    weights: Vec<Matrix>,
    biases: Vec<Matrix>,
}

// impl NeuralNetwork {
//     pub fn empty<T>(
//         input_dimension: (usize, usize),
//         hidden_dimensions: &[(usize, usize)],
//         output_dimension: (usize, usize),
//     ) -> Self
//     {
//         let input = Matrix::empty(input_dimension.0, input_dimension.1);

//         let hidden = hidden_dimensions
//             .into_iter()
//             .map(|(rows, columns)| Matrix::empty(*rows, *columns))
//             .collect::<Vec<Matrix>>();

//         let output = Matrix::empty(output_dimension.0, output_dimension.1);
//     }
// }
