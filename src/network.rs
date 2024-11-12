use crate::matrix::Matrix;

#[derive(Debug)]
pub struct NeuralNetwork {
    input: Matrix,
    hidden: Vec<Matrix>,
    output: Matrix,
    weights: Vec<Matrix>,
    biases: Vec<Matrix>,
}

impl NeuralNetwork {
    pub fn new<F>(
        input_dimensions: (usize, usize),
        output_dimensions: (usize, usize),
        hidden_dimensions: &[(usize, usize)],
        weight_fn: F,
        bias_fn: F,
    ) -> ()
    where
        F: Fn() -> f64,
    {
        let input = Matrix::empty(input_dimensions.0, input_dimensions.1);

        let hidden = hidden_dimensions
            .into_iter()
            .map(|(rows, columns)| Matrix::empty(*rows, *columns))
            .collect::<Vec<Matrix>>();

        let output = Matrix::empty(output_dimensions.0, output_dimensions.1);

        let (input_weight, input_bias) = (
            Matrix::random(hidden_dimensions[0].0, output_dimensions.1, &weight_fn),
            Matrix::random(hidden_dimensions[0].0, 1, &bias_fn),
        );

        let (hidden_weights, input_weights) = (
            hidden_dimensions
                .windows(2)
                .skip(1)
                .map(|window| Matrix::random(window[1].0, window[0].0, &weight_fn)),
            hidden_dimensions
                .windows(2)
                .skip(1)
                .map(|window| Matrix::random(window[1].0, 1, &bias_fn)),
        );
    }
}
