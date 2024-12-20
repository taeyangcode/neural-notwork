#![allow(dead_code)]

use crate::matrix::Matrix;

pub struct NeuralNetworkBuilder<WeightF, BiasF, ActivationF, LossF>
where
    WeightF: Fn() -> f64,
    BiasF: Fn() -> f64,
    ActivationF: Fn(f64) -> f64,
    LossF: Fn(f64, f64) -> f64,
{
    input: Option<Matrix>,
    hidden: Option<Vec<Matrix>>,
    output: Option<Matrix>,
    weights: Option<Vec<Matrix>>,
    biases: Option<Vec<Matrix>>,
    learning_rate: Option<f64>,
    activation_fn: Option<ActivationF>,
    loss_fn: Option<LossF>,

    weight_fn: Option<WeightF>,
    bias_fn: Option<BiasF>,
}

impl<WeightF, BiasF, ActivationF, LossF> NeuralNetworkBuilder<WeightF, BiasF, ActivationF, LossF>
where
    WeightF: Fn() -> f64,
    BiasF: Fn() -> f64,
    ActivationF: Fn(f64) -> f64,
    LossF: Fn(f64, f64) -> f64,
{
    pub fn default() -> NeuralNetworkBuilder<WeightF, BiasF, ActivationF, LossF> {
        Self {
            input: None,
            hidden: None,
            output: None,
            weights: None,
            biases: None,
            learning_rate: None,
            activation_fn: None,
            loss_fn: None,

            weight_fn: None,
            bias_fn: None,
        }
    }

    pub fn with_input(
        mut self,
        input: Matrix,
    ) -> NeuralNetworkBuilder<WeightF, BiasF, ActivationF, LossF> {
        self.input = Some(input);
        self
    }

    pub fn with_hidden(
        mut self,
        hidden: Vec<Matrix>,
    ) -> NeuralNetworkBuilder<WeightF, BiasF, ActivationF, LossF> {
        self.hidden = Some(hidden);
        self
    }

    pub fn with_output(
        mut self,
        output: Matrix,
    ) -> NeuralNetworkBuilder<WeightF, BiasF, ActivationF, LossF> {
        self.output = Some(output);
        self
    }

    pub fn with_weights(
        mut self,
        weights: Vec<Matrix>,
    ) -> NeuralNetworkBuilder<WeightF, BiasF, ActivationF, LossF> {
        self.weights = Some(weights);
        self
    }

    pub fn with_random_weights(
        mut self,
        weight_fn: WeightF,
    ) -> NeuralNetworkBuilder<WeightF, BiasF, ActivationF, LossF> {
        self.weight_fn = Some(weight_fn);
        self
    }

    pub fn with_biases(
        mut self,
        biases: Vec<Matrix>,
    ) -> NeuralNetworkBuilder<WeightF, BiasF, ActivationF, LossF> {
        self.biases = Some(biases);
        self
    }

    pub fn with_random_biases(
        mut self,
        bias_fn: BiasF,
    ) -> NeuralNetworkBuilder<WeightF, BiasF, ActivationF, LossF> {
        self.bias_fn = Some(bias_fn);
        self
    }

    pub fn with_learning_rate(
        mut self,
        learning_rate: f64,
    ) -> NeuralNetworkBuilder<WeightF, BiasF, ActivationF, LossF> {
        self.learning_rate = Some(learning_rate);
        self
    }

    pub fn with_activation_fn(
        mut self,
        activation_fn: ActivationF,
    ) -> NeuralNetworkBuilder<WeightF, BiasF, ActivationF, LossF> {
        self.activation_fn = Some(activation_fn);
        self
    }

    pub fn with_loss_fn(
        mut self,
        loss_fn: LossF,
    ) -> NeuralNetworkBuilder<WeightF, BiasF, ActivationF, LossF> {
        self.loss_fn = Some(loss_fn);
        self
    }

    pub fn build(mut self) -> NeuralNetwork<ActivationF, LossF> {
        let random_weights = self.random_weights();
        let random_biases = self.random_biases();

        NeuralNetwork {
            input: self.input.unwrap(),
            hidden: self.hidden.unwrap(),
            output: self.output.unwrap(),
            weights: self.weights.unwrap_or(random_weights),
            biases: self.biases.unwrap_or(random_biases),
            learning_rate: self.learning_rate.unwrap(),
            activation_fn: self.activation_fn.unwrap(),
            loss_fn: self.loss_fn.unwrap(),
        }
    }

    fn random_weights(&mut self) -> Vec<Matrix> {
        let input = self.input.as_ref().unwrap();
        let output = self.output.as_ref().unwrap();
        let weight_fn = self.weight_fn.as_ref().unwrap();

        match self.hidden.as_ref() {
            Some(hidden) if !hidden.is_empty() => {
                let hidden_last = hidden.last().unwrap();

                let input_weight =
                    std::iter::once(Matrix::random(input.columns, hidden[0].columns, weight_fn));
                let hidden_weights = hidden
                    .windows(2)
                    .map(|window| Matrix::random(window[0].columns, window[1].columns, weight_fn));
                let output_weight = std::iter::once(Matrix::random(
                    hidden_last.columns,
                    output.columns,
                    weight_fn,
                ));

                input_weight
                    .chain(hidden_weights)
                    .chain(output_weight)
                    .collect()
            }
            _ => vec![Matrix::random(input.columns, output.columns, weight_fn)],
        }
    }

    fn random_biases(&mut self) -> Vec<Matrix> {
        let output = self.output.as_ref().unwrap();
        let bias_fn = self.bias_fn.as_ref().unwrap();

        match self.hidden.as_ref() {
            Some(hidden) if !hidden.is_empty() => {
                let hidden_biases = hidden.into_iter().map(|hidden| {
                    Matrix::random(hidden.rows, 1, bias_fn)
                        * Matrix::random(1, hidden.columns, || 1.0_f64)
                });
                let output_bias = std::iter::once(
                    Matrix::random(output.rows, 1, bias_fn)
                        * Matrix::random(1, output.columns, || 1.0_f64),
                );

                hidden_biases.chain(output_bias).collect()
            }
            _ => vec![
                Matrix::random(output.rows, 1, bias_fn)
                    * Matrix::random(1, output.columns, || 1.0_f64),
            ],
        }
    }
}

#[derive(Debug)]
pub struct NeuralNetwork<ActivationF, LossF>
where
    ActivationF: Fn(f64) -> f64,
    LossF: Fn(f64, f64) -> f64,
{
    input: Matrix,
    hidden: Vec<Matrix>,
    output: Matrix,
    weights: Vec<Matrix>,
    biases: Vec<Matrix>,
    learning_rate: f64,
    activation_fn: ActivationF,
    loss_fn: LossF,
}

impl<ActivationF, LossF> NeuralNetwork<ActivationF, LossF>
where
    ActivationF: Fn(f64) -> f64,
    LossF: Fn(f64, f64) -> f64,
{
    pub fn propogate(&mut self) {
        let layers = std::iter::once(&self.input)
            .chain(&self.hidden)
            .chain(std::iter::once(&self.output));

        let adjusted_layers = layers.zip(self.weights.iter()).zip(self.biases.iter()).map(
            |((layer, weights), biases)| (layer * weights + biases).apply(&self.activation_fn),
        );

        dbg!(adjusted_layers.collect::<Vec<_>>());
    }
}

#[cfg(test)]
mod test {
    use super::NeuralNetworkBuilder;
    use crate::matrix::Matrix;

    #[test]
    fn test() {
        let weight_fn = || rand::random::<f64>();
        let bias_fn = || 1.0_f64;

        let input = Matrix {
            rows: 5,
            columns: 1,
            entries: vec![1_f64, 1_f64, 1_f64, 1_f64, 1_f64],
        };

        let hidden = vec![Matrix::random(5, 1, || rand::random::<f64>())];

        let output = Matrix {
            rows: 5,
            columns: 1,
            entries: vec![2_f64, 3_f64, 4_f64, 5_f64, 6_f64],
        };

        let learning_rate = 0.01_f64;

        let sigmoid = |x: f64| 1.0_f64 / (1.0_f64 + core::f64::consts::E.powf(x));

        let mut network = NeuralNetworkBuilder::default()
            .with_input(input)
            .with_hidden(hidden)
            .with_output(output)
            .with_random_weights(weight_fn)
            .with_random_biases(bias_fn)
            .with_learning_rate(learning_rate)
            .with_activation_fn(sigmoid)
            .with_loss_fn(|a, b| 1.0_f64)
            .build();

        network.propogate();
    }
}
