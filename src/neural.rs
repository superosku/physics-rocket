
use rand::Rng;

#[derive(Debug, Clone)]
pub struct neural_net {
    weights: Vec<Vec<f32>>,
    layer_values: Vec<Vec<f32>>,
    layer_sizes: Vec<u32>,
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

impl neural_net {
    pub fn new(layer_sizes: Vec<u32>) -> neural_net {
        let mut rng = rand::thread_rng();

        let mut weights: Vec<Vec<f32>> = Vec::new();
        let mut layer_values: Vec<Vec<f32>> = Vec::new();

        for i in 0..layer_sizes.len() {
            let mut layer = Vec::new();
            for _j in 0..layer_sizes[i] {
                layer.push(0.0);
            }
            layer_values.push(layer);
        }
        for i in 0..layer_sizes.len() - 1 {
            let mut layer: Vec<f32> = Vec::new();

            // Add +1 to layer_sizes[i] to account for bias
            let weight_layer_size = (layer_sizes[i] + 1) * layer_sizes[i + 1];

            for _j in 0..weight_layer_size {
                layer.push(rng.gen::<f32>() * 2.0 - 1.0);
            }
            weights.push(layer);
        }
        neural_net {
            weights: weights,
            layer_values: layer_values,
            layer_sizes: layer_sizes,
        }
    }

    pub fn mix_randomly_with_other(&self, other: &neural_net) -> neural_net {
        // Returns new neural_net that is a mix of self and other
        let mut rng = rand::thread_rng();

        let mut new_net = neural_net::new(self.layer_sizes.clone());

        for i in 0..self.weights.len() {
            for j in 0..self.weights[i].len() {
                if rng.gen::<f32>() < 0.5 {
                    new_net.weights[i][j] = self.weights[i][j];
                } else {
                    new_net.weights[i][j] = other.weights[i][j];
                }
            }
        }

        new_net
    }

    pub fn clone_mutated(&self, lr: f32) -> neural_net {
        // Returns new neural_net that has slightly mutated values
        let mut rng = rand::thread_rng();

        let mut new_net = neural_net::new(self.layer_sizes.clone());

        for i in 0..self.weights.len() {
            for j in 0..self.weights[i].len() {
                // if rng.gen::<f32>() < 0.2 {
                    new_net.weights[i][j] = self.weights[i][j] + (rng.gen::<f32>() - 0.5) * lr;
                // }
            }
        }

        new_net
    }

    pub fn set_first_layer(&mut self, values: Vec<f32>) {
        for i in 0..values.len() {
            self.layer_values[0][i] = values[i];
        }
    }

    pub fn get_last_layer(&self) -> Vec<f32> {
        self.layer_values[self.layer_values.len() - 1].clone()
    }

    pub fn forward_propagate(&mut self) {
        for to_layer_index in 1..self.layer_sizes.len() {
            let from_layer_index = to_layer_index - 1;
            let weight_array = &self.weights[from_layer_index as usize];

            let from_size = self.layer_sizes[from_layer_index as usize];
            let to_size = self.layer_sizes[to_layer_index as usize];

            for to_cell_index in 0..to_size {
                let mut sum = 0.0;
                // Normal stuff
                for from_cell_index in 0..from_size {
                    let from_input_value = self.layer_values[from_layer_index as usize][from_cell_index as usize];
                    let weight = weight_array[((from_size + 1) * to_cell_index + from_cell_index) as usize];
                    sum += from_input_value * weight;
                }
                // Bias
                let weight = weight_array[((from_size + 1) * to_cell_index + from_size - 1) as usize];
                sum += weight;

                self.layer_values[to_layer_index as usize][to_cell_index as usize] = sigmoid(sum);
            }
        }
    }
}
