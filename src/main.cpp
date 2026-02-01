#include <torch/torch.h>
#include <iostream>
#include <vector>

/* 
  1. Three inputs: minimum value, minimum angle point, inversion
  2. medium_thickness_or_radius, scatterer_radius
  3. Batch size will likely only start out as 1 until I can generate much more examples
  4. Random must use a uniform distribution
  5. Make sure to implement softmax to handle all values
  6. Loss function to adjust for offset.
  7. Optimization
  8. Back propagation
*/

constexpr int input_count = 3;

struct Model : torch::nn::Module
{
  torch::nn::Linear fc1{nullptr};
  torch::nn::Linear fc2{nullptr};
  torch::nn::Linear fc3{nullptr};

  Model()
  {
    fc1 = register_module("fc1", torch::nn::Linear(input_count, 16));
    fc2 = register_module("fc1", torch::nn::Linear(16, 32));
    fc3 = register_module("fc1", torch::nn::Linear(32, 3));
  }

};

int main() {

  int batch_size = 1;

  return 0;
}