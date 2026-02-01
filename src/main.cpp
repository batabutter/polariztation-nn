#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <memory>

/*
  1. Three inputs: minimum value, minimum angle point, inversion
  2. outputs = medium_thickness_or_radius, scatterer_radius
*/

struct Model : torch::nn::Module
{
  torch::nn::Linear fc1{nullptr};
  torch::nn::Linear fc2{nullptr};
  torch::nn::Linear fc3{nullptr};

  Model()
  {
    fc1 = register_module("fc1", torch::nn::Linear(3, 16));
    fc2 = register_module("fc2", torch::nn::Linear(16, 16));
    fc3 = register_module("fc3", torch::nn::Linear(16, 3));
  }
  // 5,3
  torch::Tensor forward(torch::Tensor x)
  {
    x = torch::relu(fc1(x));
    x = torch::relu(fc2(x));
    x = fc3(x);
    return x;
  }
};

int main()
{

  auto model = std::make_shared<Model>();

  // Shape: (5,3)
  torch::Tensor x = torch::tensor({
    {-0.03888, 0.25, 7.0},
    {-0.03888, 0.25, 7.0},
    {-0.01821, 0.15, 5.0},
    {-0.022925, 0.15, 6.0},
    {-0.042534, 0.25, 0.02}
  }, torch::kFloat32);
  
  auto y = model->forward(x);

  std::cout << y << std::endl;

  return 0;
}