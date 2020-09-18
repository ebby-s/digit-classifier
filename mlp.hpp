using namespace std;

// use stochastic learning or batches of 10
// layers = 784, 32, 28, 10
// for each pixel, input = val/128 - 1
// add twisting term ax to sigmoid
// add dropout

float sigmoid(float x){
  return 1.7159 * tanh(x*2/3);
}

class MLP{
private:
  vector<vector<float>> weight;
  vector<vector<float>> bias;
  vector<int> layers;
  float learning_rate;
public:
  MLP(int input_layer, float new_learning_rate){
    layers.push_back(input_layer);
    learning_rate = new_learning_rate;
  }
  void set_learning_rate(float new_learning_rate){
    learning_rate = new_learning rate;
  }
  get_weight();
  get_bias();
  add_layer();
  predict();
  calculate_loss();
  train();
};
