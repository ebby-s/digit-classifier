using namespace std;

// TODO add definition for train

// use stochastic learning or batches of 10
// layers = 784, 32, 28, 10
// for each pixel, input = val/128 - 1
// add twisting term ax to sigmoid
// add dropout

float sigmoid(float x){
  return 1.7159 * tanh(x*2/3);
}

float calculate_loss(vector<float> y_hat, vector<float> y){
  float loss = 0;
  for(int i=0; i<y.size(); i++){
    loss += pow((y[i] - y_hat[i]), 2);
  }
  return loss;
}

class MLP{     // multilayer perceptron
private:
  vector<Matrix*> weight;    // weights of each layer
  vector<Matrix*> bias;         // biases of each layer
  vector<int> layers;     // size of each layer
  float learning_rate;
public:
  MLP(int input_layer, float new_learning_rate){
    layers.push_back(input_layer);
    learning_rate = new_learning_rate;
  }
  void set_learning_rate(float new_learning_rate){
    learning_rate = new_learning_rate;
  }
  Matrix* get_weight(int layer){  // get pointer to weight matrix of a layer
    return weight[layer];
  }
  Matrix* get_bias(int layer){  // get pointer to bias matrix of a layer
    return bias[layer];
  }

  void add_layer(int size){    // add a layer with randomly initialized parameters
    Matrix* new_weight;
    Matrix* new_bias;
    new_weight = new Matrix(layers.back(), size);
    new_bias = new Matrix(1, size);
    new_weight->load_random();
    new_bias->load_random();
    weight.push_back(new_weight);
    bias.push_back(new_bias);

    layers.push_back(size);
  }

  Matrix predict(Matrix x) const{
    for(int i=0; i<layers.size(); i++){
      x.multiply(weight[i]);
      x.add(bias[i]);
    }
    return x;
  }

  float train(vector<Matrix*> x, vector<int> y);
};
