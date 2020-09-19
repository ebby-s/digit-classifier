using namespace std;

// TODO store derivatives in matrices

// use stochastic learning or batches of 10
// layers = 784, 32, 28, 10
// for each pixel, input = val/128 - 1
// add twisting term ax to sigmoid
// add dropout

float sigmoid(float x){
  return 1.7159 * tanh(x*2/3);
}
float sigmoid_gradient(float x){
  return 1.1439 * pow(1/cosh(x*2/3), 2);
}

float calculate_loss(Matrix* y_hat, Matrix* y){
  float loss = 0;
  for(int i=0; i<10; i++){
    loss += pow((y->get_value(0,i) - y_hat->get_value(0,i)), 2);
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

  void predict(Matrix* x) const{         // passes input x through network
    for(int i=0; i<layers.size()-1; i++){
      x->multiply(weight[i]);
      x->add(bias[i]);
      for(int j=0; j<layers[i+1]; j++){
        x->set_value(0, j, sigmoid(x->get_value(0, j)));
      }
    }
  }

  void train(Matrix* x, Matrix* y){
    vector<Matrix*> a, z;
    Matrix y_hat(x->get_cols(), x->get_rows());
    vector<float> dadz, dzda, dadw, next_dadz;
    float layer_learn_rate;

    y_hat = *x;

    for(int i=0; i<layers.size()-1; i++){
      y_hat.multiply(weight[i]);
      y_hat.add(bias[i]);
      a.push_back(new Matrix(y_hat));
      for(int j=0; j<layers[i+1]; j++){
        y_hat.set_value(0, j, sigmoid(y_hat.get_value(0, j)));
      }
      z.push_back(new Matrix(y_hat));
    }

    dadz.assign(layers.back(),0);
    for(int i=0; i<layers.back(); i++){
      dadz[i] = 2*(z.back()->get_value(0,i) - y->get_value(0,1));
    }

    for(int i=layers.size()-1; i>0; i--){
      layer_learn_rate = learning_rate * pow(layers[i-1], 0.5);

      dzda.assign(layers[i],0);
      for(int j=0; j<layers[i]; j++){
        dzda[j] = sigmoid_gradient(a[i-1]->get_value(0, j));
      }

      dadw.assign(layers[i-1], 0);
      for(int j=0; j<layers[i-1]; j++){
        if(i==1){
          dadw[j] = x->get_value(0,j);
        }else{
          dadw[j] = z[i-2]->get_value(0,j);
        }
      }

      if(i!=1){
        next_dadz.assign(layers[i-1], 0);
        for(int j=0; j<layers[i-1]; j++){
          for(int k=0; k<layers[i]; k++){
            next_dadz[j] += weight[i-1]->get_value(j, k);
          }
        }
      }

      for(int j=0; j<layers[i]; j++){
        bias[i-1]->set_value(0, j, bias[i-1]->get_value(0, j) - dadz[j] * dzda[j] * layer_learn_rate);
        for(int k=0; k<layers[i-1]; k++){
          weight[i-1]->set_value(k, j, weight[i-1]->get_value(k, j) - dadz[j] * dzda[j] * dadw[k] * layer_learn_rate);
        }
      }

      dadz = next_dadz;
    }

    for(int i=0; i<a.size(); i++){
      delete a[i];
      delete z[i];
    }
  }
};
