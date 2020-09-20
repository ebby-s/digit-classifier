using namespace std;

// TODO train with mini-batches of 10

// use stochastic learning or batches of 10
// layers = 784, 32, 28, 10
// for each pixel, input = val/128 - 1

float sigmoid(float x){        // sigmoid used as activation function
  return 1.7159 * tanh(x*2/3);
}
float sigmoid_gradient(float x){     // derivative of sigmoid function
  return 1.1439 * pow(1/cosh(x*2/3), 2);
}

float calculate_loss(Matrix* y_hat, Matrix* y){   // calculates loss for one training example
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

  void train_batch(vector<Matrix*> x_batch, vector<Matrix*> y_batch){
    vector<Matrix*> z, a;            // holds results from forward pass
    vector<float> dzdw, dzdz, next_dzdz;       // holds derivatives
    float layer_learn_rate;             // holds adjusted learning rate for each layer
    Matrix y_hat(28, 28);

    Matrix* x, *y;
    vector<Matrix*> delta_weight, delta_bias;     // holds total change to parameters over all samples
    for(int i=0; i<weight.size(); i++){
      delta_weight.push_back(new Matrix(weight[i]->get_cols(), weight[i]->get_rows()));
      delta_bias.push_back(new Matrix(bias[i]->get_cols(), bias[i]->get_rows()));
    }

    for(int n=0; n<x_batch.size(); n++){     // iterate over batch size n
      x = x_batch[n];
      y = y_batch[n];

      y_hat = *x;
      a.push_back(new Matrix(*x));

      for(int i=0; i<layers.size()-1; i++){        // forward pass
        y_hat.multiply(weight[i]);
        y_hat.add(bias[i]);
        z.push_back(new Matrix(y_hat));
        for(int j=0; j<layers[i+1]; j++){
          y_hat.set_value(0, j, sigmoid(y_hat.get_value(0, j)));
        }
        a.push_back(new Matrix(y_hat));
      }

      dzdz.assign(layers.back(),0);             // calculate derivative of loss with respect to final layer z values
      for(int i=0; i<layers.back(); i++){
        dzdz[i] = 2*(a.back()->get_value(0,i) - y->get_value(0,i)) * sigmoid_gradient(z.back()->get_value(0, i));
      }

      for(int i=layers.size()-1; i>0; i--){     // repeat for each layer

        layer_learn_rate = learning_rate * pow(layers[i-1], 0.5);     // adjusted learning rate for deeper layers

        dzdw.assign(layers[i-1], 0);           // calculate derivative of z with respect to weights
        for(int j=0; j<layers[i-1]; j++){
          dzdw[j] = a[i-1]->get_value(0,j);
        }
        //if(i==1){for(int l=0; l<dzdz.size(); l++){cerr << dzdz[l] << endl;}}

        if(i!=1){                               // calculate derivative of z with respect to z of previous layer
          next_dzdz.assign(layers[i-1], 0);
          for(int j=0; j<layers[i-1]; j++){
            for(int k=0; k<layers[i]; k++){
              next_dzdz[j] += weight[i-1]->get_value(j, k);
            }
            next_dzdz[j] *= sigmoid_gradient(z[i-1]->get_value(0, j));
          }
        }

        for(int j=0; j<layers[i]; j++){          // accumulate changes to weights and biases
          delta_bias[i-1]->set_value(0, j, delta_bias[i-1]->get_value(0, j) - dzdz[j] * layer_learn_rate);
          for(int k=0; k<layers[i-1]; k++){
            delta_weight[i-1]->set_value(k, j, delta_weight[i-1]->get_value(k, j) - dzdz[j] * dzdw[k] * layer_learn_rate);
          }
        }
        dzdz = next_dzdz;
      }
      delete a.back();          // delete all new matrices created during the process
      delete x;
      for(int i=0; i<z.size(); i++){
        delete a[i];
        delete z[i];
      a.clear();
      z.clear();
      }
    }

    for(int i=0; i<weight.size(); i++){   // update parameters
      //weight[i]->print(cerr);
      delta_bias[i]->print(cerr);
      weight[i]->add(delta_weight[i]);
      bias[i]->add(delta_bias[i]);
    }
  }
};
