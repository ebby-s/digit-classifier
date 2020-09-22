using namespace std;

// TODO train with mini-batches of 10

// use stochastic learning or batches of 10
// layers = 784, 32, 28, 10
// for each pixel, input = val/128 - 1

float sigmoid(float x){return 1.0/(1.0 + exp(-x));}        // sigmoid used as activation function
float sigmoid_gradient(float y){return y * (1.0-y);}     // derivative of sigmoid function

float tanh_sigmoid(float x){return 1.7159 * tanh(x*2/3);}    // tanh sigmoids to compare
float tanh_sigmoid_gradient(float x){return 1.1439 * pow(1/cosh(x*2/3), 2);}

float calculate_loss(Matrix* y_hat, Matrix* y){   // calculates loss for one training batch
  float loss = 0;
  for(int i=0; i<10; i++){
    loss += pow((y->get_value(0,i) - y_hat->get_value(0,i)), 2);
  }
  return loss;
}

class MLP{     // multilayer perceptron
private:
  vector<Matrix*> weight;    // weights of each layer
  vector<Matrix*> delta_weight;      // change to weights every train cycle
  vector<int> layers;     // size of each layer
  float learning_rate;
  float momentum;
public:
  MLP(int input_layer, float new_learning_rate, float new_momentum){
    layers.push_back(input_layer);
    learning_rate = new_learning_rate;
    momentum = new_momentum;
  }
  Matrix* get_weight(int layer){return weight[layer];}  // get pointer to weight matrix of a layer

  void clear_momentum(){                          // resets momentum
    for(int i=0; i<delta_weight.size(); i++){
      delta_weight[i]->clear();
    }
  }

  void add_layer(int size){    // add a layer with randomly initialized parameters
    weight.push_back(new Matrix(layers.back(), size));
    weight.back()->load_random();
    delta_weight.push_back(new Matrix(layers.back(), size));
    layers.push_back(size);
  }

  void predict(Matrix* x) const{         // passes input x through network
    for(int i=0; i<layers.size()-1; i++){
      x->multiply(weight[i]);
      for(int j=0; j<layers[i+1]; j++){
        x->set_value(0, j, sigmoid(x->get_value(0, j)));
      }
    }
  }

  void train(Matrix* x, Matrix* y, int n_cycles){
    vector<Matrix*> z, a;            // holds results from forward pass
    vector<float> dzdw, dzdz, next_dzdz;       // holds derivatives
    float layer_learn_rate;             // holds adjusted learning rate for each layer
    Matrix y_hat(28, 28);

    for(int n=0; n<n_cycles; n++){     // iterate over batch size n

      y_hat = *x;
      a.push_back(new Matrix(*x));

      for(int i=0; i<layers.size()-1; i++){        // forward pass
        y_hat.multiply(weight[i]);
        z.push_back(new Matrix(y_hat));
        for(int j=0; j<layers[i+1]; j++){
          y_hat.set_value(0, j, sigmoid(y_hat.get_value(0, j)));
        }
        a.push_back(new Matrix(y_hat));
      }

      dzdz.assign(layers.back(),0);             // calculate derivative of loss with respect to final layer z values
      for(int i=0; i<layers.back(); i++){
        dzdz[i] = 2*(a.back()->get_value(0,i) - y->get_value(0,i)) * sigmoid_gradient(a.back()->get_value(0, i));
      }

      for(int i=layers.size()-1; i>0; i--){     // repeat for each layer

        layer_learn_rate = learning_rate;// * pow(layers[i-1], 0.5);     // adjusted learning rate for deeper layers

        dzdw.assign(layers[i-1], 0);           // calculate derivative of z with respect to weights
        for(int j=0; j<layers[i-1]; j++){
          dzdw[j] = a[i-1]->get_value(0,j);
        }

        if(i!=1){                               // calculate derivative of z with respect to z of previous layer
          next_dzdz.assign(layers[i-1], 0);
          for(int j=0; j<layers[i-1]; j++){
            for(int k=0; k<layers[i]; k++){
              next_dzdz[j] += weight[i-1]->get_value(j, k);
            }
            next_dzdz[j] *= sigmoid_gradient(a[i-1]->get_value(0, j));
          }
        }

        for(int j=0; j<layers[i]; j++){          // accumulate changes to weights and biases
          for(int k=0; k<layers[i-1]; k++){
            delta_weight[i-1]->set_value(k, j, momentum * delta_weight[i-1]->get_value(k, j) - dzdz[j] * dzdw[k] * layer_learn_rate);
          }
        }
        weight[i-1]->add(delta_weight[i-1]);
        dzdz = next_dzdz;
      }
      delete a.back();          // delete all new matrices created during the process
      for(int i=0; i<z.size(); i++){
        delete a[i];
        delete z[i];
      a.clear();
      z.clear();
      }
    }

    //for(int i=0; i<weight.size(); i++){   // update parameters
      //weight[i]->add(delta_weight[i]);
    //}
  }
};
