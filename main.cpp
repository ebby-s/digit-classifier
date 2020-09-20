#include<iostream>
#include<fstream>
#include<vector>
#include<cmath>
#include<random>
#include "matrix.hpp"
#include "frame.hpp"
#include "dataset.hpp"
#include "mlp.hpp"

using namespace std;

int main(){
  MLP model(784, 0.0001);
  model.add_layer(32);
  model.add_layer(28);
  model.add_layer(10);

  vector<string> files;
  string data_path = "./data/data";
  for(int i=0; i<10; i++){
    data_path.append(to_string(i));
    files.push_back(data_path);
    data_path.pop_back();
  }
  Dataset data(files, 1000, 800);

  vector<Matrix*> testing_batch, training_batch, y_batch;
  Matrix* x, *y, *y_hat;

  for(int k=0; k<10; k++){
    y = new Matrix(1, 10);
    for(int j=0; j<10; j++){ y->set_value(0,j,(float)(k==j)); }
    y_batch.push_back(y);
  }

  Frame debug_frame(28, 28, 255);
  vector<float> losses;
  losses.assign(10,0);

  for(int i=0; i<1; i++){

    if(i%1==0){
      testing_batch = data.get_testing_batch();
      for(int k=0; k<10; k++){
        y = y_batch[k];
        y_hat = testing_batch[k];
        model.predict(y_hat);
        losses[k] = calculate_loss(y_hat, y);
        delete y_hat;
      }
      cerr << accumulate(losses.begin(), losses.end(), 0)/losses.size() << endl;
    }

    training_batch = data.get_training_batch();
    model.train_batch(training_batch, y_batch);
  }

  y_hat = model.get_weight(0);
  debug_frame.load_from_matrix(y_hat);
  debug_frame.to_pgm(2);
  y_hat = data.get_testing_digit(5);
  for(int j=0; j<10; j++){
    y->set_value(0,j,(float)(j==5));
  }
  model.predict(y_hat);
  y_hat->print(cerr);
}
