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
  Frame frame(28, 28, 255);

  MLP model(784, 0.01);
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

  vector<Matrix*> test_batch, train_batch;
  Matrix* x, *y, *y_hat;
  y = new Matrix(1, 10);

  vector<float> losses;

  for(int i=0; i<100; i++){
    losses.assign(10,0);
    train_batch = data.get_training_batch();
    if(i%100==0){test_batch = data.get_testing_batch();}

    for(int k=0; k<10; k++){
      for(int j=0; j<10; j++){
        y->set_value(0,j,(float)(k==j));
      }
      x = train_batch[k];
      if(i%100==0){
        y_hat = test_batch[k];
        model.predict(y_hat);
        losses[k] = calculate_loss(y_hat, y);
        cout << losses[k] << endl;
      }
      model.train(x, y);
    }
  }
  cout << accumulate(losses.begin(), losses.end(), 0)/losses.size() << endl;
}
