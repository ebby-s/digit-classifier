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
  MLP model(784, 0.00001, 0.7);
  model.add_layer(128);
  model.add_layer(10);

  vector<string> files;
  string data_path = "./data/data";
  for(int i=0; i<10; i++){
    data_path.append(to_string(i));
    files.push_back(data_path);
    data_path.pop_back();
  }
  Dataset data(files, 1000, 800);

  Matrix* x, *y, *y_hat;
  y = new Matrix(1, 10);
  int sample_digit;

  Frame debug_frame(28, 28, 255);
  vector<float> losses;
  losses.assign(10,0);


  for(int i=0; i<100; i++){        // main loop i
    sample_digit = rand()%10;
    for(int l=0; l<10; l++){y->set_value(0,l,(float)(sample_digit==l));}
    x = data.get_training_digit(sample_digit);
    model.train(x, y, 500);
    model.clear_momentum();

    if(i%8==0){
      y_hat = data.get_testing_digit((sample_digit+3)%10);
      model.predict(y_hat);
      //cerr << i << ' ' << calculate_loss(y_hat, y) << endl;
      cerr << i << ' ' << sample_digit << endl;
      y_hat->print(cerr);
      delete y_hat;
    }
  }

  y_hat = model.get_weight(0);
  debug_frame.load_from_matrix(y_hat);
  debug_frame.to_pgm(2);

}
