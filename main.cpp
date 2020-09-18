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
  Dataset data(1200, 800);
  MLP model(784, 0.01);
  model.add_layer(32);
  model.add_layer(28);
  model.add_layer(10);

  Matrix* a;
  Matrix* b;
  for(int j=0; j<1; j++){
    a = data.get_testing_digit(0);
  }
  b = new Matrix(28,28);

  for(int i=0; i<784; i++){
    b->set_value(i%28, i/28, a->get_value(0,i));
  }

  frame.load_from_matrix(b);
  //frame.to_pgm(10);
}
