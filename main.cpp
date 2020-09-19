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

  Matrix* a;
  Matrix* b;

  a = new Matrix(1,4);
  a->load_random();
  a->print(cout);
  cout << endl;
  b = new Matrix(4,3);
  b->load_random();
  b->print(cout);
  cout << endl;
  a->multiply(b);
  a->print(cout);
}
