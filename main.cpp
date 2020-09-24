#include<iostream>
#include<fstream>
#include<vector>
#include<cmath>
#include<utility>
#include "matrix.hpp"
#include "dataset.hpp"
#include "mlp.hpp"

using namespace std;

int main(){
  MLP model(784, 0.001, 0.9, 512, 0.002);
  model.add_layer(128, 6, 0, 10);
  model.add_layer(10, 10, 1, 100);
  string model_file_name = "./model_weights.txt";

  string train_data_name = "data/train-images.idx3-ubyte";
  string train_label_name = "data/train-labels.idx1-ubyte";
  string test_data_name = "data/t10k-images.idx3-ubyte";
  string test_label_name = "data/t10k-labels.idx1-ubyte";
  Dataset data(train_data_name, train_label_name, test_data_name, test_label_name);

  pair<Matrix*,Matrix*> sample;
  pair<int,float> train_result;      // stores iterations trained and final error after a training cycle

  for(int i=0; i<60000; i++){

    sample = data.get_training_sample();

    cerr << "Data:" << endl;
    for(int j=0; j<784; j++){
      if(j%28==0){cerr << endl;}
      cerr << get<0>(sample)->get_value(0,j);
    }
    cerr << endl << "Label:" << endl;
    for(int j=0; j<10; j++){
      cerr << get<1>(sample)->get_value(0,j);
    }
    cerr << endl;

    train_result = model.train(get<0>(sample), get<1>(sample));

    cerr << "Sample " << i << endl;
    cerr << "Error: " << get<1>(train_result) << " after " << get<0>(train_result) << " iterations." << endl;
		if (i%100==0){
			cerr << "Saving the network" << endl;
			model.save_to_file(model_file_name);
		}
  }
  data.close();
}
