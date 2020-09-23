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
  MLP model(784, 0.0001, 0.9);
  model.add_layer(128, 6, 0, 10);
  model.add_layer(10, 10, 1, 100);

  string train_data_name = "./data/train-images-idx3-ubyte.gz";
  string train_label_name = "./data/train-labels-idx1-ubyte.gz";
  string test_data_name = "./data/t10k-images-idx3-ubyte.gz";
  string test_label_name = "./data/t10k-labels-idx1-ubyte.gz";
  Dataset data(train_data_name, train_label_name, test_data_name, test_label_name);

  pair<Matrix*,Matrix*> sample;
  pair<int,float> train_result;      // stores iterations trained and final error after a training cycle

  Frame debug_frame(28, 28, 255);

  for(int i=0; i<60000; i++){

    sample = data.get_training_sample();

    cerr << "Data:" << endl;
    for int j=0; j<28; j++){
      for(int k=0; k<28; k++){
        cerr << sample[0]->get_value(j,k);
      }
      cerr << endl;
    }
    cerr << "Label:" << endl;
    for(int j=0; j<10; j++){
      cerr << sample[1]->get_value(0,j);
    }
    cerr << endl;

    train_result = model.train(sample[0], sample[1], 512);

    cerr << "Sample " << i << endl;
		if (sample % 100 == 0) {   // Save the current network (weights)
			cout << "Saving the network to " << model_fn << " file." << endl;
			write_matrix(model_fn);
		}

  }

  y_hat = model.get_weight(0);
  debug_frame.load_from_matrix(y_hat);
  debug_frame.to_pgm(2);

}
