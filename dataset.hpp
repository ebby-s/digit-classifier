using namespace std;

class Dataset{
private:
  vector<int>* train_data;
  vector<int>* test_data;
  int label_split;       // number of samples of each digit
  int test_train_split;      // number of training samples of each digit
  vector<int> next_train_sample;
  vector<int> next_test_sample;
public:
  Dataset(vector<string> filenames, int new_label_split, int new_test_train_split){
    u_char inchar;
    int file = 0;
    ifstream source(filenames[file]);

    train_data = new vector<int>;
    test_data = new vector<int>;
    label_split = new_label_split;
    test_train_split = new_test_train_split;
    next_train_sample.assign(10,0);
    next_test_sample.assign(10,0);

    for(int j=0; j<10; j++){
      for(int i=0; i<784*test_train_split; i++){
        source.read((char*)&inchar, sizeof(inchar));
        if(source.eof()){
          file++;
          source.close();
          source.clear();
          source.open(filenames[file]);
          source.read((char*)&inchar, sizeof(inchar));
        }
        train_data->push_back((int)inchar);
      }
      for(int i=0; i<784*(label_split-test_train_split); i++){
        source.read((char*)&inchar, sizeof(inchar));
        if(source.eof()){
          file++;
          source.close();
          source.clear();
          source.open(filenames[file]);
          source.read((char*)&inchar, sizeof(inchar));
        }
        test_data->push_back((int)inchar);
      }
    }
  }

  vector<Matrix*> get_training_batch(){
    vector<Matrix*> batch;
    Matrix* sample;
    for(int j=0; j<next_train_sample.size(); j++){
      sample = new Matrix(1,784);
      for(int i=0; i<784; i++){
        sample->set_value(0, i, ((float)(*train_data)[j*test_train_split*784 + next_train_sample[j]*784 + i])/255.0);
      }
      batch.push_back(sample);
      next_train_sample[j] = (next_train_sample[j]+1) % test_train_split;
    }
    return batch;
  }

  Matrix* get_training_digit(int digit){
    Matrix* sample;
    sample = new Matrix(1,784);
    for(int i=0; i<784; i++){
      sample->set_value(0, i, ((float)((*train_data)[digit*test_train_split*784 + next_train_sample[digit]*784 + i]))/255.0);
    }
    next_train_sample[digit] = (next_train_sample[digit]+1) % test_train_split;
    return sample;
  }

  vector<Matrix*> get_testing_batch(){
    vector<Matrix*> batch;
    Matrix* sample;
    for(int j=0; j<next_test_sample.size(); j++){
      sample = new Matrix(1,784);
      for(int i=0; i<784; i++){
        sample->set_value(0, i, ((float)(*test_data)[j*(label_split-test_train_split)*784 + next_test_sample[j]*784 + i])/255.0);
      }
      batch.push_back(sample);
      next_test_sample[j] = (next_test_sample[j]+1) % (label_split-test_train_split);
    }
    return batch;
  }

  Matrix* get_testing_digit(int digit){
    Matrix* sample;
    sample = new Matrix(1,784);
    for(int i=0; i<784; i++){
      sample->set_value(0, i, ((float)(*test_data)[digit*(label_split-test_train_split)*784 + next_test_sample[digit]*784 + i])/255.0);
    }
    next_test_sample[digit] = (next_test_sample[digit]+1) % (label_split-test_train_split);
    return sample;
  }
};
