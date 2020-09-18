using namespace std;

class Dataset{
private:
  vector<int> train_data;
  vector<int> test_data;
  int label_split;       // number of samples of each digit
  int test_train_split;      // number of training samples of each digit
  vector<int> next_sample;
public:
  Dataset(istream &source, int new_label_split, int new_test_train_split){
    u_char inchar;
    label_split = new_label_split;
    test_train_split = new_test_train_split;

    for(int j=0; j<10; j++){
      for(int i=0; i<28*28*test_train_split; i++){
        source >> inchar;
        train_data.push_back((int)inchar);
      }
      for(int i=0; i<28*28*(label_split-test_train_split); i++){
        source >> inchar;
        test_data.push_back((int)inchar);
      }
    }
  }

  vector<int> get_training_batch(){
    vector<int> batch;
    for(int j=0; j<next_sample.size(); j++){
      for(int i=0; i<784; i++){
        batch.push_back(train_data[j*label_split*784 + next_sample[j]*784 + i]);
      }
      next_sample[j] = (next_sample[j]+1) % test_train_split;
    }
    return batch;
  }

  vector<int> get_training_digit(int digit){
    vector<int> sample;
  }
  vector<int> get_testing_batch();
  vector<int> get_testing_digit(int digit);
};
