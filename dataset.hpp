using namespace std;

class Dataset{
private:
  ifstream train_data;
  ifstream train_labels;
  ifstream test_data;
  ifstream test_labels;
public:
  Dataset(string train_data_name, string train_label_name, string test_data_name, string test_label_name){
    char inchar;
    train_data.open(train_data_name);
    train_labels.open(train_label_name);
    test_data.open(test_data_name);
    test_labels.open(test_label_name);

    for (int i=0; i<16; i++) {                        // read file headers
      train_data.read(&inchar, sizeof(inchar));
      test_data.read(&inchar, sizeof(inchar));
    }
    for (int i=0; i<8; i++) {
      train_labels.read(&inchar, sizeof(inchar));
      test_labels.read(&inchar, sizeof(inchar));
    }
  }

  void close(){
    train_data.close();
    train_labels.close();
    test_data.close();
    test_labels.close();
  }

  pair<Matrix*, Matrix*> get_training_sample(){
    pair<Matrix*, Matrix*> sample;
    get<0>(sample) = new Matrix(1,784);
    get<1>(sample) = new Matrix(1,10);

    char inchar;
    for (int i=0; i<784; i++){
      train_data.read(&inchar, sizeof(inchar));
      get<0>(sample)->set_value(0,i,inchar!=0);
    }

    train_labels.read(&inchar, sizeof(inchar));
    get<1>(sample)->set_value(0,(int)(inchar),1);

    return sample;
  }

  pair<Matrix*, Matrix*> get_testing_sample(){
    pair<Matrix*, Matrix*> sample;
    get<0>(sample) = new Matrix(1,784);
    get<1>(sample) = new Matrix(1,10);

    char inchar;
    for (int i=0; i<784; i++){
      test_data.read(&inchar, sizeof(inchar));
      get<0>(sample)->set_value(0,i,inchar!=0);
    }
    test_labels.read(&inchar, sizeof(inchar));
    get<1>(sample)->set_value(0,(int)(inchar),1);

    return sample;
  }
};
