using namespace std;

class Matrix{
private:
  vector<int> dims;        // dimensions of matrix in format Rows, Columns
  vector<double>* values;        // pointer to values in matrix
public:
  Matrix(int cols, int rows){     // dimensions determined in constructor
    dims.push_back(rows);
    dims.push_back(cols);
    values = new vector<double>;
    values->assign(rows*cols,0);
  }
  Matrix(const Matrix &obj){       // copy constructor
    dims = obj.dims;
    values = new vector<double>;
    values->assign(dims[0] * dims[1], 0);
    for(int i=0; i<values->size(); i++){
      (*values)[i] = (*(obj.values))[i];
    }
  }
  ~Matrix(){           // deconstructor
    delete values;
  }
  Matrix& operator=(const Matrix &obj){    // assignment operator overloaded
    dims = obj.dims;
    values = new vector<double>;
    values->assign(dims[0] * dims[1], 0);
    for(int i=0; i<values->size(); i++){
      (*values)[i] = (*(obj.values))[i];
    }
  }
  int get_rows() const{
    return dims[0];
  }
  int get_cols() const{
    return dims[1];
  }
  double get_value(int col, int row) const{
    return (*values)[dims[1]*row + col];
  }
  void set_value(int col, int row, double val){
    (*values)[dims[1]*row + col] = val;
  }

  void load_random(int rand_a, int rand_b, int rand_c){     // loads matrix with random values centered around 0
    for(int i=0; i<values->size(); i++){
      (*values)[i] = (2.0*(double)(rand()%2)-1.0) * ((double)(rand()%rand_a)+rand_b)/rand_c;
    }
  }

  void clear(){         // sets all values to zero
    values->assign(dims[0]*dims[1], 0);
  }

  void add(const Matrix* other){           // add other matrix to self
    for(int j=0; j<dims[0]; j++){
      for(int i=0; i<dims[1]; i++){
        (*values)[j*dims[1] + i] += other->get_value(i,j);
      }
    }
  }

  void multiply(const Matrix* other){        // self = other * self
    vector<int> output_dims;
    vector<double>* output_values;
    output_dims.push_back(other->get_rows());
    output_dims.push_back(dims[1]);
    output_values = new vector<double>;
    output_values->assign(other->get_rows()*dims[1],0);

    for(int k=0; k<output_dims[1]; k++){
      for(int j=0; j<output_dims[0]; j++){
        for(int i=0; i<dims[0]; i++){
          (*output_values)[j*output_dims[1]+k] += (*values)[dims[1]*i+k] * other->get_value(i,j);
        }
      }
    }
    dims = output_dims;
    delete values;
    values = output_values;
  }

  void print(ostream &dst){               // prints matrix to ostream
    for(int i=0; i<values->size(); i++){
      if(i%dims[1]==0){dst << '[' << ' ';}
      dst << (*values)[i] << ' ';
      if((i+1)%dims[1]==0){dst << ']' << endl;}
    }
  }

  void write_to_file(ofstream &dst){        // writes matrix to a file
    for(int i=0; i<values->size(); i++){
      dst << (*values)[i] << " ";
      if(i%dims[1]==0){dst << endl;}
    }
  }
};
