using namespace std;

class Matrix{
private:
  int size;     // total elements
  int* dims;        // dimensions of matrix in format Rows, Columns
  double* values;        // pointer to values in matrix
public:
  Matrix(int cols, int rows){     // dimensions determined in constructor
    size = rows*cols;
    dims = new int[2];
    dims[0] = rows;
    dims[1] = cols;
    values = new double[size];
    for(int i=0; i<size; i++){
      values[i] = 0;
    }
  }
  Matrix(const Matrix &obj){       // copy constructor
    size = obj.size;
    if(dims){delete[] dims;};
    dims = new int[2];
    dims[0] = obj.dims[0];
    dims[1] = obj.dims[1];
    if(values){delete[] values;}
    values = new double[size];
    for(int i=0; i<size; i++){
      values[i] = obj.values[i];
    }
  }
  ~Matrix(){           // deconstructor
    delete[] dims;
    delete[] values;
  }
  Matrix& operator=(const Matrix &obj){    // assignment operator overloaded
    size = obj.size;
    if(dims){delete[] dims;};
    dims = new int[2];
    dims[0] = obj.dims[0];
    dims[1] = obj.dims[1];
    if(values){delete[] values;}
    values = new double[size];
    for(int i=0; i<size; i++){
      values[i] = obj.values[i];
    }
  }
  int get_rows() const{
    return dims[0];
  }
  int get_cols() const{
    return dims[1];
  }
  double get_value(int col, int row) const{
    return values[dims[1]*row + col];
  }
  void set_value(int col, int row, double val){
    values[dims[1]*row + col] = val;
  }

  void load_random(int rand_a, int rand_b, int rand_c){     // loads matrix with random values centered around 0
    for(int i=0; i<size; i++){
      values[i] = (2.0*(double)(rand()%2)-1.0) * ((double)(rand()%rand_a)+rand_b)/rand_c;
    }
  }

  void clear(){         // sets all values to zero
    for(int i=0; i<size; i++){
      values[i] == 0;
    }
  }

  void add(const Matrix* other){           // add other matrix to self
    for(int j=0; j<dims[0]; j++){
      for(int i=0; i<dims[1]; i++){
        values[j*dims[1] + i] += other->get_value(i,j);
      }
    }
  }

  void multiply(const Matrix* other){        // self = other * self
    int output_size = other->get_rows()*dims[1];
    int* output_dims;
    double* output_values;
    output_dims = new int[2];
    output_values = new double[output_size];
    output_dims[0] = other->get_rows();
    output_dims[1] = dims[1];
    for(int i=0; i<output_size; i++){
      output_values[i] = 0;
    }

    for(int k=0; k<output_dims[1]; k++){
      for(int j=0; j<output_dims[0]; j++){
        for(int i=0; i<dims[0]; i++){
          output_values[j*output_dims[1]+k] += values[dims[1]*i+k] * other->get_value(i,j);
        }
      }
    }
    size = output_size;
    delete[] dims;
    dims = output_dims;
    delete[] values;
    values = output_values;
  }

  void print(ostream &dst){               // prints matrix to ostream
    for(int i=0; i<size; i++){
      if(i%dims[1]==0){dst << '[' << ' ';}
      dst << values[i] << ' ';
      if((i+1)%dims[1]==0){dst << ']' << endl;}
    }
  }

  void write_to_file(ofstream &dst){        // writes matrix to a file
    for(int i=0; i<size; i++){
      if(i%dims[1]==0){dst << endl;}
      dst << values[i] << " ";
    }
  }
};
