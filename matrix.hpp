using namespace std;

class Matrix{
private:
  vector<int> dims;
  vector<float>* values;
public:
  Matrix(int cols, int rows){
    dims.push_back(rows);
    dims.push_back(cols);
    values = new vector<float>;
    values->assign(rows*cols,0)
  }
  int get_rows() const{
    return dims[0];
  }
  int get_cols() const{
    return dims[1];
  }
  float get_value(int col, int row) const{
    return (*values)[dims[1]*row + col];
  }
  void set_value(int col, int row, float val){
    (*values)[dims[1]*row + col] = val;
  }

  void add(const Matrix* other){
    for(int j=0; j<dims[0]; j++){
      for(int i=0; i<dims[1]; i++){
        (*values)[j*dims[1] + i] += other->get_value(i,j);
      }
    }
  }

  void multiply(const Matrix* other){
    vector<int> output_dims;
    vector<float>* output_values;
    output_dims.push_back(other->get_rows());
    output_dims.push_back(dims[1]);
    output_values = new vector<float>;
    output_values->assign(other->get_rows()*dims[1],0);

    for(int k=0; k<output_dims[1]; k++){
      for(int j=0; j<output_dims[0]; j++){
        for(int i=0; i<dims[1]; i++){
          (*output_values)[j*output_dims[1]+k] += (*values)[dims[1]*i+k] * other->get_value(j,i);
        }
      }
    }
    dims = output_dims;
    delete[] values;
    values = output_values;
  }
};