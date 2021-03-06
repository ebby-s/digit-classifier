using namespace std;

class Frame{         // stores an array of pixels to represent the final image
private:
  int width;
  int height;
  int max_val;
  vector<int> pixels;
public:
  Frame(int new_width, int new_height, int new_max_val){   // dimensions can't be adjusted later
    width = new_width;
    height = new_height;
    max_val = new_max_val;
    pixels.assign(width*height, 0);
  }
  int get_height() const{
    return height;
  }
  int get_width() const{
    return width;
  }
  int get_pixel(int posx, int posy) const{
    return pixels[posy*width + posx];
  }
  void set_pixel(int posx, int posy, int val){
    pixels[posy*width + posx] = val;
  }
  void clear(){
    pixels.assign(width*height, 0);
  }

  void load_from_matrix(Matrix* a){
    width = a->get_cols();
    height = a->get_rows();
    pixels.assign(width*height, 0);
    max_val = 255;

    for(int j=0; j<height; j++){
      for(int i=0; i<width; i++){
        pixels[j*width+i] = min(max((int)(a->get_value(i,j)*255),0),255);
      }
    }
  }

  void load_from_vector(vector<float> x){
    width = x.size();
    height = 1;
    pixels.assign(width, 0);
    max_val = 255;

    for(int j=0; j<width; j++){
      pixels[j] = min(max((int)(x[j]*255),0),255);
    }
  }

  void to_pgm(int scale) const{         // saves image as a pgm file
    cout << "P2" << endl;
    cout << width*scale << " " << height*scale << endl;
    cout << max_val << endl;
    for(int j=0; j<height*scale; j++){
      for(int i=0; i<width*scale; i++){
        cout << pixels[(j/scale)*width+(i/scale)] << ' ';
      }
      cout << endl;
    }
  }
};
