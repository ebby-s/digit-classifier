#include<iostream>
#include<vector>
#include<cmath>
#include "frame.hpp"
#include "dataset.hpp"
#include "mlp.hpp"

using namespace std;

int main(){
  Dataset data(cin, 1000, 800);
  MLP model(784, 0.01);

  
}
