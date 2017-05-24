// reading a text file
#include <iostream>
#include <fstream>
#include <string>

using namespace std;

int main () {
  string line;
  float array[10];
  int count = 0 ; 
  ifstream myfile ("actualExFast.txt");
  std::string::size_type sz;   
  if (myfile.is_open())
  {
    
    while ( getline (myfile,line) )
    {
      array[count] = std::stod(line);
      count++;
      
    }
    myfile.close();
  }

  else cout << "Unable to open file"; 

  for(int i=0;i<2;i++)
    cout<<"Output "<<array[i]<<endl;
  return 0;
}