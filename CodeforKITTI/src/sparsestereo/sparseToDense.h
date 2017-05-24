using namespace cv;
using namespace std;

struct labelOfPoints
{
	vector<Point> points;
	int label;
};

struct ySortedPoints
{
	vector<Point> p ; 
	vector<double> y;
};