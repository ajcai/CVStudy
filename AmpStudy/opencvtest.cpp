#include <opencv2\opencv.hpp>
using namespace cv;
using namespace std;
int extract(){
	vector<Mat> channels;

	Mat img=imread("data/license.jpg"),ycbcrImg,yimg,dstImg;
	imshow("origin",img);
	cvtColor(img,ycbcrImg,COLOR_RGB2YCrCb);
	split(ycbcrImg,channels);
	yimg=channels.at(0);
	imshow("Y",yimg);
	Mat k1=(Mat_<double>(1,4)<<1,0,0,-1);
	Mat k2=(Mat_<double>(1,7)<<0.5,0,0,-1,0,0,0.5);
	cout<<k1<<endl;
	
	filter2D(yimg,dstImg,yimg.depth(),k1);
	imshow("dst1",dstImg);
	filter2D(yimg,dstImg,yimg.depth(),k1.t());
	imshow("dst2",dstImg);
	filter2D(yimg,dstImg,yimg.depth(),k2);
	imshow("dst3",dstImg);
	filter2D(yimg,dstImg,yimg.depth(),k2.t());
	imshow("dst4",dstImg);
	return 1;
}
/*int main(){
	extract();
	waitKey();
}*/
