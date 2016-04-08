#include <opencv2\opencv.hpp>
#include <iostream>
#include <armadillo>
#include <array>
#include <amp.h>
#include <amp_math.h>
extern "C"{
#include <vl/kdtree.h>
#include <vl/generic.h>
#include <vl/kdtree.h>
}



//using namespace cv;
using namespace std;
using namespace arma;
using namespace concurrency;
using namespace concurrency::fast_math;

//convert unsigned int cv::mat to arma::Mat<double>
static void Cv_mat_to_arma_mat(const cv::Mat& cv_mat_in, arma::mat& arma_mat_out)
{
    for(int r=0;r<cv_mat_in.rows;r++){
		for(int c=0;c<cv_mat_in.cols;c++){
			arma_mat_out(r,c)=cv_mat_in.data[r*cv_mat_in.cols+c]/255.0;
		}
	}
};
//convert unsigned int arma::Mat to cv::Mat
template<typename T>
static void Arma_mat_to_cv_mat(const arma::Mat<T>& arma_mat_in,cv::Mat_<T>& cv_mat_out)
{
    cv::transpose(cv::Mat_<T>(static_cast<int>(arma_mat_in.n_cols),
                              static_cast<int>(arma_mat_in.n_rows),
                              const_cast<T*>(arma_mat_in.memptr())),
                  cv_mat_out);
};
//write arma::mat to file
static void Write_arma_mat(field<mat> mats){
	for(int i=0;i<mats.n_elem;i++){
		cv::Mat_<double> cv_mat_show;
		Arma_mat_to_cv_mat(mats(i),cv_mat_show);
		cv_mat_show=cv_mat_show*255;
		char filename[100];
		sprintf(filename,"data/result/filter_result_%d.bmp",i);
		cv::imwrite(filename,cv_mat_show);
	}
};
static void Write_arma_mat(arma::mat mat,char* filename){
	cv::Mat_<double> cv_mat_show;
	Arma_mat_to_cv_mat(mat,cv_mat_show);
	cv_mat_show=cv_mat_show*255;
	cv::imwrite(filename,cv_mat_show);
};

static arma::mat Collect_features(const arma::mat& arma_mat_in,field<mat>& filters,SizeMat& window,SizeMat& overlap){
	//filter the image
	field<mat> feature_imgs(filters.n_elem);
	for(int i=0;i<filters.n_elem;i++){
		mat rst=conv2(arma_mat_in,filters(i),"same");
		feature_imgs(i)=rst;
		char filename[100];
		sprintf(filename,"data/result/filter_result_%d.bmp",i);
		Write_arma_mat(rst,filename); //保存滤波中间结果
	}
	//sample grid
	int patch_size=window(0)*window(1);
	SizeMat skip=window-overlap;
	int sample_row_num=(arma_mat_in.n_rows-window(0))/skip(0)+1;//sample grid of row numbers
	int sample_col_num=(arma_mat_in.n_cols-window(1))/skip(1)+1;//sample grid of col numbers
	int sample_patch_num=sample_row_num*sample_col_num;// number of patches
	uvec grid_row(sample_row_num);//row of grid top-left point of grid
	for(int i=0;i<grid_row.n_elem;i++){grid_row(i)=i*skip(0);}
	uvec grid_col(sample_col_num);//col of grid top-left point of grid
	for(int i=0;i<grid_col.n_elem;i++){grid_col(i)=i*skip(1);}

	arma::mat features(patch_size*filters.n_elem,sample_patch_num);		//所有特征
	for(int i=0;i<sample_patch_num;i++){
		vec patch_feature(window(0)*window(1)*filters.n_elem);
		for(int f=0;f<filters.n_elem;f++){//取对应patch的特征，将多个滤波结果合为一个特征
			arma::mat patch=feature_imgs(f)(i%sample_row_num,i/sample_row_num,window);
			patch_feature(span(patch_size*f,patch_size*(f+1)-1))=vectorise(patch);
		}
		features.col(i)=patch_feature;
	}
	return features;
}
static void run_jor(){
	//------------------------------------------------
	//read img and get gray image
	vector<cv::Mat> channels;
	cv::Mat cv_img=cv::imread("data/license.jpg"),ycbcrImg,grayImg,cbImg,crImg,interpolatedImg;
	cv::resize(cv_img,interpolatedImg,cv::Size(),3,3,cv::INTER_CUBIC);		//双立方插值放大
	cvtColor(interpolatedImg,ycbcrImg,cv::COLOR_RGB2YCrCb);
	split(ycbcrImg,channels);
	grayImg=channels.at(0);
	cbImg=channels.at(1);
	crImg=channels.at(2);
	//---------------------------------------------------
	//convert cv::Mat to arma::mat
	mat img(grayImg.rows,grayImg.cols);
	Cv_mat_to_arma_mat(grayImg,img);
	//----------------------------------------------------
	//initial configurations
	mat G;
	G<<1<<0<<0<<-1<<endr;
	mat L;
	L<<0.5<<0<<0<<-1<<0<<0<<0.5;
	field<mat> filters(4);
	filters(0)=G;
	filters(1)=G.t();
	filters(2)=L;
	filters(3)=L.t();
	SizeMat window=size(3,3);
	SizeMat overlap=size(2,2);
	int upscale_factor=3;
	//-----------------------------------------------------
	//get features
	mat features=Collect_features(img,filters,window*upscale_factor,overlap*upscale_factor);
	cout<<"all features size"<<size(features)<<endl;
	//--------------------------------------------------------
	//PCA
	mat pca;
	pca.load(".\\model_vars\\pca.data");
	features=pca.t()*features;
	cout<<"pca features size"<<size(features)<<endl;
	//---------------------------------------------------------

	/*cv::Mat mergedImage,recoveryImg;
	vector<cv::Mat> merged_channels;
	merged_channels.push_back(grayImg);
	merged_channels.push_back(cbImg);
	merged_channels.push_back(crImg);
	cv::merge(merged_channels,mergedImage);
	cvtColor(mergedImage,recoveryImg,cv::COLOR_YCrCb2RGB);
	cv::imshow("merged",recoveryImg);
	cvWaitKey();*/
}

int main(){
	//run_jor();

	//VL_PRINT("Hello World!\n");
	///*prepare training data*/
	////read file
	//clock_t start,end;double dur_sec;//计时
	//start=clock();
	//arma::Mat<float> lowpatches;
	//lowpatches.load(".\\model_vars\\lowpatches.data");
	//end=clock();
	//dur_sec=double(end-start)/CLOCKS_PER_SEC;
	//printf("read data file:%fs\n",dur_sec);

	//int feat_dims=lowpatches.n_rows;//特征维数
	//int feat_nums=lowpatches.n_cols;//特征数目

	///*prepare query data*/
	//int qry_num=10;//查询数量
	//arma::Mat<float> querymat(feat_dims,qry_num,fill::zeros);
	///*build the kdtree*/
	//start=clock();
	//VlKDForest* kdforest = vl_kdforest_new(VL_TYPE_FLOAT,feat_dims,1,VlDistanceL1);
	//vl_kdforest_build(kdforest,feat_nums,lowpatches.memptr());
	//end=clock();
	//dur_sec=double(end-start)/CLOCKS_PER_SEC;
	//printf("build tree runtime:%fs\n",dur_sec);

	///* Searcher object */
	//vl_kdforest_set_max_num_comparisons(kdforest,1500);
	//VlKDForestSearcher* searcher = vl_kdforest_new_searcher(kdforest);
	//const int NN_K=16;
	//umat knn_indexes(NN_K,qry_num);
	//VlKDForestNeighbor neighbours[NN_K];
	//for(int q=0;q<qry_num;q++){
	//	vl_kdforestsearcher_query(searcher,neighbours,NN_K,querymat.colptr(q));
	//	for(int k=0;k<NN_K;k++){
	//		knn_indexes(k,q)=neighbours[k].index;
	//	}
	//}
	//knn_indexes.print("knn_indexs");
	

	arma::Mat<float> lowpatches_labels_f;
	lowpatches_labels_f.load(".\\model_vars\\lowpatches_labels.data");

	umat lowpatches_labels=conv_to<umat>::from(lowpatches_labels_f);
	lowpatches_labels(span(0,99),0).print("lowpatches_labels");
	printf("finish!\n");

}