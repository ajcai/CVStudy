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

clock_t start,end;double dur_sec;//计时

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
static void Sample_grid(const SizeMat& img_size,const SizeMat& window,const SizeMat& overlap,uvec& grid_row,uvec& grid_col){//grid_row,grid_col为返回参数，表示网格左上角点
	//sample grid
	SizeMat skip=window-overlap;
	grid_row=regspace<uvec>(0,skip(0),img_size(0)-window(0));
	grid_col=regspace<uvec>(0,skip(1),img_size(1)-window(1));
}
static arma::mat Collect_features(const arma::mat& arma_mat_in,const field<mat>& filters,const SizeMat& window,const SizeMat& overlap){
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
	uvec grid_row; //=regspace<uvec>(0,skip(0),size(arma_mat_in,0)-window(0));
	uvec grid_col; //=regspace<uvec>(0,skip(1),size(arma_mat_in,1)-window(1));
	Sample_grid(size(arma_mat_in),window,overlap,grid_row,grid_col);
	int sample_row_num=grid_row.n_elem;
	int sample_col_num=grid_col.n_elem;
	int sample_patch_num=grid_row.n_elem*grid_col.n_elem;

	arma::mat features(patch_size*filters.n_elem,sample_patch_num);		//所有特征
	clock_t start,end;double dur_sec;//计时
	start=clock();
	parallel_for(0,sample_patch_num,[&](int i){
		vec patch_feature(window(0)*window(1)*filters.n_elem);
		for(int f=0;f<filters.n_elem;f++){//取对应patch的特征，将多个滤波结果合为一个特征
			arma::mat patch=feature_imgs(f)(i%sample_row_num,i/sample_row_num,window);
			patch_feature(span(patch_size*f,patch_size*(f+1)-1))=vectorise(patch);
		}
		features.col(i)=patch_feature;
	});
	end=clock();
	dur_sec=double(end-start)/CLOCKS_PER_SEC;
	printf("grid patches:%fs\n",dur_sec);
	return features;
}
static arma::mat Overlap_add(const arma::SizeMat& img_size,const mat& recovered_features,const SizeMat& window,const SizeMat& overlap){//特征矩阵转化为图像
	uvec grid_row; //=regspace<uvec>(0,skip(0),size(arma_mat_in,0)-window(0));
	uvec grid_col; //=regspace<uvec>(0,skip(1),size(arma_mat_in,1)-window(1));
	Sample_grid(img_size,window,overlap,grid_row,grid_col);
	int sample_row_num=grid_row.n_elem;
	int sample_col_num=grid_col.n_elem;
	int sample_patch_num=grid_row.n_elem*grid_col.n_elem;

	arma::mat overlap_img(img_size,fill::zeros);		//合成图像
	umat weight(img_size,fill::ones);
	clock_t start,end;double dur_sec;//计时
	start=clock();
	parallel_for(0,sample_patch_num,[&](int i){
		overlap_img(i%sample_row_num,i/sample_row_num,window)=reshape(recovered_features,window);
		weight(i%sample_row_num,i/sample_row_num,window)+=1;
	});
	overlap_img=overlap_img/weight;
	end=clock();
	dur_sec=double(end-start)/CLOCKS_PER_SEC;
	printf("overlap patches:%fs\n",dur_sec);
	return overlap_img;
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
	fmat pca;
	pca.load(".\\model_vars\\pca.data");
	clock_t start,end;double dur_sec;//计时
	start=clock();
	features=pca.t()*features;
	mat features_knn=normalise(features,1,0);
	end=clock();
	dur_sec=double(end-start)/CLOCKS_PER_SEC;
	printf("matrix multiply:%fs\n",dur_sec);
	cout<<"pca features size"<<size(features)<<endl;
	//---------------------------------------------------------


	//find knn
	//VL_PRINT("Hello World!\n");
	//prepare training data
	//read file

	start=clock();
	arma::Mat<float> lowpatches;
	lowpatches.load(".\\model_vars\\lowpatches.data");
	end=clock();
	dur_sec=double(end-start)/CLOCKS_PER_SEC;
	printf("load lowpatches data:%fs\n",dur_sec);

	int feat_dims=lowpatches.n_rows;//特征维数
	int feat_nums=lowpatches.n_cols;//特征数目

	////prepare query data
	//int qry_num=10;//查询数量
	//arma::Mat<float> querymat(feat_dims,qry_num,fill::zeros);
	//build the kdtree
	start=clock();
	VlKDForest* kdforest = vl_kdforest_new(VL_TYPE_FLOAT,feat_dims,1,VlDistanceL1);
	vl_kdforest_build(kdforest,feat_nums,lowpatches.memptr());
	end=clock();
	dur_sec=double(end-start)/CLOCKS_PER_SEC;
	printf("build tree runtime:%fs\n",dur_sec);

	// Searcher object 
	int qry_num=size(features_knn,1);
	vl_kdforest_set_max_num_comparisons(kdforest,1500);
	VlKDForestSearcher* searcher = vl_kdforest_new_searcher(kdforest);
	const int NN_K=1;
	umat knn_indexes(NN_K,qry_num);//k个最近邻的patch
	VlKDForestNeighbor neighbours[NN_K];
	for(int q=0;q<qry_num;q++){
		vl_kdforestsearcher_query(searcher,neighbours,NN_K,features_knn.colptr(q));
		for(int k=0;k<NN_K;k++){
			knn_indexes(k,q)=neighbours[k].index;
		}
	}

	cout<<"knn_indexs size"<<size(knn_indexes)<<endl;
	//-------------------------------------------------------
	//load labels
	arma::Mat<float> lowpatches_labels_f;
	lowpatches_labels_f.load(".\\model_vars\\lowpatches_labels.data");
	umat lowpatches_labels=conv_to<umat>::from(lowpatches_labels_f)-1;//need to be improved,the saved file should be uint type
	//find best reg
	umat knn_reg(size(knn_indexes));
	parallel_for(0,(int)knn_indexes.n_cols,[&](int i){
		knn_reg.col(i)=lowpatches_labels(knn_indexes.col(i));
	});
	cout<<"knn_reg"<<size(knn_reg)<<endl;
	//---------------------------------------------------------
	//load PPs
	field<mat> PPs;
	PPs.load(".\\model_vars\\PPs.data");
	//-------------------------------------------------------
	//恢复细节
	start=clock();
	mat recovered_features(size(PPs(0),0),size(features,1));
	cout<<"recovered_features"<<size(recovered_features)<<endl;
	parallel_for(0,(int)size(features,1),[&](int i){
		recovered_features.col(i)=PPs(knn_reg(0,i))*features.col(i);
	});
	end=clock();
	dur_sec=double(end-start)/CLOCKS_PER_SEC;
	printf("recover detail time:%fs\n",dur_sec);
	cout<<"recovered_features: mean"<<mean(mean(recovered_features))<<endl;
	recovered_features.col(1).print("recovered_features");

	mat detail_img=Overlap_add(size(img),recovered_features,window,overlap);
	cout<<"detail img size:"<<size(detail_img)<<endl;

	//cv::Mat_<double> cv_detail;
	//Arma_mat_to_cv_mat(detail_img,cv_detail);
	//cv::imshow("detail",cv_detail*255);
	//cvWaitKey();


	/*合并颜色空间，得到彩色图像
	cv::Mat mergedImage,recoveryImg;
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
	run_jor();
	
}