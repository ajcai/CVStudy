
#include <iostream>
#include <array>
#include <amp.h>
#include <amp_math.h>
using namespace std;
using namespace concurrency;
using namespace concurrency::fast_math;

void work_sin(){
	const int size =10;
	std::array<float,size> seq;
	generate(begin(seq),end(seq),[]
	{
		return (float)rand()/(float)RAND_MAX;
	});
	for(int i = 0;i<size;i++){
		std::cout<<seq[i]<<std::endl;
	}

	array_view<float,1> seq_view(size,seq);

	parallel_for_each(seq_view.extent,[=](index<1> idx) restrict(amp)
	{
		seq_view[idx] = sin(seq_view[idx]);
	});

	for(int i = 0;i<size;i++){
		std::cout<<seq_view[i]<<std::endl;
	}
}
vector<int> create_matrix(int);
void gpu_mattix_add(){
	const int rows = 100;
	const int columns = 100;
	vector<int> a = create_matrix(rows*columns);
	vector<int> b = create_matrix(rows*columns);
	vector<int> c(rows*columns);

	array_view<const int,2> av(rows,columns,a);
	array_view<const int,2> bv(rows,columns,b);
	array_view<int,2> cv(rows,columns,c);

	cv.discard_data();//表明我们对它包装的vector对象的数据不感兴趣，不必把它们从系统内存复制到显存

	parallel_for_each(cv.extent,[=](index<2> idx) restrict(amp){
		cv[idx] = av[idx] + bv[idx];
	});
	//cv.synchronize_async().then([=]{
		index<2> idx(14,12);
		std::cout << cv[idx]<<std::endl;
	//});
}
vector<int> create_matrix(int size){
	vector<int> matrix(size);
	generate(begin(matrix),end(matrix),[]{
		return rand()%100;
	});
	return matrix;
}
//int main(){
//	const int size=10000000;
//	vector<int> matrix(size);
//	clock_t start,end;double dur_sec;//计时
//	//---------------------------------------------
//	start=clock();
//	for(int idx=0;idx<size;idx++){
//		matrix[idx]=idx;
//	}
//	end=clock();
//	dur_sec=double(end-start)/CLOCKS_PER_SEC;
//	printf("CPU串行用时:%fs\n",dur_sec);
//	//----------------------------------------------
//	start=clock();
//	parallel_for(0,size,[&matrix](int idx){
//		matrix[idx]=idx;
//	});
//	end=clock();
//	dur_sec=double(end-start)/CLOCKS_PER_SEC;
//	printf("CPU并行用时:%fs\n",dur_sec);
//	//----------------------------------------------
//	start=clock();
//	array_view<int,1> am(size,matrix);
//	am.discard_data();
//	parallel_for_each(am.extent,[=](index<1> idx) restrict(amp){
//		am[idx] = idx.rank;
//	});
//	end=clock();
//	dur_sec=double(end-start)/CLOCKS_PER_SEC;
//	printf("GPU并行用时:%fs\n",dur_sec);
//
//
//	for (int i=0; i<size; i++){  
//		index<1> idx(i);
//		std::cout << am[idx]<<" ";
//	 }  
//	//for(int i=0;i<size;i++){
//	//	cout<<matrix[i]<<" ";
//	//}
//
//
//	return 0;
//}


