
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

	cv.discard_data();//�������Ƕ�����װ��vector��������ݲ�����Ȥ�����ذ����Ǵ�ϵͳ�ڴ渴�Ƶ��Դ�

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



