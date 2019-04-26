all : manky my

manky:
	nvcc -lm -std=c++11 man/main_cuda.cu man/lab3_cuda.cu man/lab3_io.cu -o man_pca

my: 
	 nvcc -lm -std=c++11 main_cuda.cu lab3_cuda.cu lab3_io.cu -o pca
