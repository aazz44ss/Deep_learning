#include <iostream>
#include <random>
#include <algorithm>
#include <string>
#include <time.h>
using namespace std;

#include "hdf5.h"


template <typename Dtype>
class Blob {
public:
	Dtype **data;
	int shape[4];
	int count_with_pad;
	int pad[2];
	int stride[2];

	int mini_batch;
	Blob(int m) {
		mini_batch = m;
		data = new Dtype*[mini_batch];
	}

	void get_shuffled_set(Blob<Dtype> &All, int *shuffle_index, int current_index, int total) {

		if (current_index + mini_batch >= total) {
			shape[0] = total - current_index;
		}
		else {
			shape[0] = mini_batch;
		}
		shape[1] = All.shape[1];
		shape[2] = All.shape[2];
		shape[3] = All.shape[3];
		count_with_pad = All.count_with_pad;
		pad[0] = All.pad[0];
		pad[1] = All.pad[1];
		stride[0] = All.stride[0];
		stride[1] = All.stride[1];

		for (int i = 0; i < shape[0]; i++) {
			data[i] = All.data[shuffle_index[current_index + i]];
		}
	}

	Blob() {};
	Blob(int m, int Channel, int Height, int Width,int stride_Height=1, int stride_Width=1, int pad_Height = 0, int pad_Width = 0) {
		pad[0] = pad_Height;
		pad[1] = pad_Width;
		stride[0] = stride_Height;
		stride[1] = stride_Width;
		shape[0] = m;
		shape[1] = Channel;
		shape[2] = Height + 2*pad_Height;
		shape[3] = Width + 2*pad_Width;
		
		data = new Dtype*[m];

		count_with_pad = (Height + 2 * pad_Height)*(Width + 2 * pad_Width)*Channel;

		for (int i = 0; i < m; i++) {
			data[i] = new Dtype[count_with_pad];
			for (int j = 0; j < count_with_pad; j++) {
				data[i][j] = 0;}}
	}

	void init() {
		for (int i = 0; i < shape[0]; i++) {
			for (int j = 0; j < count_with_pad; j++) {
				data[i][j] = 0;}}
	}
};

template <typename Dtype>
class Blob_maxPool: public Blob<Dtype> {
public:
	int pool[2];
	int **maxPool_cache; // cache the maximum value index of Z_prev
	Blob_maxPool(int m, int Channel, int Height, int Width, int pool_h, int pool_w, int stride_h = 1, int stride_w = 1, int pad_Height = 0, int pad_Width = 0): Blob<Dtype>(m,Channel,Height,Width, stride_h, stride_w,pad_Height,pad_Width){
		pool[0] = pool_h;
		pool[1] = pool_w;
		maxPool_cache = new int*[m];

		for (int i = 0; i < m; i++) {
			maxPool_cache[i] = new int[Blob<Dtype>::count_with_pad];
		}

	}
};

template <typename Dtype>
void convert_to_one_hot(Blob<Dtype> &Y_one_hot, int *Y, int index) {
	int m = Y_one_hot.shape[0];
	for (int i = 0; i<m; i++) {
		Y_one_hot.data[i][Y[i]] = 1;}
}

template <typename Dtype>
class parameter {
public:
	Dtype *W;
	Dtype *b;
	int shape[4];
	parameter(int dims, int prev_dims, int fH, int fW) {
		shape[0] = dims;
		shape[1] = prev_dims;
		shape[2] = fH;
		shape[3] = fW;
		
		W = new Dtype[dims*prev_dims*fH*fW];
		b = new Dtype[dims];
		for (int i = 0; i < shape[0]; i++) {
			for (int j = 0; j < shape[1]; j++) {
				for (int k = 0; k < shape[2]; k++) {
					for (int l = 0; l < shape[3]; l++) {
						W[((i * shape[1] + j)*shape[2] + k)*shape[3] + l] = ((Dtype)rand() / (RAND_MAX) * 2 - 1)*sqrt(2.0/shape[1]/shape[2]/shape[3]);}}}
			b[i] = 0;}
	}
};

template <typename Dtype>
class gradient {
public:
	Dtype *dW, *vW, *sW;
	Dtype *db, *vb, *sb;
	Dtype beta1, beta2, lambd;
	int shape[4];

	gradient(int dims, int prev_dims, int fH, int fW) {
		shape[0] = dims;
		shape[1] = prev_dims;
		shape[2] = fH;
		shape[3] = fW;
		
		beta1 = 0.9;
		beta2 = 0.999;
		lambd = 0;

		dW = new Dtype[dims*prev_dims*fH*fW];
		vW = new Dtype[dims*prev_dims*fH*fW];
		sW = new Dtype[dims*prev_dims*fH*fW];
		db = new Dtype[dims];
		vb = new Dtype[dims];
		sb = new Dtype[dims];
		for (int i = 0; i < shape[0]; i++) {
			for (int j = 0; j < shape[1]; j++) {
				for (int k = 0; k < shape[2]; k++) {
					for (int l = 0; l < shape[3]; l++) {
						dW[((i * shape[1] + j)*shape[2] + k)*shape[3] + l] = 0;
						vW[((i * shape[1] + j)*shape[2] + k)*shape[3] + l] = 0;
						sW[((i * shape[1] + j)*shape[2] + k)*shape[3] + l] = 0;}}}
			db[i] = 0;
			vb[i] = 0;
			sb[i] = 0;}
	}

	void init() {
		for (int i = 0; i < shape[0]; i++) {
			for (int j = 0; j < shape[1]; j++) {
				for (int k = 0; k < shape[2]; k++) {
					for (int l = 0; l < shape[3]; l++) {
						dW[((i * shape[1] + j)*shape[2] + k)*shape[3] + l] = 0;}}}
			db[i] = 0;}
	}
};

template <typename Dtype>
void softmax(int m, Blob<Dtype> &Z_prev) {
	int index = Z_prev.shape[1];;
	Dtype temp;
	for (int i = 0; i<m; i++) {
		temp = 0;
		for (int j = 0; j<index; j++) {
			Z_prev.data[i][j] = exp(Z_prev.data[i][j]);
			temp += Z_prev.data[i][j];}
		for (int j = 0; j<index; j++) {
			Z_prev.data[i][j] /= temp;}
	}
}

template <typename Dtype>
void softmax_cost_backward(int m, Blob<Dtype> &Z, Blob<int> &Y) {
	for (int i = 0; i<m; i++) {
		Z.data[i][Y.data[i][0]] -= 1;}
}

template <typename Dtype>
void update_parameter(parameter<Dtype> &W, gradient<Dtype> &dW, int t, Dtype learning_rate) {
	int dims = W.shape[0];
	int prev_dims = W.shape[1];
	int f_Height = W.shape[2];
	int f_Width = W.shape[3];
	
	Dtype beta1 = dW.beta1;
	Dtype beta2 = dW.beta2;
	Dtype corrected;
	corrected = (1 - pow(beta2, t)) / (1 - pow(beta1, t));

	int p, h, index;
	for (int i = 0; i < dims; i++) {
		p = i*prev_dims;
		for (int j = 0; j < prev_dims; j++) {
			h = (p + j)*f_Height;
			for (int k = 0; k < f_Height; k++) {
				index = (h + k)*f_Width;
				for (int l = 0; l < f_Width; l++) {
					dW.vW[index + l] = beta1 * dW.vW[index + l] + (1 - beta1)*dW.dW[index + l];
					dW.sW[index + l] = beta2 * dW.sW[index + l] + (1 - beta2)*pow(dW.dW[index + l], 2);
					W.W[index + l] = W.W[index + l] - learning_rate * corrected * dW.vW[index + l] / (sqrt(dW.sW[index + l]) + 0.0000001);
				}
			}
		}
	}

	for (int i = 0; i < dims; i++) {
		dW.vb[i] = beta1 * dW.vb[i] + (1 - beta1)*dW.db[i];
		dW.sb[i] = beta2 * dW.sb[i] + (1 - beta2)*pow(dW.db[i], 2);
		W.b[i] = W.b[i] - learning_rate * corrected * dW.vb[i] / (sqrt(dW.sb[i]) + 0.0000001);
	}

}

template <typename Dtype>
Dtype cost_function(int m, Blob<Dtype> &Z, Blob<int> &Y) {
	Dtype cost = 0;

	for (int i = 0; i < m; i++) {
		cost += -log(Z.data[i][Y.data[i][0]]+0.0000001);
	}
	cost /= m;
	return cost;
}

template <typename Dtype>
void Convolution(int m, Blob<Dtype> &Z, parameter<Dtype> &W, Blob<Dtype> &Z_prev) {
	int prev_dims = Z_prev.shape[1];
	int prev_Height_pad = Z_prev.shape[2];
	int prev_Width_pad = Z_prev.shape[3];
	int Kernel = W.shape[0];
	int f_Height = W.shape[2];
	int f_Width = W.shape[3];
	int Height_pad = Z.shape[2];
	int Width_pad = Z.shape[3];
	int pad_h = Z.pad[0];
	int pad_w = Z.pad[1];
	int stride_h = Z.stride[0];
	int stride_w = Z.stride[1];
	int fk_start,fd_start,fh_start;
	int prev_h_start, prev_w_start, prev_slice_c_start, prev_slice_h_start;
	int k_start, h_start;
	Dtype Z_data_cache;

	// CHECK
	if (Height_pad - 2 * pad_h != ((prev_Height_pad - f_Height) / stride_h + 1)) {
		printf("Conv layer doesn't match height \n");
		printf("Z_prev_h_pad:%d, stride:%d \n", prev_Height_pad, stride_h);
		printf("Z_h_pad:%d, pad_h:%d \n", Height_pad, pad_h);
		printf("Filter_Height:%d \n\n", f_Height);
	}
	if (Width_pad - 2 * pad_w != ((prev_Width_pad - f_Width) / stride_w + 1)) {
		printf("Conv layer doesn't match Width \n");
		printf("Z_prev_w_pad:%d, stride:%d \n", prev_Width_pad, stride_w);
		printf("Z_w_pad:%d, pad_w:%d \n", Width_pad, pad_w);
		printf("Filter_Width:%d \n\n", f_Width);
	}

	//compute Z = b first can have Z.init() += b result
	//Z.init();
	for (int i = 0; i < m; i++) {
		for (int k = 0; k < Kernel; k++){
			k_start = k * Height_pad * Width_pad;
			for (int h = pad_h; h < Height_pad-pad_h; h++) {
				h_start = k_start + h * Width_pad;
				for (int w = pad_w; w < Width_pad-pad_w; w++) {
					Z.data[i][h_start + w] = W.b[k];
				}
			}
		}
	}

	for (int i = 0; i < m; i++) {
		for (int k = 0; k < Kernel; k++) {
			fk_start = k*prev_dims*f_Height*f_Width;
			k_start = k * Height_pad * Width_pad;
			for (int h = pad_h; h < Height_pad-pad_h; h++) {
				h_start = k_start + h * Width_pad;
				prev_h_start = (h-pad_h) * stride_h * prev_Width_pad;
				for (int w = pad_w; w < Width_pad-pad_w; w++) {
					Z_data_cache = 0;
					prev_w_start = prev_h_start + (w-pad_w) * stride_w;
					for (int c = 0; c < prev_dims; c++) {
						fd_start = fk_start + c*f_Height*f_Width;
						prev_slice_c_start = prev_w_start + c * prev_Height_pad * prev_Width_pad;
						for (int fH = 0; fH < f_Height; fH++) {
							fh_start = fd_start + fH * f_Width;
							prev_slice_h_start = prev_slice_c_start + fH * prev_Width_pad;
							for (int fW = 0; fW < f_Width; fW++) {
								Z_data_cache += W.W[fh_start + fW] * Z_prev.data[i][prev_slice_h_start + fW];
							}
						}
					}
					Z.data[i][h_start + w] += Z_data_cache;
				}
			}
		}
	}

}

template <typename Dtype>
void Convolution_backward(int m, Blob<Dtype> &dZ_prev, Blob<Dtype> &Z_prev, Blob<Dtype> &dZ, parameter<Dtype> &W, gradient<Dtype> &dW, int t) {
	int prev_dims = Z_prev.shape[1];
	int prev_Height_pad = Z_prev.shape[2];
	int prev_Width_pad = Z_prev.shape[3];
	int Height_pad = dZ.shape[2];
	int Width_pad = dZ.shape[3];
	int Kernel = W.shape[0];
	int f_Height = W.shape[2];
	int f_Width = W.shape[3];
	int pad_h = dZ.pad[0];
	int pad_w = dZ.pad[1];
	int stride_h = dZ.stride[0];
	int stride_w = dZ.stride[1];

	int fk_start,fd_start,fh_start;
	int prev_h_start, prev_w_start, prev_slice_c_start, prev_slice_h_start;
	int k_start, h_start;

	Dtype dZ_data_cache;
	Dtype lambd = dW.lambd;

	dW.init();
	dZ_prev.init();
	for (int i = 0; i < m; i++) {
		for (int k = 0; k < Kernel; k++) {
			k_start = k * Height_pad * Width_pad;
			fk_start = k*prev_dims*f_Height*f_Width;
			for (int h = pad_h; h < Height_pad-pad_h; h++) {
				h_start = k_start + h * Width_pad;
				prev_h_start = (h-pad_h) * stride_h * prev_Width_pad;
				for (int w = pad_w; w < Width_pad-pad_w; w++) {
					prev_w_start = prev_h_start + (w-pad_w) * stride_w;
					dZ_data_cache = dZ.data[i][h_start + w];
					for (int c = 0; c < prev_dims; c++) {
						fd_start = fk_start + c*f_Height*f_Width;
						prev_slice_c_start = prev_w_start + c * prev_Height_pad * prev_Width_pad;
						for (int fH = 0; fH < f_Height; fH++) {
							fh_start = fd_start + fH * f_Width;
							prev_slice_h_start = prev_slice_c_start + fH * prev_Width_pad;
							for (int fW = 0; fW < f_Width; fW++) {
								dW.dW[fh_start + fW] += (dZ_data_cache * Z_prev.data[i][prev_slice_h_start + fW] + lambd * W.W[fh_start + fW]) / m;
								dZ_prev.data[i][prev_slice_h_start + fW] += W.W[fh_start + fW] * dZ_data_cache;
							}
						}
					}
					dW.db[k] += dZ_data_cache /m;
				}
			}
		}
	}
	update_parameter(W, dW, t, 0.001);
}

template <typename Dtype>
void maxPool(int m, Blob_maxPool<Dtype> &Z, Blob<Dtype> &Z_prev) {
	int prev_dims = Z_prev.shape[1];
	int prev_Height_pad = Z_prev.shape[2];
	int prev_Width_pad = Z_prev.shape[3];

	int prev_pad_h = Z_prev.pad[0];
	int prev_pad_w = Z_prev.pad[1];

	int Height_pad = Z.shape[2];
	int Width_pad = Z.shape[3];
	int pad_h = Z.pad[0];
	int pad_w = Z.pad[1];
	int pool_h = Z.pool[0];
	int pool_w = Z.pool[1];
	int stride_h = Z.stride[0];
	int stride_w = Z.stride[1];
	int prev_c_start, prev_h_start, prev_w_start, prev_slice_h_start;
	int c_start, h_start;
	Dtype temp_prev_slice_max;


	// CHECK
	if (Height_pad - 2 * pad_h != ((prev_Height_pad - pool_h) / stride_h + 1)) {
		printf("Pool layer doesn't match height \n");
		printf("Z_prev_h_pad:%d, prev_pad_h:%d, stride_h:%d \n", prev_Height_pad, prev_pad_h, stride_h);
		printf("Z_h_pad:%d, pad_h:%d \n\n", Height_pad, pad_h);
	}
	if (Width_pad - 2 * pad_w != ((prev_Width_pad - pool_w) / stride_w + 1)) {
		printf("Pool layer doesn't match Width \n");
		printf("Z_prev_w_pad:%d, prev_pad_w:%d, stride_w:%d \n", prev_Width_pad, prev_pad_w, stride_w);
		printf("Z_w_pad:%d, pad_w:%d \n\n", Width_pad, pad_w);
	}


	for (int i = 0; i < m; i++) {
		for (int c = 0; c < prev_dims; c++) {
			c_start = c * Height_pad * Width_pad;
			prev_c_start = c * prev_Height_pad * prev_Width_pad;
			for (int h = pad_h; h < Height_pad - pad_h; h++) {
				h_start = c_start + h * Width_pad;
				prev_h_start = prev_c_start + (h - pad_h) * stride_h * prev_Width_pad;
				for (int w = pad_w; w < Width_pad - pad_w; w++) {
					prev_w_start = prev_h_start + (w - pad_w) * stride_w;
					temp_prev_slice_max = -INFINITY;
					for (int fH = 0; fH < pool_h; fH++) {
						prev_slice_h_start = prev_w_start + fH * prev_Width_pad;
						for (int fW = 0; fW < pool_w; fW++) {
							if (Z_prev.data[i][prev_slice_h_start + fW] > temp_prev_slice_max) {
								temp_prev_slice_max = Z_prev.data[i][prev_slice_h_start + fW];
							}
						}
					}
					Z.data[i][h_start + w] = temp_prev_slice_max;
				}
			}
		}
	}
}

template <typename Dtype>
void maxPool2(int m, Blob_maxPool<Dtype> &Z, Blob<Dtype> &Z_prev) {
	int prev_dims = Z_prev.shape[1];
	int prev_Height_pad = Z_prev.shape[2];
	int prev_Width_pad = Z_prev.shape[3];

	int prev_pad_h = Z_prev.pad[0];
	int prev_pad_w = Z_prev.pad[1];

	int Height_pad = Z.shape[2];
	int Width_pad = Z.shape[3];
	int pad_h = Z.pad[0];
	int pad_w = Z.pad[1];
	int pool_h = Z.pool[0];
	int pool_w = Z.pool[1];
	int stride_h = Z.stride[0];
	int stride_w = Z.stride[1];
	int prev_c_start, prev_h_start, prev_w_start, prev_slice_h_start;
	int c_start, h_start;
	Dtype temp_prev_slice_max;
	int temp_prev_slice_max_index;

	// CHECK
	if (Height_pad - 2 * pad_h != ((prev_Height_pad - pool_h) / stride_h + 1)) {
		printf("Pool layer doesn't match height \n");
		printf("Z_prev_h_pad:%d, prev_pad_h:%d, stride_h:%d \n", prev_Height_pad, prev_pad_h, stride_h);
		printf("Z_h_pad:%d, pad_h:%d \n\n", Height_pad, pad_h);
	}
	if (Width_pad - 2 * pad_w != ((prev_Width_pad - pool_w) / stride_w + 1)) {
		printf("Pool layer doesn't match Width \n");
		printf("Z_prev_w_pad:%d, prev_pad_w:%d, stride_w:%d \n", prev_Width_pad, prev_pad_w, stride_w);
		printf("Z_w_pad:%d, pad_w:%d \n\n", Width_pad, pad_w);
	}


	for (int i = 0; i < m; i++) {
		for (int c = 0; c < prev_dims; c++) {
			prev_c_start = c * prev_Height_pad * prev_Width_pad;
			c_start = c * Height_pad * Width_pad;
			for (int h = pad_h; h < Height_pad - pad_h; h++) {
				prev_h_start = prev_c_start + (h - pad_h) * stride_h * prev_Width_pad;
				h_start = c_start + h * Width_pad;
				for (int w = pad_w; w < Width_pad - pad_w; w++) {
					prev_w_start = prev_h_start + (w - pad_w) * stride_w;
					temp_prev_slice_max = -INFINITY;
					for (int fH = 0; fH < pool_h; fH++) {
						prev_slice_h_start = prev_w_start + fH * prev_Width_pad;
						for (int fW = 0; fW < pool_w; fW++) {
							if (Z_prev.data[i][prev_slice_h_start + fW] > temp_prev_slice_max) {
								temp_prev_slice_max = Z_prev.data[i][prev_slice_h_start + fW];
								temp_prev_slice_max_index = prev_slice_h_start + fW;  //cache the index of max value, that can be used when computing backward propagation
							}
						}
					}
					Z.data[i][h_start + w] = temp_prev_slice_max;
					Z.maxPool_cache[i][h_start + w] = temp_prev_slice_max_index;
				}
			}
		}
	}
}

template <typename Dtype>
void maxPool_backward(int m, Blob<Dtype> &dZ_prev, Blob<Dtype> &Z_prev, Blob_maxPool<Dtype> &dZ) {
	int prev_dims = dZ_prev.shape[1];
	int prev_Height_pad = dZ_prev.shape[2];
	int prev_Width_pad = dZ_prev.shape[3];

	int prev_pad_h = dZ_prev.pad[0];
	int prev_pad_w = dZ_prev.pad[1];

	int Height_pad = dZ.shape[2];
	int Width_pad = dZ.shape[3];
	int pad_h = dZ.pad[0];
	int pad_w = dZ.pad[1];
	int pool_h = dZ.pool[0];
	int pool_w = dZ.pool[1];
	int stride_h = dZ.stride[0];
	int stride_w = dZ.stride[1];
	int prev_c_start, prev_h_start, prev_w_start, prev_slice_h_start;
	int c_start, h_start;
	Dtype temp_prev_slice_max;
	int Z_prev_index;

	dZ_prev.init();
	for (int i = 0; i < m; i++) {
		for (int c = 0; c < prev_dims; c++) {
			c_start = c * Height_pad * Width_pad;
			prev_c_start = c * prev_Height_pad * prev_Width_pad;
			for (int h = pad_h; h < Height_pad - pad_h; h++) {
				h_start = c_start + h * Width_pad;
				prev_h_start = prev_c_start + (h-pad_h) * stride_h * prev_Width_pad;
				for (int w = pad_w; w < Width_pad - pad_w; w++) {
					prev_w_start = prev_h_start + (w-pad_w) * stride_w;
					temp_prev_slice_max = -INFINITY;
					for (int fH = 0; fH < pool_h; fH++) {
						prev_slice_h_start = prev_w_start + fH * prev_Width_pad;
						for (int fW = 0; fW < pool_w; fW++) {
							if (Z_prev.data[i][prev_slice_h_start + fW] > temp_prev_slice_max) {
								temp_prev_slice_max = Z_prev.data[i][prev_slice_h_start + fW];
								Z_prev_index = prev_slice_h_start + fW;
							}
						}
					}
					dZ_prev.data[i][Z_prev_index] += dZ.data[i][h_start + w];
				}
			}
		}
	}
}

template <typename Dtype>
void maxPool_backward2(int m, Blob<Dtype> &dZ_prev, Blob_maxPool<Dtype> &Z, Blob_maxPool<Dtype> &dZ) {
	int prev_dims = dZ_prev.shape[1];
	int Height_pad = dZ.shape[2];
	int Width_pad = dZ.shape[3];
	int pad_h = dZ.pad[0];
	int pad_w = dZ.pad[1];
	int c_start, h_start;

	dZ_prev.init();
	for (int i = 0; i < m; i++) {
		for (int c = 0; c < prev_dims; c++) {
			c_start = c * Height_pad * Width_pad;
			for (int h = pad_h; h < Height_pad - pad_h; h++) {
				h_start = c_start + h * Width_pad;
				for (int w = pad_w; w < Width_pad - pad_w; w++) {
					dZ_prev.data[i][ Z.maxPool_cache[i][h_start + w] ] += dZ.data[i][h_start + w];
				}
			}
		}
	}
}

template <typename Dtype>
void FC_layer(int m, Blob<Dtype> &Z, parameter<Dtype> &W, Blob<Dtype> &Z_prev) {
	int dims = W.shape[0];
	int prev_dims = W.shape[1];
	int W_index_cache;
	Dtype Z_data_cache;

	//CHECK
	if (prev_dims != Z_prev.count_with_pad) {
		printf("FC_layer has different neuron with Z_prev.shape=[%d,%d,%d,%d] neuraon:%d, W.shape=[%d,%d,%d] neuron:%d\n", Z_prev.shape[0], Z_prev.shape[1], Z_prev.shape[2], Z_prev.shape[3], Z_prev.count_with_pad, W.shape[0], W.shape[1], W.shape[2], prev_dims);
	}
	if (dims != Z.count_with_pad) {
		printf("FC_layer has different neuron with Z.shape=[%d,%d,%d,%d] neuraon:%d, W dims:%d\n", Z.shape[0], Z.shape[1], Z.shape[2], Z.shape[3], Z.count_with_pad, dims);
	}

	//compute Z = b first can have Z.init() += b result
	Z.init();
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < dims; j++) {
			Z.data[i][j] = W.b[j];}}
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < dims; j++) {
			W_index_cache = j*prev_dims;
			Z_data_cache = 0;
			for (int k = 0; k < prev_dims; k++) {
				Z_data_cache += Z_prev.data[i][k] * W.W[W_index_cache + k];
			}
			Z.data[i][j] += Z_data_cache;
		}
	}

}

template <typename Dtype>
void FC_layer_backward(int m, Blob<Dtype> &dZ_prev, Blob<Dtype> &Z_prev, Blob<Dtype> &dZ, parameter<Dtype> &W, gradient<Dtype> &dW, int t) {
	int dims = W.shape[0];
	int prev_dims = W.shape[1];
	Dtype lambd = dW.lambd;
	Dtype dZ_data_cache;
	int W_index_cache;

	if (prev_dims != Z_prev.count_with_pad) {
		printf("FC_layer_backward has different neuron with Z_prev.shape=[%d,%d,%d,%d] neuraon:%d, W.shape=[%d,%d,%d,%d] \n", Z_prev.shape[0], Z_prev.shape[1], Z_prev.shape[2], Z_prev.shape[3], Z_prev.count_with_pad, W.shape[0], W.shape[1], W.shape[2], W.shape[3]);
	}
	if (dims != dZ.count_with_pad) {
		printf("FC_layer_backward has different neuron with dZ.shape=[%d,%d,%d,%d] neuraon:%d, W.shape=[%d,%d,%d,%d] \n", dZ.shape[0], dZ.shape[1], dZ.shape[2], dZ.shape[3], dZ.count_with_pad, W.shape[0], W.shape[1], W.shape[2], W.shape[3]);
	}

	// compute dZ_prev
	dZ_prev.init();// every time when use +=... and with m>1 size, need to be init to 0;
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < dims; j++) {
			W_index_cache = j*prev_dims;
			dZ_data_cache = dZ.data[i][j];
			for (int k = 0; k < prev_dims; k++) {
				dZ_prev.data[i][k] += W.W[W_index_cache + k] * dZ_data_cache;
			}
		}
	}

	// compute dW, db
	dW.init();
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < dims; j++) {
			dW.db[j] += dZ.data[i][j] / m;
		}
	}
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < dims; j++) {
			W_index_cache = j*prev_dims;
			dZ_data_cache = dZ.data[i][j];
			for (int k = 0; k < prev_dims; k++) {
				dW.dW[W_index_cache + k] += (dZ_data_cache * Z_prev.data[i][k] + lambd * W.W[W_index_cache + k]) / m;
			}
		}
	}

	update_parameter(W, dW, t, 0.001);

}

template <typename Dtype>
void relu(int m, Blob<Dtype> &Z_prev) {
	int neuron = Z_prev.count_with_pad;

	for (int i = 0; i < m; i++) {
		for (int j = 0; j < neuron; j++) {
			if (Z_prev.data[i][j] < 0)
				Z_prev.data[i][j] = 0;
		}
	}
}

template <typename Dtype>
void relu_backward(int m, Blob<Dtype> &dZ_prev, Blob<Dtype> &Z_prev) {

	int prev_dims = Z_prev.shape[1];
	int prev_Height_pad = Z_prev.shape[2];
	int prev_Width_pad = Z_prev.shape[3];
	int prev_pad_h = Z_prev.pad[0];
	int prev_pad_w = Z_prev.pad[1];
	int Height_pad = dZ_prev.shape[2];
	int Width_pad = dZ_prev.shape[3];
	int pad_h = dZ_prev.pad[0];
	int pad_w = dZ_prev.pad[1];

	for (int i = 0; i < m; i++) {
		for (int c = 0; c < prev_dims; c++) {
			for (int h = prev_pad_h; h < prev_Height_pad - prev_pad_h; h++) {
				for (int w = prev_pad_w; w < prev_Width_pad - prev_pad_w; w++) {
					if (Z_prev.data[i][(c*prev_Height_pad + h)*prev_Width_pad + w] <= 0) {
						dZ_prev.data[i][(c*prev_Height_pad + h)*prev_Width_pad + w] = 0;
					}
				}
			}
		}
	}
}

template <typename Dtype>
Dtype compute_accuracy(int m, Blob<Dtype> &Z, Blob<int> &Y) {
	int index = Z.shape[1];
	int maxIndex;
	Dtype temp, accurate = 0;
	for (int i = 0; i < m; i++) {
		temp = 0;
		for (int j = 0; j < index; j++) {
			if (Z.data[i][j] >= temp) {
				maxIndex = j;
				temp = Z.data[i][j];}}
		if (Y.data[i][0] == maxIndex) {
			accurate++;}
	}
	return accurate / m;
}

template <typename Dtype>
void read_HDF5_4D(Blob<Dtype> *&container, std::string file_path, std::string data_name, Dtype normalize, int stride_h = 1, int stride_w = 1, int pad_h = 0, int pad_w = 0) {
	hid_t file_id, dataset_id, dataspace;
	int status;
	file_id = H5Fopen(file_path.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
	dataset_id = H5Dopen2(file_id, data_name.c_str(), H5P_DEFAULT);
	dataspace = H5Dget_space(dataset_id);

	hsize_t dim[4];              /* memory space dimensions */
	for (int i = 0; i < 4; i++) {
		dim[i] = 1;
	}
	status = H5Sget_simple_extent_dims(dataspace, dim, NULL);

	int total_size = 1;
	for (int i = 0; i < 4; i++) {
		total_size *= dim[i];
	}
	int *data_flatten;
	data_flatten = new int[total_size];

	status = H5Dread(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
		data_flatten);

	container = new Blob<Dtype>(dim[0], dim[3], dim[1], dim[2], stride_h, stride_w, pad_h, pad_w);

	//convert to (m,c,h,w)
	for (int i = 0; i < dim[0]; i++) {
		for (int h = 0; h < dim[1]; h++) {
			for (int w = 0; w < dim[2]; w++) {
				for (int c = 0; c < dim[3]; c++) {
					container->data[i][(c*(dim[1] + 2 * pad_h) + h + pad_h)*(dim[2] + 2 * pad_w) + w + pad_w] = (data_flatten[((i*dim[1] + h)*dim[2] + w)*dim[3] + c]) / normalize;
				}
			}
		}
	}

	/* Close the dataset. */
	status = H5Dclose(dataset_id);
	/* Close the file. */
	status = H5Fclose(file_id);
	/* Close the dataspace. */
	status = H5Sclose(dataspace);
	delete data_flatten;
}

int main() {

	Blob<double> *X,*X_test;
	Blob<int> *Y,*Y_test;
	read_HDF5_4D<double>(X, "..\\data\\signs_train.h5", "train_set_x",/*normalize*/255,/*stride*/1, 1,/*Pad*/2,2);
	read_HDF5_4D<int>(Y, "..\\data\\signs_train.h5", "train_set_y",1);
	read_HDF5_4D<double>(X_test, "..\\data\\signs_test.h5", "test_set_x", 255,0,0);
	read_HDF5_4D<int>(Y_test, "..\\data\\signs_test.h5", "test_set_y", 1);

	int m;
	double mini_cost,accuracy;
	int *shuffle_index;
	m = X->shape[0];
	shuffle_index = new int[m];
	for (int i = 0; i < m; i++) {
		shuffle_index[i] = i;}
	
	int mini_batch = 256, batch_count;
	batch_count = ceil((double)m / (double)mini_batch);

	// shuffle data container
	Blob<int> *mini_Y = NULL;
	mini_Y = new Blob<int>(mini_batch);
	Blob<double> *Z0 = NULL;
	Blob<double> *dZ0 = NULL;
	Z0  = new Blob<double>(mini_batch);
	dZ0 = new Blob<double>(mini_batch, 3, 64, 64, 2, 2);

	// Convolution layer
	Blob<double> *Z1_conv = NULL, *Z2_conv = NULL;
	Blob_maxPool<double> *Z1_pool = NULL, *Z2_pool = NULL;
	Blob<double> *dZ1_conv = NULL, *dZ2_conv = NULL;
	Blob_maxPool<double> *dZ1_pool = NULL, *dZ2_pool = NULL;
	Z1_conv  = new Blob<double>(mini_batch, 8, 64, 64); /*((batch, channel,height,width, stride_h=1,stride_w=1, pad_h=0,pad_w=0))*/
	dZ1_conv = new Blob<double>(mini_batch, 8, 64, 64);
	Z1_pool  = new Blob_maxPool<double>(mini_batch,8,16,16, 4,4, 4,4, 1,1);
	dZ1_pool = new Blob_maxPool<double>(mini_batch,8,16,16, 4,4, 4,4, 1,1);
	Z2_conv  = new Blob<double>(mini_batch, 16, 16, 16);
	dZ2_conv = new Blob<double>(mini_batch, 16, 16, 16);
	Z2_pool  = new Blob_maxPool<double>(mini_batch,16,8,8, 2,2, 2,2);
	dZ2_pool = new Blob_maxPool<double>(mini_batch,16,8,8, 2,2, 2,2);

	parameter<double> *W1_conv = NULL, *W2_conv = NULL;
	gradient<double> *dW1_conv = NULL, *dW2_conv = NULL;
	W1_conv = new parameter<double>(8, 3, 5, 5);
	dW1_conv = new gradient<double>(8, 3, 5, 5);
	W2_conv = new parameter<double>(16, 8, 3, 3);
	dW2_conv = new gradient<double>(16, 8, 3, 3);

	// FC layer
	Blob<double> *Z3 = NULL, *Z4 = NULL, *Z5 = NULL;
	Blob<double> *dZ3 = NULL, *dZ4 = NULL, *dZ5 = NULL;
	Z3  = new Blob<double>(mini_batch, 300, 1, 1);
	dZ3 = new Blob<double>(mini_batch, 300, 1, 1);
	Z4  = new Blob<double>(mini_batch, 60, 1, 1);
	dZ4 = new Blob<double>(mini_batch, 60, 1, 1);
	Z5  = new Blob<double>(mini_batch, 6, 1, 1);
	dZ5 = new Blob<double>(mini_batch, 6, 1, 1);

	parameter<double> *W3 = NULL, *W4 = NULL, *W5 = NULL;
	gradient<double> *dW3 = NULL, *dW4 = NULL, *dW5 = NULL;
	W3 = new parameter<double>(300, 1024, 1, 1);
	dW3 = new gradient<double>(300, 1024, 1, 1);
	W4 = new parameter<double>(60, 300, 1, 1);
	dW4 = new gradient<double>(60, 300, 1, 1);
	W5 = new parameter<double>(6, 60, 1, 1);
	dW5 = new gradient<double>(6, 60, 1, 1);

	

	int t=0;
	int current_batch;

	double timer=0;
	clock_t tic, toc;
	tic = clock();

	for (int i = 0; i < 10000; i++) {
		mini_cost = 0;
		random_shuffle(&shuffle_index[0], &shuffle_index[m - 1]);
		for (int j=0; j<m; j+=mini_batch) {
			// data prepare
			t++; // adam counter
			mini_Y->get_shuffled_set(*Y, shuffle_index, j, m);
			Z0->get_shuffled_set(*X, shuffle_index, j, m);
			current_batch = Z0->shape[0];

			// start
			//L1
			Convolution(current_batch, *Z1_conv, *W1_conv, *Z0);
			relu(current_batch, *Z1_conv);
			maxPool(current_batch,*Z1_pool, *Z1_conv);
			//maxPool2(current_batch, *Z1_pool, *Z1_conv);

			//L2
			Convolution(current_batch, *Z2_conv, *W2_conv, *Z1_pool);
			relu(current_batch, *Z2_conv);
			maxPool(current_batch, *Z2_pool, *Z2_conv);
			//maxPool2(current_batch, *Z2_pool, *Z2_conv);

			//L3
			FC_layer(current_batch, *Z3, *W3, *Z2_pool);
			relu(current_batch, *Z3);
			//L4
			FC_layer(current_batch, *Z4, *W4, *Z3);
			relu(current_batch, *Z4);
			//L5
			FC_layer(current_batch, *Z5, *W5, *Z4);
			softmax(current_batch, *Z5);

			mini_cost += cost_function(current_batch, *Z5, *mini_Y);
			accuracy = compute_accuracy(current_batch, *Z5, *mini_Y);

			//L5
			softmax_cost_backward(current_batch, *Z5, *mini_Y);
			FC_layer_backward(current_batch, *dZ4, *Z4, *Z5, *W5, *dW5, t);
			//L4
			relu_backward(current_batch, *dZ4, *Z4);
			FC_layer_backward(current_batch, *dZ3, *Z3, *dZ4, *W4, *dW4, t);
			//L3
			relu_backward(current_batch, *dZ3, *Z3);
			FC_layer_backward(current_batch, *dZ2_pool, *Z2_pool, *dZ3, *W3, *dW3, t);

			//L2
			//maxPool_backward2(current_batch, *dZ2_conv, *Z2_pool, *dZ2_pool);
			maxPool_backward(current_batch, *dZ2_conv, *Z2_conv, *dZ2_pool);
			relu_backward(current_batch, *dZ2_conv, *Z2_conv);
			Convolution_backward(current_batch, *dZ1_pool, *Z1_pool, *dZ2_conv, *W2_conv, *dW2_conv, t);
			
			//L1
			//maxPool_backward2(current_batch, *dZ1_conv, *Z1_pool, *dZ1_pool);
			maxPool_backward(current_batch,*dZ1_conv,*Z1_conv, *dZ1_pool);
			relu_backward(current_batch, *dZ1_conv, *Z1_conv);
			Convolution_backward(current_batch, *dZ0, *Z0, *dZ1_conv, *W1_conv, *dW1_conv, t);
		}
		toc = clock();
		timer = (double)(toc - tic) / CLOCKS_PER_SEC;
		printf("after %d epoches: cost=%.3f, accuracy=%.3f, time=%.3f sec \n", i, mini_cost/batch_count, accuracy, timer);
	}
	return 0;
}

