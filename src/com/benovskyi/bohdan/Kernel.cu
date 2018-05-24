extern "C"
__global__ void matrixAdd(int *a, int *b, int *c) {
	int i = threadIdx.x;
	int j = threadIdx.y;
	int index = i + j * 5;
	c[index] = (a[index] + b[index]) * 5 + 8;
}