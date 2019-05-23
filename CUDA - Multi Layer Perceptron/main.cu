#include <iostream>
#include <stdio.h>
#include <stdlib.h> 
#include <time.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 1
//Global variable set up
const int radius = 3;
const int numSamples = 100;
const double learningRate = 0.15;
const int epochs = 1;
const int numNeurons =20;
//Set up neurons
double inputLayer[2][numNeurons] = { 0 }; //takes input and weights
double outputLayer[1][numNeurons] = { 0 }; //takes weights and outputs
double * matrixA; //Temporarily initialised arrays which are allocated to aid in gpu memory allocation
double * matrixB;
double * matrixC;
double * matrixD;
double * matrixE;

//Calculates dot product of two arrays from a given pointer and returns a total - must be same size
double dotProduct(double *array1, double *array2, int size) {
	double total = 0;
	for (int i = 0; i < size; i++) {
		total += array1[i] * array2[i];
	}
	return total;
}

//Does an element by element multiplication but keeps the size of the array the same- alters an array at a given pointer to contain this
void elementMultiply(double *array1, double *array2, double *output, int rows) {
	for (int i = 0; i < rows; i++) {
		output[i] = array1[i] * array2[i];
	}
}
//Calculates the sigmoid derivative for every position in an array then alters an array at a pointer to contain this 
void sigmoidDerivativeMatrix(double* inputArray, int rows, double *output) {
	for (int i = 0; i < rows; i++) {
		output[i] = inputArray[i] * (1 - inputArray[i]);
	}
}
//Calculates the sigmoid derivative of a singular value and returns it as a double
double sigmoidDerivativeScalar(double inputVal) {
	double sigmoidValue;
	sigmoidValue = inputVal * (1 - inputVal);
	return sigmoidValue;
}

//Calculates the sigmoid value of a single input and returns it as a double
double sigmoidScalar(double inputVal) {
	double sigmoidValue;
	sigmoidValue = 1 / (1 + exp(-inputVal));
	return sigmoidValue;
}
//Calculates the sigmoid for every position in an array then alters an array at a pointer to contain this 
void sigmoidMatrix(double *inputArray, int rows, double *output) {
	for (int i = 0; i < rows; i++) {
		output[i] = 1 / (1 + exp(-inputArray[i]));
	}
}
//Completes the matrix multiplication of two arrays and puts the values in a given pointer output
__global__ void MatMultKernel(double *array1, double *array2, double *output, int arr1_rows, int arr1_cols, int arr2_cols) {
	double result = 0;
	__shared__ double subArray1[BLOCK_SIZE][BLOCK_SIZE]; //Setting up the shared memory into sub arrays for more efficient computation 
	__shared__ double subArray2[BLOCK_SIZE][BLOCK_SIZE];
	int bIDx = blockIdx.x, bIDy = blockIdx.y, tIDx = threadIdx.x, tIDy = threadIdx.y; //Setting up variables to identify threads uniquely
	int row = bIDy * BLOCK_SIZE + tIDy; //Setting the given row of a thread
	int col = bIDx * BLOCK_SIZE + tIDx; //Setting the given col of a thread
	for (int i = 0; i < (arr1_cols-1)/BLOCK_SIZE+1; i++) { //Iterating through every chunk of columns proportional to block size
		if (row < arr1_rows && i*BLOCK_SIZE+tIDx<arr1_cols) {
			subArray1[tIDy][tIDx] = array1[row*arr1_cols + i * BLOCK_SIZE + tIDx]; //Setting up sub array1 to contain relevant pieces of array1
		}else {
			subArray1[tIDy][tIDx] = 0; //0ing values to prevent miscalculation if not relevant 
		}
		if (col < arr2_cols && i*BLOCK_SIZE+tIDy<arr1_cols) {
			subArray2[tIDy][tIDx] = array2[(i * BLOCK_SIZE + tIDy)*arr2_cols+col]; //Setting up sub array2 to contain relevant pieces of array2
		}else {
			subArray2[tIDy][tIDx] = 0;//0ing values to prevent miscalculation if not relevant 
		}
		__syncthreads(); //Blocking to ensure sub arrays are built
		for (int ii = 0; ii < BLOCK_SIZE; ii++) {
			result += subArray1[tIDy][ii] * subArray2[ii][tIDx]; //Calculating result for this chunk utilising many threads simultaneously
		}
		__syncthreads(); //Ensure result calculation is done 
	}
	if (row < arr1_rows&&col < arr2_cols) {
		output[row*arr2_cols + col] = result; //Calculate overall output in position
	}
	
}
//Invokes kernel and copies result from kernel to pointer then exits
void cudaMatrixMultiply(double *array1, double *array2, double *output, int arr1_rows, int arr1_cols, int arr2_cols,double * matrixP1,double * matrixP2,double * matrixP3) {

	cudaMemcpy(matrixP1, array1, sizeof(double)*arr1_rows*arr1_cols, cudaMemcpyHostToDevice);//Taking input array from parameter to gpu
	cudaMemcpy(matrixP2, array2, sizeof(double)*arr2_cols*arr1_cols, cudaMemcpyHostToDevice);
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE,1);
	dim3 dimGrid((arr2_cols - 1) / BLOCK_SIZE + 1, (arr1_rows - 1) / BLOCK_SIZE + 1);//Ensuring dim grid is properly sized
	MatMultKernel<<<dimGrid,dimBlock>>> (matrixP1, matrixP2, matrixP3, arr1_rows, arr1_cols, arr2_cols);//Kernel invocation
	cudaMemcpy(output, matrixP3, sizeof(double)*arr1_rows*arr2_cols, cudaMemcpyDeviceToHost);//Copy results to pointer
}	
//Adds two matricies together and returns in pointer of the first matrix
void addMatrix(double *targetArray, double *addArray, int rows) {
	for (int i = 0; i < rows; i++) {
		targetArray[i] = targetArray[i] + addArray[i];
	}
}
//Creates input data and returns a pointer to the data set 
double* createInput(int dataSize, double* inputData) {//Takes in data to ensure inputs match outputs
	int setSize = dataSize * 2;
	double* dataSet = new double[setSize];
	for (int i = 0; i < dataSize; i++) {
		dataSet[i] = radius * cos(inputData[i]); //Uses trig to build data set 
		dataSet[(dataSize + i)] = radius * sin(inputData[i]);//As dataset has two co-ords was more efficient to return in one array of double size
	}
	return dataSet;
}
//Creates an output of random angles between 0 and 1 with 4 decimal places
double* createOutput(int dataSize) {
	double* dataSet = new double[dataSize];
	for (int i = 0; i < dataSize; i++) {
		double theta = rand() % 1000; //randomly seeded number generator
		theta = theta / 1000.0;
		dataSet[i] = theta;
	}
	return dataSet;
}

//Does the training and error calculation for the Multi layer perceptron - taking in inputs and expected outputs to perform supervised learning 
void trainMultiLayerPerceptron(double* inputData, double* expectedOutputData, int maxiter) {
	for (int j = 0; j < maxiter; j++) {
		double layerOneAdjustment[2][numNeurons] = { 0 }; //Initalise the adjustment arrays to nothing
		double layerTwoAdjustment[1][numNeurons] = { 0 };
		double errorSum = 0.0;//reset error to 0
		for (int i = 0; i < numSamples; i++) {
			//Initialise arrays which hold temporary variables and returns from matrix multiplications to 0.0
			double layer1output[1][numNeurons] = { 0 };
			double inputDataArray[1][2] = { inputData[i] ,inputData[numSamples + i] };
			double transposeInputData[2][1] = { inputData[i],inputData[numSamples + i] };
			double layerOneAdjustmentTmp[2][numNeurons] = { 0 };
			double layerTwoAdjustmentTmp[1][numNeurons] = { 0 };
			double layer2delta[1][1] = { 0 };
			double layer1error[1][numNeurons] = { 0 };
			double layer1delta[1][numNeurons] = { 0 };
			double layer1outputSigmoid[1][numNeurons] = { 0 };
			cudaMatrixMultiply(*inputDataArray, *inputLayer, *layer1output, 1, 2, numNeurons,matrixA,matrixB,matrixD);//Calculate layer 1 output and pass designated pre-allocated GPU memory
			sigmoidMatrix(*layer1output, numNeurons, *layer1output);//Sigmoid of output
			double layer2output = sigmoidScalar(dotProduct(*layer1output, *outputLayer, numNeurons));//Calculate sigmoid of dot product for layer 2 output
			double layer2error = expectedOutputData[i] - layer2output;//Calculate layer 2 error
			layer2delta[0][0] = layer2error * sigmoidDerivativeScalar(layer2output); //Calculate layer 2 delta into an array 
			cudaMatrixMultiply(*layer2delta, *outputLayer, *layer1error, 1, 1, numNeurons,matrixC,matrixD,matrixD);//Calculate layer 1 error using matrix multi of level2delta and outputlayer
			sigmoidDerivativeMatrix(*layer1output, numNeurons, *layer1outputSigmoid);//Calculate sigmoid derivative of layer 1 output
			elementMultiply(*layer1error, *layer1outputSigmoid, *layer1delta, numNeurons); //Element by element multiply layer 1 error by sigmoid
			cudaMatrixMultiply(*transposeInputData, *layer1delta, *layerOneAdjustmentTmp, 2, 1, numNeurons,matrixE,matrixD,matrixB);//Calculate layer one adjustment into a temporary array
			cudaMatrixMultiply(*layer2delta, *layer1output, *layerTwoAdjustmentTmp, 1, 1, numNeurons,matrixC,matrixD,matrixD);//Calculate layer two adjustment into a temporary array
			for (int ii = 0; ii < numNeurons; ii++) {//Iterate through and update layer adjustments using temporary array 
				layerOneAdjustment[0][ii] = layerOneAdjustment[0][ii] + layerOneAdjustmentTmp[0][ii];
				layerOneAdjustment[1][ii] = layerOneAdjustment[1][ii] + layerOneAdjustmentTmp[1][ii];
				layerTwoAdjustment[0][ii] = layerOneAdjustment[0][ii] + layerTwoAdjustmentTmp[0][ii];
			}
			errorSum = errorSum + (layer2error * layer2error);//Calculate error sum
			if (j == (maxiter - 1)) {
				printf("input 1 %lf\n", inputData[i]);
				printf("input 2 %lf\n", inputData[numSamples + i]);
				printf("Expected output %lf\n", expectedOutputData[i]);
				printf("output 1 %lf\n", layer2output);
				printf("error sum %lf\n", errorSum);
			}
			
			for (int ii = 0; ii < numNeurons; ii++) {//Update layer weights by learning rate times the adjustment 
				inputLayer[0][ii] = inputLayer[0][ii] + learningRate * layerOneAdjustment[0][ii];
				inputLayer[1][ii] = inputLayer[1][ii] + learningRate * layerOneAdjustment[1][ii];
				outputLayer[0][ii] = outputLayer[0][ii] + learningRate * layerTwoAdjustment[0][ii];
			}
		}


	}
}
int main(int argc, char **argv){
	srand(time(NULL));//Seed random
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	double* expectedOutputData = createOutput(numSamples);//Generate expected results
	double* inputData = createInput(numSamples, expectedOutputData); //Generate inputs
	cudaMalloc(&matrixA, (1 * 2 * sizeof(double))); //Pre-allocate GPU memory to known size of matricies for later multiplications
	cudaMalloc(&matrixB, (2 * numNeurons * sizeof(double)));
	cudaMalloc(&matrixC, (1 * 1 * sizeof(double)));
	cudaMalloc(&matrixD, (1 * numNeurons * sizeof(double)));
	cudaMalloc(&matrixE, (2 * 1 * sizeof(double)));
	for (int i = 0; i < epochs; i++) {
		for (int ii = 0; ii < numNeurons; ii++) {//Loop to setup Neural network weights to random values with bias -0.5
			inputLayer[0][ii] = { ((rand() % 1000) / 1000.0) - 0.5 };
			inputLayer[1][ii] = { ((rand() % 1000) / 1000.0) - 0.5 };
			outputLayer[0][ii] = { ((rand() % 1000) / 1000.0) - 0.5 };
		}
		cudaEventRecord(start, 0);
		trainMultiLayerPerceptron(inputData, expectedOutputData, 1000);//Train network
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		printf("Milliseconds  %fn", milliseconds);
	}
	cudaFree(matrixA);//Release memory to avoid leaks
	cudaFree(matrixB);
	cudaFree(matrixC);
	cudaFree(matrixD);
	cudaFree(matrixE);
	return 0;
}

