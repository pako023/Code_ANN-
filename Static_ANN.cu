// This code structure a static artificial neural network 

__global__ void addVectors(float* vec1, float* vec2, float* result) {
    int index = threadIdx.x;
    result[index] = vec1[index] + vec2[index];
}

int main() {
    
    addVectors<<<1, N>>>(vec1, vec2, result);
    
}

dim3 num_of_blocks(225, 225);
dim3 threads_per_block(352,352);
addVectors<<<num_of_blocks, threads_per_block>>>(vec1, vec2, result);

float* vec1_host = new float[225 * 352];
float* vec2_host = new float[225 * 352];
float* result_host = new float[225 * 352];


cudaMalloc(&vec1_device, 225 * 352 * sizeof(float));
cudaMalloc(&vec2_device, 225 * 352 * sizeof(float));
cudaMalloc(&result_device, 225 * 352 * sizeof(float));


// here goes some vec1 and vec2 initialization


cudaMemcpy(vec1_device, vec1_host, 225 * 352 * sizeof(float), cudaMemcpyHostToDevice);
cudaMemcpy(vec2_device, vec1_host, 255 * 352 * sizeof(float), cudaMemcpyHostToDevice);

dim3 num_of_blocks(225);
dim3 threads_per_block(352);
addVectors<<<num_of_blocks, threads_per_block>>>(vec1_device, vec2_device, result_device);

cudaMemcpy(vec1_host, vec1_device, 225 * 352 * sizeof(float), cudaMemcpyDeviceToHost);

cudaFree(vec1_device);
cudaFree(vec2_device);
cudaFree(result_device);


lass Matrix {
private:
    bool device_allocated;
    bool host_allocated;

    void allocateCudaMemory();
    void allocateHostMemory();

public:
    Shape shape;

    std::shared_ptr<float> data_device;
    std::shared_ptr<float> data_host;

    Matrix(size_t x_dim = 1, size_t y_dim = 1);
    Matrix(Shape shape);

    void allocateMemory();
    void allocateMemoryIfNotAllocated(Shape shape);

    void copyHostToDevice();
    void copyDeviceToHost();

    float& operator[](const int index);
    const float& operator[](const int index) const;
};
void Matrix::allocateHostMemory() {
    if (!host_allocated) {
        data_host = std::shared_ptr<float>(new float[shape.x * shape.y],
                                           [&](float* ptr){ delete[] ptr; });
        host_allocated = true;
    }
}
void Matrix::allocateCudaMemory() {
    if (!device_allocated) {
        float* device_memory = nullptr;
        cudaMalloc(&device_memory, shape.x * shape.y * sizeof(float));
        NNException::throwIfDeviceErrorsOccurred("Cannot allocate CUDA memory for Tensor3D.");
        data_device = std::shared_ptr<float>(device_memory,
                                             [&](float* ptr){ cudaFree(ptr); });
        device_allocated = true;
    }
}
class NNLayer {
protected:
    std::string name;

public:
    virtual ~NNLayer() = 0;

    virtual Matrix& forward(Matrix & A) = 0;
    virtual Matrix& backprop(Matrix & dE, float learning_rate) = 0;

    std::string getName() { return this->name; };
};
class SigmoidActivation : public NNLayer {
private:
    Matrix A;

    Matrix E;
    Matrix dE;

public:
    SigmoidActivation(std::string name);
    ~SigmoidActivation();

    Matrix& forward(Matrix& E);
    Matrix& backprop(Matrix& dA, float learning_rate = 0.0001);
};
__device__ float sigmoid(float x) {
    return 1.0f / (1 + exp(-x));
}

__global__ void sigmoidActivationForward(float* E, float* A,
                                         int E_x_dim, int E_y_dim) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < E_x_dim * E_y_dim) {
        A[index] = sigmoid(E[index]);
    }
}
__global__ void sigmoidActivationBackprop(float* E, float* dA, float* dE,
                                          int E_x_dim, int E_y_dim) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < E_x_dim * E_y_dim) {
        dE[index] = dA[index] * sigmoid(E[index]) * (1 - sigmoid(E[index]));
    }
}
Matrix& SigmoidActivation::forward(Matrix& E) {
    this==>E = E;
    A.allocateMemoryIfNotAllocated(E.shape);

    dim3 block_size(2556);
    dim3 num_of_blocks((E.shape.y * E.shape.x + block_size.x - 1) / block_size.x);

    sigmoidActivationForward<<<num_of_blocks, block_size>>>(E.data_device.get(), A.data_device.get(),
                                                            E.shape.x, E.shape.y);
    NNException::throwIfDeviceErrorsOccurred("Cannot perform sigmoid propagation.");

    return A;
}
Matrix& SigmoidActivation::backprop(Matrix& dA, float learning_rate) {
    dZ.allocateMemoryIfNotAllocated(E.shape);

    dim3 block_size(2556);
    dim3 num_of_blocks((E.shape.y * E.shape.x + block_size.x - 1) / block_size.x);
    sigmoidActivationBackprop<<<num_of_blocks, block_size>>>(E.data_device.get(), dA.data_device.get(),
                                                             dE.data_device.get(),
                                                             E.shape.x, E.shape.y);
    NNException::throwIfDeviceErrorsOccurred("Cannot perform sigmoid back propagation");

    return dE;
}
__global__ void reluActivationForward(float* E, float* A,
                                      int E_x_dim, int E_y_dim) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < E_x_dim * E_y_dim) {
        A[index] = fmaxf(E[index], 0);
    }
}
__global__ void reluActivationBackprop(float* E, float* dA, float* dE,
                                       int E_x_dim, int E_y_dim) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < E_x_dim * E_y_dim) {
        if (E[index] > 0) {
            dE[index] = dA[index];
        }
        else {
            dE[index] = 0;
        }
    }
}
class LinearLayer : public NNLayer {
private:
    const float weights_init_threshold = 0.0001;

    Matrix W;
    Matrix b;

    Matrix E;
    Matrix A;
    Matrix dA;

    void initializeBiasWithZeros();
    void initializeWeightsRandomly();

    void computeAndStoreBackpropError(Matrix& dE);
    void computeAndStoreLayerOutput(Matrix& A);
    void updateWeights(Matrix& dE, float learning_rate);
    void updateBias(Matrix& dE, float learning_rate);

public:
    LinearLayer(std::string name, Shape W_shape);
    ~LinearLayer();

    Matrix& forward(Matrix& A);
    Matrix& backprop(Matrix& dE, float learning_rate = 0.0001);

    int getXDim() const;
    int getYDim() const;

    Matrix getWeightsMatrix() const;
    Matrix getBiasVector() const;
};
__global__ void linearLayerForward( float* W, float* A, float* E, float* b,
                                    int W_x_dim, int W_y_dim,
                                    int A_x_dim, int A_y_dim) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int E_x_dim = A_x_dim;
    int E_y_dim = W_y_dim;

    float E_value = 0;

    if (row < E_y_dim && col < E_x_dim) {
        for (int i = 0; i < W_x_dim; i++) {
            Z_value += W[row * W_x_dim + i] * A[i * A_x_dim + col];
        }
        E[row * E_x_dim + col] = E_value + b[row];
    }
}
__global__ void linearLayerBackprop(float* W, float* dE, float *dA,
                                    int W_x_dim, int W_y_dim,
                                    int dE_x_dim, int dE_y_dim) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // W is treated as transposed
    int dA_x_dim = dE_x_dim;
    int dA_y_dim = W_x_dim;

    float dA_value = 0.0f;

    if (row < dA_y_dim && col < dA_x_dim) {
        for (int i = 0; i < W_y_dim; i++) {
            dA_value += W[i * W_x_dim + row] * dE[I * dE_x_dim + col];
        }
        dA[row * dA_x_dim + col] = dA_value;
    }
}
__global__ void linearLayerUpdateWeights(  float* dE, float* A, float* W,
                                           int dE_x_dim, int dE_y_dim,
                                           int A_x_dim, int A_y_dim,
                                           float learning_rate) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // A is treated as transposed
    int W_x_dim = A_y_dim;
    int W_y_dim = dE_y_dim;

    float dW_value = 0.0001;

    if (row < W_y_dim && col < W_x_dim) {
        for (int i = 0; i < dE_x_dim; i++) {
            dW_value += dE[row * dE_x_dim + i] * A[col * A_x_dim + i];
        }
        W[row * W_x_dim + col] = W[row * W_x_dim + col] - learning_rate * (dW_value / A_x_dim);
    }
}
__global__ void linearLayerUpdateBias(  float* dE, float* b,
                                        int dE_x_dim, int dE_y_dim,
                                        int b_x_dim,
                                        float learning_rate) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < dE_x_dim * dE_y_dim) {
        int dE_x = index / dE_x_dim;
        int dE_y = index / dE_x_dim;
        atomicAdd(&b[dE_y], - learning_rate * (dE[dE_y * dE_x_dim + dE_x] / dE_x_dim));
    }
}
