#include "simple_ml_ext.hpp"

DataSet::DataSet(size_t images_num, size_t input_dim)
    : images_num(images_num), input_dim(input_dim)
{
    images_matrix = new float[images_num * input_dim];
    labels_array = new unsigned char[images_num];
}

DataSet::~DataSet()
{
    delete[] images_matrix;
    delete[] labels_array;
}

uint32_t swap_endian(uint32_t val)
{
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    return (val << 16) | (val >> 16);
}

/**
 *Read an images and labels file in MNIST format.  See this page:
 *http://yann.lecun.com/exdb/mnist/ for a description of the file format.
 *Args:
 *    image_filename (str): name of images file in MNIST format (idx3-ubyte)
 *    label_filename (str): name of labels file in MNIST format (idx1-ubyte)
 **/
DataSet *parse_mnist(const std::string &image_filename, const std::string &label_filename)
{
    std::ifstream images_file(image_filename, std::ios::in | std::ios::binary);
    std::ifstream labels_file(label_filename, std::ios::in | std::ios::binary);
    uint32_t magic_num, images_num, rows_num, cols_num;

    images_file.read(reinterpret_cast<char *>(&magic_num), 4);
    labels_file.read(reinterpret_cast<char *>(&magic_num), 4);

    images_file.read(reinterpret_cast<char *>(&images_num), 4);
    labels_file.read(reinterpret_cast<char *>(&images_num), 4);
    images_num = swap_endian(images_num);

    images_file.read(reinterpret_cast<char *>(&rows_num), 4);
    rows_num = swap_endian(rows_num);
    images_file.read(reinterpret_cast<char *>(&cols_num), 4);
    cols_num = swap_endian(cols_num);

    DataSet *dataset = new DataSet(images_num, rows_num * cols_num);

    labels_file.read(reinterpret_cast<char *>(dataset->labels_array), images_num);
    unsigned char *pixels = new unsigned char[images_num * rows_num * cols_num];
    images_file.read(reinterpret_cast<char *>(pixels), images_num * rows_num * cols_num);
    for (size_t i = 0; i < images_num * rows_num * cols_num; i++)
    {
        dataset->images_matrix[i] = static_cast<float>(pixels[i]) / 255;
    }

    delete[] pixels;

    return dataset;
}

/**
 *Print Matrix
 *Print the elements of a matrix A with size m * n.
 *Args:
 *      A (float*): Matrix of size m * n
 **/
void print_matrix(float *A, size_t m, size_t n)
{
    for (size_t i = 0; i < m; ++i)
    {
        for (size_t j = 0; j < n; ++j)
        {
            std::cout << A[i * n + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}

/**
 * Matrix Dot Multiplication
 * Efficiently compute C = A.dot(B)
 * Args:
 *     A (const float*): Matrix of size m * n
 *     B (const float*): Matrix of size n * k
 *     C (float*): Matrix of size m * k
 **/
void matrix_dot(const float *A, const float *B, float *C, size_t m, size_t n, size_t k)
{
    // Initialize C
    for (size_t o = 0; o < m * k; ++o){
        C[o] = 0.0;
    }
    // Do Additions (Locality improved)
    for (size_t i = 0; i < m; ++i)
    {
        for (size_t j = 0; j < n; ++j)
        {
            float a = A[i*n + j];
            for (size_t l = 0; l < k; ++l){
                C[i * k + l] += a * B[j * k + l];
            }
        }
    }
}

/**
 * Matrix Dot Multiplication Trans Version
 * Efficiently compute C = A.T.dot(B)
 * Args:
 *     A (const float*): Matrix of size n * m
 *     B (const float*): Matrix of size n * k
 *     C (float*): Matrix of size m * k
 **/
void matrix_dot_trans(const float *A, const float *B, float *C, size_t n, size_t m, size_t k)
{
    // Initialize C
    for (size_t o = 0; o < m * k; ++o){
        C[o] = 0.0;
    }
    // Do Additions (Locality improved)
    for (size_t i = 0; i < m; ++i){
        for(size_t l = 0; l < n; ++l){
            for(size_t j = 0; j < k; ++j){
                C[i*k + j] += A[l*m + i] * B[l*k +j];
            }
        }
    }
}




/**
 * Matrix Dot Multiplication Trans Version 2
 * Efficiently compute C = A.dot(B.T)
 * Args:
 *     A (const float*): Matrix of size m * n
 *     B (const float*): Matrix of size k * n
 *     C (float*): Matrix of size m * k
 **/
void matrix_trans_dot(const float *A, const float *B, float *C, size_t m, size_t n, size_t k)
{
    // Initialize C
    for (size_t o = 0; o < m * k; ++o){
        C[o] = 0.0;
    }
    // Do Additions (Locality improved)
    for (size_t i = 0; i < m; ++i){
        for(size_t j = 0; j < k; ++j){
            for (size_t l = 0;l < n; ++l){
                C[i * k + j] += A[i * n + l] * B[j * n + l];
            }
        }
    }
}

/**
 * Matrix Minus
 * Efficiently compute A = A - B
 * For each element A[i], B[i] of A and B, A[i] -= B[i]
 * Args:
 *     A (float*): Matrix of size m * n
 *     B (const float*): Matrix of size m * n
 **/
void matrix_minus(float *A, const float *B, size_t m, size_t n)
{
    // Directly addition
    for (size_t o = 0; o < m * n; ++o){
        A[o] = A[o]-B[o];
    }
}

/**
 * Matrix Multiplication Scalar
 * For each element C[i] of C, C[i] *= scalar
 * Args:
 *     C (float*): Matrix of size m * n
 *     scalar (float)
 **/
void matrix_mul_scalar(float *C, float scalar, size_t m, size_t n)
{
    // Directly mul
    for (size_t o = 0; o < m * n; ++o){
        C[o] = C[o] * scalar;
    }    
}

/**
 * Matrix Division Scalar
 * For each element C[i] of C, C[i] /= scalar
 * Args:
 *     C (float*): Matrix of size m * n
 *     scalar (float)
 **/
void matrix_div_scalar(float *C, float scalar, size_t m, size_t n)
{
    // Directly div
    for (size_t o = 0; o < m * n; ++o){
        C[o] /= scalar;
    }  
}

/**
 * Matrix Softmax Normalize
 * For each row of the matrix, we do softmax normalization
 * Args:
 *     C (float*): Matrix of size m * n
 **/
void matrix_softmax_normalize(float *C, size_t m, size_t n)
{
    // Get exp value
    for (size_t i = 0; i < m * n; ++i){
        C[i] = exp(C[i]);
    }
    // Divide
    for (size_t j = 0; j < m; ++j){
        float total = 0.0;
        for (size_t l = 0; l < n; ++l){
            total += C[j * n + l];
        }
        matrix_div_scalar(C + j * n, total, 1, n);
    }
}

/**
 * Vector to One-Hot Matrix
 * Transform a label vector y to the one-hot encoding matrix Y
 * Args:
 *     y (unsigned char *): vector of size m * 1
 *     Y (float*): Matrix of size m * k
 *     k: Number of Classes
 **/
void vector_to_one_hot_matrix(const unsigned char *y, float *Y, size_t m, size_t k)
{
    // Initialize Y
    for (size_t o = 0; o < m * k; ++o){
        Y[o] = 0.0;
    }
    for (size_t i = 0; i < m; ++i){
        Y[i * k + static_cast<int>(y[i])] = 1.0;
    }
}





/**
 * A C++ version of the softmax regression epoch code.  This should run a
 * single epoch over the data defined by X and y (and sizes m,n,k), and
 * modify theta in place.  Your function will probably want to allocate
 * (and then delete) some helper arrays to store the logits and gradients.
 *
 * Args:
 *     X (const float *): pointer to X data, of size m*n, stored in row
 *          major (C) format
 *     y (const unsigned char *): pointer to y data, of size m
 *     theta (float *): pointer to theta data, of size n*k, stored in row
 *          major (C) format
 *     m (size_t): number of examples
 *     n (size_t): input dimension
 *     k (size_t): number of classes
 *     lr (float): learning rate / SGD step size
 *     batch (int): size of SGD batch
 *
 * Returns:
 *     (None)
 */
void softmax_regression_epoch_cpp(const float *X, const unsigned char *y, float *theta, size_t m, size_t n, size_t k, float lr, size_t batch)
{
    // Allocate memory for logits and gradients
    float *Z = new float[batch * k]; // logits and probilities
    float *Y = new float[batch * k]; // one-hot matrix
    float *G = new float[n * k]; // gradients

    for (size_t i = 0; i < m; i += batch)
    {
        // Select a batch of data
        const float *X_b = X + i * n;
        const unsigned char *y_b = y + i;
        size_t act_batch = batch;
        if (i + batch > m){
            act_batch = m - i;
        }
        // Compute the unnormalized log probabilities
        matrix_dot(X_b, theta, Z, act_batch, n, k);
        matrix_softmax_normalize(Z, act_batch, k);

        // Convert true labels to one-hot encoding
        vector_to_one_hot_matrix(y_b, Y, act_batch, k);

        // Do minus
        matrix_minus(Z, Y, act_batch, k);

        // Compute gradients
        matrix_dot_trans(X_b, Z, G, act_batch, n, k);
        matrix_div_scalar(G, static_cast<float>(act_batch), n, k);
        matrix_mul_scalar(G, lr, n, k);



        // Update theta
        matrix_minus(theta, G, n, k);
    }

    // Free allocated memory
    delete[] Z;
    delete[] Y;
    delete[] G;
}

/**
 *Example function to fully train a softmax classifier
 **/
void train_softmax(const DataSet *train_data, const DataSet *test_data, size_t num_classes, size_t epochs, float lr, size_t batch)
{
    size_t size = train_data->input_dim * num_classes;
    float *theta = new float[size];
    memset(theta, 0, size * sizeof(float));
    float *train_result = new float[train_data->images_num * num_classes];
    float *test_result = new float[test_data->images_num * num_classes];
    float train_loss, train_err, test_loss, test_err;
    std::cout << "| Epoch | Train Loss | Train Err | Test Loss | Test Err |" << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    for (size_t epoch = 0; epoch < epochs; epoch++)
    {
        // Train theta       
        softmax_regression_epoch_cpp(train_data->images_matrix, train_data->labels_array, theta, train_data->images_num, train_data->input_dim, num_classes, lr, batch);
        // result = X @ theta
        matrix_dot(train_data->images_matrix, theta, train_result, train_data->images_num, train_data->input_dim, num_classes);
        matrix_dot(test_data->images_matrix, theta, test_result, test_data->images_num, test_data->input_dim, num_classes);

        train_loss = mean_softmax_loss(train_result, train_data->labels_array, train_data->images_num, num_classes);
        test_loss = mean_softmax_loss(test_result, test_data->labels_array, test_data->images_num, num_classes);
        train_err = mean_err(train_result, train_data->labels_array, train_data->images_num, num_classes);
        test_err = mean_err(test_result, test_data->labels_array, test_data->images_num, num_classes);
        std::cout << "|  " << std::setw(4) << std::right << epoch << " |    "
                  << std::fixed << std::setprecision(5) << train_loss << " |   "
                  << std::fixed << std::setprecision(5) << train_err << " |   "
                  << std::fixed << std::setprecision(5) << test_loss << " |  "
                  << std::fixed << std::setprecision(5) << test_err << " |" << std::endl;
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_time -
                                                              start_time);
    std::cout << "Execution Time: " << elapsed_time.count()
              << " milliseconds\n";
    delete[] theta;
    delete[] train_result;
    delete[] test_result;
}

/*
 *Return softmax loss.  Note that for the purposes of this assignment,
 *you don't need to worry about "nicely" scaling the numerical properties
 *of the log-sum-exp computation, but can just compute this directly.
 *Args:
 *    result (const float *): 1D array of shape
 *        (batch_size x num_classes), containing the logit predictions for
 *        each class.
 *    labels_array (const unsigned char *): 1D array of shape (batch_size, )
 *        containing the true label of each example.
 *Returns:
 *    Average softmax loss over the sample.
 */
float mean_softmax_loss(const float *result, const unsigned char *labels_array, size_t images_num, size_t num_classes)
{
    float total_loss = 0.0;
    for (size_t i = 0; i < images_num; ++i){
        float true_prediction_logit = result[i * num_classes + static_cast<int>(labels_array[i])];
        float sum = 0.0;
        float single_loss = 0.0;
        for(size_t j = 0; j < num_classes; ++j){
            sum += exp(result[i * num_classes + j] - true_prediction_logit);
        }
        single_loss = log(sum);
        total_loss += single_loss;
    }
    return total_loss / images_num;
}

/*
 *Return error.
 *Args:
 *    result (const float *): 1D array of shape
 *        (batch_size x num_classes), containing the logit predictions for
 *        each class.
 *    labels_array (const unsigned char *): 1D array of shape (batch_size, )
 *        containing the true label of each example.
 *Returns:
 *    Average error over the sample.
 */
float mean_err(float *result, const unsigned char *labels_array, size_t images_num, size_t num_classes)
{
    float errors = 0.0;
    for (size_t i = 0; i < images_num; ++i){
        float max_logit = result[i * num_classes];
        size_t max_index = 0;
        for (size_t j = 1; j < num_classes; ++j){
            if (result[i * num_classes + j] > max_logit){
                max_logit = result[i * num_classes + j];
                max_index = j;
            }
        }
        if (max_index != static_cast<size_t>(labels_array[i])){
            errors += 1.0;
        }
    }
    return errors / images_num;
}

/**
 * Matrix Multiplication
 * Efficiently compute A = A * B
 * For each element A[i], B[i] of A and B, A[i] *= B[i]
 * Args:
 *     A (float*): Matrix of size m * n
 *     B (const float*): Matrix of size m * n
 **/
void matrix_mul(float *A, const float *B, size_t size)
{
    for (size_t i = 0; i < size; ++i){
        A[i] *= B[i];
    }
}

/*
Run a single epoch of SGD for a two-layer neural network defined by the
weights W1 and W2 (with no bias terms):
    logits = ReLU(X * W1) * W2
The function should use the step size lr, and the specified batch size (and
again, without randomizing the order of X).  It should modify the
W1 and W2 matrices in place.
Args:
    X: 1D input array of size
        (num_examples x input_dim).
    y: 1D class label array of size (num_examples,)
    W1: 1D array of first layer weights, of shape
        (input_dim x hidden_dim)
    W2: 1D array of second layer weights, of shape
        (hidden_dim x num_classes)
    m: num_examples
    n: input_dim
    l: hidden_dim
    k: num_classes
    lr (float): step size (learning rate) for SGD
    batch (int): size of SGD batch
*/
void nn_epoch_cpp(const float *X, const unsigned char *y, float *W1, float *W2, size_t m, size_t n, size_t l, size_t k, float lr, size_t batch)
{
    // Allocate memory
    float *Z1 = new float[batch * l]; // Z1
    float *Z2 = new float[batch * k]; //Z2
    float *Y = new float[batch * k]; // one_hot matrix Y
    float *G1 = new float[batch * l]; // G1
    float *Boole_Z1 = new float[batch * l]; // 1Z1 > 0
    float *W1_l = new float[n*l]; // gradients of W1
    float *W2_l = new float[l*k]; // gradients of W2

    for (size_t i = 0; i < m; i += batch)
    {
        // Select a batch of data
        const float *X_b = X + i * n;
        const unsigned char *y_b = y + i;
        size_t act_batch = batch;
        if (i + batch > m){
            act_batch = m - i;
        }

        // Z1 = X dot W1
        matrix_dot(X_b, W1, Z1, act_batch, n, l);
        // Z1 = ReLU(Z1)
        for (size_t j = 0; j < act_batch*l; ++j){
            if (Z1[j] < 0.0){
                Z1[j] = 0.0;
            }
        }

        // Z2 = normalize(exp(Z1 dot W2))
        matrix_dot(Z1, W2, Z2, act_batch, l, k);
        matrix_softmax_normalize(Z2, act_batch, k);

        // Y = vector_to_one_hot_matrix(y)
        vector_to_one_hot_matrix(y_b, Y, act_batch, k);

        // G1 = (Z2-Y) dot (W2.T)*Boole_Z1
        matrix_minus(Z2, Y, act_batch, k);
        matrix_trans_dot(Z2, W2, G1, act_batch, k, l);
        for (size_t h = 0; h < act_batch * l; ++h){
            if (Z1[h] > 0){
                Boole_Z1[h] = 1.0;
            } else {
                Boole_Z1[h] = 0.0;
            }
        }
        matrix_mul(G1, Boole_Z1, act_batch * l);

        // W1_l = (X_b.T dot G1)/batch * lr
        matrix_dot_trans(X_b, G1, W1_l, act_batch, n, l);
        matrix_div_scalar(W1_l, static_cast<float>(act_batch), n, l);
        matrix_mul_scalar(W1_l, lr, n, l);



        // W2_l = (Z1.T dot (Z2-Y))/batch * lr
        matrix_dot_trans(Z1, Z2, W2_l, act_batch, l, k);
        matrix_div_scalar(W2_l, static_cast<float>(act_batch), l, k);
        matrix_mul_scalar(W2_l, lr, l, k); 



        //Update W1, W2
        matrix_minus(W1, W1_l, n, l);
        matrix_minus(W2, W2_l, l, k);
    }
    delete[] Z1;
    delete[] Z2;
    delete[] Y;
    delete[] G1;
    delete[] Boole_Z1;
    delete[] W1_l;
    delete[] W2_l;
}


/**
 *Example function to fully train a nn classifier
 **/
void train_nn(const DataSet *train_data, const DataSet *test_data, size_t num_classes, size_t hidden_dim, size_t epochs, float lr, size_t batch)
{
    size_t size_w1 = train_data->input_dim * hidden_dim;
    size_t size_w2 = hidden_dim * num_classes;
    float *W1 = new float[size_w1];
    float *W2 = new float[size_w2];
    float *Temp_tr = new float[train_data->images_num * hidden_dim];
    float *Temp_te = new float[test_data->images_num * hidden_dim];
    std::mt19937 rng;
    rng.seed(0);
    std::normal_distribution<float> dist(0.0, 1.0);
    for (size_t i = 0; i < size_w1; i++)
    {
        W1[i] = dist(rng);
    }
    for (size_t i = 0; i < size_w2; i++)
    {
        W2[i] = dist(rng);
    }
    matrix_div_scalar(W1, sqrtf(hidden_dim), train_data->input_dim, hidden_dim);
    matrix_div_scalar(W2, sqrtf(num_classes), hidden_dim, num_classes);
    float *train_result = new float[train_data->images_num * num_classes];
    float *test_result = new float[test_data->images_num * num_classes];
    float train_loss, train_err, test_loss, test_err;
    std::cout << "| Epoch | Train Loss | Train Err | Test Loss | Test Err |" << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    for (size_t epoch = 0; epoch < epochs; epoch++)
    {
        nn_epoch_cpp(train_data->images_matrix, train_data->labels_array, W1, W2, train_data->images_num, train_data->input_dim, hidden_dim, num_classes, lr, batch);
        
        // result = relu(X @ W1) @ W2
        // Train
        matrix_dot(train_data->images_matrix, W1, Temp_tr, train_data->images_num, train_data->input_dim, hidden_dim);
        for (size_t j = 0; j < train_data->images_num * hidden_dim; ++j){
            if (Temp_tr[j] < 0.0){
                Temp_tr[j] = 0.0;
            }
        }
        matrix_dot(Temp_tr, W2, train_result, train_data->images_num, hidden_dim, num_classes);

        // Test
        matrix_dot(test_data->images_matrix, W1, Temp_te, test_data->images_num, test_data->input_dim, hidden_dim);
        for (size_t j = 0; j < test_data->images_num * hidden_dim; ++j){
            if (Temp_te[j] < 0.0){
                Temp_te[j] = 0.0;
            }
        }
        matrix_dot(Temp_te, W2, test_result, test_data->images_num, hidden_dim, num_classes);
        
        train_loss = mean_softmax_loss(train_result, train_data->labels_array, train_data->images_num, num_classes);
        test_loss = mean_softmax_loss(test_result, test_data->labels_array, test_data->images_num, num_classes);
        train_err = mean_err(train_result, train_data->labels_array, train_data->images_num, num_classes);
        test_err = mean_err(test_result, test_data->labels_array, test_data->images_num, num_classes);
        std::cout << "|  " << std::setw(4) << std::right << epoch << " |    "
                  << std::fixed << std::setprecision(5) << train_loss << " |   "
                  << std::fixed << std::setprecision(5) << train_err << " |   "
                  << std::fixed << std::setprecision(5) << test_loss << " |  "
                  << std::fixed << std::setprecision(5) << test_err << " |" << std::endl;
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_time -
                                                              start_time);
    std::cout << "Execution Time: " << elapsed_time.count()
              << " milliseconds\n";
    delete[] W1;
    delete[] W2;
    delete[] train_result;
    delete[] test_result;
    delete[] Temp_tr;
    delete[] Temp_te;
}
