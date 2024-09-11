#include "simple_ml_openacc.hpp"

void matrix_dot_openacc(const float *A, const float *B,
                        float *C, size_t m, size_t n, size_t k)
{
    #pragma acc data copyin(A[0:m*n], B[0:n*k]) copy(C[0:m*k])
    {
        // Initialize C
        #pragma acc parallel loop
        for (size_t o = 0; o < m * k; ++o){
            C[o] = 0.0;
        }
        // Do Additions (Locality improved)
        #pragma acc parallel loop collapse(2)
        for (size_t i = 0; i < m; ++i)
        {
            for (size_t j = 0; j < n; ++j)
            {
                for (size_t l = 0; l < k; ++l){
                    #pragma acc atomic update
                    C[i * k + l] += A[i*n + j] * B[j * k + l];
                }
            }
        }
    }
}

void matrix_dot_trans_openacc(const float *A, const float *B, float *C, size_t n, size_t m, size_t k)
{
    #pragma acc data copyin(A[0:m*n], B[0:n*k]) copy(C[0:m*k])
    {
        // Initialize C
        #pragma acc parallel loop
        for (size_t o = 0; o < m * k; ++o){
            C[o] = 0.0;
        }
        // Do Additions (Locality improved)
        #pragma acc parallel loop collapse (2)
        for (size_t i = 0; i < m; ++i)
        {
            for(size_t l = 0; l < n; ++l)
            {
                for(size_t j = 0; j < k; ++j){
                    #pragma acc atomic update
                    C[i*k + j] += A[l*m + i] * B[l*k +j];
                }
            }
        }
    }
}

void matrix_trans_dot_openacc(const float *A, const float *B, float *C, size_t m, size_t n, size_t k)
{
    #pragma acc data copyin(A[0:m*n], B[0:n*k]) copy(C[0:m*k])
    {
        // Initialize C
        #pragma acc parallel loop
        for (size_t o = 0; o < m * k; ++o){
            C[o] = 0.0;
        }
        // Do Additions (Locality improved)
        #pragma acc parallel loop collapse (2)
        for (size_t i = 0; i < m; ++i){
            for(size_t j = 0; j < k; ++j){
                for (size_t l = 0;l < n; ++l){
                    #pragma acc atomic update
                    C[i * k + j] += A[i * n + l] * B[j * n + l];
                }
            }
        }
    }
}

void matrix_minus_openacc(float *A, const float *B, size_t m, size_t n)
{
    #pragma acc data copyin(B[0:m*n]) copy(A[0:m*n])
    {
        // Directly addition
        #pragma acc parallel loop
        for (size_t o = 0; o < m * n; ++o){
            A[o] = A[o]-B[o];
        }
    }
}

void matrix_mul_scalar_openacc(float *C, float scalar, size_t m, size_t n)
{
    #pragma acc data copy(C[0:m*n])
    {
        // Directly mul
        #pragma acc parallel loop
        for (size_t o = 0; o < m * n; ++o){
            C[o] = C[o] * scalar;
        }
    }
}

void matrix_div_scalar_openacc(float *C, float scalar, size_t m, size_t n)
{
    #pragma acc data copy(C[0:m*n])
    {
        // Directly div
        #pragma acc parallel loop
        for (size_t o = 0; o < m * n; ++o){
            C[o] /= scalar;
        } 
    }
}

void matrix_softmax_normalize_openacc(float *C, size_t m, size_t n)
{
    #pragma acc data copy(C[0:m*n])
    {
        // Normalize
        #pragma acc parallel loop
        for (size_t j = 0; j < m; ++j){
            float total = 0.0;
            for (size_t l = 0; l < n; ++l){
                C[j * n + l] = exp(C[j * n + l]);
                total += C[j * n + l];
            }
            for (size_t l = 0; l < n; ++l){
                C[j * n + l] /= total;
            }
        }
    }
}

void vector_to_one_hot_matrix_openacc(const unsigned char *y, float *Y, size_t m, size_t k)
{
    #pragma acc data copy(y[0:m],Y[0:m*k])
    {
        // Initialize Y
        #pragma acc parallel loop
        for (size_t o = 0; o < m * k; ++o){
            Y[o] = 0.0;
        }
        #pragma acc parallel loop
        for (size_t i = 0; i < m; ++i){
            Y[i * k + static_cast<int>(y[i])] = 1.0;
        }
    }
}

void softmax_regression_epoch_openacc(const float *X, const unsigned char *y,
                                      float *theta, size_t m, size_t n, size_t k,
                                      float lr, size_t batch)
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
        matrix_dot_openacc(X_b, theta, Z, act_batch, n, k);
        matrix_softmax_normalize_openacc(Z, act_batch, k);

        // Convert true labels to one-hot encoding
        vector_to_one_hot_matrix_openacc(y_b, Y, act_batch, k);

        // Do minus
        matrix_minus_openacc(Z, Y, act_batch, k);

        // Compute gradients
        matrix_dot_trans_openacc(X_b, Z, G, act_batch, n, k);
        matrix_div_scalar_openacc(G, static_cast<float>(act_batch), n, k);
        matrix_mul_scalar_openacc(G, lr, n, k);

        // Update theta
        matrix_minus_openacc(theta, G, n, k);
    }

    // Free allocated memory
    delete[] Z;
    delete[] Y;
    delete[] G;
}

void train_softmax_openacc(const DataSet *train_data, const DataSet *test_data, size_t num_classes, size_t epochs, float lr, size_t batch)
{
    size_t size = train_data->input_dim * num_classes;
    float *theta = new float[size];
    memset(theta, 0, size * sizeof(float));
    size_t size_tr = train_data->images_num * num_classes;
    size_t size_te = test_data->images_num * num_classes;
    float *train_result = new float[size_tr];
    float *test_result = new float[size_te];
    float train_loss, train_err, test_loss, test_err;
    std::cout << "| Epoch | Train Loss | Train Err | Test Loss | Test Err |" << std::endl;
    std::chrono::milliseconds elapsed_time;
    // BEGIN YOUR CODE

    auto start_time = std::chrono::high_resolution_clock::now();
    for (size_t epoch = 0; epoch < epochs; epoch++)
    {
        softmax_regression_epoch_openacc(train_data->images_matrix, train_data->labels_array, theta, train_data->images_num, train_data->input_dim, num_classes, lr, batch);
        matrix_dot_openacc(train_data->images_matrix, theta, train_result, train_data->images_num, train_data->input_dim, num_classes);
        matrix_dot_openacc(test_data->images_matrix, theta, test_result, test_data->images_num, test_data->input_dim, num_classes);

        train_loss = mean_softmax_loss_openacc(train_result, train_data->labels_array, train_data->images_num, num_classes);
        test_loss = mean_softmax_loss_openacc(test_result, test_data->labels_array, test_data->images_num, num_classes);
        train_err = mean_err_openacc(train_result, train_data->labels_array, train_data->images_num, num_classes);
        test_err = mean_err_openacc(test_result, test_data->labels_array, test_data->images_num, num_classes);
        std::cout << "|  " << std::setw(4) << std::right << epoch << " |    "
                  << std::fixed << std::setprecision(5) << train_loss << " |   "
                  << std::fixed << std::setprecision(5) << train_err << " |   "
                  << std::fixed << std::setprecision(5) << test_loss << " |  "
                  << std::fixed << std::setprecision(5) << test_err << " |" << std::endl;
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    elapsed_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_time -
                                                              start_time);
    std::cout << "Execution Time: " << elapsed_time.count()
              << " milliseconds\n";
    #pragma acc exit data copyout()
   // END YOUR CODE
    delete[] theta;
    delete[] train_result;
    delete[] test_result;
}

float mean_softmax_loss_openacc(const float *result, const unsigned char *labels_array, size_t images_num, size_t num_classes)
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

float mean_err_openacc(const float *result, const unsigned char *labels_array, size_t images_num, size_t num_classes)
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

void matrix_mul_openacc(float *A, const float *B, size_t size)
{
    #pragma acc data copyin(B[0:size]) copy(A[0:size])
    {
        // Directly addition
        #pragma acc parallel loop
        for (size_t o = 0; o < size; ++o){
            A[o] *= B[o];
        }
    }
}

void nn_epoch_openacc(const float *X, const unsigned char *y, float *W1, float *W2, size_t m, size_t n, size_t l, size_t k, float lr, size_t batch)
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
        matrix_dot_openacc(X_b, W1, Z1, act_batch, n, l);
        // Z1 = ReLU(Z1)
        for (size_t j = 0; j < act_batch*l; ++j){
            if (Z1[j] < 0.0){
                Z1[j] = 0.0;
            }
        }

        // Z2 = normalize(exp(Z1 dot W2))
        matrix_dot_openacc(Z1, W2, Z2, act_batch, l, k);
        matrix_softmax_normalize_openacc(Z2, act_batch, k);

        // Y = vector_to_one_hot_matrix(y)
        vector_to_one_hot_matrix_openacc(y_b, Y, act_batch, k);

        // G1 = (Z2-Y) dot (W2.T)*Boole_Z1
        matrix_minus_openacc(Z2, Y, act_batch, k);
        matrix_trans_dot_openacc(Z2, W2, G1, act_batch, k, l);
        for (size_t h = 0; h < act_batch * l; ++h){
            if (Z1[h] > 0){
                Boole_Z1[h] = 1.0;
            } else {
                Boole_Z1[h] = 0.0;
            }
        }
        matrix_mul_openacc(G1, Boole_Z1, act_batch * l);

        // W1_l = (X_b.T dot G1)/batch * lr
        matrix_dot_trans_openacc(X_b, G1, W1_l, act_batch, n, l);
        matrix_div_scalar_openacc(W1_l, static_cast<float>(act_batch), n, l);
        matrix_mul_scalar_openacc(W1_l, lr, n, l);



        // W2_l = (Z1.T dot (Z2-Y))/batch * lr
        matrix_dot_trans_openacc(Z1, Z2, W2_l, act_batch, l, k);
        matrix_div_scalar_openacc(W2_l, static_cast<float>(act_batch), l, k);
        matrix_mul_scalar_openacc(W2_l, lr, l, k); 



        //Update W1, W2
        matrix_minus_openacc(W1, W1_l, n, l);
        matrix_minus_openacc(W2, W2_l, l, k);
    }
    delete[] Z1;
    delete[] Z2;
    delete[] Y;
    delete[] G1;
    delete[] Boole_Z1;
    delete[] W1_l;
    delete[] W2_l;
}

void train_nn_openacc(const DataSet *train_data, const DataSet *test_data, size_t num_classes, size_t hidden_dim, size_t epochs, float lr, size_t batch)
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
    size_t size_tr = train_data->images_num * num_classes;
    size_t size_te = test_data->images_num * num_classes;
    float *train_result = new float[size_tr];
    float *test_result = new float[size_te];
    float train_loss, train_err, test_loss, test_err;
    std::cout << "| Epoch | Train Loss | Train Err | Test Loss | Test Err |" << std::endl;
    std::chrono::milliseconds elapsed_time;
    // BEGIN YOUR CODE
  
    auto start_time = std::chrono::high_resolution_clock::now();
    for (size_t epoch = 0; epoch < epochs; epoch++)
    {

        nn_epoch_openacc(train_data->images_matrix, train_data->labels_array, W1, W2, train_data->images_num, train_data->input_dim, hidden_dim, num_classes, lr, batch);
        
        // result = relu(X @ W1) @ W2
        // Train
        matrix_dot_openacc(train_data->images_matrix, W1, Temp_tr, train_data->images_num, train_data->input_dim, hidden_dim);
        for (size_t j = 0; j < train_data->images_num * hidden_dim; ++j){
            if (Temp_tr[j] < 0.0){
                Temp_tr[j] = 0.0;
            }
        }
        matrix_dot_openacc(Temp_tr, W2, train_result, train_data->images_num, hidden_dim, num_classes);

        // Test
        matrix_dot_openacc(test_data->images_matrix, W1, Temp_te, test_data->images_num, test_data->input_dim, hidden_dim);
        for (size_t j = 0; j < test_data->images_num * hidden_dim; ++j){
            if (Temp_te[j] < 0.0){
                Temp_te[j] = 0.0;
            }
        }
        matrix_dot_openacc(Temp_te, W2, test_result, test_data->images_num, hidden_dim, num_classes);
        



        train_loss = mean_softmax_loss_openacc(train_result, train_data->labels_array, train_data->images_num, num_classes);
        test_loss = mean_softmax_loss_openacc(test_result, test_data->labels_array, test_data->images_num, num_classes);
        train_err = mean_err_openacc(train_result, train_data->labels_array, train_data->images_num, num_classes);
        test_err = mean_err_openacc(test_result, test_data->labels_array, test_data->images_num, num_classes);
        std::cout << "|  " << std::setw(4) << std::right << epoch << " |    "
                  << std::fixed << std::setprecision(5) << train_loss << " |   "
                  << std::fixed << std::setprecision(5) << train_err << " |   "
                  << std::fixed << std::setprecision(5) << test_loss << " |  "
                  << std::fixed << std::setprecision(5) << test_err << " |" << std::endl;
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    elapsed_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_time -
                                                              start_time);
    std::cout << "Execution Time: " << elapsed_time.count()
              << " milliseconds\n";
  
    // END YOUR CODE
    delete[] W1;
    delete[] W2;
    delete[] train_result;
    delete[] test_result;
    delete[] Temp_tr;
    delete[] Temp_te;
}
