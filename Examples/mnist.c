#include "../Mainline/nn.h"

int main() {
    FILE* file = fopen("../Data/mnist_train.csv", "r");
    char header[10000];
    fgets(header, sizeof(header), file);
    float* data = (float*)malloc(60000*784*sizeof(float));
    float* labels = (float*)malloc(60000*10*sizeof(float));
    for (int i = 0; i < 60000; i++) {
        char line_buffer[2500];
        fgets(line_buffer, sizeof(line_buffer), file);
        char* label = strtok(line_buffer, ",");
        for (int j = 0; j < 10; j++) {
            labels[i*10+j] = 0.0;
        }
        int change_spot = i*10+atoi(label);
        labels[change_spot] = 1.0;
        for (int j = 0; j < 784; j++) {
            data[i*784+j] = atof(strtok(NULL, ","))/256.0;
        }
    }
    
    int data_shape[] = {60000, 784};
    int labels_shape[] = {60000, 10};
    Tensor* data_tensor = createTensor(data, data_shape, 2);
    Tensor* labels_tensor = createTensor(labels, labels_shape, 2);
    int split_count;
    int batch_size = 256;
    Tensor** data_batches = split(data_tensor, batch_size, &split_count);
    Tensor** labels_batches = split(labels_tensor, batch_size, &split_count);

    int net_shape[] = {10};
    NeuralNet* netty = createNet(784, net_shape, 1, reLU, last_softmax);
    int epochs = 100;
    float lr = 3e-3;
    for (int i = 0; i < epochs; i++) {
        printf("Onto epoch %i\n", i);
        for (int j = 0; j < split_count; j++) {
            printf("Batch number %i\n", j);
            Tensor* out = netForward(netty, data_batches[j]);
            Tensor* entropy = crossEntropy(labels_batches[j], out, -1);
            Tensor* loss = fullMean(entropy);
            printf("Loss is %f\n", item(loss));
            backward(loss);
            step(loss, lr);
        }
    }
    return 0;
}
