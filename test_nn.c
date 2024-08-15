#include <stdio.h>
#include <stdlib.h>
#include "Mainline/nn.h"

int main() {
    int layer_dims[] = {12,6,1};
    NeuralNet* netty = createNet(4, layer_dims, 3, reLU, sigmoid);
    for (int i = 0; i<100; i++) {
        float a[] = {1,2,3,4};
        int a_shape[] = {1,4};
        Tensor* at = createTensor(a, a_shape, 2);
        Tensor* out = netForward(netty, at);
        printf("The value of out is %f\n", item(out));
        out->grad = scalarMultCore(-1.0, out->data, 1);
        printf("Out grad is %f\n", out->grad[0]);
        backward(out);
        step(out, 0.01);
    }
    return 0;
}