#include <stdio.h>
#include <stdlib.h>
#include "Mainline/nn.h"

int main() {
    int layer_dims[] = {12,6,1};
    NeuralNet* netty = createNet(4, layer_dims, 3, reLU, sigmoid);
    for (int i = 0; i<10; i++) {
        float a[] = {1,2,3,4};
        int a_shape[] = {1,4};
        Tensor* at = createTensor(a, a_shape, 2);
        Tensor** holdy = netForward(netty, at);
        int last = 3*netty->num_layers;
        Tensor* out = holdy[last];
        out->grad = out->data;
        printf("The value of out is %f\n", item(out));
        backward(out);
        step(out, 0.1);
        free(out);
        free(holdy);
        free(at);
    }
    return 0;
}