#include <stdio.h>
#include <stdlib.h>
#include "Mainline/tensor.h"

int main() {
    float a[] = {1,2,3,4,5,6};
    float b[] = {7,8,11,12};
    float d[] = {9,10};
    int a_shape[] = {3,2};
    int b_shape[] = {2,2};
    int d_shape[] = {2,1};
    Tensor* at = createTensor(a, a_shape, 2);
    Tensor* bt = createTensor(b, b_shape, 2);
    Tensor* dt = createTensor(d, d_shape, 2);
    Tensor* ct = matmul(at, bt);
    for (int i=0; i < ct->num_dims; i++) {
        printf("%i\n", ct->shape[i]);
    }
    printf("Past matmul1\n");
    Tensor* et = reLU(ct);
    for (int i=0; i < et->num_dims; i++) {
        printf("%i\n", et->shape[i]);
    }
    printf("%i\n", et->numel);
    printf("%i\n", et->num_dims);
    Tensor* ft = matmul(et, dt);
    printf("Past matmul2\n");
    Tensor* gt = sigmoid(ft);
    Tensor* mt = full_mean(gt);
    printf("The mean is %f\n", item(mt));
    backward(mt);
    for (int i=0; i < at->numel; i++) {
        printf("%f\n", at->grad[i]);
    }
    printf("--------\n");
    for (int i=0; i < bt->numel; i++) {
        printf("%f\n", bt->grad[i]);
    }
}