#include <stdio.h>
#include <stdlib.h>
#include "Mainline/tensor.h"

int main() {
    float a[] = {1,2,3,4,5,6};
    float b[] = {7,8,12,12};
    int a_shape[] = {3,2};
    int b_shape[] = {2,2};
    Tensor* at = createTensor(a, a_shape, 2);
    Tensor* bt = createTensor(b, b_shape, 2);
    Tensor* ct = matmul(at, bt);
    printf("%f\n", item(full_mean(ct))); 
    return 0;
    float ct_grad[] = {1.0/6,1.0/6,1.0/6,1.0/6,1.0/6,1.0/6};
    ct->grad = ct_grad;
    matmulBackward(ct);
    for (int i = 0; i < bt->numel; i++) {
        printf("%f\n", bt->grad[i]);
    }
}