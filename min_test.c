#include <stdio.h>
#include <stdlib.h>
#include "Mainline/min_tensor.h"

int main() {
    float a[] = {1,2,3,4,5,6};
    float b[] = {7,8,11,12};
    int a_shape[] = {3,2};
    int b_shape[] = {2,2};
    Tensor* at = createTensor(a, a_shape, 2);
    Tensor* bt = createTensor(b, b_shape, 2);
    Tensor* ct = matmul(at, bt);
    Tensor* et = reLU(ct);
    printf("%iaftersigfunc\n", et->shape[0]);
    printf("%iaftersigfunc\n", et->shape[1]);
    return 0;
}