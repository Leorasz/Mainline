#include <stdio.h>
#include <stdlib.h>
#include "Mercury/tensor.h"

int main() {
    float a[] = {1,2,3};
    float b[] = {4,5,6};
    float d[] = {3,2,1};
    Tensor* at = createTensor(a, 3);
    Tensor* bt = createTensor(b, 3);
    Tensor* ct = sum(at,bt);
    Tensor* dt = createTensor(d, 3);
    Tensor* et = dot(ct, dt);
    // backward(et);
    // printf("%f\n", at->grad[0]);
}