#include <stdio.h>
#include <stdlib.h>
#include "Mainline/tensor.h"


int main() {
    float a[] = {0.5};
    int a_shape[] = {1};
    Tensor* at = createTensor(a, a_shape, 1);
    Tensor* bt = reLU(at);
    backward(bt);
    printFloatP(at->grad, 1);
    return 0;

}
