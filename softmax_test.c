#include <stdio.h>
#include <stdlib.h>
#include "Mainline/nn.h"

int main() {
    float a[] = {1,2,3,4};
    int a_shape[] = {2,2};
    Tensor* at = createTensor(a, a_shape, 2);
    Tensor* bt = sum(at, 1);
    printFloatP(bt->data, bt->numel);
    return 0;
}