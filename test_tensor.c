#include <stdio.h>
#include <stdlib.h>
#include "Mainline/tensor2.h"

int main() {
    float a[] = {1,2,3,4,5,6};
    float b[] = {-1,0,-2,1};
    float c[] = {-1,1};
    int a_shape[] = {3,2};
    int b_shape[] = {2,2};
    int c_shape[] = {2,1};
    Tensor* at = createTensor(a, a_shape, 2);
    Tensor* bt = createTensor(b, b_shape, 2);
    Tensor* ct = createTensor(c, c_shape, 2); 
    Tensor* dt = matmul(at,bt);
    printTensor(dt);
    Tensor* ft = matmul(dt, ct);
    float g[] = {0,0,1};
    int g_shape[] = {3,1};
    Tensor* gt = createTensor(g, g_shape, 2);
    Tensor* ht = elementwiseMult(ft, gt);
    Tensor* final = fullSum(ht);
    final->grad = final->data;
    backward(final);
    printf("h\n");
    printGrad(ht);
    printf("g\n");
    printGrad(gt);
    printf("f\n");
    printGrad(ft);
    printf("d\n");
    printGrad(dt);
    printf("c\n");
    printGrad(ct);
    printf("b\n");
    printGrad(bt);
    printf("a\n");
    printGrad(at);
    step(final, 1);
    printTensor(at);
    printTensor(bt);
    printTensor(ct);
    return 0; 

}