#include <stdio.h>
#include <stdlib.h>
#include "Mainline/tensor.h"

// void printFloatP(float* a, int numel) {
//     for (int i = 0; i < numel; i++) {
//         printf("%f\n", a[i]);
//     }
//     printf("----------\n");
// }
int main() {
    float a[] = {1,2,-2,1,3,-1};
    float b[] = {-1,0,-2,1};
    float d[] = {-1,1};
    int a_shape[] = {3,2};
    int b_shape[] = {2,2};
    int d_shape[] = {2,1};
    Tensor* at = createTensor(a, a_shape, 2);
    Tensor* bt = createTensor(b, b_shape, 2);
    Tensor* dt = createTensor(d, d_shape, 2);
    Tensor* ct = matmul(at, bt);
    printf("C:\n");
    printFloatP(ct->data, ct->numel);
    Tensor* et = reLU(ct);
    printf("e:\n");
    printFloatP(et->data, et->numel);
    Tensor* ft = matmul(et, dt);
    printf("f:\n");
    printFloatP(ft->data, ft->numel);
    Tensor* gt = sigmoid(ft);
    printf("g:\n");
    printFloatP(gt->data, gt->numel);
    Tensor* mt = full_mean(gt);
    printf("The mean is %f\n", item(mt));
    backward(mt);
    printf("g grad\n");
    printFloatP(gt->grad, gt->numel);
    printf("f grad\n");
    printFloatP(ft->grad, ft->numel);
    printf("e grad\n");
    printFloatP(et->grad, et->numel);
    printf("d grad\n");
    printFloatP(dt->grad, dt->numel);
    printf("c grad\n");
    printFloatP(ct->grad, ct->numel);

}