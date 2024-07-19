#include <stdio.h>
#include <stdlib.h>

typedef struct Tensor {
    float* data;
    float* grad;
    int* shape;
    int num_dims;
    int numel;
    struct Tensor** prev;
    void (*backer)(struct Tensor*);
} Tensor;

Tensor* createTensor(float* data, int* shape, int num_dims) {
    printf("%i begin func\n", shape[0]);
    printf("%i begin func\n", shape[1]);
    Tensor* a = (Tensor*)malloc(sizeof(Tensor));
    a->data = data;
    int numel = 1;
    for (int i = 0; i < num_dims; i++) {
        numel = numel * shape[i];
    }
    a->numel = numel;
    float* grad = (float*)calloc(numel, sizeof(float));
    a->grad = grad;
    printf("%i before shape\n", shape[0]);
    printf("%i before shape\n", shape[1]);
    a->shape = (int*)malloc(num_dims * sizeof(int));
    for (int i = 0; i < num_dims; i++) {
        a->shape[i] = shape[i];
    }
    printf("%i aftershape\n", shape[0]);
    printf("%i aftershape\n", shape[1]);
    a->num_dims = num_dims;
    a->prev = NULL;
    a->backer = NULL;
    return a;
}


Tensor* matmul(Tensor* a, Tensor* b) {
    if (a->num_dims != 2 || b->num_dims != 2) {
        fprintf(stderr, "Error: either a or b don't have two dims for matmul, a: %i, b: %i\n", a->num_dims, b->num_dims);
        exit(1);
    }
    if (a->shape[1] != b->shape[0]) {
        fprintf(stderr, "Error: shapes don't match for matmul, a1 is %i and b0 is %i\n", a->shape[1], b->shape[0]);
        exit(1);
    }
    int c_numel = a->shape[0] * b->shape[1];
    float* c = (float*)malloc(c_numel*sizeof(float));
    int pos_counter = 0;
    for (int i = 0; i < a->shape[0]; i++) {
        for (int j = 0; j < b->shape[1]; j++) {
            float entry = 0;
            for (int k = 0; k < a->shape[1]; k++) {
                entry += a->data[i*a->shape[1]+k]*b->data[k*b->shape[0]+j];
            }
            c[pos_counter] = entry;
            pos_counter++;
        }
    }
    int c_shape[] = {a->shape[0],b->shape[1]};
    Tensor* res = createTensor(c, c_shape, 2);
    return res;

}

Tensor* reLU(Tensor* a) {
    float* b = a->data;
    for (int i = 0; i < a->numel; i++) {
        if (b[i] < 0) {
            b[i] = 0;
        }
    }
    printf("%i before create\n", a->shape[0]);
    printf("%i before create\n", a->shape[1]);
    Tensor* res = createTensor(b, a->shape, a->num_dims);
    return res;
}
