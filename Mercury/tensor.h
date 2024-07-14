#include <stdio.h>
#include <stdlib.h>

typedef struct Tensor {
    float* data;
    float* grad;
    int size;
    struct Tensor** prev;
    void (*backer)(struct Tensor*, float*, float*);
} Tensor;

Tensor** bin(Tensor* a, Tensor* b) {
    Tensor** c = malloc(2*sizeof(Tensor*));
    c[0] = a;
    c[1] = b;
    return c;
}

Tensor* createTensor(float* data, int data_size) {
    Tensor* a = malloc(sizeof(Tensor));
    a->data = data;
    float* grad = (float*)calloc(data_size, sizeof(Tensor));
    a->grad = grad;
    a->size = data_size;
    a->prev = NULL;
    a->backer = NULL;
    return a;
}

float item(Tensor* a) {
    if (a->size > 1) {
        fprintf(stderr, "Error: can't take item of tensor, size is %i", a->size);
    }
    return a->data[0];
}

float* scalarMultiplyF(float c, float* a, int a_size) {
    float* res = malloc(a_size*sizeof(float));
    for (int i = 0; i < a_size; i++) {
        res[i] = c * a[i];
    }
    return res;
}

void continueBack(Tensor* a) {
    if (a->prev != NULL) {
        a->prev[0]->backer(a->prev[0],a->grad,a->prev[1]->data);
        a->prev[1]->backer(a->prev[1],a->grad,a->prev[0]->data);
    }
}

void dotBackward(struct Tensor* a, float* ahead, float* other) {
    a->grad = scalarMultiplyF(ahead[0], other, a->size);
    // continueBack(a);
}

void sumBackward(struct Tensor* a, float* ahead, float* other) {
    a->grad = ahead;
    continueBack(a);
}

Tensor* dot(Tensor* a, Tensor* b) {
    if (a->size != b->size) {
        fprintf(stderr, "Error: a is of size %i while b is of size %i, can't dot product\n", a->size, b->size);
    }
    float c[] = {0};
    for (int i = 0; i < a->size; i++) {
        c[0] += a->data[i] * b->data[i];
    }
    Tensor* res = createTensor(c, 1); 
    res->prev = bin(a,b);
    res->backer = dotBackward;
    return res;

}

Tensor* sum(Tensor* a, Tensor* b) {
    if (a->size != b->size) {
        fprintf(stderr, "Error: a is of size %i while b is of size %i, can't dot product\n", a->size, b->size);
    }
    float* c = malloc(a->size*sizeof(float));
    for (int i = 0; i < a->size; i++) {
        c[i] = a->data[i] + b->data[i];
    }
    Tensor* res = createTensor(c, a->size);
    res->prev = bin(a,b);
    res->backer = sumBackward;
    return res;
}

void backward(Tensor* a) {
    float grad[] = {1};
    a->grad = grad;
    continueBack(a);
}

