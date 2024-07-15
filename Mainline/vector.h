#include <stdio.h>
#include <stdlib.h>

typedef struct Vector {
    float* data;
    float* grad;
    int size;
    struct Vector** prev;
    void (*backer)(struct Vector*, float*, float*);
} Vector;

Vector** binVector(Vector* a, Vector* b) {
    Vector** c = malloc(2*sizeof(Vector*));
    c[0] = a;
    c[1] = b;
    return c;
}

Vector* createVector(float* data, int data_size) {
    Vector* a = malloc(sizeof(Vector));
    a->data = data;
    float* grad = (float*)calloc(data_size, sizeof(Vector));
    a->grad = grad;
    a->size = data_size;
    a->prev = NULL;
    a->backer = NULL;
    return a;
}

float item(Vector* a) {
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

void continueBack(Vector* a) {
    if (a->backer != NULL) {
        a->backer(a->prev[0],a->grad,a->prev[1]->data);
        a->backer(a->prev[1],a->grad,a->prev[0]->data);
    }
}

void dotBackward(struct Vector* a, float* ahead, float* other) {
    a->grad = scalarMultiplyF(ahead[0], other, a->size);
    continueBack(a);
}

void sumBackward(struct Vector* a, float* ahead, float* other) {
    a->grad = ahead;
    continueBack(a);
}

Vector* dot(Vector* a, Vector* b) {
    if (a->size != b->size) {
        fprintf(stderr, "Error: a is of size %i while b is of size %i, can't dot product\n", a->size, b->size);
    }
    float c[] = {0};
    for (int i = 0; i < a->size; i++) {
        c[0] += a->data[i] * b->data[i];
    }
    Vector* res = createVector(c, 1); 
    res->prev = bin(a,b);
    res->backer = dotBackward;
    return res;

}

Vector* sum(Vector* a, Vector* b) {
    if (a->size != b->size) {
        fprintf(stderr, "Error: a is of size %i while b is of size %i, can't sum\n", a->size, b->size);
    }
    float* c = malloc(a->size*sizeof(float));
    for (int i = 0; i < a->size; i++) {
        c[i] = a->data[i] + b->data[i];
    }
    Vector* res = createVector(c, a->size);
    res->prev = bin(a,b);
    res->backer = sumBackward;
    return res;
}

void backward(Vector* a) {
    float grad[] = {1};
    a->grad = grad;
    continueBack(a);
}

