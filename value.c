#include <stdio.h>
#include <stdlib.h>

typedef struct Value {
    float data;
    float grad;
    struct Value** prev;
    int genesis;
} Value;

Value* Val(float data) {
    Value* a = malloc(sizeof(Value));
    a->data = data;
    a->grad = 0.0;
    a->prev = NULL;
    a->genesis = 0;
    return a;
}

Value** Bin(Value* a, Value* b) {
    Value** bin = malloc(2*sizeof(Value*));
    bin[0] = a;
    bin[1] = b;
    return bin;
}

Value* add(Value* a, Value* b) {
    Value* c = Val(a->data + b->data);
    c->prev = Bin(a,b);
    a->genesis = 1;
    b->genesis = 1;
    return c;
}

Value* mult(Value* a, Value* b) {
    Value* c = Val(a->data * b->data);
    c->prev = Bin(a,b);
    a->genesis = 2;
    b->genesis = 2;
    return c;
}

void _backward(Value* a, float ahead, Value* other) {
    switch (a->genesis) {
        case 1:
            a->grad = ahead;
            break;
        case 2: 
            a->grad = other->data * ahead;
            break;
    }
    if (a->prev != NULL) {
        _backward(a->prev[0], a->grad, a->prev[1]);
        _backward(a->prev[1], a->grad, a->prev[0]);
    }
}

void backward(Value* a) {
    a->grad = 1.0;
    _backward(a->prev[0], 1.0, a->prev[1]);
    _backward(a->prev[1], 1.0, a->prev[0]);
}

int main() {
    Value* a = Val(4.0);
    Value* b = Val(2.0);
    Value* c = add(a,b);
    Value* d = Val(3.0);
    Value* e = mult(c,d);
    backward(e);
    printf("%f\n", d->grad);
    return 0;
}