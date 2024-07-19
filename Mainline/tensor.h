#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void printFloatP(float* a, int numel) {
    for (int i = 0; i < numel; i++) {
        printf("%f\n", a[i]);
    }
    printf("----------\n");
}

typedef struct Tensor {
    float* data;
    float* grad;
    int* shape;
    int num_dims;
    int numel;
    struct Tensor** prev;
    int num_prev;
    void (*backer)(struct Tensor*);
} Tensor;

Tensor** singlify(Tensor* a) {
    Tensor** b = (Tensor**)malloc(sizeof(Tensor*));
    b[0] = a;
    return b;
}
Tensor** bin(Tensor* a, Tensor* b) {
    Tensor** c = (Tensor**)malloc(2*sizeof(Tensor*));
    c[0] = a;
    c[1] = b;
    return c;
}

float sum(float* a, int size) {
    float collector = 0;
    for (int i=0; i < size; i++) {
        collector += a[i];
    }
    return collector;
}

float* addF(float* a, float* b, int a_numel) {
    float* c = (float*)malloc(a_numel*sizeof(float));
    for (int i=0; i < a_numel; i++) {
        c[i] = a[i] + b[i];
    }
    return c;
}

float* oneArray(int numel) {
    float* res = (float*)malloc(numel*sizeof(float));
    for (int i=0; i < numel; i++) {
        res[i] = 1.0;
    }
    return res;
}

float* scalarMultiplyF(float c, float* a, int a_size) {
    float* res = malloc(a_size*sizeof(float));
    for (int i=0; i < a_size; i++) {
        res[i] = c * a[i];
    }
    return res;
}

int* shapeSwap(int* a) {
    int* res = (int*)malloc(2*sizeof(int));
    res[0] = a[1];
    res[1] = a[0];
    return res;
}

float* transposeF(float* a, int* a_shape) {
    float* res = (float*)malloc(a_shape[0]*a_shape[1]*sizeof(float));
    for (int i=0; i < a_shape[0]; i++) {
        for (int j = 0; j < a_shape[1]; j++) {
            res[j*a_shape[0]+i] = a[i*a_shape[1]+j];
        }
    }
    return res;
}

float sigmoidCore(float a) {
    return 1.0 / (1.0 + exp(-a));
}

Tensor* createTensor(float* data, int* shape, int num_dims) {
    Tensor* a = (Tensor*)malloc(sizeof(Tensor));
    a->data = data;
    int numel = 1;
    for (int i = 0; i < num_dims; i++) {
        numel = numel * shape[i];
    }
    a->numel = numel;
    float* grad = (float*)calloc(numel, sizeof(float));
    a->grad = grad;
    a->shape = (int*)malloc(num_dims * sizeof(int));
    for (int i = 0; i < num_dims; i++) {
        a->shape[i] = shape[i];
    }
    a->num_dims = num_dims;
    a->prev = NULL;
    a->num_prev = 0;
    a->backer = NULL;
    return a;
}

Tensor* randomTensor(int* shape, int num_dims) {
    int numel = 1;
    for (int i = 0; i < num_dims; i++) {
        numel = numel * shape[i];
    }
    float* data = (float*)malloc(numel*sizeof(float));
    for (int i = 0; i < numel; i++) {
        data[i] = ((float)rand()/(float)(RAND_MAX)) * 2.0 - 1.0;
    }
    Tensor* res = createTensor(data, shape, num_dims);
    return res;
}

float item(Tensor* a) {
    if (a->numel > 1) {
        fprintf(stderr, "Error: can't take item of tensor, numel is %i\n", a->numel);
        exit(1);
    }
    return a->data[0];
}

void shapeCompare(Tensor* a, Tensor* b, char* name) {
    if (a->num_dims != b->num_dims) {
        fprintf(stderr, "Error: number of dimensions for %s args don't match, %i and %i\n", name, a->num_dims, b->num_dims);
        exit(1);
    }
    for (int i; i < a->num_dims; i++) {
        if (a->shape[i] != b->shape[i]) {
            fprintf(stderr, "Error: sizes of dimension %i for %s args don't match, %i and %i\n", i, name, a->shape[i], b->shape[i]);
            exit(1);
        }
    }
}

void meanBackward(Tensor* a) {
    float squeezer = (float)a->numel / a->prev[0]->numel;
    float* grad = oneArray(a->prev[0]->numel);
    grad = scalarMultiplyF(squeezer, grad, a->prev[0]->numel);
    a->prev[0]->grad = addF(a->prev[0]->grad,grad,a->prev[0]->numel);
    if (a->prev[0]->backer != NULL) {
        a->prev[0]->backer(a->prev[0]);
    }
}

void addBackward(Tensor* a) {
    a->prev[0]->grad = addF(a->prev[0]->grad,a->grad,a->numel);
    a->prev[1]->grad = addF(a->prev[1]->grad,a->grad,a->numel);
    if (a->prev[0]->backer != NULL) {
        a->prev[0]->backer(a->prev[0]);
    }
    if (a->prev[1]->backer != NULL) {
        a->prev[1]->backer(a->prev[1]);
    }
}

float* matmulBackCore(float* a_grad, int* a_shape, float* ahead, float* other, int* other_shape) {
    for (int i = 0; i < other_shape[0]; i++) { //row of other
        for (int j = 0; j < other_shape[1]; j++) {
            int pos = i*other_shape[1]+j;
            for (int k = 0; k < a_shape[0]; k++) {
                a_grad[k*a_shape[1]+i] += other[pos] * ahead[k*other_shape[1]+j];
            }
        }
    }
    return a_grad;
}

float* matmulBackFlip(float* a_grad, int* a_shape, float* ahead, float* other, int* other_shape) {
    float* grad_flip = transposeF(a_grad, a_shape);
    int* shape_flip = shapeSwap(a_shape);
    float* other_flip = transposeF(other, other_shape);
    int* other_shape_flip = shapeSwap(other_shape);
    float* unflip_res = matmulBackCore(grad_flip, shape_flip, ahead, other_flip, other_shape_flip);
    float* res = transposeF(unflip_res, shape_flip);
    return res;
}

void matmulBackward(Tensor* a) {
    a->prev[0]->grad = matmulBackCore(a->prev[0]->grad, a->prev[0]->shape, a->grad, a->prev[1]->data, a->prev[1]->shape);
    if (a->prev[0]->backer != NULL) {
        a->prev[0]->backer(a->prev[0]);
    }
    a->prev[1]->grad = matmulBackFlip(a->prev[1]->grad, a->prev[1]->shape, a->grad, a->prev[0]->data, a->prev[0]->shape);
    if (a->prev[1]->backer != NULL) {
        a->prev[1]->backer(a->prev[1]);
    }
}

void reluBackward(Tensor* a) {
    float* grad = (float*)malloc(a->numel*sizeof(float));
    for (int i = 0; i < a->numel; i++) {
        if (a->prev[0]->data[i] > 0) {
            grad[i] = a->grad[i];
        } else {
            grad[i] = 0;
        }
    }
    a->prev[0]->grad = addF(a->prev[0]->grad, grad, a->numel);
    if (a->prev[0]->backer != NULL) {
        a->prev[0]->backer(a->prev[0]);
    }
}

void sigmoidBackward(Tensor* a) {
    float* grad = (float*)malloc(a->numel*sizeof(float));
    float* data = a->data;
    for (int i = 0; i < a->numel; i++) {
        grad[i] = data[i] * (1 - data[i]) * a->grad[i];
    }
    a->prev[0]->grad = addF(a->prev[0]->grad, grad, a->numel);
    if (a->prev[0]->backer != NULL) {
        a->prev[0]->backer(a->prev[0]);
    }
}

void backward(Tensor* a) {
    if (a->numel != 1) {
        fprintf(stderr, "Error: cannot backward a tensor with more than one element");
        exit(1);
    }
    float grad[] = {1};
    a->grad = grad;
    if (a->backer != NULL) {
        a->backer(a);
    } else {
        fprintf(stderr, "Error: cannot take backward of original tensor");
        exit(1);
    }
    float grad2[] = {1};
    a->grad = grad;
}

void step(Tensor* a, float lr) {
    float* diff = scalarMultiplyF(-lr, a->grad, a->numel);
    a->data = addF(a->data, diff, a->numel);
    for (int i = 0; i < a->num_prev; i++) {
        step(a->prev[i], lr);
    }
}

Tensor* add(Tensor* a, Tensor* b) {
    char name[] = "add";
    shapeCompare(a,b,name);
    float* c = (float*)malloc(a->numel*sizeof(float));
    for (int i = 0; i < a->numel; i++) {
        c[i] = a->data[i] + b->data[i];
    }
    Tensor* res = createTensor(c, a->shape, a->num_dims);
    res->prev = bin(a,b);
    res->num_prev = 2;
    res->backer = addBackward;
    return res;
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
                entry += a->data[i*a->shape[1]+k]*b->data[k*b->shape[1]+j];
            }
            c[pos_counter] = entry;
            pos_counter++;
        }
    }
    int c_shape[] = {a->shape[0],b->shape[1]};
    Tensor* res = createTensor(c, c_shape, 2);
    res->prev = bin(a,b);
    res->num_prev = 2;
    res->backer = matmulBackward;
    return res;

}

Tensor* mean(Tensor* a, int dim) {
    if (dim > a->num_dims-1) {
        fprintf(stderr, "Error: trying to take %i dim mean of tensor when it only has %i dims\n", dim, a->num_dims);
        exit(1);
    }
    int skip = 1;
    int skip_changer = a->num_dims-1;
    while (skip_changer > dim) {
        skip *= a->shape[skip_changer];
        skip_changer--;
    }
    int* new_shape = (int*)malloc((a->num_dims-1)*sizeof(int));
    int hit = 0;
    //skipping over the dimension that the mean is taken of
    for (int i = 0; i < a->num_dims; i++) {
        if (i == dim) {
            hit = 1;
        } else if (hit == 0) {
            new_shape[i] = a->shape[i];
        } else {
            new_shape[i-1] = a->shape[i];
        }
    }

    int new_numel = 1;
    for (int i = 0; i < a->num_dims -1; i++) {
        new_numel *= new_shape[i];
    }

    float* b = (float*)malloc(new_numel*sizeof(float));
    int* start_spots = (int*)malloc(new_numel*sizeof(int));
    int block_count = 0;
    int spot_count = 0;
    //finding the spots to start counting the mean from 
    for (int i=0; i < a->numel; spot_count++) {
        start_spots[spot_count] = i;
        block_count++;
        if (block_count >= skip) {
            i += -skip+skip*a->shape[dim];
            block_count = 0;
        } 
        i++;
    }
    //going to each spot and skipping the right amount for each element to account for a higher dimension being represented in 1D
    int divider = a->shape[dim];
    for (int i=0; i < new_numel; i++) {
        float collector = 0;
        for (int j=start_spots[i]; j < start_spots[i] + a->shape[dim]*skip; j+=skip) {
            collector += a->data[j];
        }
        collector /= divider;
        b[i] = collector;
    } 

    Tensor* res = createTensor(b, new_shape, a->num_dims-1);
    res->prev = singlify(a);
    res->num_prev = 1;
    res->backer = meanBackward;
    return res;

}

Tensor* full_mean(Tensor* a) {
    float collector[] = {0};
    for (int i = 0; i < a->numel; i++) {
        collector[0] += a->data[i];
    }
    collector[0] = collector[0] / a->numel;
    int res_shape[] = {1};
    Tensor* res = createTensor(collector, res_shape, 1);
    res->prev = singlify(a);
    res->num_prev = 1;
    res->backer = meanBackward;
    return res;
}

Tensor* reLU(Tensor* a) {
    float* b = (float*)malloc(a->numel*sizeof(float));
    for (int i = 0; i < a->numel; i++) {
        if (a->data[i] < 0) {
            b[i] = 0;
        } else {
            b[i] = a->data[i];
        }
    }
    Tensor* res = createTensor(b, a->shape, a->num_dims);
    res->prev = singlify(a);
    res->num_prev = 1;
    res->backer = reluBackward;
    return res;
}

Tensor* sigmoid(Tensor* a) {
    float* b = (float*)malloc(a->numel*sizeof(float));
    for (int i = 0; i < a->numel; i++) {
        b[i] = sigmoidCore(a->data[i]);
    }
    Tensor* res = createTensor(b, a->shape, a->num_dims);
    res->prev = singlify(a);
    res->num_prev = 1;
    res->backer = sigmoidBackward;
    return res;
}