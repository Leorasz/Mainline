#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/*
FUNCTION NAMING CONVENTIONS:
F- float -> float
Core- float* -> float*
Nothing- operation that either is input or outputs tensor(s), for user
Back- equation used for backprop
Backward- apply a back function to a tensor, returns void
*/

void printFloatP(float* a, int numel) {
    for (int i = 0; i < numel; i++) {
        printf("%f\n", a[i]);
    }
    printf("----------\n");
}

void printIntP(int* a, int numel) {
    for (int i = 0; i < numel; i++) {
        printf("%i\n", a[i]);
    }
    printf("----------\n");
}

float* fullSumCore(float* a, int numel) {
    float collector = 0;
    for (int i=0; i < numel; i++) {
        collector += a[i];
    }
    float* res = (float*)malloc(sizeof(float));
    res[0] = collector;
    return res;
}

float* dimSumCore(float* a, int* shape, int num_dims, int numel, int dim) {
    if (dim > num_dims-1) {
        fprintf(stderr, "Error: trying to take %i dim sum of float when it only has %i dims\n", dim, num_dims);
        exit(1);
    }
    int new_dim = dim;
    if (dim < 0) {
        new_dim = num_dims+dim;
        if (new_dim < 0) {
            fprintf(stderr, "Error: Dimension input is too negative for dim sum, dim was %i, turned into %i with num_dims of %i", dim, new_dim ,num_dims);
            exit(1);
        }
    }
    int skip = 1;
    int skip_changer = num_dims-1;
    while (skip_changer > new_dim) {
        skip *= shape[skip_changer];
        skip_changer--;
    }
    int* new_shape = (int*)malloc((num_dims-1)*sizeof(int));
    int hit = 0;
    //skipping over the dimension that the sum is taken of to create new shape
    for (int i = 0; i < num_dims; i++) {
        if (i == new_dim) {
            hit = 1;
        } else if (hit == 0) {
            new_shape[i] = shape[i];
        } else {
            new_shape[i-1] = shape[i];
        }
    }

    // int new_numel = 1;
    // for (int i = 0; i < num_dims -1; i++) {
    //     new_numel *= new_shape[i];
    // }
    int new_numel = numel / shape[new_dim];
    
    int* start_spots = (int*)malloc(new_numel*sizeof(int));
    int block_count = 0;
    int spot_pos = 0;
    //finding the spots to start counting the sum from 
    for (int i=0; i < numel; i++) {
        start_spots[i] = spot_pos;
        block_count++;
        if (block_count >= skip) {
            spot_pos += -skip+skip*shape[new_dim];
            block_count = 0;
        } 
        spot_pos++;
    }
    //going to each spot and skipping the right amount for each element to account for a higher dimension being represented in 1D
    int divider = shape[new_dim];
    float* res = (float*)malloc(new_numel*sizeof(float));
    for (int i=0; i < new_numel; i++) {
        float collector = 0;
        for (int j=start_spots[i]; j < start_spots[i] + shape[new_dim]*skip; j+=skip) {
            collector += a[j];
        }
        res[i] = collector;
    } 
    return res;
}

float* fullExpandCore(float a, int new_numel) {
    float* res = (float*)malloc(new_numel*sizeof(float));
    for (int i = 0; i < new_numel; i++) {
        res[i] = a;
    }
    return res;
}

float* dimExpandCore(float* a, int* shape, int numel, int num_dims, int expand_size, int expand_spot) {
    //in a's shape array, expand_size will be inserted at expand_spot
    if (expand_spot > num_dims) {
        fprintf(stderr, "Error: Can't expand at that dimension %i, there's only %i dimensions\n", expand_spot, num_dims);
        exit(1);
    }
    int new_spot = expand_spot;
    if (expand_spot < 0) {
        new_spot = num_dims+expand_spot+1;
        if (new_spot < 0) {
            fprintf(stderr, "Error: Expansion spot is too negative for expansion, expansion spot was %i, turned into %i with num_dims of %i\n", expand_spot, new_spot, num_dims);
            exit(1);
        }
    }
    int new_numel = expand_size*numel;
    float* res = (float*)malloc(new_numel*sizeof(float));
    int skip = 1;
    int skip_changer = num_dims-1;
    while (skip_changer >= new_spot) {
        skip *= shape[skip_changer];
        skip_changer--;
    }

    int* start_spots = (int*)malloc(numel*sizeof(int));
    int block_count = 0;
    int spot_pos = 0;
    //finding the spots to start filling from
    for (int i=0; i < numel; i++) {
        start_spots[i] = spot_pos;
        block_count++; 
        if (block_count>=skip) {
            spot_pos += -skip+skip*expand_size;
            block_count = 0;
        }
        spot_pos++;
    }

    for (int i = 0; i < numel; i++) {
        for (int j = start_spots[i]; j < start_spots[i]+skip*expand_size; j+=skip) {
            res[j] = a[i];
        }
    }

    return res;

}

float* addCore(float* a, float* b, int numel) {
    float* res = (float*)malloc(numel*sizeof(float));
    for (int i=0; i < numel; i++) {
        res[i] = a[i] + b[i];
    }
    return res;
}

float* elementwiseMultCore(float* a, float* b, int a_numel) {
    float* res = (float*)malloc(a_numel*sizeof(float));
    for (int i=0; i < a_numel; i++) {
        res[i] = a[i] * b[i];
    }
    return res;
}

float* scalarMultCore(float c, float* a, int a_numel) {
    float* res = (float*)malloc(a_numel*sizeof(float));
    for (int i=0; i < a_numel; i++) {
        res[i] = c * a[i];
    }
    return res;
}

float* onesCore(int numel) {
    float* res = (float*)malloc(numel*sizeof(float));
    for (int i=0; i < numel; i++) {
        res[i] = 1.0;
    }
    return res;
}

float* randomCore(int numel) {
    float* res = (float*)malloc(numel*sizeof(float));
    for (int i=0; i < numel; i++) {
        res[i] = ((float)rand()/(float)(RAND_MAX)) * 2.0 - 1.0;
    }
    return res;
}

int* shapeSwap2D(int* a) {
    int* res = (int*)malloc(2*sizeof(int));
    res[0] = a[1];
    res[1] = a[0];
    return res;
}

float* transposeCore(float* a, int* a_shape) {
    float* res = (float*)malloc(a_shape[0]*a_shape[1]*sizeof(float));
    for (int i=0; i < a_shape[0]; i++) {
        for (int j = 0; j < a_shape[1]; j++) {
            res[j*a_shape[0]+i] = a[i*a_shape[1]+j];
        }
    }
    return res;
}

//for the activation functions i do two approaches, the first uses less code but might be slower, the second uses more code but might be faster, to be tested
float sigmoidF(float a) {
    return 1.0 / (1.0 + exp(-a));
}

float sigmoidBackF(float a) {
    float sig = sigmoidF(a);
    return sig * (1 - sig);
}

float reLUF(float a) {
    if (a >= 0) {
        return a;
    } else {
        return 0.0;
    }
}

float reLUBackF(float a) {
    if (a >= 0) {
        return 1.0;
    } else {
        return 0.0;
    }
}

float* elementwiseCore(float* a, int numel, float (*funcF)(float)) {
    float* res = (float*)malloc(numel*sizeof(float));
    for (int i = 0; i < numel; i++) {
        res[i] = funcF(a[i]);
    }
    return res;
}

float* sigmoidCore(float* a, int numel) {
    float* res = (float*)malloc(numel*sizeof(float));
    for (int i = 0; i < numel; i++) {
        res[i] = 1.0 / (1.0 + exp(-a[i]));
    }
    return res;
}

float* sigmoidBackCore(float* a, int numel) {
    float* res = (float*)malloc(numel*sizeof(float));
    for (int i = 0; i < numel; i++) {
        float sig = 1.0 / (1.0 + exp(-a[i]));
        res[i] = sig * (1 - sig);
    }
    return res;
}

float* reLUCore(float* a, int numel) {
    float* res = (float*)malloc(numel*sizeof(float));
    for (int i = 0; i < numel; i++) {
        if (a[i] > 0) {
            res[i] = a[i];
        } else {
            res[i] = 0.0;
        }
    }
    return res;
}

float* reLUBackCore(float* a, int numel) {
    float* res = (float*)malloc(numel*sizeof(float));
    for (int i = 0; i < numel; i++) {
        if (a[i] > 0) {
            res[i] = a[i];
        } else {
            res[i] = 0.0;
        }
    }
    return res;
}

float* exponentiateCore(float* a, int numel) {
    float* res = (float*)malloc(numel*sizeof(float));
    for (int i = 0; i < numel; i++) {
        res[i] = exp(a[i]);
    }
    return res;
}

float* powerCore(float exponent, float* a, int numel) {
    float* res = (float*)malloc(numel*sizeof(float));
    for (int i = 0; i < numel; i++) {
        res[i] = powf(a[i], exponent);
    }
    return res;
} 

float* naturalLogCore(float* a, int numel) {
    float* res = (float*)malloc(numel*sizeof(float));
    for (int i = 0; i < numel; i++) {
        res[i] = log(a[i]);
    }
    return res;
}

float* matmulCore(float* a, float* b, int* a_shape, int* b_shape) {
    if (a_shape[1] != b_shape[0]) {
        fprintf(stderr, "Error: shapes don't match for matmul, a1 is %i and b0 is %i\n", a_shape[1], b_shape[0]);
        exit(1);
    }
    int res_numel = a_shape[0] * b_shape[1];
    float* res = (float*)malloc(res_numel*sizeof(float));
    int pos_counter = 0;
    for (int i = 0; i < a_shape[0]; i++) {
        for (int j = 0; j < b_shape[1]; j++) {
            float entry = 0;
            for (int k = 0; k < a_shape[1]; k++) {
                entry += a[i*a_shape[1]+k]*b[k*b_shape[1]+j];
            }
            res[pos_counter] = entry;
            pos_counter++;
        }
    }
    return res;
}

typedef struct Tensor {
    float* data;
    float* grad;
    int* shape;
    int num_dims;
    int numel;
    int requires_grad;
    struct Tensor** prev;
    int num_prev;
    void (*backer)(struct Tensor*);
    int other_data;
} Tensor;

void printTensor(Tensor* a) {
    printFloatP(a->data, a->numel);
    printIntP(a->shape, a->num_dims);
}

void printGrad(Tensor* a) {
    printFloatP(a->grad, a->numel);
}

Tensor** _single(Tensor* a) { //for putting a single tensor in prev
    Tensor** res = (Tensor**)malloc(sizeof(Tensor*));
    res[0] = a;
    return res;
}
Tensor** _double(Tensor* a, Tensor* b) { //for putting two tensors in prev
    Tensor** res = (Tensor**)malloc(2*sizeof(Tensor*));
    res[0] = a;
    res[1] = b;
    return res;
}

Tensor* createTensor(float* data, int* shape, int num_dims) {
    Tensor* res = (Tensor*)malloc(sizeof(Tensor));
    res->data = data;
    int numel = 1;
    for (int i = 0; i < num_dims; i++) {
        numel = numel * shape[i];
    }
    res->numel = numel;
    float* grad = (float*)calloc(numel, sizeof(float));
    res->grad = grad;
    res->shape = (int*)malloc(num_dims * sizeof(int));
    for (int i = 0; i < num_dims; i++) {
        res->shape[i] = shape[i];
    }
    res->num_dims = num_dims;
    res->requires_grad = 1;
    res->prev = NULL;
    res->num_prev = 0;
    res->backer = NULL;
    res->other_data = 0;
    return res;
}

Tensor* onesTensor(int* shape, int num_dims) {
    int numel = 1;
    for (int i = 0; i < num_dims; i++) {
        numel = numel * shape[i];
    } //this numel calculation gets repeated, but idk what to do
    float* data = onesCore(numel);
    Tensor* res = createTensor(data, shape, num_dims);
    return res;
}

Tensor* randomTensor(int* shape, int num_dims) {
    int numel = 1;
    for (int i = 0; i < num_dims; i++) {
        numel = numel * shape[i];
    } //this numel calculation gets repeated, but idk what to do
    float* data = randomCore(numel);
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

int elementwiseShapeCompare(Tensor* a, Tensor* b) {
    if (a->num_dims != b->num_dims) {
        return 1;
    }
    for (int i = 0; i < a->num_dims; i++) {
        if (a->shape[i] != b->shape[i]) {
            return 1;
        }
    }
    return 0;
}

void next_step(Tensor* a) {
    for (int i = 0; i < a->num_prev; i++) {
        if (a->prev[i]->backer != NULL && a->prev[i]->requires_grad == 1) {
            a->prev[i]->backer(a->prev[i]);
        }
    }
}

void powerBackward(Tensor* a) {
    //0 is the exponent, 1 is the base
    //d/dx a^x = a^x * ln(x)
    float* new_grad0 = elementwiseMultCore(a->grad, a->data, a->numel);
    float* nat = naturalLogCore(a->prev[1]->data, a->numel);
    new_grad0 = elementwiseMultCore(new_grad0, nat, a->numel);
    a->prev[0]->grad = fullSumCore(new_grad0, a->numel);
    float* new_grad1 = powerCore(item(a->prev[0])-1, a->prev[1]->data, a->numel);
    new_grad1 = scalarMultCore(item(a->prev[0]), new_grad1, a->numel);
    a->prev[1]->grad = elementwiseMultCore(new_grad1, a->grad, a->numel);
    next_step(a);
}

void exponentiateBackward(Tensor* a) {
    a->prev[0]->grad = elementwiseMultCore(a->grad, a->data, a->numel);
    next_step(a);
}

void dimExpandBackward(Tensor* a) {
    a->prev[0]->grad = dimSumCore(a->grad, a->shape, a->num_dims, a->numel, a->other_data);
    next_step(a);
}

void fullSumBackward(Tensor* a) {
    a->prev[0]->grad = fullExpandCore(item(a), a->prev[0]->numel);
    next_step(a);
}

void dimSumBackward(Tensor* a) {
    a->prev[0]->grad = dimExpandCore(a->grad, a->shape, a->numel, a->num_dims, a->prev[0]->shape[a->other_data],a->other_data);
    next_step(a);
}

void addBackward(Tensor* a) {
    a->prev[0]->grad = a->grad;
    a->prev[1]->grad = a->grad;
    next_step(a);
}

void elementwiseMultBackward(Tensor* a) {
    a->prev[0]->grad = elementwiseMultCore(a->grad, a->prev[1]->data, a->numel);
    a->prev[1]->grad = elementwiseMultCore(a->grad, a->prev[0]->data, a->numel);
    next_step(a);
}

void scalarMultBackward(Tensor* a) {
    float* new_grad0 = fullSumCore(a->prev[1]->data, a->numel);
    float* new_grad1 = fullExpandCore(a->prev[0]->data[0], a->numel);
    a->prev[0]->grad = elementwiseMultCore(a->grad, new_grad0, a->numel);
    a->prev[1]->grad = elementwiseMultCore(a->grad, new_grad1, a->numel);
    next_step(a);
}

void matmulBackward(Tensor* a) {
    float* flipped_other = transposeCore(a->prev[1]->data, a->prev[1]->shape);
    int* flipped_other_shape = shapeSwap2D(a->prev[1]->shape);
    a->prev[0]->grad = matmulCore(a->grad, flipped_other, a->shape, flipped_other_shape);
    flipped_other = transposeCore(a->prev[0]->data, a->prev[0]->shape);
    flipped_other_shape = shapeSwap2D(a->prev[0]->shape);
    a->prev[1]->grad = matmulCore(flipped_other, a->grad, flipped_other_shape, a->shape);
    next_step(a);
}

void sigmoidBackward(Tensor* a) {
    float* new_grad = sigmoidBackCore(a->prev[0]->data, a->numel);
    new_grad = elementwiseMultCore(new_grad, a->grad, a->numel);
    a->prev[0]->grad = new_grad;
    next_step(a);
}

void reLUBackward(Tensor* a) {
    float* new_grad = reLUBackCore(a->prev[0]->data, a->numel);
    new_grad = elementwiseMultCore(new_grad, a->grad, a->numel);
    a->prev[0]->grad = new_grad;
    next_step(a);
}

void backward(Tensor* a) {
    if (a->numel != 1) {
        fprintf(stderr, "Error: cannot backward a tensor with more than one element");
        exit(1);
    }
    if (a->backer != NULL) {
        a->backer(a);
    } else {
        fprintf(stderr, "Error: cannot take backward of original tensor");
        exit(1);
    }
} 

void step(Tensor* a, float lr) {
    if (a->requires_grad == 1) {
        float* diff = scalarMultCore(-lr, a->grad, a->numel);
        a->data = addCore(a->data, diff, a->numel);
        for (int i = 0; i < a->num_prev; i++) {
            step(a->prev[i], lr);
        }
    }
}

Tensor* power(Tensor* exponent, Tensor* a) {
    float* res_data = powerCore(item(exponent), a->data, a->numel);
    Tensor* res = createTensor(res_data, a->shape, a->num_dims);
    res->prev = _double(exponent, a);
    res->num_prev = 2;
    res->backer = powerBackward;
    return res;
}

Tensor* exponentiate(Tensor* a) {
    float* res_data = exponentiateCore(a->data, a->numel);
    Tensor* res = createTensor(res_data, a->shape, a->num_dims);
    res->prev = _single(a);
    res->num_prev = 1;
    res->backer = exponentiateBackward;
    return res;
}

Tensor* fullExpand(float a, int* shape, int num_dims) {
    int numel = 1;
    for (int i = 0; i < num_dims; i++) {
        numel *= shape[i];
    }
    float* res_data = fullExpandCore(a, numel);
    Tensor* res = createTensor(res_data, shape, num_dims);
    return res;
}

Tensor* dimExpand(Tensor* a, int expand_size, int expand_spot) {
    float* res_data = dimExpandCore(a->data, a->shape, a->numel, a->num_dims, expand_size, expand_spot);
    int new_spot = expand_spot;
    if (expand_spot < 0) {
        new_spot = a->num_dims+expand_spot+1;
    }
    int new_num_dims = a->num_dims+1;
    int* new_shape = (int*)malloc((new_num_dims)*sizeof(int));
    int hit = 0;
    for (int i = 0; i < new_num_dims; i++) {
        if (i == new_spot) {
            new_shape[i] = expand_size;
            hit = 1;
        }
        if (hit) {
            new_shape[i+1] = a->shape[i];
        } else {
            new_shape[i] = a->shape[i];
        }
    }
    Tensor* res = createTensor(res_data, new_shape, new_num_dims);
    res->prev = _single(a);
    res->num_prev = 1;
    res->backer = dimExpandBackward;
    res->other_data = expand_spot;
    return res;
}

Tensor* fullSum(Tensor* a) {
    float* res_data = fullSumCore(a->data, a->numel);
    int new_shape[] = {1};
    Tensor* res = createTensor(res_data, new_shape, 1);
    res->prev = _single(a);
    res->num_prev = 1;
    res->backer = fullSumBackward;
    return res;
}

Tensor* dimSum(Tensor* a, int dim) {
    float* res_data = dimSumCore(a->data, a->shape, a->num_dims, a->numel, dim);
    int new_dim = dim;
    if (dim < 0) {
        new_dim = a->num_dims+dim;
    }
    int new_num_dims = a->num_dims-1;
    int* new_shape = (int*)malloc((new_num_dims)*sizeof(int));
    int hit = 0;
    for (int i = 0; i < new_num_dims; i++) {
        if (i == new_dim) {
            hit = 1;
        }
        if (hit) {
            new_shape[i] = a->shape[i+1];
        } else {
            new_shape[i] = a->shape[i];
        }
    }
    Tensor* res = createTensor(res_data, new_shape, new_num_dims);
    res->prev = _single(a);
    res->num_prev = 1;
    res->backer = dimSumBackward;
    res->other_data = new_dim;
    return res;
}

Tensor* add(Tensor* a, Tensor* b) {
    if (elementwiseShapeCompare(a,b)) {
        fprintf(stderr, "Error: Can't add tensors because shapes don't match, a's shape is\n");
        for (int i = 0; i < a->num_dims; i++) {
            fprintf(stderr, "%i\n", a->shape[i]);
        }
        fprintf(stderr, "and b's shape is\n");
        for (int i = 0; i < a->num_dims; i++) {
            fprintf(stderr, "%i\n", a->shape[i]);
        }
        exit(1);
    }
    float* res_data = addCore(a->data, b->data, a->numel);
    Tensor* res = createTensor(res_data, a->shape, a->num_dims);
    res->prev = _double(a,b);
    res->num_prev = 2;
    res->backer = addBackward;
    return res;
}

Tensor* elementwiseMult(Tensor* a, Tensor* b) {
    if (elementwiseShapeCompare(a,b)) {
        fprintf(stderr, "Error: Can't multiply tensors because shapes don't match, a's shape is\n");
        for (int i = 0; i < a->num_dims; i++) {
            fprintf(stderr, "%i\n", a->shape[i]);
        }
        fprintf(stderr, "and b's shape is\n");
        for (int i = 0; i < a->num_dims; i++) {
            fprintf(stderr, "%i\n", a->shape[i]);
        }
        exit(1);
    }
    float* res_data = elementwiseMultCore(a->data, b->data, a->numel);
    Tensor* res = createTensor(res_data, a->shape, a->num_dims);
    res->prev = _double(a,b);
    res->num_prev = 2;
    res->backer = elementwiseMultBackward;
    return res; 
}

Tensor* dimMean(Tensor* a, int dim) {
    Tensor* sums = dimSum(a, dim);
    int new_dim = dim;
    if (dim < 0) {
        new_dim = a->num_dims+dim;
    }
    float divider = 1/a->shape[new_dim];
    Tensor* dividerT = fullExpand(divider, sums->shape, sums->num_dims);
    Tensor* res = elementwiseMult(sums, dividerT);
    return res;
}

Tensor* scalarMult(Tensor* a, Tensor* b) {
    if (a->numel != 1) {
        fprintf(stderr, "Error: Can't scalar multiply using A as scalar, its numel is %i, not 1", a->numel);
        exit(1);
    }
    float* res_data = scalarMultCore(item(a), b->data, a->numel);
    Tensor* res = createTensor(res_data, a->shape, a->num_dims);
    res->prev = _double(a,b);
    res->num_prev = 2;
    res->backer = scalarMultBackward;
    return res; 
}

Tensor* matmul(Tensor* a, Tensor* b) {
    if (a->num_dims != 2) {
        if (a->num_dims == 1) {
            fprintf(stderr, "Error: A only has one dimension, unable to perform matmul");
            exit(1);
        } else {
            fprintf(stderr, "Error: Matmul using tensors with more than two dimensions not supported yet, A has %i", a->num_dims);
            exit(1);
        }
    }
    if (b->num_dims != 2) {
        if (b->num_dims == 1) {
            fprintf(stderr, "Error: B only has one dimension, unable to perform matmul");
            exit(1);
        } else {
            fprintf(stderr, "Error: Matmul using tensors with more than two dimensions not supported yet, B has %i", a->num_dims);
            exit(1);
        }
    }
    float* res_data = matmulCore(a->data, b->data, a->shape, b->shape);
    int res_shape[] = {a->shape[0], b->shape[1]};
    Tensor* res = createTensor(res_data, res_shape, 2);
    res->prev = _double(a,b);
    res->num_prev = 2;
    res->backer = matmulBackward;
    return res;
}

Tensor* sigmoid(Tensor* a) {
    float* res_data = sigmoidCore(a->data, a->numel);
    Tensor* res = createTensor(res_data, a->shape, a->num_dims);
    res->prev = _single(a);
    res->num_prev = 1;
    res->backer = sigmoidBackward;
    return res;
}

Tensor* reLU(Tensor* a) {
    float* res_data = reLUCore(a->data, a->numel);
    Tensor* res = createTensor(res_data, a->shape, a->num_dims);
    res->prev = _single(a);
    res->num_prev = 1;
    res->backer = reLUBackward;
    return res;
}

Tensor* softmax(Tensor* a, int dim) {
    Tensor* ed = exponentiate(a);
    Tensor* summed = dimSum(ed, dim);
    float minus_one[] = {-1.0};
    int minus_shape[] = {1};
    Tensor* minus = createTensor(minus_one, minus_shape, 1);
    minus->requires_grad = 0;
    Tensor* flipped_summed = power(minus, summed);
    int new_dim = dim;
    if (dim < 0) {
        new_dim = a->num_dims+dim;
    }
    Tensor* re_expanded = dimExpand(flipped_summed, ed->shape[new_dim], new_dim);
    Tensor* res = elementwiseMult(ed, re_expanded);
    return res;
}