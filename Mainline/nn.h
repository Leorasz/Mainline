#include <stdio.h>
#include <stdlib.h>
#include "tensor.h"

typedef struct NeuralNet {
    struct Tensor** weights;
    struct Tensor** biases;
    int num_layers;
    struct Tensor* (*activator)(struct Tensor*);
    struct Tensor* (*last_activator)(struct Tensor*);
} NeuralNet;

NeuralNet* createNet(int input_d, int* layer_dims, int num_layers, Tensor* (*activator)(struct Tensor*), Tensor* (*last_activator)(struct Tensor*)) {
    NeuralNet* res = (NeuralNet*)malloc(sizeof(NeuralNet));
    Tensor** weights = (Tensor**)malloc(num_layers*sizeof(Tensor*));
    Tensor** biases = (Tensor**)malloc(num_layers*sizeof(Tensor*));
    for (int i = 0; i < num_layers; i++) {
        int* weights_shape = (int*)malloc(2*sizeof(int));
        if (i==0) {
            weights_shape[0] = input_d;
        } else {
            weights_shape[0] = layer_dims[i-1];
        }
        weights_shape[1] = layer_dims[i];
        weights[i] = randomTensor(weights_shape, 2);
        int* bias_shape = (int*)malloc(2*sizeof(int));
        bias_shape[0] = 1;
        bias_shape[1] = layer_dims[i];
        biases[i] = randomTensor(bias_shape, 2);
    }
    res->weights = weights;
    res->biases = biases;
    res->num_layers = num_layers;
    res->activator = activator;
    res->last_activator = last_activator;
    return res;
}

Tensor** netForward(NeuralNet* net, Tensor* input) {
    if (input->shape[1] != net->weights[0]->shape[0]) {
        fprintf(stderr, "Input dimension is off, input dimension 1 is %i while net weight 0 dimension 0 is %i", input->shape[1], net->weights[0]->shape[0]);
        exit(1);
    }
    int num_hold_spots = 3*(net->num_layers)+1;
    Tensor** holder = (Tensor**)malloc(num_hold_spots*sizeof(Tensor));
    holder[0] = input;
    int counter = 1;
    for (int i = 0; i < net->num_layers; i++) {
        Tensor* product = matmul(holder[counter-1],net->weights[i]);
        holder[counter] = product;
        counter++;
        Tensor* sum = add(product, net->biases[i]);
        holder[counter] = sum;
        counter++;
        Tensor* activated;
        if (i == net->num_layers - 1) {
            activated = net->last_activator(sum);
        } else {
            activated = net->activator(sum);
        }
        holder[counter] = activated;
        counter++;
    }
    return holder;
}