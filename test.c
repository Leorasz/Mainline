#include <stdio.h>
#include <stdlib.h>
#include "Mainline/vector.h"

int main() {
    float a[] = {1,2,3};
    float b[] = {4,5,6};
    float d[] = {3,2,1};
    Vector* at = createVector(a, 3);
    Vector* bt = createVector(b, 3);
    Vector* ct = sum(at,bt);
    Vector* dt = createVector(d, 3);
    Vector* et = dot(ct, dt);
    backward(et);
    printf("%f\n", dt->grad[1]);
}
