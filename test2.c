#include <stdio.h>
#include <stdlib.h>

float* funky() {
    float res[] = {1,2,3};
    return res;
}
int main() {
    float* a = funky();
    printf("aopidjgqpowiejgfqwe\n");
    printf("aiopwdjgapowijg\n");
    float b[] = {1,2,3,4,5,6,7};
    for (int i=0;i<3;i++) {
        printf("%f\n",a[i]);
    }
    return 0;
}