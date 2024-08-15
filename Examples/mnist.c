#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main() {
    FILE *trainp = fopen("../Data/mnist_train.csv", "r");
    FILE *testp = fopen("../Data/mnist_test.csv", "r");
    if (!trainp) {
        printf("Can't open train data\n");
        return 1;
    }
    if (!testp) {
        printf("Can't open test data\n");
        return 1;
    }

    size_t batch_size = 256;
    size_t input_size = 784;
    size_t buffer_size = batch_size*(input_size+1);

    char buf[buffer_size];

    while (fgets(buf, buffer_size, trainp)) {
        //for each batch
        printf("new batch\n");
        int counter = 0;
        int* num = (int*)malloc((input_size+1)*sizeof(int));
        char* field = strtok(buf, ",");
        while(field) {
            num[counter] = (int)field;
            field = strtok(NULL, ",");
            counter++;
        }
        for (int i = 0; i < input_size+1; i++) {
            printf("%i\n",num[i]);
        }
        printf("the counter is %i\n", counter); 
        break;
    }

    fclose(trainp);
    fclose(testp);
    return 0;
}
