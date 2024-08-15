#include <stdio.h>
#include "mather.h"

int main(int argc, char **argv)
{
	mather_t *m = mather_create(4);
	mather_add(m, 6);
	printf("%d\n", mather_val(m));
	mather_destroy(m);
	return 0;
}