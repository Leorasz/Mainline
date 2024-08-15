#ifndef __MATHER_H__
#define __MATHER_H__

#ifdef __cplusplus
extern "C" {
#endif

struct mather;
typedef struct mather mather_t;

mather_t *mather_create(int start);
void mather_destroy(mather_t *m);

void mather_add(mather_t *m, int val);
int mather_val(mather_t *m);

#ifdef __cplusplus
}
#endif

#endif /* __MATHER_H__ */