#include <stdio.h>
#include <stdlib.h>

#define TBB_ATOMICS

#if defined(TBB_ATOMICS)

#include <tbb/atomic.h>
typedef tbb::atomic<uint32_t> atomic_t;

#define get_atomic_val(m) \
({                                                      \
    m;                                                  \
})

#define atomic_set(m,val)                               \
({                                                      \
    m = val;                                            \
})
// effect: m++
#define atomic_inc(m)                                   \
({                                                      \
    m.fetch_and_increment();                            \
})
// effect:
//
#define atomic_lock(m)                                  \
({                                                      \
    while (m.compare_and_swap(1,0));                    \
})
// effect:
// if m=0 then m=1 and return 0
// if m=1 then         return 1
#define atomic_trylock(m)                               \
({                                                      \
    !m.compare_and_swap(1,0);                           \
})
// effect: m=0
#define atomic_unlock(m)                                \
({                                                      \
    m = 0;                                              \
})

#elif defined(BGP_ATOMICS)

#include <bpcore/bgp_atomic_ops.h>
typedef _BGP_Atomic atomic_t;

#define get_atomic_val(m) \
({                                                      \
    m.atom;                                             \
})

#define atomic_set(m, val)                              \
({                                                      \
    m.atom = val;                                       \
})
// effect: m++
#define atomic_inc(m)                                   \
({                                                      \
    _bgp_fetch_and_add(&m, 1);                          \
})
// effect:
// 
#define atomic_lock(m)                                  \
({                                                      \
  uint32_t zero = 0;                                    \
  while (!_bgp_compare_and_swap(&m, &zero, 1)) {zero=0;}\
})
// effect:
// if m=0 then m=1 and return 0
// if m=1 then         return 1
#define atomic_trylock(m)                               \
({                                                      \
  uint32_t zero = 0;                                    \
  _bgp_compare_and_swap(&m, &zero, 1);                  \
})
// effect: m=0
#define atomic_unlock(m)                                \
({                                                      \
    m.atom = 0;                                         \
})

#endif // TBB_ATOMICS


int main(){

    uint32_t i,j;
    atomic_t _lock,_copy;
    uint32_t l,c;
    uint32_t rc;

    printf("==================================\n");
    for (i=0;i<10;i++){
        j=i%2;
        atomic_set(_lock,j);
        l=get_atomic_val(_lock);
        printf("after atomic_set(_lock,%d), _lock = %d\n",j,l);
    }
    
    printf("==================================\n");
    for (i=0;i<10;i++){
        j=i%2;
        atomic_set(_lock,j);
        _copy=_lock;
        c=get_atomic_val(_copy);
        rc = atomic_trylock(_lock);
        l=get_atomic_val(_lock);
        printf("atomic_trylock before: _lock = %d after: _lock = %d return = %d\n",c,l,rc);
    }

    printf("==================================\n");
    for (i=0;i<10;i++){
        j=i%2;
        atomic_set(_lock,j);
        _copy=_lock;
        c=get_atomic_val(_copy);
        atomic_unlock(_lock);
        l=get_atomic_val(_lock);
        printf("atomic_unlock before: _lock = %d after: _lock = %d\n",c,l);
    }

    printf("==================================\n");
    for (i=0;i<10;i++){
        j=i%2;
        atomic_set(_lock,j);
        _copy=_lock;
        c=get_atomic_val(_copy);
        atomic_lock(_lock);
        l=get_atomic_val(_lock);
        printf("atomic_lock before: _lock = %d after: _lock = %d\n",c,l);
    }

    printf("==================================\n");
    return(0);
}














