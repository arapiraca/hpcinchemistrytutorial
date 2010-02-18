/*
see http://www.opengroup.org/onlinepubs/000095399/functions/posix_memalign.html
*/

#include <stdlib.h>
#include <string.h>
#include <assert.h>

// void *malloc(size_t size);

void* amalloc(size_t size)
{
    size_t alignment = 64;
    void* memptr;
    int status = posix_memalign(&memptr, alignment, size);
    assert(status==0);
    return memptr;
}

