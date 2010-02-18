/*
author: azutomo Yoshii kazutomo@mcs.anl.gov
from http://www.mcs.anl.gov/~kazutomo/getticks.html
*/

unsigned long long getticks(void)
{
  unsigned hi, lo;
  __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
  return ( (unsigned long long)lo)|( ((unsigned long long)hi)<<32 );
}