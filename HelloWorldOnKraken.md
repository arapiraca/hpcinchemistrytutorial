
```
> module swap PrgEnv-pgi PrgEnv-gnu
> module load acml
> scp a@cyber.chem.utk.edu:helloworld.cc .
a@cyber.chem.utk.edu's password: 
helloworld.cc                            100%  266     0.3KB/s   00:00    

> CC helloworld.cc 
/opt/cray/xt-asyncpe/3.0/bin/CC: INFO: linux target is being used

> qsub -I -V -lsize=16,walltime=00:10:00
qsub: waiting for job 37154.nid00004 to start
qsub: job 37154.nid00004 ready

> cd /lustre/scratch/wrk0168/
> cp ~/a.out .
> aprun -n 16 -d 1 ./a.out
```