#include <mpi.h>
#include <cstdio>
#include <algorithm>
using namespace std;

int main(int argc, char** argv) {
    MPI::Init(argc, argv);
    const int maxlen = 1024*1024;
    const int nproc = MPI::COMM_WORLD.Get_size();
    const int rank  = MPI::COMM_WORLD.Get_rank();
    const int left  = (rank+nproc-1)%nproc;
    const int right = (rank+1)%nproc;

    char* buf = new char[maxlen];

    if (rank == 0) {
        printf("   msglen     nloop      used    rate (byte/s)\n");
        printf("  -------  --------  --------  --------\n");
    }

    for (int msglen=1; msglen<=maxlen; msglen*=2) {
        double testim = (5e-6 + msglen*1e-9)*nproc;
        int nloop = max(0.1/testim,1.0);

        MPI::COMM_WORLD.Barrier();
        double used = MPI::Wtime();
        for (int loop=0; loop<nloop; loop++) {
            if (rank == 0) {
                MPI::COMM_WORLD.Send(buf, msglen, MPI::BYTE, right, 1);
                MPI::COMM_WORLD.Recv(buf, msglen, MPI::BYTE, left,  1);
            }
            else {
                MPI::COMM_WORLD.Recv(buf, msglen, MPI::BYTE, left,  1);
                MPI::COMM_WORLD.Send(buf, msglen, MPI::BYTE, right, 1);
            }
        }
        used = MPI::Wtime() - used;
        double rate = nloop*nproc*msglen/used;
        if (rank == 0) printf(" %8d  %8d  %.2e   %.2e\n", msglen, nloop, used, rate);
    }

    delete buf;

    MPI::Finalize();
    return 0;
}
