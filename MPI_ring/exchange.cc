#include <mpi.h>
#include <cstdio>
using namespace std;

// There is deliberate unsafe code in here ... where?

int main(int argc, char** argv) {
    MPI::Init(argc, argv);
    const int maxlen = 1024*1024;
    const int nproc = MPI::COMM_WORLD.Get_size();
    if (nproc != 2) MPI::COMM_WORLD.Abort(1);
    const int rank  = MPI::COMM_WORLD.Get_rank();
    const int other = (rank+1)%2;

    char* buf1 = new char[maxlen];
    char* buf2 = new char[maxlen];

    if (rank == 0) printf("trying 1\n");
    MPI::COMM_WORLD.Send(buf1, maxlen, MPI::BYTE, other, 1);
    MPI::COMM_WORLD.Recv(buf2, maxlen, MPI::BYTE, other, 1);

    if (rank == 0) printf("trying 2\n");
    MPI::COMM_WORLD.Sendrecv(buf1, maxlen, MPI::BYTE, other, 2,
                             buf2, maxlen, MPI::BYTE, other, 2);

    if (rank == 0) printf("trying 3\n");
    MPI::Request req = MPI::COMM_WORLD.Irecv(buf2, maxlen, MPI::BYTE, other, 3);
    MPI::COMM_WORLD.Send(buf1, maxlen, MPI::BYTE, other, 3);
    req.Wait();

    delete buf1;
    delete buf2;

    MPI::Finalize();
    return 0;
}
