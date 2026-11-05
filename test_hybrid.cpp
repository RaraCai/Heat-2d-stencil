
#include <iostream>
#include <mpi.h>
#include <omp.h>

int main(int argc, char** argv) {

    int provided;

    MPI_Init_thread(
        &argc,
        &argv,
        MPI_THREAD_FUNNELED,
        &provided
    );

    if (provided < MPI_THREAD_FUNNELED) {
        std::cerr << "MPI thread support insufficient!"
                  << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int num_threads = omp_get_num_threads();

        #pragma omp critical
        {
            std::cout
                << "Rank " << rank
                << " / " << size
                << ", Thread " << tid
                << " / " << num_threads
                << std::endl;
        }
    }

    MPI_Finalize();

    return 0;
}
