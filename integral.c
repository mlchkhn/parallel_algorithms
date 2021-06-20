#include <stdio.h>
#include <mpi.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

double func(double x) {
    return sqrt(4 - x * x);
}

double piece(int num, int rank, const int N) {
    double sum = 0;
    double a = 0, b = 2, x1, x2, h = (b - a) / (double) N;
    int i;
    for (i = rank; i < N; i += num) {
        x1 = a + i * h;
        x2 = a + (i + 1) * h;
        sum += 0.5 * h * (func(x1) + func(x2));
    }
    return sum;
}

int main(int argc, char *argv[])
{
        int num, rank; // ----------------------------------------------------- ?
        const uint8_t main_thread = 0;
        const int N = atoi(argv[1]); // как не клоннировать ------------------- ?

        MPI_Init(&argc, &argv);
        double start = MPI_Wtime();
        MPI_Comm_size(MPI_COMM_WORLD, &num);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        if (rank == main_thread) {
                MPI_Status status; // --------------------------------------------- ?
                double ans = 0;
                ans += piece(num, rank, N);
                // tag ------------------------------------------------------------ ?
                int i;
                for (i = 1; i < num; ++i) {
                        double result = 0;
                        MPI_Recv(&result, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status);
                        ans += result;
                }
                printf("ans = %.8f\n", ans);
                double end = MPI_Wtime();
                printf("time = %.6f\n", end - start);
                //
        } else {
                double result = piece(num, rank, N);
                MPI_Bsend(&result, 1, MPI_DOUBLE, main_thread, 0, MPI_COMM_WORLD);
                // данные // (размер данных)/sizeof(тип) // тип данных // куда // идентификатор сообщения // коммуникатор
        }
        MPI_Finalize();                                                                                                                                                                                             return 0;
}