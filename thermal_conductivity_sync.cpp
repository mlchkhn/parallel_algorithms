#include <iostream>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <chrono>
#include <mpi.h>

/////////////////////////////////////////////////////////////////
/// N должно быть у меня таким, чтобы height было не меньше 2 ///
/// Чётные отправляют первыми, нечётные получают первыми      ///
/////////////////////////////////////////////////////////////////

int main(int argc, char *argv[]) {
    int num, rank;
    const int N = atoi(argv[1]);
    const double T = atof(argv[2]);
    double a = 0, b = 1, c = 1, phi = 0, l = 1;
    const double h = l / N, tau = h * h * 0.3 / (c * c);
    const int M = (int) (T / tau);
    auto start = std::chrono::steady_clock::now();

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    const uint8_t main_thread = 0; // будет главным
    // rank-ый поток будет иметь столько строчек для расчёта:
    int height = N / num + (rank < (N % num) ? 1 : 0);
    std::vector<double> u_prev(height, phi);
    std::vector<double> u_next(height, 0.);

    if (num == 1) {
        u_prev[0] = a, u_prev[N - 1] = b;
        u_next[0] = a, u_next[N - 1] = b;

        for (int m = 0; m < M; ++m) {
            for (int n = 1; n < height - 1; ++n)
                u_next[n] = u_prev[n] + 0.3 * (u_prev[n + 1] - 2. * u_prev[n] + u_prev[n - 1]);
            std::move(u_next.begin(), u_next.end(), u_prev.begin());
        }
        for (int j = 0; j < height; ++j) std::cout << u_prev[j] << "\n";
    }
    else if (rank == 0) {
        u_prev[0] = u_next[0] = a;
        double bottom = 0;
        // printf("%d\n", M);
        for (int m = 0; m < M; ++m) {
            MPI_Ssend(&u_prev[height - 1], 1, MPI_DOUBLE, rank + 1, m, MPI_COMM_WORLD);
            // получаем то, что необходимо нам на m-ом шаге от нижнего соседа
            MPI_Recv(&bottom, 1, MPI_DOUBLE, rank + 1, m, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            u_next[height - 1] = u_prev[height - 1] + 0.3 * (bottom - 2 * u_prev[height - 1] + u_prev[height - 2]);

            // считаем то, что не требует инфы от соседей, в том числе верхнюю строчку
            for (int n = 1; n < height - 1; ++n) // u_next[0] = u_prev[0] = const = a
                u_next[n] = u_prev[n] + 0.3 * (u_prev[n + 1] - 2 * u_prev[n] + u_prev[n - 1]);

            std::move(u_next.begin(), u_next.end(), u_prev.begin());
        }
        for (int j = 0; j < height; ++j) std::cout << u_prev[j] << "\n";
    }
    else if (rank == num - 1) {
        u_prev[height - 1] = u_next[height - 1] = b;
        double up = 0;
        for (int m = 0; m < M; ++m) {
            if (rank % 2) {
                // получаем то, что необходимо нам на m-ом шаге от верхнего соседа
                MPI_Recv(&up, 1, MPI_DOUBLE, rank - 1, m, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Ssend(&u_prev[0], 1, MPI_DOUBLE, rank - 1, m, MPI_COMM_WORLD);
            }
            else {
                MPI_Ssend(&u_prev[0], 1, MPI_DOUBLE, rank - 1, m, MPI_COMM_WORLD);
                // получаем то, что необходимо нам на m-ом шаге от верхнего соседа
                MPI_Recv(&up, 1, MPI_DOUBLE, rank - 1, m, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            u_next[0] = u_prev[0] + 0.3 * (u_prev[1] - 2 * u_prev[0] + up);

            // считаем то, что не требует инфы от соседей, в том числе нижнюю строчку
            for (int n = 1; n < height - 1; ++n)
                u_next[n] = u_prev[n] + 0.3 * (u_prev[n + 1] - 2 * u_prev[n] + u_prev[n - 1]);

            std::move(u_next.begin(), u_next.end(), u_prev.begin());
        }
        MPI_Send( &u_prev[0], height, MPI_DOUBLE, main_thread, M + 1, MPI_COMM_WORLD);
    }
    else {
        double up = 0, bottom = 0;
        for (int m = 0; m < M; ++m) {
            if (rank % 2) {
                // получаем то, что необходимо нам на m-ом шаге от верхнего соседа
                MPI_Recv(&up, 1, MPI_DOUBLE, rank - 1, m, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Ssend(&u_prev[0], 1, MPI_DOUBLE, rank - 1, m, MPI_COMM_WORLD);
                MPI_Recv(&bottom, 1, MPI_DOUBLE, rank + 1, m, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Ssend(&u_prev[height - 1], 1, MPI_DOUBLE, rank + 1, m, MPI_COMM_WORLD);
            }
            else {
                MPI_Ssend(&u_prev[height - 1], 1, MPI_DOUBLE, rank + 1, m, MPI_COMM_WORLD);
                MPI_Recv(&bottom, 1, MPI_DOUBLE, rank + 1, m, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Ssend(&u_prev[0], 1, MPI_DOUBLE, rank - 1, m, MPI_COMM_WORLD);
                MPI_Recv(&up, 1, MPI_DOUBLE, rank - 1, m, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            u_next[0] = u_prev[0] + 0.3 * (u_prev[1] - 2 * u_prev[0] + up);
            u_next[height - 1] = u_prev[height - 1] + 0.3 * (bottom - 2 * u_prev[height - 1] + u_prev[height - 2]);

            // считаем то, что не требует инфы от соседей
            for (int n = 1; n < height - 1; ++n)
                u_next[n] = u_prev[n] + 0.3 * (u_prev[n + 1] - 2 * u_prev[n] + u_prev[n - 1]);

            std::move(u_next.begin(), u_next.end(), u_prev.begin());
        }
        MPI_Send( &u_prev[0], height, MPI_DOUBLE, main_thread, M + 1, MPI_COMM_WORLD);
    }
    if (rank == main_thread) {
        int height_temp = 0;
        for (int r = 1; r < num; ++r) {
            height_temp = N / num + (r < (N % num) ? 1 : 0);
            std::vector<double> u_prev_temp(height_temp, 0.);
            MPI_Recv(&u_prev_temp[0], height_temp, MPI_DOUBLE, r, M + 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int j = 0; j < height_temp; ++j) std::cout << u_prev_temp[j] << "\n";
        }
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        std::cout << "\nelapsed time: " << elapsed_seconds.count() << "s\n";
    }
    MPI_Finalize();

    return 0;
}