#include <array>
#include <mpi.h>

#include "2d.h"
#include "utils.h"

namespace {

struct DebugMutex {
    DebugMutex() {
        auto &info = MPIInfo::instance();
        int msg;

        if (info.rank() != 0) {
            MPI_Recv(&msg, 1, MPI_INT, info.rank() - 1, 111, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    ~DebugMutex() {
        auto &info = MPIInfo::instance();
        int msg;

        MPI_Send(&msg, 1, MPI_INT, (info.rank() + 1) % info.num_procs(), 111, MPI_COMM_WORLD);
        if (info.rank() == 0) {
            MPI_Recv(&msg, 1, MPI_INT, info.num_procs() - 1, 111, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
};

}

void summa2d(const char *path_a, const char *path_b, bool print_result,
             std::optional<double> g_value) {
    Matrix a = Matrix::read_and_distribute(path_a);
    Matrix b = Matrix::read_and_distribute(path_b);
    b.sort_by_cols();

    auto &info = MPIInfo::instance();

    {
        DebugMutex mtx;
        std::cerr << "[rank = " << info.rank() << "] a: " << a << "\n";
        std::cerr << "[rank = " << info.rank() << "] b: " << b << "\n";
    }

    int row = info.rank() / info.pc();
    int col = info.rank() % info.pc();

    MPI_Comm row_comm, col_comm;
    MPI_Comm_split(MPI_COMM_WORLD, row, col, &row_comm);
    MPI_Comm_split(MPI_COMM_WORLD, col, row, &col_comm);

    std::array<Matrix, 2> a_buffer, b_buffer;

    auto get_matrices = [&](int stage) {
        Matrix &a_cur = col == stage ? a : a_buffer[stage % 2];
        Matrix &b_cur = row == stage ? b : b_buffer[stage % 2];
        return std::pair<Matrix &, Matrix &>{a_cur, b_cur};
    };

    auto prepare_matrices = [&](int stage) {
        std::array<MPI_Request, 2> initial, broadcast;
        auto [a_cur, b_cur] = get_matrices(stage);

        a_cur.init_broadcast(col, stage, row_comm, &initial[0]);
        b_cur.init_broadcast(row, stage, col_comm, &initial[1]);

        a_cur.broadcast(col, stage, row_comm, &initial[0], &broadcast[0]);
        b_cur.broadcast(row, stage, col_comm, &initial[1], &broadcast[1]);

        return broadcast;
    };

    auto broadcast_request = prepare_matrices(0);

    std::vector<Matrix> intermediate_result(info.pc());
    for (int stage = 0; stage < info.pc(); stage++) {
        MPI_Wait(&broadcast_request[0], MPI_STATUS_IGNORE);
        MPI_Wait(&broadcast_request[1], MPI_STATUS_IGNORE);

        if (stage + 1 < info.pc())
            broadcast_request = prepare_matrices(stage + 1);

        auto [a_cur, b_cur] = get_matrices(stage);
        intermediate_result[stage] = a_cur * b_cur;
    }

    auto result = Matrix::merge(intermediate_result);

    {
        DebugMutex mtx;
        std::cerr << "[rank = " << info.rank() << "] result: " << result << "\n";
    }

    if (print_result) {
        /* TODO: printing */
    }

    if (g_value) {
        /* TODO: consider speeding this up */
        long long cnt = result.count_greater(*g_value);
        if (info.rank() == 0) {
            for (int i = 1; i < info.num_procs(); i++) {
                long long value;
                MPI_Recv(&value, 1, MPI_LONG_LONG, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                cnt += value;
            }
            std::cout << cnt << "\n";
        } else {
            MPI_Send(&cnt, 1, MPI_LONG_LONG, 0, 0, MPI_COMM_WORLD);
        }
    }
}
