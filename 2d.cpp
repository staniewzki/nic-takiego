#include <array>
#include <mpi.h>

#include "utils.h"

#include "2d.h"

Matrix summa2d(const char *path_a, const char *path_b) {
    Matrix a = Matrix::read_and_distribute(path_a, SplitAlong::Col);
    Matrix b = Matrix::read_and_distribute(path_b, SplitAlong::Row);
    b.sort_by_cols();

    auto &info = MPIInfo::instance();

    MPI_Comm row_comm, col_comm;
    MPI_Comm_split(MPI_COMM_WORLD, info.layer() * info.pc() + info.row(), info.col(), &row_comm);
    MPI_Comm_split(MPI_COMM_WORLD, info.layer() * info.pc() + info.col(), info.row(), &col_comm);

    std::array<Matrix, 2> a_buffer, b_buffer;

    auto get_matrices = [&](int stage) {
        Matrix &a_cur = info.col() == stage ? a : a_buffer[stage % 2];
        Matrix &b_cur = info.row() == stage ? b : b_buffer[stage % 2];
        return std::pair<Matrix &, Matrix &>{a_cur, b_cur};
    };

    auto prepare_matrices = [&](int stage) {
        std::array<MatrixInfo, 2> meta;
        std::array<MPI_Request, 2> init, broadcast;
        auto [a_cur, b_cur] = get_matrices(stage);

        a_cur.init_broadcast(info.col(), stage, row_comm, meta[0], init[0]);
        b_cur.init_broadcast(info.row(), stage, col_comm, meta[1], init[1]);

        MPI_Wait(&init[0], MPI_STATUS_IGNORE);
        MPI_Wait(&init[1], MPI_STATUS_IGNORE);

        a_cur.broadcast(info.col(), stage, row_comm, meta[0], broadcast[0]);
        b_cur.broadcast(info.row(), stage, col_comm, meta[1], broadcast[1]);

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

    return Matrix::merge(std::move(intermediate_result));
}
