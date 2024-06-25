#include <mpi.h>

#include "2d.h"
#include "utils.h"

#include "3d.h"

Matrix summa3d(const char *path_a, const char *path_b) {
    auto partial_result = summa2d(path_a, path_b);
    auto split = partial_result.col_split();

    auto &info = MPIInfo::instance();

    int grid = info.pc() * info.pc();
    int fiber = info.rank() % grid;

    MPI_Comm fiber_comm;
    MPI_Comm_split(MPI_COMM_WORLD, fiber, info.layer(), &fiber_comm);

    std::vector<MatrixInfo> send_meta(info.num_layers());
    std::vector<MatrixInfo> recv_meta(info.num_layers());
    for (int i = 0; i < info.num_layers(); i++)
        send_meta[i] = split[i].get_info();

    MPI_Alltoall(send_meta.data(), sizeof(MatrixInfo), MPI_BYTE,
                 recv_meta.data(), sizeof(MatrixInfo), MPI_BYTE, fiber_comm);

    std::vector<int> sendcnt(info.num_layers()), sendoff(info.num_layers());
    std::vector<int> recvcnt(info.num_layers()), recvoff(info.num_layers());
    int total_send = 0, total_recv = 0;
    for (int i = 0; i < info.num_layers(); i++) {
        sendoff[i] = total_send;
        recvoff[i] = total_recv;
        sendcnt[i] = send_meta[i].cells() * sizeof(Cell);
        recvcnt[i] = recv_meta[i].cells() * sizeof(Cell);
        total_send += sendcnt[i];
        total_recv += recvcnt[i];
    }

    std::vector<Cell> send_data;
    send_data.reserve(total_send);
    for (int i = 0; i < info.num_layers(); i++)
        for (const auto &cell : split[i].cells())
            send_data.emplace_back(cell);

    /* deallocate */
    split = {};

    std::vector<Cell> recv_data(total_recv);
    MPI_Alltoallv(send_data.data(), sendcnt.data(), sendoff.data(), MPI_BYTE,
                  recv_data.data(), recvcnt.data(), recvoff.data(), MPI_BYTE, fiber_comm);

    for (int i = 1; i < info.num_layers(); i++) {
        assert(recv_meta[0].n() == recv_meta[i].n());
        assert(recv_meta[0].m() == recv_meta[i].m());
    }

    return Matrix::merge(recv_meta[0].n(), recv_meta[0].m(), std::move(recv_data));
}