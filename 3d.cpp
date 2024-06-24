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

    std::vector<Matrix> layered(info.num_layers());
    layered[info.layer()] = std::move(split[info.layer()]);

    std::vector<MatrixInfo> init_send(info.num_layers());
    std::vector<MatrixInfo> init_recv(info.num_layers());
    for (int i = 0; i < info.num_layers(); i++) {
        if (i != info.layer()) {
            split[i].init_send(grid * i + fiber, init_send[i]);
            layered[i].init_receive(grid * i + fiber, init_recv[i]);
        }
    }

    std::vector<MPI_Request> send, recv;
    send.reserve(info.num_layers() - 1);
    recv.reserve(info.num_layers() - 1);
    for (int i = 0; i < info.num_layers(); i++) {
        if (i != info.layer()) {
            send.push_back({});
            recv.push_back({});
            split[i].send(grid * i + fiber, init_send[i], send.back());
            layered[i].receive(grid * i + fiber, init_recv[i], recv.back());
        }
    }

    MPI_Waitall(info.num_layers() - 1, send.data(), MPI_STATUS_IGNORE);
    MPI_Waitall(info.num_layers() - 1, recv.data(), MPI_STATUS_IGNORE);

    return Matrix::merge(layered);
}