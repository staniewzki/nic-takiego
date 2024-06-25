#pragma once

#include <cmath>
#include <cassert>
#include <mpi.h>

class MPIInfo {
  public:
    /**
     * @brief Returns a singleton instance.
     *
     * When this method is called first time, layers_num must be passed.
     */
    static const MPIInfo &instance(int initial_layers_num = 0) {
        static MPIInfo info(initial_layers_num);
        return info;
    }

    MPIInfo(const MPIInfo &info) = delete;

    void operator=(const MPIInfo &) = delete;

    int num_procs() const {
        return num_procs_;
    }

    int rank() const {
        return rank_;
    }

    int pc() const {
        return pc_;
    }

    int num_layers() const {
        return num_layers_;
    }

    int row() const {
        return row_;
    }

    int col() const {
        return col_;
    }

    int layer() const {
        return layer_;
    }

  private:
    MPIInfo(int initial_layers_num) : num_layers_(initial_layers_num) {
        assert(num_layers_);
        MPI_Comm_size(MPI_COMM_WORLD, &num_procs_);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        int grid = num_procs_ / num_layers_;
        pc_ = std::sqrt(grid) + 0.5;
        assert(pc_ * pc_ * num_layers_ == num_procs_);

        layer_ = rank_ / grid;
        row_ = (rank_ % grid) / pc_;
        col_ = rank_ % pc_;
    }

    int rank_, num_procs_, pc_, num_layers_;
    int row_, col_, layer_;
};
