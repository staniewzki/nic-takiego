#pragma once

#include <cmath>
#include <cassert>
#include <mpi.h>

class MPIInfo {
  public:
    static MPIInfo &instance() {
        static MPIInfo info;
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

  private:
    MPIInfo() {
        MPI_Comm_size(MPI_COMM_WORLD, &num_procs_);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        pc_ = std::sqrt(num_procs_) + 0.5;
        assert(pc_ * pc_ == num_procs_);
    }

    int rank_, num_procs_, pc_;
};
