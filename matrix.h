#pragma once

#include <array>
#include <iostream>
#include <vector>

struct Cell {
    uint32_t row, col;
    double value;
};

struct MatrixInfo {
    std::array<uint32_t, 3> data;
    MPI_Request request;

    MatrixInfo() = default;

    MatrixInfo(uint32_t n, uint32_t m, uint32_t cells) {
        data = {n, m, cells};
    }

    uint32_t n() const {
        return data[0];
    }

    uint32_t m() const {
        return data[1];
    }

    uint32_t cells() const {
        return data[2];
    }
};

enum class SplitAlong {
    Row,
    Col,
};

class Matrix {
  public:
    Matrix(uint32_t n, uint32_t m) : n_(n), m_(m) {}

    Matrix() = default;

    friend Matrix operator*(const Matrix &a, const Matrix &b);

    friend std::ostream& operator<<(std::ostream &stream, const Matrix &mat);

    static Matrix read_and_distribute(const char *filename, SplitAlong split = SplitAlong::Row);

    void sort_by_cols();

    void init_broadcast(int self, int root, MPI_Comm comm, MatrixInfo &info);

    void broadcast(int self, int root, MPI_Comm comm, MatrixInfo &info, MPI_Request &request);

    void init_send(int dest, MatrixInfo &info) const;

    void send(int dest, MatrixInfo &info, MPI_Request &request) const;

    void init_receive(int source, MatrixInfo &info);

    void receive(int source, MatrixInfo &info, MPI_Request &request);

    static Matrix merge(std::vector<Matrix> matrices);

    long long count_greater(long long value) const;

    std::vector<Matrix> col_split();

    void print() const;

  private:
    uint32_t n_, m_;
    std::vector<Cell> cells_;
};

