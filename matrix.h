#pragma once

#include <array>
#include <iostream>
#include <vector>

struct Cell {
    int row, col;
    double value;
};

class Matrix {
  public:
    Matrix(int n, int m) : n_(n), m_(m) {}

    Matrix() = default;

    friend Matrix operator*(const Matrix &a, const Matrix &b);

    friend std::ostream& operator<<(std::ostream &stream, const Matrix &mat);

    static Matrix read_and_distribute(const char *filename);

    void sort_by_cols();

    void init_broadcast(int self, int root, MPI_Comm comm, MPI_Request *request);

    void broadcast(int self, int root, MPI_Comm comm, MPI_Request *init, MPI_Request *request);

    static Matrix merge(std::vector<Matrix> matrices);

    long long count_greater(double value) const;

  private:
    int n_, m_;
    std::vector<Cell> cells_;
    std::array<int, 3> buffer_;
};

