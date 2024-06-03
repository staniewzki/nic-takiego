#include <array>
#include <algorithm>
#include <fstream>
#include <tuple>
#include <optional>
#include <queue>
#include <mpi.h>

#include "matrix.h"
#include "utils.h"

namespace {

class Partition {
  public:
    Partition(int n, int segments) {
        quotient_ = n / segments;
        rest_ = n % segments;
    }

    int starts_at(int idx) const {
        return idx < rest_ ? (quotient_ + 1) * idx : quotient_ * idx + rest_;
    }

  private:
    int quotient_, rest_;
};

/**
 * @brief Opens a stream for each array in CSR file
*/
std::tuple<std::ifstream, std::ifstream, std::ifstream, int, int, int, int>
set_csr_streams(const char *filename) {
    std::ifstream values(filename), cols(filename), cells_per_row(filename);

    int n, m, nnz, max_per_row;
    values >> n >> m >> nnz >> max_per_row;

    /* Skip the values and prepare to read COL_INDEX array */
    cols.seekg(values.tellg());
    for (int i = 0; i < nnz; i++) {
        double dummy;
        cols >> dummy;
    }

    /* Skip COL_INDEX and the first entry in ROW_INDEX array */
    cells_per_row.seekg(cols.tellg());
    for (int i = 0; i < nnz + 1; i++) {
        int dummy;
        cells_per_row >> dummy;
    }

    return {
        std::move(values),
        std::move(cols),
        std::move(cells_per_row),
        n,
        m,
        nnz,
        max_per_row
    };
}

}

Matrix Matrix::read_and_distribute(const char *filename) {
    auto &info = MPIInfo::instance();

    if (info.rank() == 0) {
        auto [values, cols, cells_per_row, n, m, nnz, max_per_row] = set_csr_streams(filename);

        Partition row_part(n, info.pc());
        Partition col_part(m, info.pc());

        /* Send coordinates to each node. */
        for (int i = 0; i < info.pc(); i++) {
            for (int j = 0; j < info.pc(); j++) {
                if (i == 0 && j == 0) continue;
                std::array<int, 4> coords = {
                    row_part.starts_at(i),
                    col_part.starts_at(j),
                    row_part.starts_at(i + 1) - row_part.starts_at(i),
                    col_part.starts_at(j + 1) - col_part.starts_at(j),
                };
                MPI_Send(coords.data(), 4, MPI_INT, i * info.pc() + j, 0, MPI_COMM_WORLD);
            }
        }

        std::vector<Cell> cur;
        Matrix mat(row_part.starts_at(1), col_part.starts_at(1));

        int row_owner = 0;
        int next_row_part = row_part.starts_at(1);
        int cells_processed = 0;
        for (int i = 0; i < n; i++) {
            if (i == next_row_part) {
                row_owner++;
                next_row_part = row_part.starts_at(row_owner + 1);
            }

            int cells_num;
            cells_per_row >> cells_num;

            if (cells_num == 0)
                continue;

            int col_owner = 0;
            int next_col_part = col_part.starts_at(1);

            auto flush_cur = [&] {
                int send_to = row_owner * info.pc() + col_owner;
                if (send_to == 0) {
                    for (auto &cell : cur)
                        mat.cells_.push_back(std::move(cell));
                } else {
                    int cells = cur.size();
                    MPI_Send(&cells, 1, MPI_INT, send_to, 0, MPI_COMM_WORLD);
                    MPI_Send(cur.data(), cells * sizeof(Cell), MPI_BYTE, send_to, 0, MPI_COMM_WORLD);
                }
                cur.clear();
            };

            for (; cells_processed < cells_num; cells_processed++) {
                double value;
                int col;

                values >> value;
                cols >> col;

                while (next_col_part <= col) {
                    if (!cur.empty())
                        flush_cur();

                    col_owner++;
                    next_col_part = col_part.starts_at(col_owner + 1);
                }

                cur.push_back(Cell {i, col, value});
            }

            if (!cur.empty())
                flush_cur();
        }

        for (int i = 1; i < info.num_procs(); i++) {
            int end_of_distribution = -1;
            MPI_Send(&end_of_distribution, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        }

        return mat;
    } else {
        std::array<int, 4> coords;
        MPI_Recv(coords.data(), 4, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        Matrix mat(coords[2], coords[3]);
        while (true) {
            int cells_num;
            MPI_Recv(&cells_num, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            if (cells_num == -1) {
                /* There are no more cells */
                break;
            }

            size_t current_size = mat.cells_.size();
            mat.cells_.resize(current_size + cells_num);

            MPI_Recv(
                mat.cells_.data() + current_size,
                cells_num * sizeof(Cell),
                MPI_BYTE,
                0,
                0,
                MPI_COMM_WORLD,
                MPI_STATUS_IGNORE
            );
        }

        for (auto &cell : mat.cells_) {
            cell.row -= coords[0];
            cell.col -= coords[1];
        }

        return mat;
    }
}

/**
 * @brief Direct multiplication
 *
 * `a` should be sorted by rows, `b` should be sorted by cols
*/
Matrix operator*(const Matrix &a, const Matrix &b) {
    assert(a.m_ == b.n_);

    int start = 0, end = -1;

    auto find_in_row = [&](int col) -> std::optional<double> {
        int l = start, r = end;
        while (l <= r) {
            int m = (l + r) / 2;
            if (a.cells_[m].col == col) {
                return a.cells_[m].value;
            } else if (a.cells_[m].col < col) {
                l = m + 1;
            } else {
                r = m - 1;
            }
        }
        return std::nullopt;
    };

    Matrix res(a.n_, b.m_);

    auto flush_row = [&] {
        int row = a.cells_[start].row;
        int col = -1;
        for (const auto &cell : b.cells_) {
            auto corresponding = find_in_row(cell.row);
            if (corresponding && cell.col == col) {
                res.cells_.back().value += *corresponding * cell.value;
            } else if (corresponding) {
                col = cell.col;
                res.cells_.push_back(Cell {row, col, *corresponding * cell.value});
            }
        }
    };

    int a_size = static_cast<int>(a.cells_.size());
    for (int i = 0; i < a_size; i++) {
        if (a.cells_[i].row != a.cells_[start].row) {
            flush_row();
            start = i;
        }
        end++;
    }

    flush_row();

    return res;
}

std::ostream& operator<<(std::ostream &stream, const Matrix &mat) {
    stream << "{";
    bool first = true;
    for (const auto &cell : mat.cells_) {
        if (first) {
            first = false;
        } else {
            stream << ", ";
        }

        stream << "(" << cell.row << ", " << cell.col << ", " << cell.value << ")";
    }
    return stream << "}";
}

void Matrix::sort_by_cols() {
    std::sort(
        cells_.begin(),
        cells_.end(),
        [](const Cell &a, const Cell &b) {
            return a.col < b.col;
        }
    );
}

void Matrix::init_broadcast(int self, int root, MPI_Comm comm, MPI_Request *request) {
    if (self == root)
        buffer_ = {n_, m_, static_cast<int>(cells_.size())};
    MPI_Ibcast(buffer_.data(), 3, MPI_INT, root, comm, request);
}

void Matrix::broadcast(int self, int root, MPI_Comm comm, MPI_Request *init, MPI_Request *request) {
    MPI_Wait(init, MPI_STATUS_IGNORE);
    if (self != root) {
        n_ = buffer_[0];
        m_ = buffer_[1];
        cells_.resize(buffer_[2]);
    }

    MPI_Ibcast(cells_.data(), cells_.size() * sizeof(Cell), MPI_BYTE, root, comm, request);
}

Matrix Matrix::merge(std::vector<Matrix> matrices) {
    size_t k = matrices.size();
    std::vector<size_t> position(k);

    std::priority_queue<
        std::tuple<int, int, int>,
        std::vector<std::tuple<int, int, int>>,
        std::greater<std::tuple<int, int, int>>> queue;

    auto add_from = [&](int idx) {
        if (position[idx] < matrices[idx].cells_.size()) {
            auto cell = matrices[idx].cells_[position[idx]];
            queue.emplace(cell.row, cell.col, idx);
            position[idx]++;
        }
    };

    for (size_t i = 0; i < k; i++)
        add_from(i);

    Matrix mat(matrices.front().n_, matrices.front().m_);
    while (!queue.empty()) {
        auto [r, c, idx] = queue.top();
        queue.pop();

        double value = matrices[idx].cells_[position[idx] - 1].value;

        if (!mat.cells_.empty() && mat.cells_.back().row == r && mat.cells_.back().col == c) {
            mat.cells_.back().value += value;
        } else {
            mat.cells_.push_back(Cell {r, c, value});
        }

        add_from(idx);
    }

    return mat;
}

long long Matrix::count_greater(double value) const {
    long long res = 0;
    for (const auto &cell : cells_) {
        if (cell.value > value)
            res++;
    }

    if (value < 0) {
        res += static_cast<long long>(n_) * m_ - cells_.size();
    }

    return res;
}