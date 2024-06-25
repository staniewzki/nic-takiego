#include <array>
#include <algorithm>
#include <fstream>
#include <tuple>
#include <optional>
#include <queue>
#include <mpi.h>

#include "utils.h"

#include "matrix.h"

namespace {

class Partition {
  public:
    Partition(uint32_t n, uint32_t segments) {
        segments_ = segments;
        quotient_ = n / segments;
        rest_ = n % segments;
    }

    /**
     * @brief Calculates the start of segment with the given index
     */
    uint32_t starts_at(uint32_t idx) const {
        return idx < rest_ ? (quotient_ + 1) * idx : quotient_ * idx + rest_;
    }

    /**
     * @brief Finds the segment given position belongs to
     */
    int belongs_to(uint32_t pos) const {
        int l = 0, r = segments_ - 1;
        while (l < r) {
            int m = (l + r + 1) / 2;
            if (starts_at(m) <= pos) {
                l = m;
            } else {
                r = m - 1;
            }
        }
        return l;
    }

  private:
    uint32_t segments_, quotient_, rest_;
};

}

Matrix Matrix::read_and_distribute(const char *filename, SplitAlong split) {
    auto &info = MPIInfo::instance();
    const int MSG_META = 2001;
    const int MSG_CELLS = 2002;
    if (info.rank() == 0) {
        std::ifstream stream(filename);

        uint32_t n, m, max_per_row;
        uint64_t nnz;
        stream >> n >> m >> nnz >> max_per_row;

        std::vector<double> values(nnz);
        std::vector<uint32_t> cols(nnz), cells_per_row(n);

        for (uint64_t i = 0; i < nnz; i++)
            stream >> values[i];

        for (uint64_t i = 0; i < nnz; i++)
            stream >> cols[i];

        // skip the starting zero
        stream >> cells_per_row[0];

        for (uint32_t i = 0; i < n; i++)
            stream >> cells_per_row[i];

        int layers_in_row = split == SplitAlong::Row ? info.num_layers() : 1;
        int layers_in_col = split == SplitAlong::Col ? info.num_layers() : 1;

        std::vector<std::vector<Cell>> matrices(info.num_procs());

        Partition row_part(n, info.pc() * layers_in_row);
        Partition col_part(m, info.pc() * layers_in_col);

        auto part_num = [&](int r, int c) {
            return r * info.pc() * layers_in_col + c;
        };

        uint32_t ptr = 0;
        for (uint32_t i = 0; i < n; i++) {
            for (; ptr < cells_per_row[i]; ptr++) {
                int r = row_part.belongs_to(i);
                int c = col_part.belongs_to(cols[ptr]);
                matrices[part_num(r, c)].emplace_back(Cell {i, cols[ptr], values[ptr]});
            }
        }

        auto row_idx = [&](int k, int i) {
            return split == SplitAlong::Row ? k + info.num_layers() * i : i;
        };

        auto col_idx = [&](int k, int j) {
            return split == SplitAlong::Col ? k + info.num_layers() * j : j;
        };

        auto proc_idx = [&](int k, int i, int j) {
            return k * info.pc() * info.pc() + i * info.pc() + j;
        };

        std::vector<MPI_Request> requests(info.num_procs());
        for (int k = 0; k < info.num_layers(); k++) {
            for (int i = 0; i < info.pc(); i++) {
                for (int j = 0; j < info.pc(); j++) {
                    int r = row_idx(k, i);
                    int c = col_idx(k, j);
                    if (r == 0 && c == 0) continue;

                    std::array meta = {
                        row_part.starts_at(r),
                        col_part.starts_at(c),
                        row_part.starts_at(r + 1) - row_part.starts_at(r),
                        col_part.starts_at(c + 1) - col_part.starts_at(c),
                        static_cast<uint32_t>(matrices[part_num(r, c)].size()),
                    };

                    std::cerr << "sending to: " << proc_idx(k, i, j) << " cells: " << matrices[part_num(r, c)].size() << "\n";
                    MPI_Send(meta.data(), meta.size(), MPI_UINT32_T, proc_idx(k, i, j),
                              MSG_META, MPI_COMM_WORLD); // , &requests[proc_idx(k, i, j)]);
                }
            }
        }

        std::vector<MPI_Request> requests2(info.num_procs());
        for (int k = 0; k < info.num_layers(); k++) {
            for (int i = 0; i < info.pc(); i++) {
                for (int j = 0; j < info.pc(); j++) {
                    int r = row_idx(k, i);
                    int c = col_idx(k, j);
                    if (r == 0 && c == 0) continue;

                    // MPI_Wait(&requests[proc_idx(k, i, j)], MPI_STATUS_IGNORE);

                    MPI_Isend(
                        matrices[part_num(r, c)].data(),
                        matrices[part_num(r, c)].size() * sizeof(Cell),
                        MPI_BYTE,
                        proc_idx(k, i, j),
                        MSG_CELLS,
                        MPI_COMM_WORLD,
                        &requests2[proc_idx(k, i, j)]
                    );
                }
            }
        }

        for (int k = 0; k < info.num_layers(); k++) {
            for (int i = 0; i < info.pc(); i++) {
                for (int j = 0; j < info.pc(); j++) {
                    if (k == 0 && i == 0 && j == 0) continue;
                    MPI_Wait(&requests2[proc_idx(k, i, j)], MPI_STATUS_IGNORE);
                }
            }
        }

        Matrix own(row_part.starts_at(1), col_part.starts_at(1));
        own.cells_ = std::move(matrices[0]);

        // MPI_Waitall(info.num_procs() - 1, &requests2[1], MPI_STATUSES_IGNORE);
        return own;
    } else {
        MPI_Request request;
        std::array<uint32_t, 5> meta;
        MPI_Irecv(meta.data(), 5, MPI_UINT32_T, 0, MSG_META, MPI_COMM_WORLD, &request);
        MPI_Wait(&request, MPI_STATUS_IGNORE);
        std::cerr << "received: " << info.rank() << " cells: " << meta[4] << "\n";

        Matrix mat(meta[2], meta[3]);
        mat.cells_.resize(meta[4]);

        MPI_Irecv(
            mat.cells_.data(),
            meta[4] * sizeof(Cell),
            MPI_BYTE,
            0,
            MSG_CELLS,
            MPI_COMM_WORLD,
            &request
        );
        MPI_Wait(&request, MPI_STATUS_IGNORE);

        for (auto &cell : mat.cells_) {
            cell.row -= meta[0];
            cell.col -= meta[1];
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

    /* Indexes of the row in matrix a that is currently processed */
    uint32_t start = 0, end = 0;

    Matrix res(a.n_, b.m_);

    /* Multiply cells from a the current row in matrix A */
    auto flush_row = [&] {
        if (a.cells_.size() <= start || b.cells_.empty()) return;

        uint32_t row = a.cells_[start].row;
        uint32_t current_col = b.cells_[0].col;
        long long ptr = start;
        bool created = false;

        for (const auto &cell : b.cells_) {
            if (cell.col != current_col) {
                current_col = cell.col;
                ptr = start;
                created = false;
            }

            while (ptr < end && a.cells_[ptr].col < cell.row)
                ptr++;

            if (ptr < end && a.cells_[ptr].col == cell.row) {
                if (created) {
                    res.cells_.back().value += a.cells_[ptr].value * cell.value;
                } else {
                    res.cells_.push_back(Cell {row, current_col, a.cells_[ptr].value * cell.value});
                }
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
            return std::pair(a.col, a.row) < std::pair(b.col, b.row);
        }
    );
}

void Matrix::init_broadcast(int self, int root, MPI_Comm comm, MatrixInfo &info, MPI_Request &request) {
    if (self == root)
        info = get_info();
    MPI_Ibcast(info.data.data(), 3, MPI_UINT32_T, root, comm, &request);
}

void Matrix::broadcast(int self, int root, MPI_Comm comm, MatrixInfo &info, MPI_Request &request) {
    if (self != root) {
        n_ = info.n();
        m_ = info.m();
        cells_.resize(info.cells());
    }

    MPI_Ibcast(cells_.data(), cells_.size() * sizeof(Cell), MPI_BYTE, root, comm, &request);
}

MatrixInfo Matrix::get_info() const {
    return {n_, m_, static_cast<uint32_t>(cells_.size())};
}

Matrix Matrix::merge(std::vector<Matrix> matrices) {
    size_t k = matrices.size();
    std::vector<size_t> position(k);

    std::priority_queue<
        std::tuple<uint32_t, uint32_t, uint32_t>,
        std::vector<std::tuple<uint32_t, uint32_t, uint32_t>>,
        std::greater<std::tuple<uint32_t, uint32_t, uint32_t>>> queue;

    auto add_from = [&](uint32_t idx) {
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

Matrix Matrix::merge(uint32_t n, uint32_t m, std::vector<Cell> cells) {
    std::sort(cells.begin(), cells.end(), [&](const Cell &a, const Cell &b) {
        return std::pair(a.row, a.col) < std::pair(b.row, b.col);
    });

    Matrix result(n, m);
    for (const auto &cell : cells) {
        if (result.cells_.empty() || result.cells_.back().row != cell.row || result.cells_.back().col != cell.col)
            result.cells_.emplace_back(cell);
        else
            result.cells_.back().value += cell.value;
    }

    return result;
}

long long Matrix::count_greater(long long value) const {
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

std::vector<Matrix> Matrix::col_split() {
    auto &info = MPIInfo::instance();
    Partition part(m_, info.num_layers());

    std::vector<Matrix> res;
    for (int i = 0; i < info.num_layers(); i++)
        res.emplace_back(n_, part.starts_at(i + 1) - part.starts_at(i));

    for (auto &cell : cells_) {
        int idx = part.belongs_to(cell.col);
        cell.col -= part.starts_at(idx);
        res[idx].cells_.push_back(std::move(cell));
    }

    return res;
}

const std::vector<Cell> &Matrix::cells() const {
    return cells_;
}

void Matrix::print() const {
    auto &info = MPIInfo::instance();

    MPI_Comm row_comm;
    MPI_Comm_split(MPI_COMM_WORLD, info.row(), info.col() * info.num_layers() + info.layer(), &row_comm);

    std::vector<uint32_t> width;
    if (info.rank() == 0) {
        width.resize(info.pc() * info.num_layers());
    }

    if (info.row() == 0) {
        MPI_Gather(&m_, 1, MPI_UINT32_T, width.data(), 1, MPI_INT, 0, row_comm);
    }

    auto process = [&](int i, int j, int k) {
        return k * info.pc() * info.pc() + i * info.pc() + j;
    };

    constexpr int HEIGHT_MSG = 4001;
    constexpr int NUM_CELLS_MSG = 4002;
    constexpr int CELLS_MSG = 4003;

    if (info.rank() == 0) {
        size_t self_pos = 0, pos;
        std::vector<Cell> buffer;
        for (int i = 0; i < info.pc(); i++) {
            uint32_t height;
            if (i == 0) {
                height = n_;
            } else {
                MPI_Recv(&height, 1, MPI_UINT32_T, process(i, 0, 0), HEIGHT_MSG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }

            for (uint32_t r = 0; r < height; r++) {
                for (int j = 0; j < info.pc(); j++) {
                    for (int k = 0; k < info.num_layers(); k++) {
                        const std::vector<Cell> *data;
                        size_t *ptr;

                        if (process(i, j, k) == 0) {
                            data = &cells_;
                            ptr = &self_pos;
                        } else {
                            uint32_t num_cells = 0;
                            MPI_Recv(
                                &num_cells, 1, MPI_UINT32_T, process(i, j, k), NUM_CELLS_MSG,
                                MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                            buffer.resize(num_cells);
                            MPI_Recv(
                                buffer.data(), num_cells * sizeof(Cell), MPI_BYTE,
                                process(i, j, k), CELLS_MSG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                            data = &buffer;
                            ptr = &pos;
                            pos = 0;
                        }

                        for (uint32_t c = 0; c < width[j * info.num_layers() + k]; c++) {
                            if (*ptr < data->size() && ((*data)[*ptr].row == r) && (*data)[*ptr].col == c) {
                                std::cout << (int) (*data)[*ptr].value << " ";
                                (*ptr)++;
                            } else {
                                std::cout << 0.0 << " ";
                            }
                        }
                    }
                }

                std::cout << "\n";
            }
        }
    } else {
        if (info.col() == 0) {
            MPI_Send(&n_, 1, MPI_INT, 0, HEIGHT_MSG, MPI_COMM_WORLD);
        }

        uint32_t l = 0, r = 0;
        for (uint32_t i = 0; i < n_; i++) {
            while (r < cells_.size() && cells_[r].row == i)
                r++;

            uint32_t len = r - l;
            MPI_Send(&len, 1, MPI_UINT32_T, 0, NUM_CELLS_MSG, MPI_COMM_WORLD);

            MPI_Send(
                (len == 0 ? nullptr : &cells_[l]), len * sizeof(Cell),
                MPI_BYTE, 0, CELLS_MSG, MPI_COMM_WORLD);

            l = r;
        }
    }
}