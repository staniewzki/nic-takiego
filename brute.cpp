#include <bits/stdc++.h>

using Matrix = std::vector<std::vector<double>>;

Matrix read_matrix(const char *filename) {
	std::ifstream stream(filename);

	int n, m, nnz, max_per_row;
	stream >> n >> m >> nnz >> max_per_row;

	std::vector<double> values(nnz);
	std::vector<int> cols(nnz), cells_per_row(n);

	for (int i = 0; i < nnz; i++)
		stream >> values[i];

	for (int i = 0; i < nnz; i++)
		stream >> cols[i];

	stream >> cells_per_row[0];
	for (int i = 0; i < n; i++)
		stream >> cells_per_row[i];

	std::vector result(n, std::vector<double>(m));
	int pos = 0;
	for (int i = 0; i < n; i++) {
		while (pos < cells_per_row[i]) {
			result[i][cols[pos]] = values[pos];	
			pos++;
		}
	}

	return result;
}

int main(int argc, char *argv[]) {
	if (argc != 3) {
		std::cerr << "usage: " << argv[0] << " matrix_A matrix_B\n";
		return 1;
	}
	
	auto A = read_matrix(argv[1]);
	auto B = read_matrix(argv[2]);
	
	int an = (int) A.size();
	int am = (int) A[0].size();
	int bn = (int) B.size();
	int bm = (int) B[0].size();

	assert(am == bn);

	std::vector result(an, std::vector<double>(bm));
	for (int i = 0; i < an; i++) {
		for (int j = 0; j < am; j++) {
			for (int k = 0; k < bm; k++) {
				result[i][k] += A[i][j] * B[j][k];
			}
		}
	}

	for (int i = 0; i < an; i++) {
		for (int j = 0; j < bm; j++) {
			std::cout << result[i][j] << " ";
		}
		std::cout << "\n";
	}
}
