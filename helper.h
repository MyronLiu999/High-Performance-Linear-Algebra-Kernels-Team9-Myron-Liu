#include <iostream>
#include <vector>
#include <chrono>
#include <random>

// generate matrix
void allcate_matrix_row_major(double** matrix, int rows, int cols) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(-100, 100);
    *matrix = new double[rows * cols];
    for (int i = 0; i < rows * cols; i++) {
        (*matrix)[i] = distrib(gen);
    }
}

void transpose_matrix(const double* matrix, int rows, int cols, double** transposed) {
    *transposed = new double[rows * cols];
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            (*transposed)[j * rows + i] = matrix[i * cols + j];
        }
    }
}
void allocate_matrix_col_major(double** matrix, int rows, int cols) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(-100, 100);
    *matrix = new double[rows * cols];
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            (*matrix)[j * rows + i] = distrib(gen);
        }
    }
}

void allocate_vector(double** vector, int longs) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(-100, 100);
    *vector = new double[longs];
    for (int i = 0; i < longs; i++) {
        (*vector)[i] = distrib(gen);
    }
}

void multiply_mm_reordered(const double* matrixA, int rowsA, int colsA, const double* matrixB, int rowsB, int colsB,
                           double* result) {
    if (colsA != rowsB) return;

    for (int i = 0; i < rowsA; i++) {
        double* rowC = result + i * colsB;
        for (int j = 0; j < colsB; j++) {
            rowC[j] = 0.0;
        }

        for (int k = 0; k < colsA; k++) {
            const double aik = matrixA[i * colsA + k];
            const double* rowB = matrixB + k * colsB;
            for (int j = 0; j < colsB; j++) {
                rowC[j] += aik * rowB[j];
            }
        }
    }
}
// void free_2d_matrix(double** matrix, int rows) {
//     for (int i = 0; i < rows; i++) {
//         delete[] matrix[i];
//     }
//     delete[] matrix;
// }

#include <new>
#include <cstddef>

double* allocate_aligned_array(std::size_t count) {
    return static_cast<double*>(
        ::operator new[](count * sizeof(double), std::align_val_t(64))
    );
}

void free_aligned_array(double* ptr) {
    ::operator delete[](ptr, std::align_val_t(64));
}

void allocate_vector_aligned(double** vec, int n) {
    *vec = allocate_aligned_array(n);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(-100, 100);

    for (int i = 0; i < n; i++) {
        (*vec)[i] = distrib(gen);
    }
}

void allocate_matrix_row_major_aligned(double** matrix, int rows, int cols) {
    *matrix = allocate_aligned_array(static_cast<std::size_t>(rows) * cols);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(-100, 100);

    for (int i = 0; i < rows * cols; i++) {
        (*matrix)[i] = distrib(gen);
    }
}

void allocate_matrix_col_major_aligned(double** matrix, int rows, int cols) {
    *matrix = allocate_aligned_array(static_cast<std::size_t>(rows) * cols);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(-100, 100);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            (*matrix)[j * rows + i] = distrib(gen);
        }
    }
}

void transpose_matrix_aligned(const double* matrix, int rows, int cols, double** transposed) {
    *transposed = allocate_aligned_array(static_cast<std::size_t>(rows) * cols);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            (*transposed)[j * rows + i] = matrix[i * cols + j];
        }
    }
}

double get_row_major(const double* matrix, int cols, int i, int j) {
    return matrix[i * cols + j];
}

double get_col_major(const double* matrix, int rows, int i, int j) {
    return matrix[j * rows + i];
}

inline double get_row_major_inline(const double* matrix, int cols, int i, int j) {
    return matrix[i * cols + j];
}

inline double get_col_major_inline(const double* matrix, int rows, int i, int j) {
    return matrix[j * rows + i];
}
void multiply_mv_row_major_helper(const double* matrix, int rows, int cols, const double* vector, double* result) {
    for (int i = 0; i < rows; i++) {
        result[i] = 0.0;
        for (int j = 0; j < cols; j++) {
            result[i] += get_row_major(matrix, cols, i, j) * vector[j];
        }
    }
}

void multiply_mv_row_major_helper_inline(const double* matrix, int rows, int cols, const double* vector, double* result) {
    for (int i = 0; i < rows; i++) {
        result[i] = 0.0;
        for (int j = 0; j < cols; j++) {
            result[i] += get_row_major_inline(matrix, cols, i, j) * vector[j];
        }
    }
}

