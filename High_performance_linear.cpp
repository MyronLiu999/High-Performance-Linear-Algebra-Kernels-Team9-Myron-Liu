#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include "helper.h"


#include <iostream>

void multiply_mv_row_major(const double* matrix, int rows, int cols, const double* vector, double* result) {
    if (matrix == nullptr || vector == nullptr || result == nullptr) {
        std::cerr << "null pointer in multiply_mv_row_major.\n";
        return;
    }

    if (rows <= 0 || cols <= 0) {
        std::cerr << "invalid dimensions in multiply_mv_row_major.\n";
        return;
    }

    for (int i = 0; i < rows; i++) {
        result[i] = 0.0;
        for (int j = 0; j < cols; j++) {
            result[i] += matrix[i * cols + j] * vector[j];
        }
    }
}

void multiply_mv_col_major(const double* matrix, int rows, int cols, const double* vector, double* result) {
    if (matrix == nullptr || vector == nullptr || result == nullptr) {
        std::cerr << "null pointer in multiply_mv_col_major.\n";
        return;
    }

    if (rows <= 0 || cols <= 0) {
        std::cerr << "invalid dimensions in multiply_mv_col_major.\n";
        return;
    }

    for (int i = 0; i < rows; i++) {
        result[i] = 0.0;
        for (int j = 0; j < cols; j++) {
            result[i] += matrix[j * rows + i] * vector[j];
        }
    }
}

void multiply_mm_naive(const double* matrixA, int rowsA, int colsA, const double* matrixB, int rowsB, int colsB, double* result) {
    if (matrixA == nullptr || matrixB == nullptr || result == nullptr) {
        std::cerr << "null pointer in multiply_mm_naive.\n";
        return;
    }

    if (rowsA <= 0 || colsA <= 0 || rowsB <= 0 || colsB <= 0) {
        std::cerr << "invalid dimensions in multiply_mm_naive.\n";
        return;
    }

    if (colsA != rowsB) {
        std::cerr << "incompatible dimensions in multiply_mm_naive.\n";
        return;
    }

    for (int i = 0; i < rowsA; i++) {
        const double* rowsA_ptr = matrixA + i * colsA;
        double* rowC = result + i * colsB;
        for (int j = 0; j < colsB; j++) {
            double sum = 0.0;
            for (int k = 0; k < colsA; k++) {
                sum += rowsA_ptr[k] * matrixB[k * colsB + j];
            }
            rowC[j] = sum;
        }
    }
}

// rowA*colA rowB*colB
void multiply_mm_transposed_b(const double* matrixA, int rowsA, int colsA, const double* matrixB_transposed, int rowsB, int colsB, double* result) {
    if (matrixA == nullptr || matrixB_transposed == nullptr || result == nullptr) {
        std::cerr << "null pointer in multiply_mm_transposed_b.\n";
        return;
    }

    if (rowsA <= 0 || colsA <= 0 || rowsB <= 0 || colsB <= 0) {
        std::cerr << "invalid dimensions in multiply_mm_transposed_b.\n";
        return;
    }

    if (colsA != rowsB) {
        std::cerr << "incompatible dimensions in multiply_mm_transposed_b.\n";
        return;
    }

    for (int i = 0; i < rowsA; i++) {
        const double* rowA = matrixA + i * colsA;
        double* rowC = result + i * colsB;

        for (int j = 0; j < colsB; j++) {
            const double* rowBT = matrixB_transposed + j * rowsB;
            double sum = 0.0;
            for (int k = 0; k < colsA; k++) {
                sum += rowA[k] * rowBT[k];
            }
            rowC[j] = sum;
        }
    }
}

double stride_benchmark(int n, int stride, int repeat) {
    double* arr = new double[n];
    for (int i = 0; i < n; i++) {
        arr[i] = static_cast<double>(i % 100);
    }

    double total_ms = 0.0;
    volatile double sink = 0.0;

    for (int r = 0; r < repeat; r++) {
        double sum = 0.0;
        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < n; i += stride) {
            sum += arr[i];
        }

        auto end = std::chrono::high_resolution_clock::now();
        sink = sum;
        total_ms += std::chrono::duration<double, std::milli>(end - start).count();
    }

    delete[] arr;
    return total_ms / repeat;
}


int main() {
    // multiply_mv_row_major
    int ROWS = 5000;
    int COLS = 5000;

    double* A = nullptr;
    double* vector = nullptr;
    double* result = nullptr;
    allcate_matrix_row_major(&A, ROWS, COLS);
    allocate_vector(&vector, COLS);
    result = new double[ROWS];
    auto start = std::chrono::high_resolution_clock::now();
    multiply_mv_row_major(A, ROWS, COLS, vector, result);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Time_row_major_Matrix_vector_multiplication: " << duration.count() << " milliseconds" << std::endl;
    // free_2d_matrix(&A, 3);
    delete[] A;
    delete[] vector;
    delete[] result;

    // multiply_mv_col_major
    double* B = nullptr;
    double* vector1 = nullptr;
    double* result1 = nullptr;
    allocate_matrix_col_major(&B, ROWS, COLS);
    allocate_vector(&vector1, COLS);
    result1 = new double[ROWS];
    auto start1 = std::chrono::high_resolution_clock::now();
    multiply_mv_col_major(B, ROWS, COLS, vector1, result1);
    auto end1 = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1);
    std::cout << "Time_col_major_Matrix_vector_multiplication: " << duration1.count() << " milliseconds" << std::endl;
    delete[] B;
    delete[] vector1;
    delete[] result1;

    // multiply_mm_naive
    ROWS = 1500;
    COLS = 1500;
    double* C = nullptr;
    double* D = nullptr;
    double* result2 = nullptr;
    allcate_matrix_row_major(&C, ROWS, COLS);
    allcate_matrix_row_major(&D, ROWS, COLS);
    result2 = new double[ROWS*COLS];
    auto start2 = std::chrono::high_resolution_clock::now();
    multiply_mm_naive(C, ROWS, COLS, D, ROWS, COLS, result2);
    auto end2 = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2);
    std::cout << "Time_col_major_Matrix_Matrix_multiplication: " << duration2.count() << " milliseconds" << std::endl;
    delete[] C;
    delete[] D;
    delete[] result2;    

    // multiply_mm_transposed_b
    double* C1 = nullptr;
    double* D1 = nullptr;
    double* E = nullptr;
    double* result3 = nullptr;
    allcate_matrix_row_major(&C1, ROWS, COLS);
    allcate_matrix_row_major(&D1, ROWS, COLS);
    transpose_matrix(D1, ROWS, COLS, &E);
    result3 = new double[ROWS*COLS];
    auto start3 = std::chrono::high_resolution_clock::now();
    multiply_mm_transposed_b(C1, ROWS, COLS, E, COLS, ROWS, result3);
    auto end3 = std::chrono::high_resolution_clock::now();
    auto duration3 = std::chrono::duration_cast<std::chrono::milliseconds>(end3 - start3);
    std::cout << "Time_col_major_Matrix_Matrix_multiplication_transposed: " << duration3.count() << " milliseconds" << std::endl;
    delete[] C1;
    delete[] D1;
    delete[] E;
    delete[] result3; 


    std::cout << "\n=== Stride Benchmark ===\n";
    const int N = 1 << 24;
    int strides[] = {1, 2, 4, 8, 16, 32, 64};
    const int repeat_stride = 5;

    for (int s : strides) {
        double t = stride_benchmark(N, s, repeat_stride);
        std::cout << "Stride " << s << ": " << t << " ms\n";
    }

    // test align
    double* A_aligned = nullptr;
    double* vector_aligned = nullptr;
    double* result_aligned = nullptr;

    allocate_matrix_row_major_aligned(&A_aligned, ROWS, COLS);
    allocate_vector_aligned(&vector_aligned, COLS);
    result_aligned = allocate_aligned_array(ROWS);

    auto startA = std::chrono::high_resolution_clock::now();
    multiply_mv_row_major(A_aligned, ROWS, COLS, vector_aligned, result_aligned);
    auto endA = std::chrono::high_resolution_clock::now();

    auto durationA = std::chrono::duration_cast<std::chrono::milliseconds>(endA - startA);
    std::cout << "Aligned row-major MV: " << durationA.count() << " ms\n";

    free_aligned_array(A_aligned);
    free_aligned_array(vector_aligned);
    free_aligned_array(result_aligned);


    double* B_aligned = nullptr;
    double* vector1_aligned = nullptr;
    double* result1_aligned = nullptr;

    allocate_matrix_col_major_aligned(&B_aligned, ROWS, COLS);
    allocate_vector_aligned(&vector1_aligned, COLS);
    result1_aligned = allocate_aligned_array(ROWS);

    auto startB = std::chrono::high_resolution_clock::now();
    multiply_mv_col_major(B_aligned, ROWS, COLS, vector1_aligned, result1_aligned);
    auto endB = std::chrono::high_resolution_clock::now();

    auto durationB = std::chrono::duration_cast<std::chrono::milliseconds>(endB - startB);
    std::cout << "Aligned col-major MV: " << durationB.count() << " ms\n";

    free_aligned_array(B_aligned);
    free_aligned_array(vector1_aligned);
    free_aligned_array(result1_aligned);

    double* C_aligned = nullptr;
    double* D_aligned = nullptr;
    double* result2_aligned = nullptr;

    allocate_matrix_row_major_aligned(&C_aligned, ROWS, COLS);
    allocate_matrix_row_major_aligned(&D_aligned, ROWS, COLS);
    result2_aligned = allocate_aligned_array(static_cast<std::size_t>(ROWS) * COLS);

    auto startC = std::chrono::high_resolution_clock::now();
    multiply_mm_naive(C_aligned, ROWS, COLS, D_aligned, ROWS, COLS, result2_aligned);
    auto endC = std::chrono::high_resolution_clock::now();

    auto durationC = std::chrono::duration_cast<std::chrono::milliseconds>(endC - startC);
    std::cout << "Aligned naive MM: " << durationC.count() << " ms\n";

    free_aligned_array(C_aligned);
    free_aligned_array(D_aligned);
    free_aligned_array(result2_aligned);


    double* C1_aligned = nullptr;
    double* D1_aligned = nullptr;
    double* E_aligned = nullptr;
    double* result3_aligned = nullptr;

    allocate_matrix_row_major_aligned(&C1_aligned, ROWS, COLS);
    allocate_matrix_row_major_aligned(&D1_aligned, ROWS, COLS);
    transpose_matrix_aligned(D1_aligned, ROWS, COLS, &E_aligned);
    result3_aligned = allocate_aligned_array(static_cast<std::size_t>(ROWS) * COLS);

    auto startD = std::chrono::high_resolution_clock::now();
    multiply_mm_transposed_b(C1_aligned, ROWS, COLS, E_aligned, COLS, ROWS, result3_aligned);
    auto endD = std::chrono::high_resolution_clock::now();

    auto durationD = std::chrono::duration_cast<std::chrono::milliseconds>(endD - startD);
    std::cout << "Aligned transposed-B MM: " << durationD.count() << " ms\n";

    free_aligned_array(C1_aligned);
    free_aligned_array(D1_aligned);
    free_aligned_array(E_aligned);
    free_aligned_array(result3_aligned);

    std::cout << "\n=== Inlining Benchmark ===\n";

    int ROWS_inline = 5000;
    int COLS_inline = 5000;

    double* A_inline_test = nullptr;
    double* x_inline_test = nullptr;
    double* result_no_inline = nullptr;
    double* result_inline = nullptr;

    allcate_matrix_row_major(&A_inline_test, ROWS_inline, COLS_inline);
    allocate_vector(&x_inline_test, COLS_inline);

    result_no_inline = new double[ROWS_inline];
    result_inline = new double[ROWS_inline];

    const int repeat_inline = 5;
    double total_no_inline = 0.0;
    double total_inline = 0.0;

    for (int t = 0; t < repeat_inline; t++) {
        auto start_no_inline = std::chrono::high_resolution_clock::now();
        multiply_mv_row_major_helper(A_inline_test, ROWS_inline, COLS_inline, x_inline_test, result_no_inline);
        auto end_no_inline = std::chrono::high_resolution_clock::now();
        total_no_inline += std::chrono::duration<double, std::milli>(end_no_inline - start_no_inline).count();

        auto start_inline = std::chrono::high_resolution_clock::now();
        multiply_mv_row_major_helper_inline(A_inline_test, ROWS_inline, COLS_inline, x_inline_test, result_inline);
        auto end_inline = std::chrono::high_resolution_clock::now();
        total_inline += std::chrono::duration<double, std::milli>(end_inline - start_inline).count();
    }

    std::cout << "MV helper (no inline): "
              << total_no_inline / repeat_inline << " ms" << std::endl;

    std::cout << "MV helper (inline): "
              << total_inline / repeat_inline << " ms" << std::endl;

    delete[] A_inline_test;
    delete[] x_inline_test;
    delete[] result_no_inline;
    delete[] result_inline;

    return 0;
}