#include <algorithm>
#include "SingletonLogger.hpp"
#include "mkl.h"

int main() {
    auto& logger = SingletonLogger::instance();
    try {
        std::vector<std::string> hs { "0.833332", "0.416666", "0.208333", "0.104167", "0.052083", "0.026042" };
        auto h = hs[logger.opt("choose $h$", hs)];
        if (std::stod(h) < .2) logger.timeUnits = SingletonLoggerTimeUnits::min;
        std::vector<std::string> epsilons { "0.0001", "0.00001" };
        auto epsilon = epsilons[logger.opt("choose $\\epsilon$", epsilons)];
        std::vector<std::string> FEs { "P1P1", "P2P1" };
        auto FE = FEs[logger.opt("choose FE", FEs)];
        double lambda_min, lambda_max;
        logger.inp("set $\\lambda_{min}$", lambda_min);
        logger.inp("set $\\lambda_{max}$", lambda_max);
        double percent;
        logger.inp("set percent of eigs to be found", percent);
        std::vector<std::string> stabTypes;
        if (logger.yes("compute eigs for A_0")) stabTypes.emplace_back("0");
        if (logger.yes("compute eigs for A_full")) stabTypes.emplace_back("full");
        if (logger.yes("compute eigs for A_n")) stabTypes.emplace_back("n");
        for (auto const & stab : stabTypes) {
            logger.beg("process A_" + stab);
                logger.beg("import matrices");
                    auto mtx2coord = [&](std::string const & path) {
                        std::ifstream input(path);
                        struct {
                            MKL_INT size;
                            MKL_INT nnz_sym;
                            MKL_INT nnz;
                            std::vector<MKL_INT> row_indx;
                            std::vector<MKL_INT> col_indx;
                            std::vector<double> values;
                        } res;
                        std::string comment;
                        std::getline(input, comment);
                        logger.buf << "importing " << path << ":\n" << comment << '\n';
                        if (comment.find("symmetric") == std::string::npos)
                            throw std::invalid_argument("matrix " + path + " is not symmetric");
                        std::getline(input, comment);
                        logger.buf << comment << '\n';
                        input >> res.size >> res.size >> res.nnz_sym;
                        res.nnz = 2 * (res.nnz_sym - res.size) + res.size;
                        logger.buf
                                << "mtx size: " << res.size << '\n'
                                << "nnz:      " << res.nnz_sym << " -> " << res.nnz;
                        logger.log();
                        res.row_indx.reserve(res.nnz);
                        res.col_indx.reserve(res.nnz);
                        res.values.reserve(res.nnz);
                        MKL_INT row_indx, col_indx;
                        double value;
                        size_t currentElementIndex = 1;
                        while (input >> row_indx >> col_indx >> value) {
                            res.row_indx.push_back(row_indx);
                            res.col_indx.push_back(col_indx);
                            res.values.push_back(value);
                            if (row_indx != col_indx) {
                                res.row_indx.push_back(col_indx);
                                res.col_indx.push_back(row_indx);
                                res.values.push_back(value);
                            }
                            logger.pro(currentElementIndex++, res.nnz_sym);
                        }
                        logger.buf
                            << "first elem: " << res.row_indx.front() << ' ' << res.col_indx.front() << ' ' << res.values.front() << '\n'
                            << "last elem:  " << res.row_indx.back() << ' ' << res.col_indx.back() << ' ' << res.values.back();
                        logger.log();
                        res.nnz = res.values.size();
                        return res;
                    };
                    logger.beg("import A");
                        auto A_mtx = mtx2coord("output/" + FE + "/h=" + h + "_A_block_" + stab + ".mtx");
                    logger.end();
                    logger.beg("import M");
                        auto M_mtx = mtx2coord("output/" + FE + "/h=" + h + "_eps=" + epsilon + "_M_block_" + stab + ".mtx");
                    logger.end();
                logger.end();
                logger.beg("convert: coord -> csr");
                    sparse_matrix_t A_coord, A, M_coord, M;
                    auto status = mkl_sparse_d_create_coo(&A_coord, SPARSE_INDEX_BASE_ONE, A_mtx.size, A_mtx.size, A_mtx.nnz, A_mtx.row_indx.data(), A_mtx.col_indx.data(), A_mtx.values.data());
                    if (status != SPARSE_STATUS_SUCCESS)
                        throw std::invalid_argument("error while reading A: status = " + std::to_string(status));
                    status = mkl_sparse_convert_csr(A_coord, SPARSE_OPERATION_NON_TRANSPOSE, &A);
                    if (status != SPARSE_STATUS_SUCCESS)
                        throw std::invalid_argument("error while converting A to CSR: status = " + std::to_string(status));
                    status = mkl_sparse_d_create_coo(&M_coord, SPARSE_INDEX_BASE_ONE, M_mtx.size, M_mtx.size, M_mtx.nnz, M_mtx.row_indx.data(), M_mtx.col_indx.data(), M_mtx.values.data());
                    if (status != SPARSE_STATUS_SUCCESS)
                        throw std::invalid_argument("error while reading M: status = " + std::to_string(status));
                    status = mkl_sparse_convert_csr(M_coord, SPARSE_OPERATION_NON_TRANSPOSE, &M);
                    if (status != SPARSE_STATUS_SUCCESS)
                        throw std::invalid_argument("error while converting M to CSR: status = " + std::to_string(status));
                    logger.beg("load csr");
                        sparse_index_base_t A_indexing;
                        MKL_INT             A_rows;
                        MKL_INT             A_cols;
                        MKL_INT*            A_rows_start;
                        MKL_INT*            A_rows_end;
                        MKL_INT*            A_col_indx;
                        double*             A_values;
                        status = mkl_sparse_d_export_csr(A, &A_indexing, &A_rows, &A_cols, &A_rows_start, &A_rows_end, &A_col_indx, &A_values);
        //                logger.log("A_indexing = " + std::to_string(A_indexing));
        //                logger.buf << A_rows << ' ' << A_cols;
        //                logger.log();
        //                {
        //                    logger.buf << "A values:\n";
        //                    for (size_t i = 0; i < A_mtx.size; ++i) {
        //                        for (size_t k = A_rows_start[i]; k < A_rows_start[i + 1]; ++k)
        //                            logger.buf << A_values[k - 1] << ' ';
        //                        logger.buf << '\n';
        //                    }
        //                    logger.log();
        //                    logger.buf << "A row ptr:\n";
        //                    for (size_t i = 0; i < A_mtx.size; ++i) {
        //                        logger.buf << i + 1 << ": " << A_rows_start[i+1] - A_rows_start[i] << '\n';
        //                    }
        //                    logger.buf << '\n';
        //                    for (size_t i = 0; i < A_mtx.size + 1; ++i) {
        //                        logger.buf << A_rows_start[i] << ' ';
        //                    }
        //                    logger.log();
        //                }
                        if (status != SPARSE_STATUS_SUCCESS)
                            throw std::invalid_argument("error while exporting A to CSR: status = " + std::to_string(status));
                        sparse_index_base_t M_indexing;
                        MKL_INT             M_rows;
                        MKL_INT             M_cols;
                        MKL_INT             *M_rows_start;
                        MKL_INT             *M_rows_end;
                        MKL_INT             *M_col_indx;
                        double              *M_values;
                        status = mkl_sparse_d_export_csr(M, &M_indexing, &M_rows, &M_cols, &M_rows_start, &M_rows_end, &M_col_indx, &M_values);
                        if (status != SPARSE_STATUS_SUCCESS)
                            throw std::invalid_argument("error while exporting M to CSR: status = " + std::to_string(status));
                    logger.end();
                logger.end();
                logger.beg("find eigs in ($\\lambda_{min}$, $\\lambda_{max}$)");
                    char UPLO = 'F';
                    MKL_INT fpm[128]; // https://software.intel.com/sites/default/files/mkl-2019-developer-reference-c.pdf#unique_957
                    feastinit(fpm);
                    fpm[0] = 1;
                    fpm[1] = 32;
                    fpm[3] = 100;
                    fpm[5] = 1;
                    double epsout = 0.;
                    MKL_INT loop = 0, m0 = percent * A_mtx.size, m = m0, info = 0;
                    std::vector<double> lambdas, x, residuals;
                    while (true) {
                        lambdas.resize(m0, std::numeric_limits<double>::max());
                        x.resize(m0 * A_mtx.size);
                        residuals.resize(m0);
                        dfeast_scsrgv(
                                &UPLO,   /* IN: UPLO = 'F', stores the full matrix */
                                &A_mtx.size,      /* IN: Size of the problem */
                                A_values,     /* IN: CSR matrix A, values of non-zero elements */
                                A_rows_start,    /* IN: CSR matrix A, index of the rist non-zero in row */
                                A_col_indx,    /* IN: CSR matrix A, columns indeces for each non-zero element */
                                M_values,    /* IN: CSR matrix B, values of non-zero elements */
                                M_rows_start,   /* IN: CSR matrix B, index of the rist non-zero in row */
                                M_col_indx,   /* IN: CSR matrix A, columns indeces for each non-zero element */
                                fpm,     /* IN: Array is used to pass parameters to Intel(R) MKL Extended Eigensolvers */
                                &epsout, /* OUT: Relative error of on the trace */
                                &loop,   /* OUT: Contains the number of refinement loop executed */
                                &lambda_min,   /* IN: Lower bound of search interval */
                                &lambda_max,   /* IN: Upper bound of search interval */
                                &m0,     /* IN/OUT: The initial guess for subspace dimension to be used. */
                                lambdas.data(),       /* OUT: The first M entries of Eigenvalues */
                                x.data(),       /* OUT: The first M entries of Eigenvectors */
                                &m,      /* OUT: The total number of eigenvalues found in the interval */
                                residuals.data(),     /* OUT: The first M components contain the relative residual vector */
                                &info    /* OUT: Code of error */
                        );
                        if (info == 0) break;
                        m0 += percent * A_mtx.size;
                        if (m0 > A_mtx.size) {
                            logger.wrn("info = " + std::to_string(info) + ", m0 = " + std::to_string(m0) + "; break");
                            break;
                        }
                        m0 = std::min(m0, A_mtx.size);
                        logger.wrn("info = " + std::to_string(info) + ", m0 -> " + std::to_string(m0));
                    }
                    auto minOrMax = lambda_max >= 0. ? "min" : "max";
                    auto outPath = "output/" + FE + "/h=" + h + "_eps=" + epsilon + "_lambda_" + stab + "_" + minOrMax + ".txt";
                    logger.log("export eigs to " + outPath);
                    std::ofstream out(outPath);
                    if (info == 0) {
                        logger.log("info = " + std::to_string(info));
                        // std::sort(lambdas.begin(), lambdas.end(), [](double i, double j) { return abs(i) < abs(j); });
                        logger.buf << "m = " << m << '\n';
                        logger.buf.precision(3);
                        for (size_t i = 0; i < m; ++i)
                            logger.buf << "$\\lambda_" << i + 1 << "$ = " << std::scientific << lambdas[i] << '\n';
                        logger.buf
                            << "trace residual = " << epsout << '\n'
                            << "loop           = " << loop;
                        logger.log();
                        out.precision(16);
                        for (size_t i = 0; i < m; ++i)
                            out << std::scientific << lambdas[i] << '\n';
                    }
                    else {
                        logger.wrn("info = " + std::to_string(info));
                        out << "?\n";
                        if (minOrMax == "max") {
                            logger.beg("trying a different method");
                                matrix_descr A_descr;
                                A_descr.type = SPARSE_MATRIX_TYPE_GENERAL;
                                MKL_INT pm[128]; // https://software.intel.com/sites/default/files/mkl-2019-developer-reference-c.pdf#unique_957
                                mkl_sparse_ee_init(pm);
                                // pm[2] = 1;
                                pm[6] = 0; // eigenvalues only
                                char which = 'S';
                                MKL_INT k0 = percent * A_mtx.size;
                                status = SPARSE_STATUS_INTERNAL_ERROR;
                                while (status != SPARSE_STATUS_SUCCESS) {
                                    MKL_INT k = k0;
                                    std::vector<double> min_lambdas(k0, 0. /*std::numeric_limits<double>::max()*/);
                                    std::vector<double> min_eigvectors(k0 *A_mtx.size);
                                    std::vector<double> min_residuals(k0);

                                    status = mkl_sparse_d_gv(&which, pm, A, A_descr, M, A_descr, k0, &k, min_lambdas.data(),
                                                             min_eigvectors.data(), min_residuals.data());
                                    // std::sort(lambdas.begin(), lambdas.end(), [](double i, double j) { return abs(i) < abs(j); });

                                    if (status != SPARSE_STATUS_SUCCESS) {
                                        k0 += percent * A_mtx.size;
                                        k0 = std::min(k0, A_mtx.size);
                                        logger.wrn("status = " + std::to_string(status) + ", k0 -> " + std::to_string(k0));
                                    }
                                    else {
                                        logger.buf << "k = " << k << '\n';
                                        for (size_t i = 0; i < k; ++i)
                                            logger.buf << "$\\lambda_" << i + 1 << "$ = " << min_lambdas[i] << '\n';
                                        logger.log();
                                        for (size_t i = 0; i < k; ++i)
                                            out << std::scientific << min_lambdas[i] << '\n';
                                    }
                                }
                            logger.end();
                        }
                    }
                logger.end();
            logger.end();
        }

        //auto status = mkl_sparse_d_gv('S', );

//        double *A, *B, *C;
//        int m, n, k, i, j;
//        double alpha, beta;
//
//        printf("\n This example computes real matrix C=alpha*A*B+beta*C using \n"
//               " Intel(R) MKL function dgemm, where A, B, and  C are matrices and \n"
//               " alpha and beta are double precision scalars\n\n");
//
//        m = 2000, k = 200, n = 1000;
//        printf(" Initializing data for matrix multiplication C=A*B for matrix \n"
//               " A(%ix%i) and matrix B(%ix%i)\n\n", m, k, k, n);
//        alpha = 1.0;
//        beta = 0.0;
//
//        printf(" Allocating memory for matrices aligned on 64-byte boundary for better \n"
//               " performance \n\n");
//        A = (double *) mkl_malloc(m * k * sizeof(double), 64);
//        B = (double *) mkl_malloc(k * n * sizeof(double), 64);
//        C = (double *) mkl_malloc(m * n * sizeof(double), 64);
//        if (A == NULL || B == NULL || C == NULL) {
//            printf("\n ERROR: Can't allocate memory for matrices. Aborting... \n\n");
//            mkl_free(A);
//            mkl_free(B);
//            mkl_free(C);
//            return 1;
//        }
//
//        printf(" Intializing matrix data \n\n");
//        for (i = 0; i < (m * k); i++) {
//            A[i] = (double) (i + 1);
//        }
//
//        for (i = 0; i < (k * n); i++) {
//            B[i] = (double) (-i - 1);
//        }
//
//        for (i = 0; i < (m * n); i++) {
//            C[i] = 0.0;
//        }
//
//        printf(" Computing matrix product using Intel(R) MKL dgemm function via CBLAS interface \n\n");
//        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
//                    m, n, k, alpha, A, k, B, n, beta, C, n);
//        printf("\n Computations completed.\n\n");
//
//        printf(" Top left corner of matrix A: \n");
//        for (i = 0; i < min(m, 6); i++) {
//            for (j = 0; j < min(k, 6); j++) {
//                printf("%12.0f", A[j + i * k]);
//            }
//            printf("\n");
//        }
//
//        printf("\n Top left corner of matrix B: \n");
//        for (i = 0; i < min(k, 6); i++) {
//            for (j = 0; j < min(n, 6); j++) {
//                printf("%12.0f", B[j + i * n]);
//            }
//            printf("\n");
//        }
//
//        printf("\n Top left corner of matrix C: \n");
//        for (i = 0; i < min(m, 6); i++) {
//            for (j = 0; j < min(n, 6); j++) {
//                printf("%12.5G", C[j + i * n]);
//            }
//            printf("\n");
//        }
//
//        printf("\n Deallocating memory \n\n");
//        mkl_free(A);
//        mkl_free(B);
//        mkl_free(C);
//
//        printf(" Example completed. \n\n");
        logger.exp("stdin.txt");
    }
    catch (std::exception const & e) {
        logger.err(e.what());
    }
    return 0;
}