
#include "INA.h"
#include <time.h>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <numeric>
#include "circ_shift.h"
#include "eigen-3.4.0/Eigen/Core"
#include "eigen-3.4.0/Eigen/Eigen"
#include "eigen-3.4.0/Eigen/Dense"
#include "tuple"
#include <chrono>
#include "ckdtree/src/ckdtree_decl.h"
#include <algorithm>    // std::random_shuffle
#include <fstream>

using namespace Eigen;

void _post_init_traverse(ckdtree *self, ckdtreenode *node) {
    if (node->split_dim == -1) {
        node->less = NULL;
        node->greater = NULL;
    } else {
        node->less = self->ctree + node->_less;
        node->greater = self->ctree + node->_greater;
        _post_init_traverse(self, node->less);
        _post_init_traverse(self, node->greater);
    }
}

Eigen::MatrixXd CKD_find_KNN4_radius_per_point(Eigen::MatrixXd &Stack, const int k, const int leaves) {

    MatrixXd P_mins = Stack.rowwise().minCoeff();
    MatrixXd P_maxes = Stack.rowwise().maxCoeff();

    std::vector<intptr_t> X_raw_indices(Stack.cols());
    std::iota(X_raw_indices.begin(), X_raw_indices.end(), 0); // aranges from 0 to len of columns of points P
    intptr_t *raw_indices = X_raw_indices.data();

    // Pointers
    double *P_ptr = Stack.data();       // (travels columnwise point per point)
    double *maxes_ptr = P_maxes.data(); // (travels m)
    double *mins_ptr = P_mins.data();   // (travels m)
    double *raw_boxsize = NULL;         // not applicable for CPP --> NULL

    // ckdtree initialization
    ckdtree *KD1 = (ckdtree *) malloc(sizeof(ckdtree));
    ckdtreenode *ctree = (ckdtreenode *) malloc(sizeof(ckdtreenode));
    std::vector<ckdtreenode> *tree_buffer = new std::vector<ckdtreenode>;

    ctree->split_dim = -1; // initiate ctree with -1, as root
    KD1->raw_data = P_ptr; // pointer to data --> travels columwise
    KD1->n = Stack.cols(); // number of points -- opposite to numpy for numpy its 1point per row and 1 dim per column
    KD1->m = Stack.rows(); // dimension of points -- opposite to numpy for numpy its 1point per row and 1 dim per column
    KD1->leafsize = leaves; // number of leaves per node
    KD1->raw_maxes = maxes_ptr; // max values per dimension (m)
    KD1->raw_mins = mins_ptr;   // min values per dimension (m)
    KD1->raw_indices = raw_indices; // indices of the points arange(0, n)
    KD1->raw_boxsize_data = raw_boxsize; // NULL Not applicable in cPP
    KD1->tree_buffer = tree_buffer;

    const int balanced_tree = 1;
    const int compact_tree = 0; // breaks does not work since it requires quiet a lot of memory and breaks on results vector
    build_ckdtree(KD1, 0, Stack.cols(), maxes_ptr, mins_ptr, balanced_tree, compact_tree);

    // transfer information to ctree
    KD1->ctree = KD1->tree_buffer->data();
    KD1->size = KD1->tree_buffer->size();

    //post init traverse --> traverse the tree to update the less and greater node attributes very important!
    _post_init_traverse(KD1, KD1->ctree);

    ////////////////// QUERY KNN /////////////////
    // n: number of points (ROWS)
    // m: dimensions of point (COLS)
    const int KNN = k;

    std::vector<intptr_t> k_c(KNN);
    std::iota(k_c.begin(), k_c.end(), 1);

    const intptr_t *kk = k_c.data();
    const intptr_t k_max = KNN;

    double *pxx = Stack.data();
    const intptr_t nn = Stack.cols(); // number of queries points

    MatrixXd DD = MatrixXd::Zero(KNN, nn); // collects distances for each KNN
    std::vector<intptr_t> II(KNN * nn);
    double *pdd = DD.data();
    intptr_t *pii = II.data();

    double p = std::numeric_limits<double>::infinity(); // infty - Chebyshev, 2 Minkowski...
    double upper_bound = std::numeric_limits<double>::infinity(); // maximal distance between points taken into account

    int nk = KNN; //number of closest neighbours checked

    query_knn(KD1, pdd, pii, pxx, nn, kk, nk, k_max, 0, p, upper_bound);

    MatrixXd radiuses = DD.block(KNN - 1, 0, 1, nn);

    free(KD1);
    free(ctree);
    delete tree_buffer;

    return radiuses;
}

Eigen::MatrixXd
CKD_count_NNs_with_radius_search(Eigen::MatrixXd &Y1Y_stack, Eigen::MatrixXd &radiuses, const int leaves) {

    MatrixXd P_mins = Y1Y_stack.rowwise().minCoeff();
    MatrixXd P_maxes = Y1Y_stack.rowwise().maxCoeff();

    std::vector<intptr_t> X_raw_indices(Y1Y_stack.cols());
    std::iota(X_raw_indices.begin(), X_raw_indices.end(), 0); // aranges from 0 to len of columns of points P
    intptr_t *raw_indices = X_raw_indices.data();

    double *P_ptr = Y1Y_stack.data();  // (travels columnwise point per point)
    double *maxes_ptr = P_maxes.data(); // (travels m)
    double *mins_ptr = P_mins.data();  // (travels m)
    double *raw_boxsize = NULL; // not applicable for CPP --> NULL

    ckdtree *KD1 = (ckdtree *) malloc(sizeof(ckdtree));
    ckdtreenode *ctree = (ckdtreenode *) malloc(sizeof(ckdtreenode));
    std::vector<ckdtreenode> *tree_buffer = new std::vector<ckdtreenode>;


    ctree->split_dim = -1; // initiate ctree with -1, as root
    KD1->raw_data = P_ptr; // pointer to data --> travels columwise
    KD1->n = Y1Y_stack.cols(); // number of points -- opposite to numpy for numpy its 1point per row and 1 dim per column
    KD1->m = Y1Y_stack.rows(); // dimension of points -- opposite to numpy for numpy its 1point per row and 1 dim per column
    KD1->leafsize = leaves; // number of leaves per node
    KD1->raw_maxes = maxes_ptr; // max values per dimension (m)
    KD1->raw_mins = mins_ptr;   // min values per dimension (m)
    KD1->raw_indices = raw_indices; // indices of the points arange(0, n)
    KD1->raw_boxsize_data = raw_boxsize; // NULL Not applicable in cPP
    KD1->tree_buffer = tree_buffer;

    const int balanced_tree = 1;
    const int compact_tree = 0; // breaks does not work since it requires quiet a lot of memory and breaks on results vector
    build_ckdtree(KD1, 0, Y1Y_stack.cols(), maxes_ptr, mins_ptr, balanced_tree, compact_tree);

    // transfer information to ctree
    KD1->ctree = KD1->tree_buffer->data();
    KD1->size = KD1->tree_buffer->size();

    //post init traverse --> traverse the tree to update the less and greater node attributes very important!
    _post_init_traverse(KD1, KD1->ctree);

    const intptr_t nn = Y1Y_stack.cols(); // number of queries points
    double p = std::numeric_limits<double>::infinity(); // infty - Chebyshev, 2 Minkowski...
    double *pxx = Y1Y_stack.data();
    double *prr = radiuses.data();
    const double epsilon = 0;

    // Results vector of vectors
    std::vector<std::vector<ckdtree_intp_t>> *Vres = new std::vector<std::vector<ckdtree_intp_t>>(Y1Y_stack.cols() + 1);
    std::vector<ckdtree_intp_t> *results = Vres->data();
    query_ball_point(KD1, pxx, prr, p, epsilon, nn, results, true, false);

    MatrixXd NN_NUMBER = MatrixXd::Zero(1, nn);
    for (int i = 0; i < nn; i++) {
        NN_NUMBER(0, i) = results[i][0]; // IMPORTANT: This accounts for the <DIGAMMA(PSI()+1)>
    }

    free(KD1);
    free(ctree);
    delete tree_buffer;
    delete Vres;

    return NN_NUMBER;

}

// standardization functions -- inplace methods
void standardize_timeseries(Eigen::MatrixXd &mat) {

    double mean;
    double var;
    double std;
    MatrixXd row;
    // standardizes each row seperately in case Z has multiple rows composed of more than 1 timeseries
    for (int i = 0; i < mat.rows(); i++) {
        row = mat.block(i, 0, 1, mat.cols());
        mean = row.sum() / row.cols();
        var = (((row.array() - mean).array().pow(2).sum()) / row.cols());
        std = pow(var, 0.5);
        mat.block(i, 0, 1, mat.cols()) = (row.array() - mean) / std;
    }

}

void add_noise_timeseries(Eigen::MatrixXd &mat, const int power) {

    const double noise_scale = pow(10, power);
    std::default_random_engine engine;
    engine.seed(std::chrono::system_clock::now().time_since_epoch().count());
    std::normal_distribution<> dist{0, noise_scale};
    auto normal = [&]() { return dist(engine); };
    MatrixXd gaussian = MatrixXd::NullaryExpr(mat.rows(), mat.cols(), normal);

    mat += gaussian;
}

// determine the indexes of nearest neighbours
Eigen::MatrixXd CKD_find_KNN4_indexes(Eigen::MatrixXd &Stack, const int k, const int leaves) {

    MatrixXd P_mins = Stack.rowwise().minCoeff();
    MatrixXd P_maxes = Stack.rowwise().maxCoeff();

    std::vector<intptr_t> X_raw_indices(Stack.cols());
    std::iota(X_raw_indices.begin(), X_raw_indices.end(), 0); // aranges from 0 to len of columns of points P
    intptr_t *raw_indices = X_raw_indices.data();

    // Pointers
    double *P_ptr = Stack.data();  // (travels columnwise point per point)
    double *maxes_ptr = P_maxes.data(); // (travels m)
    double *mins_ptr = P_mins.data();  // (travels m)
    double *raw_boxsize = NULL;             // not applicable for CPP --> NULL

    // initialize ckdtree -- sourced from Scipy
    ckdtree *KD1 = (ckdtree *) malloc(sizeof(ckdtree));
    ckdtreenode *ctree = (ckdtreenode *) malloc(sizeof(ckdtreenode));
    std::vector<ckdtreenode> *tree_buffer = new std::vector<ckdtreenode>;

    ctree->split_dim = -1; // initiate ctree with -1, as root
    KD1->raw_data = P_ptr; // pointer to data --> travels columwise
    KD1->n = Stack.cols(); // number of points -- opposite to numpy for numpy its 1point per row and 1 dim per column
    KD1->m = Stack.rows(); // dimension of points -- opposite to numpy for numpy its 1point per row and 1 dim per column
    KD1->leafsize = leaves; // number of leaves per node
    KD1->raw_maxes = maxes_ptr; // max values per dimension (m)
    KD1->raw_mins = mins_ptr;   // min values per dimension (m)
    KD1->raw_indices = raw_indices; // indices of the points arange(0, n)
    KD1->raw_boxsize_data = raw_boxsize; // NULL Not applicable in cPP
    KD1->tree_buffer = tree_buffer;

    const int balanced_tree = 1;
    const int compact_tree = 0; // breaks does not work since it requires quiet a lot of memory and breaks on results vector
    build_ckdtree(KD1, 0, Stack.cols(), maxes_ptr, mins_ptr, balanced_tree, compact_tree);

    // transfer information to ctree
    KD1->ctree = KD1->tree_buffer->data();
    KD1->size = KD1->tree_buffer->size();

    //post init traverse --> traverse the tree to update the less and greater node attributes very important!
    _post_init_traverse(KD1, KD1->ctree);

    ////////////////// QUERY KNN /////////////////
    // n: number of points (ROWS)
    // m: dimensions of point (COLS)
    const int KNN = k;

    std::vector<intptr_t> k_c(KNN);
    std::iota(k_c.begin(), k_c.end(), 1); // start arange from 1 to size()

    const intptr_t *kk = k_c.data();
    const intptr_t k_max = KNN;


    double *pxx = Stack.data();
    const intptr_t nn = Stack.cols(); // number of queries points


    MatrixXd DD = MatrixXd::Zero(KNN, nn); // collects distances for each KNN
    std::vector<intptr_t> II(KNN * nn);
    double *pdd = DD.data();
    intptr_t *pii = II.data();

    double p = std::numeric_limits<double>::infinity(); // infty - Chebyshev, 2 Minkowski...
    double upper_bound = std::numeric_limits<double>::infinity(); // maximal distance between points taken into account

    int nk = KNN; //number of closest neighbours checked

    query_knn(KD1, pdd, pii, pxx, nn, kk, nk, k_max, 0, p, upper_bound);

    MatrixXd radiuses = DD.block(KNN - 1, 0, 1, nn);

    MatrixXd INDEX_collector = MatrixXd::Zero(nk, nn);

    for (int i = 0; i < II.size(); i++) {
        INDEX_collector(i % nk, i / nk) = II[i];
    }

    free(KD1);
    free(ctree);
    delete tree_buffer;

    return INDEX_collector;
}

// perform autoembedding to determine history and embedding delay for source
std::tuple<int, int>
Ragwitz_auto_embedding(Eigen::MatrixXd X, Eigen::MatrixXd Y, const int k, const int k_max, const int tau_max) {
    // Y - destination --> cols points | rows dimensions
    // X - source

    using namespace std;
    int kNN = k + 1; // since we dont want to count the point itself here

    assert(X.cols() == Y.cols());
    const int data_cols = Y.cols();

    MatrixXd Y_stack;
    MatrixXd Y_shift;
    MatrixXd temp;
    MatrixXd temp2;
    MatrixXd value_to_predict;
    MatrixXd X_stack;
    MatrixXd indexes;
    MatrixXd combined_stack;
    double predicted_value = 0;
    double total_error = 0;
    double best_total_error = 0;
    int k_best = 0;
    int tau_best = 0;
    int destination_value_index;
    int main_shift;
    int embed_shift;
    int Y_shift_cols;

    for (int k_candidate = 1; k_candidate <= k_max; k_candidate++) {
        for (int tau_candidate = 1; tau_candidate <= tau_max; tau_candidate++) {
            // resetting mean squared error for each candidate
            total_error = 0;

            // Y_shift
            Y_shift = MatrixXd::Zero(1, data_cols);
            temp = MatrixXd::Zero(1, data_cols);

            // main shift is used for Y_shift needed due to the time series truncation resulting from embedding
            main_shift = (k_candidate - 1) * tau_candidate + 1;
            temp.block(0, 0, 1, data_cols) = circShift(Y, 0, -main_shift);
            Y_shift = temp.block(0, 0, 1, data_cols - main_shift);

            // number of time points to consider -- dynamic as th Y_shift is truncated
            Y_shift_cols = Y_shift.cols();

            // Y_stack
            Y_stack = MatrixXd::Zero(k_candidate, Y_shift_cols);
            embed_shift = main_shift - 1;
            for (int i = 0; i < k_candidate; ++i) {
                temp2 = circShift(Y, 0, -(embed_shift - tau_candidate * i)); // so far correcr k=1 k=2 k=3
                Y_stack.block(i, 0, 1, Y_shift_cols) = temp2.block(0, 0, 1, Y_shift_cols);
            }
            // truncate source by the number of columns lost in Y_shift
            X_stack = X.block(0, 0, X.rows(), Y_shift_cols);

            // stacking up source and destination that is not shifted
            combined_stack = MatrixXd::Zero(Y_stack.rows() + X_stack.rows(), Y_stack.cols());
            combined_stack << Y_stack, X_stack;

            // find indexes of the nearest neighbours of the embedded stack that will be used to determine the corresponding destination values
            indexes = CKD_find_KNN4_indexes(combined_stack, kNN, 100);

            for (int t = 0; t < Y_shift_cols; t++) {

                predicted_value = 0; // zeroing out prediction value for each time point
                value_to_predict = Y_shift.block(0, t, 1,
                                                 1); // destination value that is being predicted by embedded stack

                // determine the mean predictions of the NNs
                for (int nn = 1; nn < kNN; nn++) {
                    destination_value_index = indexes(nn, t);
                    predicted_value += Y_shift(0, destination_value_index);
                }
                // normalized by the number of nearest neighbours
                predicted_value /= (kNN - 1); // -1 because we did not account for NN(0) which is the point itself

                // mean squared prediction error which is summed over all time points
                total_error += (value_to_predict(0, 0) - predicted_value) * (value_to_predict(0, 0) - predicted_value);
            }

            // mean squared prediction error normalized by the number of time points used for the estimate
            // generalized normalization needs to account for smaller number of observations when tau>1 or k>1
            total_error /= Y_shift.cols();

            // setup best_score for k=1 and tau=1, then change if any lower error is obtained
            if (k_candidate == 1 && tau_candidate == 1) {
                best_total_error = total_error;
                k_best = k_candidate;
                tau_best = tau_candidate;
            } else if (total_error < best_total_error) {
                best_total_error = total_error;
                k_best = k_candidate;
                tau_best = tau_candidate;
            }
            // no point for different delay embedding for k=1 this can be achieved with simple DELAY
            if (k_candidate == 1) { break; }
        }
    }
    return std::make_tuple(k_best, tau_best);
}


// function used to calculate the "actual" transfer entropy from source X to destination Y,
// with the specified parameters
double TE_X_to_Y_inner(Eigen::MatrixXd &X, Eigen::MatrixXd &Y,
                       const int X_hist, const int Y_hist,
                       const int X_tau, const int Y_tau, const int DELAY,
                       const int k, const int leaves) {

    double base = exp(1); // base for entropy values

    // asserting that input data is correct
    assert(X.rows() == Y.rows());
    assert(X.cols() == Y.cols());

    // dimension of data
    const int data_cols = X.cols();

    // main rotation of data and hence data truncation
    int main_shift = std::max({(X_hist - 1) * X_tau + DELAY, (Y_hist - 1) * Y_tau + DELAY});
    main_shift -= 1;

    // shifting Y destination
    MatrixXd temp = circShift(Y, 0, -main_shift);
    MatrixXd Y_shift = temp.block(0, 0, 1, data_cols - main_shift);

    // the final number of columns that will be used in estimation
    int Y_shift_cols = Y_shift.cols();

    // stacking history of X variable
    int embed_shift = main_shift - DELAY; // Delay for the source
    MatrixXd X_HIST_STACK = MatrixXd::Zero(X_hist, Y_shift_cols);
    for (int i = 0; i < X_hist; i++) {
        temp = circShift(X, 0, -(embed_shift - X_tau * i));
        X_HIST_STACK.block(i, 0, 1, Y_shift_cols) = temp.block(0, 0, 1, Y_shift_cols);
    }


    embed_shift = main_shift - 1; // delay does not affect the Y_hist_stack
    // stacking history of Y variable
    MatrixXd Y_HIST_STACK = MatrixXd::Zero(Y_hist, Y_shift_cols);
    for (int i = 0; i < Y_hist; i++) {
        temp = circShift(Y, 0, -(embed_shift - Y_tau * i));
        Y_HIST_STACK.block(i, 0, 1, Y_shift_cols) = temp.block(0, 0, 1, Y_shift_cols);
    }

    // pre-stacking all necessary stacks
    MatrixXd Y1XY_stack(1 + X_hist + Y_hist, Y_shift_cols);
    Y1XY_stack << Y_shift, X_HIST_STACK, Y_HIST_STACK;  // stacks all matrices on top of each other

    MatrixXd Y1Y_stack(1 + Y_hist, Y_shift_cols);
    Y1Y_stack << Y_shift, Y_HIST_STACK;  // stacks all matrices on top of each other

    MatrixXd XY_stack(X_hist + Y_hist, Y_shift_cols);
    XY_stack << X_HIST_STACK, Y_HIST_STACK;  // stacks all matrices on top of each other


    // extract distances to the kth NN
    MatrixXd radiuses = CKD_find_KNN4_radius_per_point(Y1XY_stack, k + 1, leaves);
    radiuses = radiuses.array() - pow(10, -15); // needed to induce find NNs with distances smaller than radiuses
    radiuses = (radiuses.array() < 0).select(0, radiuses); // if radiuses are negative it breaks ckdtree

    // count NNs within the radius defined by radiuses
    MatrixXd NNcount_Y1Y = CKD_count_NNs_with_radius_search(Y1Y_stack, radiuses, leaves);
    MatrixXd NNcount_XY = CKD_count_NNs_with_radius_search(XY_stack, radiuses, leaves);
    MatrixXd NNcount_Y = CKD_count_NNs_with_radius_search(Y_HIST_STACK, radiuses, leaves);

    // needed for Eigen digamma function
    ArrayXXd k_matrix(1, 1);
    k_matrix(0, 0) = k;

    // IMPORTANT: <DIGAMMA(PSI()+1)> is accounted for in the loop that translates results vector into NN_NUMBER
    // the NN is actually counting NN+1 hence all NNs are NN+1 !!!!!!! This is correct implementation, yet its not
    // explicitly written like the equation is.
    double digamma_Y1Y = digamma(NNcount_Y1Y.array()).mean();
    double digamma_XY = digamma(NNcount_XY.array()).mean();
    double digamma_Y = digamma(NNcount_Y.array()).mean();
    double digamma_k = digamma(k_matrix).mean();

    double TE_XtoY_Kraskov =
            (digamma_k - digamma_Y1Y - digamma_XY + digamma_Y) / log(base); // logbase defines the units of the metric

    return TE_XtoY_Kraskov;
}

// function used for permutation testing
double TE_X_to_Y_permute(Eigen::MatrixXd &X, Eigen::MatrixXd &Y, double TE_actual,
                         const int X_hist, const int Y_hist,
                         const int X_tau, const int Y_tau,
                         const int DELAY,
                         const int permutation, const double min_pval,
                         const int k, const int leaves) {

    double base = exp(1); // base for entropy values
    double count = 0;

    // asserting that input data is correct
    assert(X.rows() == Y.rows());
    assert(X.cols() == Y.cols());

    // dimension of data
    const int data_cols = X.cols();


    //data truncation due to time series rotation
    int main_shift = std::max({DELAY, (X_hist - 1) * X_tau + DELAY, (Y_hist - 1) * Y_tau + DELAY});
    main_shift -= 1;

    // shifting Y destination
    MatrixXd temp = circShift(Y, 0, -main_shift);
    MatrixXd Y_shift = temp.block(0, 0, 1, data_cols - main_shift);

    int Y_shift_cols = Y_shift.cols();

    int embed_shift = main_shift - DELAY;
    // stacking history of X variable
    MatrixXd X_HIST_STACK = MatrixXd::Zero(X_hist, Y_shift_cols);
    for (int i = 0; i < X_hist; i++) {
        temp = circShift(X, 0, -(embed_shift - X_tau * i));
        X_HIST_STACK.block(i, 0, 1, Y_shift_cols) = temp.block(0, 0, 1, Y_shift_cols);
    }


    embed_shift = main_shift - 1; // delay does not affect the Y_hist_stack
    // stacking history of Y variable
    MatrixXd Y_HIST_STACK = MatrixXd::Zero(Y_hist, Y_shift_cols);
    for (int i = 0; i < Y_hist; i++) {
        temp = circShift(Y, 0, -(embed_shift - Y_tau * i));
        Y_HIST_STACK.block(i, 0, 1, Y_shift_cols) = temp.block(0, 0, 1, Y_shift_cols);
    }

    // PERMUTATION
    std::default_random_engine engine;
    engine.seed(std::chrono::system_clock::now().time_since_epoch().count());
    PermutationMatrix<Dynamic, Dynamic> perm(X_HIST_STACK.cols());
    perm.setIdentity();

    for (int i = 0; i < permutation; i++) {
        // Permute the X source matrix columnwise-- the history embedding stays untouched
        std::shuffle(perm.indices().data(), perm.indices().data() + perm.indices().size(), engine);
        X_HIST_STACK = X_HIST_STACK * perm;

        // pre-stacking all necessary stacks
        MatrixXd Y1XY_stack(1 + X_hist + Y_hist, Y_shift_cols);
        Y1XY_stack << Y_shift, X_HIST_STACK, Y_HIST_STACK;  // stacks all matrices on top of each other

        MatrixXd Y1Y_stack(1 + Y_hist, Y_shift_cols);
        Y1Y_stack << Y_shift, Y_HIST_STACK;  // stacks all matrices on top of each other

        MatrixXd XY_stack(X_hist + Y_hist, Y_shift_cols);
        XY_stack << X_HIST_STACK, Y_HIST_STACK;  // stacks all matrices on top of each other

        // extract distances to the kth NN
        MatrixXd radiuses = CKD_find_KNN4_radius_per_point(Y1XY_stack, k + 1, leaves);
        radiuses = radiuses.array() - pow(10, -15); // needed to induce find NNs with distances smaller than radiuses
        radiuses = (radiuses.array() < 0).select(0, radiuses); // if radiuses are negative it breaks ckdtree

        // count NNs within the radius defined by radiuses
        MatrixXd NNcount_Y1Y = CKD_count_NNs_with_radius_search(Y1Y_stack, radiuses, leaves);
        MatrixXd NNcount_XY = CKD_count_NNs_with_radius_search(XY_stack, radiuses, leaves);
        MatrixXd NNcount_Y = CKD_count_NNs_with_radius_search(Y_HIST_STACK, radiuses, leaves);

        // needed for Eigen digamma function
        ArrayXXd k_matrix(1, 1);
        k_matrix(0, 0) = k;

        // IMPORTANT: <DIGAMMA(PSI()+1)> is accounted for in the loop that translates results vector into NN_NUMBER
        // the NN is actually counting NN+1 hence all NNs are NN+1 !!!!!!! This is correct implementation, yet its not
        // explicitly written like the equation is.
        double digamma_Y1Y = digamma(NNcount_Y1Y.array()).mean();
        double digamma_XY = digamma(NNcount_XY.array()).mean();
        double digamma_Y = digamma(NNcount_Y.array()).mean();
        double digamma_k = digamma(k_matrix).mean();


        double TE_XtoY_surrogate = (digamma_k - digamma_Y1Y - digamma_XY + digamma_Y) /
                                   log(base); // logbase defines the units of the metric

        // count
        if (TE_XtoY_surrogate >= TE_actual) {
            count += 1;
        }
        // break the loop if the count is high enough not to meet the p_value criteria
        // - no point of checking more surrogates
        if (count / permutation > min_pval) {
            count = -permutation;
            break;
        }

    }

    return count / permutation;
}


// the highest level function of Transfer Entropy
returnVector_TE TE_X_to_Y_INA(Eigen::MatrixXd X,
                              Eigen::MatrixXd Y,
                              int X_hist, int Y_hist,
                              int X_tau, int Y_tau, int DELAY,
                              bool AUTOEMB, int HIST_MAX, int TAU_MAX,
                              bool AUTODEL, int MAX_DELAY,
                              const int permutation, const double min_pval,
                              const bool STD, const int NOISE,
                              const int k, const int leaves, bool DEBUG) {

    // Transfer Entropy TE_X->Y = I(Y_i+1;X|Y) - Conditional Mutual Information

    // ensure correct input to the function
    assert(NOISE >= 0);

    // standardize or normalize input data
    if (STD) {
        standardize_timeseries(X);
        standardize_timeseries(Y);
    }
    // add noise to the data (added after normalization)
    if (NOISE != 0) {
        add_noise_timeseries(X, -NOISE);
        add_noise_timeseries(Y, -NOISE);
    }

    // determine X_hist and Y_hist and X_tau and Y_tau delay embedding -- Ragwitz criterion
    if (AUTOEMB) {

        // Embedding Destination --> Y_test
        std::tuple k_tau_destination = Ragwitz_auto_embedding(X, Y, k, HIST_MAX, TAU_MAX);
        int k_destination_auto = std::get<0>(k_tau_destination);
        int tau_destination_auto = std::get<1>(k_tau_destination);
        // Embedding Source --> X_test
        std::tuple k_tau_source = Ragwitz_auto_embedding(X, X, k, HIST_MAX, TAU_MAX);
        int k_source_auto = std::get<0>(k_tau_source);
        int tau_source_auto = std::get<1>(k_tau_source);

        X_hist = k_source_auto;
        X_tau = tau_source_auto;
        Y_hist = k_destination_auto;
        Y_tau = tau_destination_auto;

    }

    // -- After setting the embedding parameters,
    // you can then select the source-destination lag so as to
    // maximise the TE for those given parameters. (see p. 18 of Wibral et al. above)
    double TE_candidate;
    double TE_estimate;
    int delay_best;
    if (AUTODEL) {
        for (int delay_candidate = 1; delay_candidate <= MAX_DELAY; delay_candidate++) {
            TE_candidate = TE_X_to_Y_inner(X, Y, X_hist, Y_hist, X_tau, Y_tau, delay_candidate, k, leaves);

            if (delay_candidate == 1) {
                TE_estimate = TE_candidate; // overwriting the TE estimate with the new best one
                delay_best = delay_candidate;
            } else if (TE_candidate > TE_estimate) {
                TE_estimate = TE_candidate; // overwriting the TE estimate with the new best one
                delay_best = delay_candidate;
            }
        }
        // set delay that yielded highest TE estimate
        DELAY = delay_best;

    } else {
        // calculate TE value using pre-set DELAY
        TE_estimate = TE_X_to_Y_inner(X, Y,
                                      X_hist, Y_hist,
                                      X_tau, Y_tau,
                                      DELAY, k, leaves);
    }

    // perform permutation test and determine significance
    double p_value;
    if (permutation > 0) {
        p_value = TE_X_to_Y_permute(X, Y, TE_estimate,
                                    X_hist, Y_hist,
                                    X_tau, Y_tau,
                                    DELAY,
                                    permutation, min_pval, k, leaves);

        // if the results is not statistically significant no need to keep
        if (p_value == -1) {
            TE_estimate = NAN;
        }
    } else { p_value = NAN; }

    return returnVector_TE{TE_estimate, p_value, X_hist, X_tau, Y_hist, Y_tau, DELAY};
}


double CTE_X_to_Y_inner(Eigen::MatrixXd &X, Eigen::MatrixXd &Y, Eigen::MatrixXd &Z,
                        const int X_hist, const int Y_hist, Eigen::MatrixXd Z_hist,
                        const int X_tau, const int Y_tau, Eigen::MatrixXd Z_tau,
                        const int DELAY_X, Eigen::MatrixXd DELAY_Z,
                        const int k, const int leaves) {

    // Z_hist dimensions      --> nb of rows = nb of conditionals
    // Z_tau dimensions       --> nb of rows = nb of conditionals
    // DELAY_COND dimensions  --> nb of rows = nb of conditionals

    double base = exp(1); // base for entropy values

    // asserting that input data is correct
    assert(X.rows() == Y.rows());
    assert(X.cols() == Y.cols());
    assert(Z.cols() == Y.cols());

    // dimension of data
    const int data_cols = X.cols();

    // parameters for conditionals
    const int nb_of_conditionals = Z.rows();
    const int Z_max_hist = Z_hist.colwise().maxCoeff()(0,0);      // there will be always one value columnwise  because 1 column only
    const int Z_max_tau = Z_tau.colwise().maxCoeff()(0,0);        // there will be always one value columnwise  because 1 column only
    const int Z_max_delay = DELAY_Z.colwise().maxCoeff()(0,0);    // there will be always one value columnwise  because 1 column only

    // main rotation of data and hence data truncation
    // this is a safe method that account for the max Z_hist Z_tau and Z_delay and hence the rotation will always be large enough
    int main_shift = std::max({(X_hist - 1) * X_tau + DELAY_X, (Y_hist - 1) * Y_tau + DELAY_X,
                               (Z_max_hist - 1) * Z_max_tau + Z_max_delay});
    main_shift -= 1;

    // shifting Y destination
    MatrixXd temp = circShift(Y, 0, -main_shift);
    MatrixXd Y_shift = temp.block(0, 0, 1, data_cols - main_shift);

    // the final number of columns that will be used in estimation
    int Y_shift_cols = Y_shift.cols();

    // stacking history of X variable
    int embed_shift = main_shift - DELAY_X; // Delay for the source
    MatrixXd X_HIST_STACK = MatrixXd::Zero(X_hist, Y_shift_cols);
    for (int i = 0; i < X_hist; i++) {
        temp = circShift(X, 0, -(embed_shift - X_tau * i));
        X_HIST_STACK.block(i, 0, 1, Y_shift_cols) = temp.block(0, 0, 1, Y_shift_cols);
    }


    // stacking history of Y variable
    embed_shift = main_shift - 1;     // delay does not affect the Y_hist_stack
    MatrixXd Y_HIST_STACK = MatrixXd::Zero(Y_hist, Y_shift_cols);
    for (int i = 0; i < Y_hist; i++) {
        temp = circShift(Y, 0, -(embed_shift - Y_tau * i));
        Y_HIST_STACK.block(i, 0, 1, Y_shift_cols) = temp.block(0, 0, 1, Y_shift_cols);
    }

    // stacking history of Z variables - conditionals with respect to their respective parameters
    MatrixXd Z_HIST_STACK = MatrixXd::Zero(Z_hist.sum(), Y_shift_cols);
    int Z_index = 0;
    MatrixXd Z_block;
    for (int i = 0; i < nb_of_conditionals; i++) {
        embed_shift = main_shift - DELAY_Z(i, 0); // use delay specific to the particular Z conditional
        Z_block = Z.block(i, 0, 1, Z.cols());
        for (int j = 0; j < Z_hist(i, 0); j++, Z_index++) {  // using Z_hist specific to the particular Z conditional
            temp = circShift(Z_block, 0, -(embed_shift - Z_tau(i, 0) * j));
            Z_HIST_STACK.block(Z_index, 0, 1, Y_shift_cols) = temp.block(0, 0, 1, Y_shift_cols);
        }
    }


    // pre-stacking all necessary stacks
    MatrixXd Y1XY_stack(1 + X_hist + Y_hist + Z_HIST_STACK.rows(), Y_shift_cols);
    Y1XY_stack << Y_shift, X_HIST_STACK, Y_HIST_STACK, Z_HIST_STACK;  // stacks all matrices on top of each other

    MatrixXd Y1Y_stack(1 + Y_hist + Z_HIST_STACK.rows(), Y_shift_cols);
    Y1Y_stack << Y_shift, Y_HIST_STACK, Z_HIST_STACK;  // stacks all matrices on top of each other

    MatrixXd XY_stack(X_hist + Y_hist + Z_HIST_STACK.rows(), Y_shift_cols);
    XY_stack << X_HIST_STACK, Y_HIST_STACK, Z_HIST_STACK;  // stacks all matrices on top of each other

    // conditional stacking
    MatrixXd conditional_stack(Y_hist + Z_HIST_STACK.rows(), Y_shift_cols);
    conditional_stack << Y_HIST_STACK, Z_HIST_STACK;

    // extract distances to the kth NN
    MatrixXd radiuses = CKD_find_KNN4_radius_per_point(Y1XY_stack, k + 1, leaves);
    radiuses = radiuses.array() - pow(10, -15); // needed to induce find NNs with distances smaller than radiuses
    radiuses = (radiuses.array() < 0).select(0, radiuses);// if radiuses are negative it breaks ckdtree

    // count NNs within the radius defined by radiuses
    MatrixXd NNcount_Y1Y = CKD_count_NNs_with_radius_search(Y1Y_stack, radiuses, leaves);
    MatrixXd NNcount_XY = CKD_count_NNs_with_radius_search(XY_stack, radiuses, leaves);
    MatrixXd NNcount_Y = CKD_count_NNs_with_radius_search(conditional_stack, radiuses, leaves);

    // needed for Eigen digamma function
    ArrayXXd k_matrix(1, 1);
    k_matrix(0, 0) = k;

    // IMPORTANT: <DIGAMMA(PSI()+1)> is accounted for in the loop that translates results vector into NN_NUMBER
    // the NN is actually counting NN+1 hence all NNs are NN+1
    double digamma_Y1Y = digamma(NNcount_Y1Y.array()).mean();
    double digamma_XY = digamma(NNcount_XY.array()).mean();
    double digamma_Y = digamma(NNcount_Y.array()).mean();
    double digamma_k = digamma(k_matrix).mean();


    double CTE_XtoY_Kraskov = (digamma_k - digamma_Y1Y - digamma_XY + digamma_Y) / log(base); // logbase defines the units of the metric

    return CTE_XtoY_Kraskov;
}

double CTE_X_to_Y_permute(Eigen::MatrixXd &X, Eigen::MatrixXd &Y, Eigen::MatrixXd &Z, double CTE_actual,
                          const int X_hist, const int Y_hist, Eigen::MatrixXd Z_hist,
                          const int X_tau, const int Y_tau, Eigen::MatrixXd Z_tau,
                          const int DELAY_X, Eigen::MatrixXd DELAY_Z,
                          const int permutation, const double min_pval,
                          const int k, const int leaves) {

    // Z_hist dimensions      --> nb of rows = nb of conditionals
    // Z_tau dimensions       --> nb of rows = nb of conditionals
    // DELAY_COND dimensions  --> nb of rows = nb of conditionals

    double base = exp(1); // base for entropy values
    double count = 0;

    // asserting that input data is correct
    assert(X.rows() == Y.rows());
    assert(X.cols() == Y.cols());
    assert(Z.cols() == Y.cols());

    // dimension of data
    const int data_cols = X.cols();

    // parameters for conditionals
    const int nb_of_conditionals = Z.rows();
    const int Z_max_hist = Z_hist.colwise().maxCoeff()(0,0);      // there will be always one value columnwise  because 1 column only
    const int Z_max_tau = Z_tau.colwise().maxCoeff()(0,0);        // there will be always one value columnwise  because 1 column only
    const int Z_max_delay = DELAY_Z.colwise().maxCoeff()(0,0); // there will be always one value columnwise  because 1 column only


    // main rotation of data and hence data truncation
    // this is a safe method that account for the max Z_hist Z_tau and Z_delay and hence the rotation will always be large enough
    int main_shift = std::max({(X_hist - 1) * X_tau + DELAY_X, (Y_hist - 1) * Y_tau + DELAY_X,
                               (Z_max_hist - 1) * Z_max_tau + Z_max_delay});
    main_shift -= 1;

    // shifting Y destination
    MatrixXd temp = circShift(Y, 0, -main_shift);
    MatrixXd Y_shift = temp.block(0, 0, 1, data_cols - main_shift);

    // the final number of columns that will be used in estimation
    int Y_shift_cols = Y_shift.cols();

    // stacking history of X variable
    int embed_shift = main_shift - DELAY_X; // Delay for the source
    MatrixXd X_HIST_STACK = MatrixXd::Zero(X_hist, Y_shift_cols);
    for (int i = 0; i < X_hist; i++) {
        temp = circShift(X, 0, -(embed_shift - X_tau * i));
        X_HIST_STACK.block(i, 0, 1, Y_shift_cols) = temp.block(0, 0, 1, Y_shift_cols);
    }

    // stacking history of Y variable
    embed_shift = main_shift - 1;     // delay does not affect the Y_hist_stack
    MatrixXd Y_HIST_STACK = MatrixXd::Zero(Y_hist, Y_shift_cols);
    for (int i = 0; i < Y_hist; i++) {
        temp = circShift(Y, 0, -(embed_shift - Y_tau * i));
        Y_HIST_STACK.block(i, 0, 1, Y_shift_cols) = temp.block(0, 0, 1, Y_shift_cols);
    }

    // stacking history of Z variables - conditionals with respect to their respective parameters
    MatrixXd Z_HIST_STACK = MatrixXd::Zero(Z_hist.sum(), Y_shift_cols);
    int Z_index = 0;
    MatrixXd Z_block;
    for (int i = 0; i < nb_of_conditionals; i++) {
        embed_shift = main_shift - DELAY_Z(i, 0); // use delay specific to the particular Z conditional
        Z_block = Z.block(i, 0, 1, Z.cols());
        for (int j = 0; j < Z_hist(i, 0); j++, Z_index++) {  // using Z_hist specific to the particular Z conditional
            temp = circShift(Z_block, 0, -(embed_shift - Z_tau(i, 0) * j));
            Z_HIST_STACK.block(Z_index, 0, 1, Y_shift_cols) = temp.block(0, 0, 1, Y_shift_cols);
        }
    }

    // PERMUTATION
    std::default_random_engine engine;
    engine.seed(std::chrono::system_clock::now().time_since_epoch().count());
    PermutationMatrix<Dynamic, Dynamic> perm(X_HIST_STACK.cols());
    perm.setIdentity();

    // temp
    MatrixXd CTE_SURROGATES = MatrixXd::Zero(1, permutation);

    for (int i = 0; i < permutation; i++) {
        // Permute the X source matrix columnwise-- the history embedding stays untouched
        std::shuffle(perm.indices().data(), perm.indices().data() + perm.indices().size(), engine);
        X_HIST_STACK = X_HIST_STACK * perm;

        // pre-stacking all necessary stacks
        MatrixXd Y1XY_stack(1 + X_hist + Y_hist + Z_HIST_STACK.rows(), Y_shift_cols);
        Y1XY_stack << Y_shift, X_HIST_STACK, Y_HIST_STACK, Z_HIST_STACK;  // stacks all matrices on top of each other

        MatrixXd Y1Y_stack(1 + Y_hist + Z_HIST_STACK.rows(), Y_shift_cols);
        Y1Y_stack << Y_shift, Y_HIST_STACK, Z_HIST_STACK;  // stacks all matrices on top of each other

        MatrixXd XY_stack(X_hist + Y_hist + Z_HIST_STACK.rows(), Y_shift_cols);
        XY_stack << X_HIST_STACK, Y_HIST_STACK, Z_HIST_STACK;  // stacks all matrices on top of each other

        // conditional stacking
        MatrixXd conditional_stack(Y_hist + Z_HIST_STACK.rows(), Y_shift_cols);
        conditional_stack << Y_HIST_STACK, Z_HIST_STACK;

        // extract distances to the kth NN
        MatrixXd radiuses = CKD_find_KNN4_radius_per_point(Y1XY_stack, k + 1, leaves);
        radiuses = radiuses.array() - pow(10, -15); // needed to induce find NNs with distances smaller than radiuses
        radiuses = (radiuses.array() < 0).select(0, radiuses);// if radiuses are negative it breaks ckdtree

        // count NNs within the radius defined by radiuses
        MatrixXd NNcount_Y1Y = CKD_count_NNs_with_radius_search(Y1Y_stack, radiuses, leaves);
        MatrixXd NNcount_XY = CKD_count_NNs_with_radius_search(XY_stack, radiuses, leaves);
        MatrixXd NNcount_Y = CKD_count_NNs_with_radius_search(conditional_stack, radiuses, leaves);

        // needed for Eigen digamma function
        ArrayXXd k_matrix(1, 1);
        k_matrix(0, 0) = k;

        // IMPORTANT: <DIGAMMA(PSI()+1)> is accounted for in the loop that translates results vector into NN_NUMBER
        // the NN is actually counting NN+1 hence all NNs are NN+1
        double digamma_Y1Y = digamma(NNcount_Y1Y.array()).mean();
        double digamma_XY = digamma(NNcount_XY.array()).mean();
        double digamma_Y = digamma(NNcount_Y.array()).mean();
        double digamma_k = digamma(k_matrix).mean();


        double CTE_XtoY_surrogate = (digamma_k - digamma_Y1Y - digamma_XY + digamma_Y) /
                                    log(base); // logbase defines the units of the metric

        CTE_SURROGATES(0, i) = CTE_XtoY_surrogate;

        // count
        if (CTE_XtoY_surrogate >= CTE_actual) {
            count += 1;
        }
        // break the loop if the count is high enough not to meet the p_value criteria
        // - no point of checking more surrogates and this saves a lot of time
        if (count / permutation > min_pval) {
            count = -permutation;
            break;
        }

    }
    return count / permutation;
}


returnVector_CTE CTE_X_to_Y_INA(Eigen::MatrixXd X, Eigen::MatrixXd Y, Eigen::MatrixXd Z,
                                const int X_hist, const int Y_hist, Eigen::MatrixXd Z_hist,
                                const int X_tau, const int Y_tau, Eigen::MatrixXd Z_tau,
                                bool AUTODEL, int MAX_DELAY, int DELAY_X, Eigen::MatrixXd DELAY_Z,
                                const int permutation, const double min_pval,
                                const bool STD, const int NOISE, const int k, const int leaves) {

    // Conditional Transfer Entropy TE_X->Y|Z = I(Y_i+1;X|Y,Z) - Conditional Mutual Information

    // ensure correct input to the function
    assert(NOISE >= 0);

    // standardize or normalize input data
    if (STD) {
        standardize_timeseries(X);
        standardize_timeseries(Y);
        standardize_timeseries(Z); // standardizes row by row even if Z.rows()>1 thus accounts
        // for possible multiple conditionals and standardizes them seperately
    }
    // add noise to the data (added after normalization)
    if (NOISE != 0) {
        add_noise_timeseries(X, -NOISE);
        add_noise_timeseries(Y, -NOISE);
        add_noise_timeseries(Z, -NOISE); // can handle matrices as well
    }

    // -- After setting the embedding parameters,
    // you can then select the source-destination lag so as to
    // maximise the TE for those given parameters. (see p. 18 of Wibral et al. above)
    double CTE_candidate;
    double CTE_estimate;
    int delay_best;

    if (AUTODEL) {
        for (int delay_candidate = 1; delay_candidate <= MAX_DELAY; delay_candidate++) {
            CTE_candidate = CTE_X_to_Y_inner(X, Y, Z,
                                             X_hist, Y_hist, Z_hist,
                                             X_tau, Y_tau, Z_tau,
                                             delay_candidate, DELAY_Z, k, leaves);

            if (delay_candidate == 1) {
                CTE_estimate = CTE_candidate; // overwriting the TE estimate with the new best one
                delay_best = delay_candidate;
            } else if (CTE_candidate > CTE_estimate) {
                CTE_estimate = CTE_candidate; // overwriting the TE estimate with the new best one
                delay_best = delay_candidate;
            }
        }
        // if autodelay then reset DELAY_X value to best delay else will use optimal DELAY_X determined in TE estimation
        DELAY_X = delay_best;

    } else {
        // calculate value for the actual value of CTE
        CTE_estimate = CTE_X_to_Y_inner(X, Y, Z,
                                        X_hist, Y_hist, Z_hist,
                                        X_tau, Y_tau, Z_tau,
                                        DELAY_X, DELAY_Z, k, leaves);
    }

    // perform permutation test and determine significance
    double p_value;

    if (permutation > 0) {
        // performing permutation
        p_value = CTE_X_to_Y_permute(X, Y, Z, CTE_estimate,
                                     X_hist, Y_hist, Z_hist,
                                     X_tau, Y_tau, Z_tau,
                                     DELAY_X, DELAY_Z,
                                     permutation, min_pval,
                                     k, leaves);

        // if the results is not statistically significant no need to keep
        if (p_value == -1) {
            CTE_estimate = NAN;
        }
    } else { p_value = NAN; }


    return returnVector_CTE{CTE_estimate, p_value};
}


// Information Network Analysis

/** @brief Compute Transfer Entropy (TE) and Conditional Transfer Entropy (CTE) between each pair of variables.
*
*
* @param dY first difference of prices - each rows is one variable, each column corresponds to one timepoint.
* @param TE_results matrix [dY.rows() x dY.rows()] that stores the TE estimates for each pair,
 * rows correspond to sources and columns correspond to destinations.
* @param TE_pvalues matrix [dY.rows() x dY.rows()] that stores the TE estimates' pvalues directly
 * corresponding to TE_results matrix.
* @param CTE_results matrix [dY.rows() x dY.rows()] that stores the CTE estimates for each pair,
 * rows correspond to sources and columns correspond to destinations.
* @param CTE_pvalues matrix [dY.rows() x dY.rows()] that stores the CTE estimates' pvalues directly
 * corresponding to CTE_results matrix.
* @param CTE_analysis bool whether you want to include CTE analysis.
* @param AUTOEMB  bool whether you want to use Ragwitz criterion for autoembedding
 * if true the parameters X_hist, Y_hist, X_tau, Y_tau will stand as the maximum values that the autoembedding can choose.
 * if false the provided parameters will be used for all estimates.
* @param AUTODEL  bool whether you want to automatically choose delay from source to destination that maximizes TE estimate.
 * if true the parameter Delay will stand as the maximum values that the auto-delay can choose.
 * if false the value provided in Delay will be used for all estimates.
* @param X_hist  source history, minimum 1.
* @param Y_hist  destination history, minimum 1.
* @param X_tau  source history embedding delay, minimum 1.
* @param Y_tau  destination history embedding delay, minimum 1.
* @param Delay  delay from source to destination, minimum 1.
* @param permutation  number of permutations to perform for statstical significance testing
 * if set to 0 no pertmutation testing will be performed.
* @param min_pval  minimum statistcal siginificance needed, above this value permutation testing will break
 * and the corresponding p_value will be set -1. If you want to perform all permutation testings until the end set min_pvalue to 1.
* @param STD bool whether you want to standardize the data.
* @param NOISE the size of the gaussian noise added to data [noise_scale=pow(10, -NOISE)];
* @param k number of nearest neighbours to use in all KDTree estimations;
* @param leaves number of leaves to use in all KDTree estimations;
* */
returnVector_INA Information_Network_Analysis(Eigen::MatrixXd &dY,
                                              const bool CTE_analysis,
                                              const bool AUTOEMB, const int MAX_HIST, const int MAX_TAU,
                                              const bool AUTODEL_TE, const bool AUTODEL_CTE, const int MAX_DELAY,
                                              const int X_hist, const int Y_hist,
                                              const int X_tau, const int Y_tau,
                                              const int X_delay,
                                              const int permutation, const double min_pval_TE,
                                              const double min_pval_CTE,
                                              const bool STD, const int NOISE, const int k, const int leaves,
                                              const bool DEBUG) {

    // basic information
    const int nbOfDealers = dY.rows();
    const int nbOfDatapoints = dY.cols();

    // Standardize all variables
    if (STD) {
        standardize_timeseries(dY);
    }

    // Add noise to all variables
    if (NOISE != 0) {
        add_noise_timeseries(dY, -NOISE);
    }

    // initializing matrices that will hold the results and sets of optimal "source" and "destination" parameters for each variable
    MatrixXd TE_results = MatrixXd::Constant(nbOfDealers, nbOfDealers,NAN); // if the data is not filled e.g. when source=destination will stay as NAN
    MatrixXd TE_pvalues = MatrixXd::Constant(nbOfDealers, nbOfDealers, NAN);
    MatrixXd CTE_results = MatrixXd::Constant(nbOfDealers, nbOfDealers, NAN);
    MatrixXd CTE_pvalues = MatrixXd::Constant(nbOfDealers, nbOfDealers, NAN);
    MatrixXd X_hist_mat = MatrixXd::Zero(nbOfDealers, nbOfDealers);
    MatrixXd X_tau_mat = MatrixXd::Zero(nbOfDealers, nbOfDealers);
    MatrixXd Y_hist_mat = MatrixXd::Zero(nbOfDealers, nbOfDealers);
    MatrixXd Y_tau_mat = MatrixXd::Zero(nbOfDealers, nbOfDealers);
    MatrixXd DELAY_mat = MatrixXd::Zero(nbOfDealers, nbOfDealers);

    ////////////////////////// TRANSFER ENTROPY //////////////////////////
    // progress counter
    double counter = 1;
    // looping over all possible sources and their respective destinations
    for (int source = 0; source < nbOfDealers; source++) {
        for (int destination = 0; destination < nbOfDealers; destination++, counter++) {

            std::cout << "TE Progress = " << (counter) / (nbOfDealers * nbOfDealers) * 100 << "%" << std::endl;

            // source the same as destination not applicable
            if (source == destination) { continue; }

            // collect source and destination timeseries
            MatrixXd X = dY.block(source, 0, 1, nbOfDatapoints);      // source time series
            MatrixXd Y = dY.block(destination, 0, 1, nbOfDatapoints); // destination time series

            // NetworkX Notes:
            // For directed graphs, explicitly mention create_using=nx.DiGraph,
            // and entry i,j of df corresponds to an edge from i to j.
            // hence source must be row and destination must be a column

            // Transfer Entropy with autoembedding and autdelay
            auto [TE_results_temp, TE_pvalues_temp, X_hist_mat_temp,
                    X_tau_mat_temp, Y_hist_mat_temp, Y_tau_mat_temp,
                    DELAY_mat_temp] = TE_X_to_Y_INA(X, Y,
                                                    X_hist, Y_hist,
                                                    X_tau, Y_tau, X_delay,
                                                    AUTOEMB, MAX_HIST, MAX_TAU,
                                                    AUTODEL_TE, MAX_DELAY,
                                                    permutation, min_pval_TE,
                                                    0, 0, k, leaves);

            // collecting Transfer Entropy estimates and corresponding pvalues
            TE_results(source, destination) = TE_results_temp;
            TE_pvalues(source, destination) = TE_pvalues_temp;

            // collecting all the parameters determine in the autoembedding to be used
            // in the Conditional Transfer Entropy estimation.
            X_hist_mat(source, destination) = X_hist_mat_temp;
            X_tau_mat(source, destination) = X_tau_mat_temp;
            Y_hist_mat(source, destination) = Y_hist_mat_temp;
            Y_tau_mat(source, destination) = Y_tau_mat_temp;
            DELAY_mat(source, destination) = DELAY_mat_temp;

        }

    }

    ////////////////////////// CONDITIONAL TRANSFER ENTROPY //////////////////////////
    if (CTE_analysis) {

        MatrixXd X_CTE;
        MatrixXd Y_CTE;
        std::vector<int> conditional_variables_indexes;
        MatrixXd candidatesToCheck_pvalues;
        counter = 1;
        for (int source = 0; source < nbOfDealers; source++) {
            for (int destination = 0; destination < nbOfDealers; destination++, counter++) {

                std::cout << "CTE Progress = " << (counter) / (nbOfDealers * nbOfDealers) * 100 << "%" << std::endl;

                // source the same as destination skip
                if (source == destination) { continue; }

                // do not perform conditional TE analysis for a TE that was not significant based on min_pval_TE
                if (TE_pvalues(source, destination) == -1 ||
                    TE_pvalues(source, destination) > min_pval_TE) { continue; }

                // collect source and destination timeseries
                X_CTE = dY.block(source, 0, 1, nbOfDatapoints);      // source time series
                Y_CTE = dY.block(destination, 0, 1, nbOfDatapoints); // destination time series

                // determine which conditional candidates need to be included in CTE etimation
                conditional_variables_indexes.clear(); // clear the vector from indexes grabbed in the previous loop

                candidatesToCheck_pvalues = TE_pvalues.block(0, destination, nbOfDealers, 1);
                for (int candidate = 0; candidate < nbOfDealers; candidate++) {
                    // if the candidate is the destination or the source skip to the next
                    if (candidate == destination) { continue; }
                    if (candidate == source) { continue; }

                    // check if candidate's p_value is not -1 (in case early permutation break is employed)
                    // and also p_value is not > min_pval required (in case no early permutation break is employed)
                    // candidate must have significant TE
                    if (candidatesToCheck_pvalues(candidate, 0) != -1 &&
                        candidatesToCheck_pvalues(candidate, 0) <= min_pval_TE) {
                        conditional_variables_indexes.push_back(candidate);
                    }
                }

                // if there are any conditional candidates that are significant
                if (conditional_variables_indexes.size() > 0) {


                    // initialize Z_matrix with conditionals
                    int nbOfConditionals = conditional_variables_indexes.size(); // number of conditional variables that were significant
                    MatrixXd Z_CTE(nbOfConditionals, nbOfDatapoints);
                    MatrixXd Z_hist_mat(nbOfConditionals, 1);
                    MatrixXd Z_tau_mat(nbOfConditionals, 1);
                    MatrixXd DELAY_Z_mat(nbOfConditionals, 1);


                    // we treat conditionals as other sources hence we take the corresponding hist tau and  delay from X (source) matrices
                    // specific to the index (number of the variable in the dY) of the conditional variable
                    for (int i = 0; i < nbOfConditionals; i++) {
                        Z_CTE.block(i, 0, 1, nbOfDatapoints) = dY.block(conditional_variables_indexes[i], 0, 1,
                                                                        nbOfDatapoints);
                        Z_hist_mat(i, 0) = X_hist_mat(conditional_variables_indexes[i], destination);
                        Z_tau_mat(i, 0) = X_tau_mat(conditional_variables_indexes[i], destination);
                        DELAY_Z_mat(i, 0) = DELAY_mat(conditional_variables_indexes[i], destination);
                    }


                    auto [CTE_results_temp, CTE_pvalues_temp] = CTE_X_to_Y_INA(X_CTE, Y_CTE, Z_CTE,
                                                                               X_hist_mat(source, destination),
                                                                               Y_hist_mat(source, destination),
                                                                               Z_hist_mat,
                                                                               X_tau_mat(source, destination),
                                                                               Y_tau_mat(source, destination),
                                                                               Z_tau_mat,
                                                                               AUTODEL_CTE, MAX_DELAY,
                                                                               DELAY_mat(source, destination),
                                                                               DELAY_Z_mat,
                                                                               permutation, min_pval_CTE,
                                                                               0, 0, k, leaves);

                    // save results
                    CTE_results(source, destination) = CTE_results_temp;
                    CTE_pvalues(source, destination) = CTE_pvalues_temp;
                } else {
                    // if no conditionals then CTE == TE, since nothing else to condition it on:
                    CTE_results(source, destination) = TE_results(source, destination);
                    CTE_pvalues(source, destination) = TE_pvalues(source, destination);

                }

            }
        }

    }

    return returnVector_INA{TE_results, TE_pvalues, CTE_results, CTE_pvalues};
}




//////////// VALIDATION ////////////////

struct TestParams {
    int Y_hist, X_hist;
    bool AUTODEL_TE, AUTODEL_CTE, AUTOEMB;
    std::string test_name;
};

bool compare_matrices(const MatrixXd& m1, const MatrixXd& m2, double tol = 1e-6) {
    if (m1.rows() != m2.rows() || m1.cols() != m2.cols()) {
        return false;
    }

    for (int i = 0; i < m1.rows(); ++i) {
        for (int j = 0; j < m1.cols(); ++j) {
            auto val1 = m1(i, j);
            auto val2 = m2(i, j);
            if (std::isnan(val1) && std::isnan(val2)) {
                continue;
            } else if (std::abs(val1 - val2) > tol) {
                return false;
            }
        }
    }
    return true;
}


void runTest(const TestParams& test) {
    // Use parameters from test struct
    int Y_hist = test.Y_hist;
    int Y_tau = 1;
    int X_hist = test.X_hist;
    int X_tau = 1;
    int X_delay = 1;
    bool STD = 1;
    int NOISE = 12;
    bool AUTOEMB = test.AUTOEMB;
    int MAX_HIST = 5;
    int MAX_TAU = 5;
    bool AUTODEL_TE = test.AUTODEL_TE;
    bool AUTODEL_CTE = test.AUTODEL_CTE;
    int MAX_DELAY = 5;
    int permutation = 1;
    double min_pval_TE = 1;
    double min_pval_CTE = 1;
    int k = 4;
    int CTE_analysis = 1;
    bool DEBUG = 0;

    std::ifstream file("Test_Data/TestDataValidation.txt");
    MatrixXd INA_test_data = MatrixXd::Zero(5999, 4);

    for (int i = 0; i < 5999; ++i) {
        for (int j = 0; j < 4; ++j) {
            file >> INA_test_data(i, j);
        }
    }
    file.close();

    INA_test_data.transposeInPlace();

    std::cout << "\nRunnning Test: " << test.test_name << "\n" << std::endl;

    auto [TE_results, TE_pvalues, CTE_results, CTE_pvalues] = Information_Network_Analysis(
            INA_test_data, CTE_analysis, AUTOEMB, MAX_HIST, MAX_TAU,
            AUTODEL_TE, AUTODEL_CTE, MAX_DELAY, X_hist, Y_hist, X_tau, Y_tau,
            X_delay, permutation, min_pval_TE, min_pval_CTE, STD, NOISE, k, 100, DEBUG);

    std::string benchmarks_path = "ValidationBenchmarks/";
    // Load expected results from benchmark files
    std::ifstream te_file(benchmarks_path + test.test_name + "_TE_results.txt");
    std::ifstream cte_file(benchmarks_path + test.test_name + "_CTE_results.txt");

    int rows, cols;
    te_file >> rows >> cols;
    MatrixXd expected_TE_results(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            te_file >> expected_TE_results(i, j);
        }
    }
    te_file.close();

    cte_file >> rows >> cols;
    MatrixXd expected_CTE_results(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            cte_file >> expected_CTE_results(i, j);
        }
    }
    cte_file.close();

    std::cout << "\nTE Validation Results: ";
    if (compare_matrices(TE_results, expected_TE_results)) {
        std::cout << " Passed !" << std::endl;
    } else {
        std::cout << "NOT Passed! " << std::endl;
        throw std::runtime_error("TE Validation Failed for " + test.test_name);
    }

    std::cout << "\nCTE Validation Results: ";
    if (compare_matrices(CTE_results, expected_CTE_results)) {
        std::cout << "Passed !" << std::endl;
    } else {
        std::cout << "NOT Passed!" << std::endl;
        throw std::runtime_error("CTE Validation Failed for " + test.test_name);
    }
}


void RunValidationTests() {

    std::vector<TestParams> tests = {
            {1, 1, 1, 1, 0, "BENCHMARK_TEST_1"},
            {2, 1, 1, 1, 0, "BENCHMARK_TEST_2"},
            {1, 2, 1, 1, 0, "BENCHMARK_TEST_3"},
            {2, 2, 1, 1, 0, "BENCHMARK_TEST_4"},
            {3, 3, 1, 1, 1, "BENCHMARK_TEST_5"},
            {1, 1, 0, 0, 0, "BENCHMARK_TEST_6"},
    };

    for (const auto& test : tests) {
        runTest(test);
    }

    std::cout << "Validation successful!" << std::endl;
}




