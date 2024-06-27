//
// Created by Aleksander on 2/4/22.
//

#ifndef TRANSFERENTROPY_INA_H
#define TRANSFERENTROPY_INA_H

#endif //TRANSFERENTROPY_INA_H

#include "eigen-3.4.0/Eigen/Core"
#include "eigen-3.4.0/Eigen/Eigen"
#include "eigen-3.4.0/Eigen/Dense"
#include "eigen-3.4.0/unsupported/Eigen/SpecialFunctions"

#include "ckdtree/src/ckdtree_decl.h"

struct returnVector_TE {
    double TE_estimate, p_value;
    int X_hist, X_tau, Y_hist, Y_tau, DELAY;
};

struct returnVector_CTE {
    double CTE_estimate, p_value;
};

struct returnVector_INA {
    Eigen::MatrixXd TE_results, TE_pvalues, CTE_results, CTE_pvalues;
};


void _post_init_traverse(ckdtree *self, ckdtreenode *node);

Eigen::MatrixXd CKD_find_KNN4_radius_per_point(Eigen::MatrixXd &Stack, int k = 4, int leaves = 100);

Eigen::MatrixXd CKD_count_NNs_with_radius_search(Eigen::MatrixXd& Y1Y_stack, Eigen::MatrixXd& radiuses,  int leaves=100);

void standardize_timeseries(Eigen::MatrixXd &mat);

void add_noise_timeseries(Eigen::MatrixXd &mat, int power);


Eigen::MatrixXd CKD_find_KNN4_indexes(Eigen::MatrixXd &Stack, int k = 4, int leaves = 100);

std::tuple<int, int> Ragwitz_auto_embedding(Eigen::MatrixXd X, Eigen::MatrixXd Y, int k, int k_max, int tau_max);

double TE_X_to_Y_inner(Eigen::MatrixXd &X, Eigen::MatrixXd &Y,
                       int X_hist = 1, int Y_hist = 1,
                       int X_tau = 1, int Y_tau = 1, int DELAY = 1,
                       int k = 4, int leaves = 100);

double TE_X_to_Y_permute(Eigen::MatrixXd &X, Eigen::MatrixXd &Y, double TE_actual,
                         int X_hist = 1, int Y_hist = 1,
                         int X_tau = 1, int Y_tau = 1,
                         int DELAY = 1,
                         int permutation = 100, double min_pval = 0.20,
                         int k = 4, int leaves = 100);

returnVector_TE TE_X_to_Y_INA(Eigen::MatrixXd X,
                              Eigen::MatrixXd Y,
                              int X_hist = 1, int Y_hist = 1,
                              int X_tau = 1, int Y_tau = 1, int DELAY = 1,
                              bool AUTOEMB = 0, int HIST_MAX = 1, int TAU_MAX = 1,
                              bool AUTODEL = 0, int MAX_DELAY = 1,
                              int permutation = 100, double min_pval = 0.20,
                              bool STD = 0, int NOISE = 0,
                              int k = 4, int leaves = 100, bool DEBUG = 0);

double CTE_X_to_Y_inner(Eigen::MatrixXd &X, Eigen::MatrixXd &Y, Eigen::MatrixXd &Z,
                        int X_hist = 1, int Y_hist = 1, Eigen::MatrixXd Z_hist = Eigen::MatrixXd::Constant(1, 1, 1),
                        int X_tau = 1, int Y_tau = 1, Eigen::MatrixXd Z_tau = Eigen::MatrixXd::Constant(1, 1, 1),
                        int DELAY_X = 1, Eigen::MatrixXd DELAY_Z = Eigen::MatrixXd::Constant(1, 1, 1),
                        int k = 4, int leaves = 100);


double CTE_X_to_Y_permute(Eigen::MatrixXd &X, Eigen::MatrixXd &Y, Eigen::MatrixXd &Z, double CTE_actual,
                          int X_hist = 1, int Y_hist = 1, Eigen::MatrixXd Z_hist = Eigen::MatrixXd::Constant(1, 1, 1),
                          int X_tau = 1, int Y_tau = 1, Eigen::MatrixXd Z_tau = Eigen::MatrixXd::Constant(1, 1, 1),
                          int DELAY_X = 1, Eigen::MatrixXd DELAY_Z = Eigen::MatrixXd::Constant(1, 1, 1),
                          int permutation = 100, double min_pval = 0.20,
                          int k = 4, int leaves = 100);


returnVector_CTE CTE_X_to_Y_INA(Eigen::MatrixXd X, Eigen::MatrixXd Y, Eigen::MatrixXd Z,
                                const int X_hist = 1, const int Y_hist = 1,
                                Eigen::MatrixXd Z_hist = Eigen::MatrixXd::Constant(1, 1, 1),
                                const int X_tau = 1, const int Y_tau = 1,
                                Eigen::MatrixXd Z_tau = Eigen::MatrixXd::Constant(1, 1, 1),
                                bool AUTODEL = 1, int MAX_DELAY = 1, int DELAY_X = 1,
                                Eigen::MatrixXd DELAY_Z = Eigen::MatrixXd::Constant(1, 1, 1),
                                const int permutation = 100, const double min_pval = 0.20,
                                const bool STD = 0, const int NOISE = 0, const int k = 4, const int leaves = 100);


returnVector_INA Information_Network_Analysis(Eigen::MatrixXd &dY,
                                              bool CTE_analysis = 1,
                                              bool AUTOEMB = 0, int MAX_HIST = 1, int MAX_TAU = 1,
                                              bool AUTODEL_TE = 0, bool AUTODEL_CTE = 0, int MAX_DELAY = 1,
                                              int X_hist = 1, int Y_hist = 1,
                                              int X_tau = 1, int Y_tau = 1,
                                              int X_delay = 1,
                                              int permutation = 100, double min_pval_TE = 0.20,
                                              double min_pval_CTE = 0.20,
                                              bool STD = 0, int NOISE = 0, int k = 4, int leaves = 100, bool DEBUG = 0);



void RunValidationTests();