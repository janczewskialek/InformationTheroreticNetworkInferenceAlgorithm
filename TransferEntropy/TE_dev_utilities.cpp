//
// Created by Aleksander on 1/15/22.
//
#include <time.h>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include "kdtree-cpp/kdtree.hpp"
#include <random>
#include <iostream>
#include "knncpp.h"
#include "circ_shift.h"
#include "TE_dev_utilities.h"

std::vector<double> linspace(double start_in, double end_in, int num_in)
{
    std::vector<double> linspaced;

    double start = static_cast<double>(start_in);
    double end = static_cast<double>(end_in);
    double num = static_cast<double>(num_in);

    if (num == 0) { return linspaced; }
    if (num == 1)
    {
        linspaced.push_back(start);
        return linspaced;
    }

    double delta = (end - start) / (num - 1);

    for(int i=0; i < num-1; ++i)
    {
        linspaced.push_back(start + delta * i);
    }
    linspaced.push_back(end); // I want to ensure that start and end
    // are exactly the same as the input
    return linspaced;
}


void basic_simulation(double alpha, double beta, double gamma, int time, Eigen::MatrixXd& data){
    using namespace Eigen;

    // random device class instance, source of 'true' randomness for initializing random seed
    std::random_device rd;

    // Mersenne twister PRNG, initialized with seed from previous random device instance
    std::mt19937 gen(rd());
    std::normal_distribution<float> normal_random1(0, 1);
    std::normal_distribution<float> normal_random2(0, 1);

    for (int i=0; i < time-1; i++){
        double epsilon = normal_random1(gen);
        double omega = normal_random2(gen);

        data(i+1, 0) = alpha*data(i, 0) + epsilon;
        data(i+1, 1) = gamma*data(i, 0) + beta*data(i, 1) + omega;
    }
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> extract_XY(Eigen::MatrixXd& data){
    using namespace Eigen;

    MatrixXd data_T = MatrixXd::Zero(data.cols(),data.rows());
    data_T = data.transpose();

    MatrixXd X_raw = data_T.block(0,0,1,data_T.cols());
    MatrixXd Y_raw = data_T.block(1,0,1,data_T.cols());

    return std::make_tuple(X_raw, Y_raw);
}



double TE_XtoY_analytical(Eigen::MatrixXd& X_raw, Eigen::MatrixXd& Y_raw){
    using namespace Eigen;

    double TE_XtoY = 0;

    MatrixXd X_raw_shift = circShift(X_raw, 0, -1);
    MatrixXd Y_raw_shift = circShift(Y_raw, 0, -1);


    // getting rid of last values because of the shift applied
    MatrixXd X = X_raw.block(0,0,1,X_raw.cols());
    MatrixXd Y = Y_raw.block(0,0,1,Y_raw.cols());
    MatrixXd X_shift = X_raw_shift.block(0,0,1,X_raw_shift.cols());
    MatrixXd Y_shift = Y_raw_shift.block(0,0,1,Y_raw_shift.cols());


    // ASSERTIONS
    assert(X.rows() == Y.rows());
    assert(X.cols() == Y.cols());
    assert(Y_shift.rows() == Y.rows());
    assert(Y_shift.cols() == Y.cols());

    /////// TE X --> Y
    // COVARIANCE Y1 x Y
    MatrixXd Mat_Y1Y = MatrixXd::Zero(2,Y.cols());
    Mat_Y1Y.block(0,0,1,Y.cols()) = Y_shift;
    Mat_Y1Y.block(1,0,1,Y.cols()) = Y;

    MatrixXd centered_Y1Y = Mat_Y1Y.transpose().rowwise() - Mat_Y1Y.transpose().colwise().mean();
    MatrixXd Cov_Y1Y = (centered_Y1Y.adjoint() * centered_Y1Y) / double(Mat_Y1Y.transpose().rows() - 1);
    double det_Cov_Y1Y = Cov_Y1Y.determinant();

    // COVARIANCE Y x X
    MatrixXd Mat_YX = MatrixXd::Zero(2,Y.cols());
    Mat_YX.block(0,0,1,Y.cols()) = Y;
    Mat_YX.block(1,0,1,Y.cols()) = X;

    MatrixXd centered_YX = Mat_YX.transpose().rowwise() - Mat_YX.transpose().colwise().mean();
    MatrixXd Cov_YX = (centered_YX.adjoint() * centered_YX) / double(Mat_YX.transpose().rows() - 1);
    double det_Cov_YX = Cov_YX.determinant();

    // COVARIANCE Y
    MatrixXd centered_Y = Y.transpose().rowwise() - Y.transpose().colwise().mean();
    MatrixXd Cov_Y = (centered_Y.adjoint() * centered_Y) / double(Y.transpose().rows() - 1);
    double det_Cov_Y = Cov_Y(0,0);

    // COVARIANCE Y1 x Y x X
    MatrixXd Mat_Y1YX = MatrixXd::Zero(3,Y.cols());
    Mat_Y1YX.block(0,0,1,Y.cols()) = Y_shift;
    Mat_Y1YX.block(1,0,1,Y.cols()) = Y;
    Mat_Y1YX.block(2,0,1,Y.cols()) = X;

    MatrixXd centered_Y1YX = Mat_Y1YX.transpose().rowwise() - Mat_Y1YX.transpose().colwise().mean();
    MatrixXd Cov_Y1YX = (centered_Y1YX.adjoint() * centered_Y1YX) / double(Mat_Y1YX.transpose().rows() - 1);
    double det_Cov_Y1YX = Cov_Y1YX.determinant();

    TE_XtoY = 0.5 * log((det_Cov_Y1Y*det_Cov_YX)/(det_Cov_Y*det_Cov_Y1YX));
    return TE_XtoY;
}

double TE_XtoY_analytical_history(Eigen::MatrixXd& X, Eigen::MatrixXd& Y, int X_hist, int Y_hist){
    using namespace Eigen;

    // asserting that input data is correct
    assert(X.rows() == Y.rows());
    assert(X.cols() == Y.cols());

    // dimension of data
    const int data_cols = X.cols();

    MatrixXd X_shift = circShift(X, 0, -1);
    MatrixXd Y_shift = circShift(Y, 0, -1);
    assert(Y_shift.rows() == Y.rows());
    assert(Y_shift.cols() == Y.cols());

    // stacking history of X variable
    MatrixXd X_HIST_STACK = MatrixXd::Zero(X_hist, data_cols);
    for (int i = 0; i < X_hist; i++){
        X_HIST_STACK.block(i, 0, 1, data_cols)= circShift(X, 0, i);
    }

    // stacking history of Y variable
    MatrixXd Y_HIST_STACK = MatrixXd::Zero(Y_hist, data_cols);
    for (int i = 0; i < Y_hist; i++){
        Y_HIST_STACK.block(i, 0, 1, data_cols)= circShift(Y, 0, i);
    }

    // COVARIANCE Y1 x Y
    MatrixXd Mat_Y1Y = MatrixXd::Zero(1+Y_hist,data_cols);
    Mat_Y1Y << Y_shift, Y_HIST_STACK;
    MatrixXd centered_Y1Y = Mat_Y1Y.transpose().rowwise() - Mat_Y1Y.transpose().colwise().mean();
    MatrixXd Cov_Y1Y = (centered_Y1Y.adjoint() * centered_Y1Y) / double(Mat_Y1Y.transpose().rows() - 1);
    double det_Cov_Y1Y = Cov_Y1Y.determinant();

    // COVARIANCE Y x X
    MatrixXd Mat_YX = MatrixXd::Zero(X_hist+Y_hist,Y.cols());
    Mat_YX << Y_HIST_STACK, X_HIST_STACK;
    MatrixXd centered_YX = Mat_YX.transpose().rowwise() - Mat_YX.transpose().colwise().mean();
    MatrixXd Cov_YX = (centered_YX.adjoint() * centered_YX) / double(Mat_YX.transpose().rows() - 1);
    double det_Cov_YX = Cov_YX.determinant();

    // COVARIANCE Y
    MatrixXd centered_Y = Y_HIST_STACK.transpose().rowwise() - Y_HIST_STACK.transpose().colwise().mean();
    MatrixXd Cov_Y = (centered_Y.adjoint() * centered_Y) / double(Y_HIST_STACK.transpose().rows() - 1);
    double det_Cov_Y = Cov_Y(0,0);

    // COVARIANCE Y1 x Y x X
    MatrixXd Mat_Y1YX = MatrixXd::Zero(1+Y_hist+X_hist,Y.cols());
    Mat_Y1YX << Y_shift, Y_HIST_STACK, X_HIST_STACK;
    MatrixXd centered_Y1YX = Mat_Y1YX.transpose().rowwise() - Mat_Y1YX.transpose().colwise().mean();
    MatrixXd Cov_Y1YX = (centered_Y1YX.adjoint() * centered_Y1YX) / double(Mat_Y1YX.transpose().rows() - 1);
    double det_Cov_Y1YX = Cov_Y1YX.determinant();

    double TE_XtoY = 0.5 * log((det_Cov_Y1Y*det_Cov_YX)/(det_Cov_Y*det_Cov_Y1YX));

    return TE_XtoY;
}

void analytical_check_over_gamma_grid(double alpha, double beta, int time,  int gamma_granularity, Eigen::MatrixXd& data){
    using namespace Eigen;
    std::vector<double> gamma_space = linspace(-0.4,0.4, gamma_granularity);

    double gamma = 0;

    for (int i=0; i<gamma_space.size(); i++){
        gamma = gamma_space[i];
        basic_simulation(alpha, beta, gamma, time, data);

        std::tuple<MatrixXd, MatrixXd> XY = extract_XY(data);
        MatrixXd X_raw = std::get<0>(XY);
        MatrixXd Y_raw = std::get<1>(XY);

        double TE_XtoY = TE_XtoY_analytical( X_raw, Y_raw);
        double TE_YtoX = TE_XtoY_analytical( Y_raw, X_raw);
        std::cout << "GAMMA: " << gamma << " TE_XtoY: " << TE_XtoY << " TE_YtoX: " << TE_YtoX << std::endl;
    }
}


