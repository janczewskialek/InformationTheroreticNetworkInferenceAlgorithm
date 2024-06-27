//
// Created by Aleksander on 1/15/22.
//

#ifndef TRANSFERENTROPY_TE_DEV_UTILITIES_H
#define TRANSFERENTROPY_TE_DEV_UTILITIES_H





void basic_simulation(double alpha, double beta, double gamma, int time, Eigen::MatrixXd& data);
std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> extract_XY(Eigen::MatrixXd& data);
std::vector<double> linspace(double start_in, double end_in, int num_in);
double TE_XtoY_analytical(Eigen::MatrixXd& X_raw, Eigen::MatrixXd& Y_raw);
double TE_XtoY_analytical_history(Eigen::MatrixXd& X_raw, Eigen::MatrixXd& Y_raw, int X_hist=1, int Y_hist=1);

void analytical_check_over_gamma_grid(double alpha, double beta, int time,  int gamma_granularity, Eigen::MatrixXd& data);



#endif //TRANSFERENTROPY_TE_DEV_UTILITIES_H
