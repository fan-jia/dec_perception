#include "gp.h"
#include "gp_utils.h"

#include <Eigen/Dense>

using namespace libgp;

Eigen::MatrixXd extract_rows(Eigen::MatrixXd X, Eigen::VectorXi ind_train){
    Eigen::MatrixXd X_train(ind_train.size(),X.cols());

    for(int i = 0; i < ind_train.size(); i++){
        X_train.row(i) = X.row(ind_train(i));
    }
    return X_train;
}

Eigen::VectorXi sort_unique(Eigen::VectorXi a){
    sort(a.data(), a.data() + a.size(), [](int lhs, int rhs){return rhs > lhs;});

    vector<int> vec; vec.clear();
    for(int i=0;i<a.size();i++) vec.push_back(a(i));
    vector<int>::iterator it;
    it = unique(vec.begin(),vec.end());
    vec.erase(it,vec.end());
    Eigen::VectorXi b(vec.size());
    for(int i = 0; i < vec.size(); i++){
        b(i) = vec[i];
    }

    return b;
}

void gp_compute(Eigen::MatrixXd X, Eigen::VectorXd Y, Eigen::MatrixXd Xtest, Eigen::VectorXd &mu, Eigen::VectorXd &s2){
    double y;
    GaussianProcess gp(2, "CovSum ( CovSEiso, CovNoise)");
    Eigen::VectorXd params(3);
    params << 0.5, 0.5, -2.0;
    gp.covf().set_loghyper(params);

    for(int i = 0; i < Y.size(); i++){
        double x[] = {X(i,0), X(i,1)};
        y = Y(i);
        gp.add_pattern(x, y);
    }

    for(int i = 0; i < Xtest.rows(); i++){
        double x[] = {Xtest(i,0), Xtest(i,1)};
        mu(i) = gp.f(x);
        s2(i) = gp.var(x);
    }
}

void gpml_rms(Eigen::VectorXi ind_train, Eigen::MatrixXd Xs, Eigen::MatrixXd Fs, Eigen::MatrixXd Xtest_new, Eigen::VectorXd &mu, Eigen::VectorXd &s2){
/*
 * Gaussian Process for Machine Learning
 * Input:
 *       Xs   -   N x d  coordinates for all grids
 *       Fs   -   N x 1  Realization of all grids (such as temperature readings)
         ind_train - N_train x 1  index of training data
         Xtest_new  - N_test x d  coordinates for all testing grids
         Ftest_new  - N_test x 1  Realization of all testing grids (ground truth)

   output:
        mu - N_test x 1   Predicted value on all testing grids
        s2 - N_test x 1   Variance of prediction on all testing grids
        rms - 1 x 1     RMS error from ground truth
 */
    ind_train = sort_unique(ind_train);
    Eigen::MatrixXd Xs_train;
    Eigen::MatrixXd Fs_train;

    Xs_train = extract_rows(Xs, ind_train);
    Fs_train = extract_rows(Fs, ind_train);

    Eigen::MatrixXd Fs_train_mtz;
    Fs_train_mtz = Fs_train.array() - Fs_train.mean();// mean value equals to 0
    gp_compute(Xs_train, Fs_train_mtz, Xtest_new, mu, s2);
    mu = mu.array() + Fs_train.mean();
}
