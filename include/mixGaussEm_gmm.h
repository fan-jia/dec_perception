#ifndef _mixgaussem_gmm_
#define _mixgaussem_gmm_


#include <iostream>
#include <math.h>
#include <numeric>
#include <vector>
#include <limits>
#include <algorithm>
#include <Eigen/Dense>
#include <time.h>
#include <stdlib.h>
using namespace std;

struct Model{
    Eigen::MatrixXd mu;
    Eigen::MatrixXd Sigma;
    Eigen::MatrixXd w;
};

struct GTData{
/*
 * Fss represents temperature (945 x 1), Xss represents coordination (945 x 2)
 */
    Eigen::MatrixXd Xss;
    Eigen::MatrixXd Fss;
    int num_gau;
    int num_bot;
};

struct PACKETS{
    Eigen::MatrixXd alpha_K;
    Eigen::MatrixXd belta_K;
    Eigen::MatrixXd gamma_K;
};

struct BOTS{
    Eigen::MatrixXd alpha_K;
    Eigen::MatrixXd belta_K;
    Eigen::MatrixXd gamma_K;
    Eigen::MatrixXd self_alpha;
    Eigen::MatrixXd self_belta;
    Eigen::MatrixXd self_gamma;
    Eigen::MatrixXd dot_alpha_K;
    Eigen::MatrixXd dot_belta_K;
    Eigen::MatrixXd dot_gamma_K;
    Eigen::MatrixXd Xs;
    Eigen::VectorXi Nm_ind;
    Eigen::MatrixXd Fs;
    Eigen::MatrixXd mu_K;
    Eigen::MatrixXd Sigma_K;
    vector<PACKETS> packets;
    vector<int> neighbor;
};

Eigen::MatrixXd pdist2(Eigen::MatrixXd Xtest, Eigen::MatrixXd Xs){
/* Compute the euclidean distance between Xtest and Xs
 * Input:
 *      Xtest: 10 x 2, Xs: 945 x 2
 * Output:
 *      Distance: 10 x 945
 */
    Eigen::MatrixXd Distance(Xtest.rows(),Xs.rows());
    for(int i = 0; i < Xtest.rows(); i++){
        for(int j = 0; j < Xs.rows(); j++){
            Distance(i,j) = (Xtest.row(i) - Xs.row(j)).norm();
        }
    }
    return Distance;
}

Eigen::VectorXi get_idx(Eigen::MatrixXd Xtest, Eigen::MatrixXd Xs){
/* Input:
 *      Xtest: 10 x 2, Xs: 945 X 2
 * Output:
 *      idx: 10 x 1
 */
    Eigen::MatrixXd Distance = pdist2(Xtest,Xs);
    Eigen::VectorXi idx(Xtest.rows());
    Eigen::MatrixXd::Index min_index;
    for(int i = 0; i < idx.size(); i++){
        Distance.row(i).minCoeff(&min_index);
        idx(i) = min_index;
    }
    return idx;
}



Eigen::MatrixXd norm_prob(Eigen::MatrixXd X){
/*
 * X: a * b where b is the num_gau
 */
    Eigen::MatrixXd y = X.array().colwise() *  X.rowwise().sum().array().inverse().array();
    return y;
}

Eigen::MatrixXd initialization(Eigen::MatrixXd X, int init){
    int n = X.cols();
    //random init k
    if(sizeof(n) == 4){
        int k = init;
        Eigen::MatrixXd A = Eigen::MatrixXd::Random(1,n).array().abs() * k;
        Eigen::MatrixXd label = A.array().ceil();
        Eigen::MatrixXd R = Eigen::MatrixXd::Zero(n, k);

        for(int i = 0; i < label.rows(); i ++) {
            for (int j = 0; j < n; j++) {
                for(int m = 0; m < k; m++){
                    if(label(i,j) == m+1){
                        R(j,m) = 1;
                    }
                }
            }
        }
        return R;
    }//some cases missing
}

Eigen::MatrixXd loggausspdf(Eigen::MatrixXd X, Eigen::MatrixXd mu, Eigen::MatrixXd Sigma){
    int d = X.rows();
    int n = X.cols();
    for (int i = 0; i < d; i++) {
        for (int j = 0; j < n; j++) {
            X(i, j) = X(i, j) - mu(i, 0); //mu: d x 1
        }
    }
    Eigen::MatrixXd U = Sigma.llt().matrixL();
    Eigen::MatrixXd Q = U.inverse() * X;
    Eigen::MatrixXd q = Q.array() * Q.array();

    double c = d*log(2*M_PI) + 2*U.diagonal().array().log().sum();

    Eigen::MatrixXd y = -1 * (c + q.array()) / 2;
    return y.transpose();
}

void expectation(Eigen::MatrixXd X, Model model, double &llh_iter, Eigen::MatrixXd &R){
    Eigen::MatrixXd mu = model.mu;
    Eigen::MatrixXd Sigma = model.Sigma;
    Eigen::MatrixXd w = model.w;

    int d = X.rows();
    int n = X.cols();
    int k = mu.cols();
    R = Eigen::MatrixXd::Zero(n, k);

    for(int i = 0; i < k; i++){
        R.col(i) = loggausspdf(X, mu.col(i), Sigma.block<1,1>(0,i));
    }

    for(int i = 0; i < n; i++){
        for(int j = 0; j < k; j++){
            R(i,j) = R(i,j) + w.array().log()(0,j); //w: 1 x k
        }
    }
    Eigen::MatrixXd T = R.array().exp().rowwise().sum().log(); //T: n x 1
    llh_iter = T.sum() / n;
    for(int i = 0; i < n; i++){
        for(int j = 0; j < k; j++){
            R(i,j) = R(i,j) - T(i,0); //w: 1 x k
        }
    }
    R = R.array().exp();
}

Model maximization(Eigen::MatrixXd X, Eigen::MatrixXd R){
    int d = X.rows();
    int n = X.cols();
    int k = R.cols();
    Eigen::MatrixXd nk = R.colwise().sum();
    Eigen::MatrixXd w = nk/n;
    Eigen::MatrixXd temp = X * R;
    Eigen::MatrixXd mu(d,k);
    for(int i = 0 ; i < d; i ++){
        for(int j = 0; j < k; j++){
            mu(j) = temp(i,j) / nk(i,j);
        }
    }
    Eigen::MatrixXd Sigma = Eigen::MatrixXd::Zero(d,d*k);//use block to construct a 3D matrix, leftCols(d) as a first element
    Eigen::MatrixXd r = R.array().sqrt();
    for(int m = 0; m < k; m++){
        Eigen::MatrixXd Xo(d,n);
        for(int i = 0; i < d; i++){
            for(int j = 0; j < n; j++){
                Xo(i,j) = X(i,j) - mu(i,m);
            }
        }
        Xo = Xo.array() * r.col(m).transpose().array();
        Sigma.block<1,1>(0,m) = Xo * Xo.transpose() / nk(m) + Eigen::MatrixXd::Identity(d,d) * (1e-20);
    }
    Model model;
    model.mu = mu;
    model.Sigma = Sigma;
    model.w = w;
    return model;
}

vector<int> sort_indexes(Eigen::MatrixXd &v) {
    // initialize original index locations
    vector<int> idx(v.cols());
    iota(idx.begin(), idx.end(), 0);

    // sort indexes based on comparing values in v
    sort(idx.begin(), idx.end(), [&v](int i1, int i2) {return v(0,i1) < v(0,i2);});

    return idx;
}

void mixGaussEm_gmm(Eigen::MatrixXd X, int init, Eigen::VectorXi &label, Model &model){
/*Perform EM algorithm for fitting the Gaussian mixture model
 *Output: label: 1 x 945 cluster label
 * model: trained model structure
 * llh:loglikelihood
 */
    bool break_flag = false;
    double tol = 1e-6;
    int max_iter = 500;
    double inf = numeric_limits<double>::infinity();
    double llh_[max_iter];
    for(int i = 0; i < max_iter; i++){
        llh_[i] = -inf;
    }
    int count = 0;
    Eigen::MatrixXd R = initialization(X,init);
    Eigen::MatrixXd::Index max_index;
    for(int iter = 1; iter < max_iter; iter++){
        for(int i = 0; i < R.rows(); i++){
            R.row(i).maxCoeff(&max_index);
            label(i) = max_index;
        }
        model = maximization(X,R);
        expectation(X,model,llh_[iter],R); //update llh[iter] and R

        if(abs(llh_[iter] - llh_[iter-1]) < tol * abs(llh_[iter])){
            break_flag = true;
            break;
        }
        count = count + 1;
    }
    double llh[count];
    for(int i = 0; i < count; i++){
        llh[i] = llh_[i+1];
    }
    vector<int> I = sort_indexes(model.mu);
    vector<double> mu_temp, Sigma_temp, w_temp;
    for (auto i: I) {
        mu_temp.push_back(model.mu(0,i));
        Sigma_temp.push_back(model.Sigma(0,i));
        w_temp.push_back(model.w(0,i));
    }
    for(int i = 0; i < R.cols(); i++){
        model.mu(0,i) = mu_temp[i];
        model.Sigma(0,i) = Sigma_temp[i];
        model.w(0,i) = w_temp[i];
    }
    Eigen::VectorXi back_label = Eigen::VectorXi::Zero(label.size());
    for(int iij = 0; iij < R.cols(); iij++){
        for(int i = 0; i < label.size(); i++){
            if(label[i] == iij){
                back_label[i] = I[iij];
            }
        }
    }
    label = back_label;
    int k = 0;
    Eigen::MatrixXd R_temp(R.rows(),R.cols());
    for(auto i:I){
        R_temp.col(k) = R.col(i);
        k = k + 1;
    }
    R = R_temp;
}

void mixGaussPred_gmm(Eigen::MatrixXd X, Model model, Eigen::VectorXi &label, Eigen::MatrixXd &R){
/* Predict label and responsibility for Gaussian mixture model.
 * Input:
 *      X: d x n data matrix
 *      model: trained model structure outputted by the EM algorithm
 * Output:
 *      label: 1 x n cluster label
 *      R: k x n responsibility
 */
    Eigen::MatrixXd mu = model.mu;
    Eigen::MatrixXd Sigma = model.Sigma;
    Eigen::MatrixXd w = model.w;
    Eigen::MatrixXd::Index max_index;
    int n = X.cols();
    int k = mu.cols();

    for(int i = 0; i < k; i++){
        R.col(i) = loggausspdf(X, mu.col(i), Sigma.block<1,1>(0,i));
    }

    for(int i = 0; i < n; i++){
        for(int j = 0; j < k; j++){
            R(i,j) = R(i,j) + w.array().log()(0,j); //w: 1 x k
        }
    }
    Eigen::MatrixXd T = R.array().exp().rowwise().sum().log(); //T: n x 1
    for(int i = 0; i < n; i++){
        for(int j = 0; j < k; j++){
            R(i,j) = R(i,j) - T(i,0); //w: 1 x k
        }
    }
    R = R.array().exp();
    for(int i = 0; i < R.rows(); i++){
        R.row(i).maxCoeff(&max_index);
        label(i) = max_index;//Eigen::VectorXd label(X.cols());
    }
}

#endif
