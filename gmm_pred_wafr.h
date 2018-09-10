#ifndef _gmm_pred_wafr_
#define _gmm_pred_wafr_


#include "mixGaussEm_gmm.h"
using namespace std;

Eigen::MatrixXd set_NaN_tz(Eigen::MatrixXd X){
/*
 * set NaN elements to zero
 */
    for(int i = 0; i < X.rows(); i++){
        for(int j = 0; j < X.cols(); j++){
            if(isnan(X(i,j)) == 1){
                X(i,j) = 0;
            }
        }
    }
    return X;
}

Eigen::MatrixXd repmat(Eigen::VectorXd X, int n){
/*
 * Input: X: m x 1
 * Output: Y: m x n, all columns are the same to X
 */
    Eigen::MatrixXd Y(X.size(), n);
    for(int i = 0; i < n; i++){
        Y.col(i) = X;
    }
    return Y;
}

Eigen::MatrixXd gt_pred(Eigen::MatrixXd Xs,Eigen::MatrixXd R, Eigen::MatrixXd X_test){
/*
 * function for learning the gating function

    input:
           all training data Xs: n x d (10 x 2)
           all labels of probability R   n x K (10 x 3)
           all models model struct with K dimensions

    output:
        predicted probability of gating function for each cluster: n_test x K
        PP_exp:   softmax  (sum is 1): n_test x K
        PP_out:   standard normalize with + and - (sum is 1)
        PP:   raw data of gp predicted probability (sum maybe close to 1)
 */
    int K = R.cols();//get number of models, 3
    int N = Xs.rows();//10
    int n_test = X_test.rows();//202

    Eigen::MatrixXd PP = Eigen::MatrixXd::Zero(n_test, K);//202 x 3

    Eigen::VectorXi ind_train(N);
    for(int i = 0; i < N; i++){
        ind_train(i) = i;
    }

    for(int ijk = 0; ijk < K; ijk++){
        Eigen::VectorXd mu_gp(n_test);
        Eigen::VectorXd s2_gp(n_test);
        gpml_rms(ind_train, Xs, R.col(ijk), X_test, mu_gp, s2_gp);
        PP.col(ijk) = mu_gp;
    }

    Eigen::MatrixXd PP_out(n_test, K);
    for(int i = 0; i < PP_out.rows(); i++){
        PP_out.row(i) = PP.row(i).array()/PP.row(i).sum();
    }
    return PP_out;
}

void gmm_pred_wafr(Eigen::MatrixXd Xtest, Eigen::MatrixXd Ftest, BOTS bot, GTData gt_data, Eigen::VectorXd &pred_h, Eigen::VectorXd &pred_Var){
/*
 * Input:
 *      Xtest: 202 x 2, Ftest: 202 x 1
 */
    Model model;
    model.Sigma = bot.alpha_K.array().inverse().array() * bot.gamma_K.array();
    model.mu = bot.alpha_K.array().inverse().array() * bot.belta_K.array();
    model.w = norm_prob(bot.alpha_K);

    bot.Nm_ind = sort_unique(bot.Nm_ind);
    bot.Fs = extract_rows(gt_data.Fss, bot.Nm_ind);

    Eigen::VectorXd label(bot.Fs.rows());//10 x 1
    Eigen::MatrixXd R = Eigen::MatrixXd::Zero(bot.Fs.rows(), model.mu.cols());//10 x 1
    mixGaussPred_gmm(bot.Fs.transpose(), model, label, R);

    Eigen::MatrixXd mu = Eigen::MatrixXd::Zero(Ftest.rows(), gt_data.num_gau);//202 x 3
    Eigen::MatrixXd s2 = Eigen::MatrixXd::Zero(Ftest.rows(), gt_data.num_gau);//202 x 3
    Eigen::MatrixXd rms = Eigen::MatrixXd::Zero(1, gt_data.num_gau);//1 x 3

    for(int ijk = 0; ijk < gt_data.num_gau; ijk++){
        Eigen::VectorXi ind_train;
        int k = 0;
        for(int i = 0; i < label.size(); i++){
            if(label(i) == ijk){
                ind_train(k) = i;
                k = k + 1;
            }
        }
        Eigen::VectorXd mu_vec(Ftest.rows());
        Eigen::VectorXd s2_vec(Ftest.rows());
        gpml_rms(ind_train, extract_rows(gt_data.Xss, bot.Nm_ind), bot.Fs, Xtest, mu_vec, s2_vec);
        mu.col(ijk) = mu_vec;
        s2.col(ijk) = s2_vec;
    }

    Eigen::MatrixXd PP_out;
    PP_out = gt_pred(extract_rows(gt_data.Xss, bot.Nm_ind), R, Xtest);

/*
 * start to filter infeasible component
 */

    Eigen::MatrixXd pred_mu_mat = mu;//202 x 3
    Eigen::MatrixXd PP_out_tmp = PP_out.array() * (1 - pred_mu_mat.array().isNaN());
    Eigen::MatrixXd norm_PP_out(PP_out_tmp.rows(), PP_out_tmp.cols());
    for(int i = 0; i < norm_PP_out.rows(); i++){
        norm_PP_out.row(i) = PP_out_tmp.row(i).array()/PP_out_tmp.row(i).sum();
    }

    pred_mu_mat = set_NaN_tz(pred_mu_mat);//set NaN elements to 0
/*
 * end of filtering
 */

    Eigen::VectorXd muu_pp = (norm_PP_out.array() * pred_mu_mat.array()).rowwise().sum();//202 x 1

    Eigen::MatrixXd pred_s2_mat = s2;

    Eigen::MatrixXd muu_pp_rep = repmat(muu_pp, gt_data.num_gau);//202 x 3
    pred_s2_mat = s2 + (pred_mu_mat - muu_pp_rep).array() * (pred_mu_mat - muu_pp_rep).array();
    Eigen::VectorXd s2_pp = (PP_out.array() * pred_s2_mat.array()).rowwise().sum();

    pred_h = muu_pp;//202 x 1
    pred_Var = s2_pp;//202 x 1
}



#endif
