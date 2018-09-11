#ifndef _main_bot_distributed_
#define _main_bot_distributed_

#include "gmm_pred_wafr.h"

using namespace std;

Eigen::VectorXi powercellidx(Eigen::MatrixXd g, Eigen::MatrixXd s){
/*
 * Input:
 *      g: 3 x 2, s: 2 x 945
 * Output:
 *      idx: 945 x 1
 */

    Eigen::MatrixXd Distance = pdist2(s.transpose(), g);//945 x 3
    Eigen::VectorXi idx(s.cols());//945 x 1
    Eigen::MatrixXd::Index min_index;
    for(int i = 0; i < idx.size(); i++){
        Distance.row(i).minCoeff(&min_index);
        idx(i) = min_index;
    }
    return idx;
}

vector<BOTS> transmitPacket(vector<BOTS> bots, int n){
    int num_bot = bots.size();
    for(int j = 0; j < num_bot; j++){
        for(int i = 0; i < bots[n].neighbor.size(); i++){
            if(j == bots[n].neighbor[i]){
                bots[j].packets[n] = bots[n].packets[n];
            }
        }
    }
    return bots;
}

vector<BOTS> updateBotComputations(vector<BOTS> bots, int n, Eigen::MatrixXd Fss, double eta){
    int num_gau = bots[n].alpha_K.cols();
    Model model;
    model.mu = bots[n].mu_K;
    model.Sigma = bots[n].Sigma_K;
    model.w = norm_prob(bots[n].alpha_K);//norm to 0~1

    Eigen::VectorXi label(bots[n].Fs.rows());
    Eigen::MatrixXd alpha_mnk = Eigen::MatrixXd::Zero(bots[n].Fs.rows(), model.mu.cols());//10 x 3
    mixGaussPred_gmm(bots[n].Fs.transpose(), model, label, alpha_mnk);
    bots[n].self_alpha = alpha_mnk.colwise().sum();// 1 x 3, get self_alpha

    Eigen::MatrixXd temp(alpha_mnk.rows(),alpha_mnk.cols());// 10 x 3
    for(int i = 0; i < temp.cols(); i++){
        temp.col(i) = alpha_mnk.col(i).array() * bots[n].Fs.array();
    }
    bots[n].self_belta = temp.colwise().sum();// 1 x 3, get self_belta

    Eigen::MatrixXd temp1(alpha_mnk.rows(),alpha_mnk.cols());
    for(int i = 0; i < temp1.cols(); i++){
        temp1.col(i) = bots[n].Fs.col(0).array() - model.mu(0,i);
    }
    temp1 = temp1.array() * temp1.array();
    temp1 = temp1.array() * alpha_mnk.array();
    bots[n].self_gamma = temp1.colwise().sum();// 1 x 3, get self_gamma

/*
 * after compute local summary stats, we update estimate of global stats using packets
 */

    int num_neighbor = bots[n].neighbor.size();

/*
 * start consensus based dynamic estimation process
 */
    Eigen::MatrixXd stack_alpha_neighbor(bots[n].neighbor.size(), num_gau);
    Eigen::MatrixXd stack_belta_neighbor(bots[n].neighbor.size(), num_gau);
    Eigen::MatrixXd stack_gamma_neighbor(bots[n].neighbor.size(), num_gau);
    for(int i = 0; i < bots[n].neighbor.size(); i++){
        stack_alpha_neighbor.row(i) = bots[n].packets[bots[n].neighbor[i]].alpha_K;// 1 x 3
        stack_belta_neighbor.row(i) = bots[n].packets[bots[n].neighbor[i]].belta_K;// 1 x 3
        stack_gamma_neighbor.row(i) = bots[n].packets[bots[n].neighbor[i]].gamma_K;// 1 x 3
    }

    Eigen::MatrixXd diff_alpha(bots[n].neighbor.size(), num_gau);
    Eigen::MatrixXd diff_belta(bots[n].neighbor.size(), num_gau);
    Eigen::MatrixXd diff_gamma(bots[n].neighbor.size(), num_gau);

    for(int i = 0; i < diff_alpha.rows(); i++){
        diff_alpha.row(i) = stack_alpha_neighbor.row(i) - bots[n].alpha_K;
        diff_belta.row(i) = stack_belta_neighbor.row(i) - bots[n].belta_K;
        diff_gamma.row(i) = stack_gamma_neighbor.row(i) - bots[n].gamma_K;
    }
    bots[n].dot_alpha_K = diff_alpha.colwise().sum() + bots[n].self_alpha - bots[n].alpha_K;
    bots[n].dot_belta_K = diff_belta.colwise().sum() + bots[n].self_belta - bots[n].belta_K;
    bots[n].dot_gamma_K = diff_gamma.colwise().sum() + bots[n].self_gamma - bots[n].gamma_K;

    bots[n].alpha_K = bots[n].alpha_K + eta * bots[n].dot_alpha_K;
    bots[n].belta_K = bots[n].belta_K + eta * bots[n].dot_belta_K;
    bots[n].gamma_K = bots[n].gamma_K + eta * bots[n].dot_gamma_K;//all is 1 x 3

    bots[n].Sigma_K = bots[n].alpha_K.array().inverse().array() * bots[n].gamma_K.array();
    bots[n].mu_K = bots[n].alpha_K.array().inverse().array() * bots[n].belta_K.array();

    bots[n].packets[n].alpha_K = bots[n].alpha_K;
    bots[n].packets[n].belta_K = bots[n].belta_K;
    bots[n].packets[n].gamma_K = bots[n].gamma_K;

    return bots;
}

Eigen::VectorXd insert_vec(Eigen::VectorXd X, Eigen::VectorXd Y, Eigen::VectorXi idx){
/*
 * Insert X values to indices of Y (output)
 */
    for(int i = 0; i < idx.size(); i++){
        Y(idx(i)) = X(i);
    }
    return Y;
}

Eigen::VectorXi sort_unique_all_robots(vector<BOTS> bots, GTData gt_data){
    vector<int> ind_train_vec;
    for(int i = 0; i < gt_data.num_bot; i++){
        for(int j = 0; j < bots[i].Nm_ind.size(); j++){
            ind_train_vec.push_back(bots[i].Nm_ind(j));
        }
    }
    Eigen::VectorXi ind_train(ind_train_vec.size());
    for(int i = 0; i < ind_train.size(); i++){
        ind_train(i) = ind_train_vec[i];
    }

    ind_train = sort_unique(ind_train);
    return ind_train;
}

Eigen::VectorXi setdiff(Eigen::VectorXi idx_train, int init){
    vector<int> vec;
    for(int i = 0; i < init; i ++){
        vec.push_back(i);
    }
    for(int i = 0; i < idx_train.size(); i++){
        vec.erase(remove(vec.begin(),vec.end(),idx_train(i)),vec.end());
    }
    Eigen::VectorXi idx_test(vec.size());
    for(int i = 0; i < vec.size(); i++){
        idx_test(i) = vec[i];
    }
    return idx_test;
}

Eigen::VectorXd extract_vec(Eigen::VectorXd X, Eigen::VectorXi idx){
    Eigen::VectorXd Y(idx.size());
    for(int i = 0; i < idx.size(); i++){
        Y(i) = X(idx(i));
    }
    return Y;
}



#endif
