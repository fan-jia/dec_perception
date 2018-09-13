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

Eigen::VectorXd accumarray(Eigen::VectorXi k, Eigen::VectorXd phi_func, int num_bot){
    Eigen::VectorXd m = Eigen::VectorXd::Zero(num_bot);
    for(int i = 0; i < k.size(); i ++){
        for(int j = 0; j < num_bot; j++){
            if(k(i) == j){
                m(j) = m(j) + phi_func(i);
            }
        }

    }
    return m;
}

void main_bot_distribute(GTData gt_data){
/*
 * Fss represents temperature (945 x 1), Xss represents coordination (945 x 2)
 */

    //Some parameters can be changed
    int unit_sam = 3;
    int it_num = 10;
    double kp = 0.5;
    bool stop_flag = 0;
    double eta = 0.1;
    double beta = 1;
    double map_x = 20;
    double map_y = 44;

    Model model_rss;
    Eigen::VectorXi label_rss(gt_data.Fss.rows());
    mixGaussEm_gmm(gt_data.Fss.transpose(), gt_data.num_gau, label_rss, model_rss);//Output: label, model, llh

    vector<Eigen::MatrixXd> pilot_Xs_stack;
    for(int i = 0; i < gt_data.num_bot; i++){
        pilot_Xs_stack.push_back(Eigen::MatrixXd::Zero(unit_sam*gt_data.num_gau+1, 2));
    }

    /*
    pilot_Xs_stack[0] << 10,2,  9,22,  12,5,  15,41,  15,24,  6,18,  12,22,  13,11,  2,11, 14.2025,8.5429;
    pilot_Xs_stack[1] << 7,16, 4,32, 19,43, 1,25, 17,34, 1,20, 10,3, 2,11,  16,3, 7.335, 8.767;
    pilot_Xs_stack[2] << 0,12,  9,11, 0,35, 0,21, 4,10, 12,29, 16,40, 16,6, 10,9, 5.9637,13.729;
    */


    srand(200);
    for(int ikoo = 0; ikoo < gt_data.num_bot; ikoo++){
        Eigen::MatrixXd ind_ttmp = Eigen::MatrixXd::Zero(unit_sam, gt_data.num_gau);
        for(int kik = 0; kik < gt_data.num_gau; kik++){
            vector<int> sap_tmp;
            for(int i = 0; i < label_rss.size(); i++){
                if(label_rss[i] == kik){
                    sap_tmp.push_back(i);
                }
            }
            random_shuffle (sap_tmp.begin(), sap_tmp.end());
            for(int i = 0; i < unit_sam; i++){
                ind_ttmp(i,kik) = sap_tmp[i];
            }
        }
        Eigen::Map<Eigen::RowVectorXd> v1(ind_ttmp.data(), ind_ttmp.size());//reshape to unit_sam*num_gau  x 1
        for(int i = 0; i < v1.size(); i++){
            pilot_Xs_stack[ikoo].row(i) = gt_data.Xss.row(v1(i));
        }
        Eigen::VectorXd last_row(2);
        last_row << ((double) rand() / (RAND_MAX)) * map_x, ((double) rand() / (RAND_MAX))*map_y;//replace by starting points in a smaller area
        pilot_Xs_stack[ikoo].row(pilot_Xs_stack[ikoo].rows()-1) = last_row;
    }



/*
 * Initialization
 */



    vector<BOTS> bots;
    BOTS bot;
    PACKETS packet;
    bot.alpha_K = Eigen::MatrixXd::Zero(1,gt_data.num_gau);
    bot.belta_K = bot.gamma_K = bot.self_alpha = bot.self_belta = bot.self_gamma = bot.alpha_K;
    bot.dot_alpha_K = bot.dot_belta_K = bot.dot_gamma_K = bot.alpha_K;
    Eigen::MatrixXd g(gt_data.num_bot,2);

    for(int ijk = 0; ijk < gt_data.num_bot; ijk++){
        bots.push_back(bot); //initialize
    }

    for(int ijk = 0; ijk < gt_data.num_bot; ijk++){
        bots[ijk].Xs = pilot_Xs_stack[ijk];
        bots[ijk].Nm_ind = get_idx(bots[ijk].Xs,gt_data.Xss);
        g.row(ijk) = bots[ijk].Xs.row(bots[ijk].Xs.rows()-1);//in coverage control, specify generator's positions (starting positions)
        bots[ijk].Fs = extract_rows(gt_data.Fss, bots[ijk].Nm_ind);
    }




    for(int ijk = 0; ijk < gt_data.num_bot; ijk++){
        Model model;
        Eigen::VectorXi label(bots[ijk].Fs.rows());
        mixGaussEm_gmm(bots[ijk].Fs.transpose(), gt_data.num_gau, label, model);//Output is model, we don't care label

        //cout << "label is " << label << endl;

        bots[ijk].mu_K = model.mu;
        bots[ijk].Sigma_K = model.Sigma;
        bots[ijk].self_alpha = model.w;

        Eigen::MatrixXd alpha_mnk = Eigen::MatrixXd::Zero(bots[ijk].Fs.rows(), model.mu.cols());//10 x 3
        Eigen::VectorXi label_(bots[ijk].Fs.rows());
        mixGaussPred_gmm(bots[ijk].Fs.transpose(), model, label_, alpha_mnk);
        bots[ijk].alpha_K = alpha_mnk.colwise().sum();// 1 x 3
        Eigen::MatrixXd temp(alpha_mnk.rows(),alpha_mnk.cols());// 10 x 3
        for(int i = 0; i < temp.cols(); i++){
            temp.col(i) = alpha_mnk.col(i).array() * bots[ijk].Fs.array();
        }
        bots[ijk].belta_K = temp.colwise().sum();// 1 x 3

        Eigen::MatrixXd temp1(alpha_mnk.rows(),alpha_mnk.cols());
        for(int i = 0; i < temp1.cols(); i++){
            temp1.col(i) = bots[ijk].Fs.col(0).array() - model.mu(0,i);
        }
        temp1 = temp1.array() * temp1.array();
        temp1 = temp1.array() * alpha_mnk.array();
        bots[ijk].gamma_K = temp1.colwise().sum();

        for(int ijk_rec = 0; ijk_rec < gt_data.num_bot; ijk_rec++){
            bots[ijk].packets.push_back(packet);
        }

        if((ijk != (gt_data.num_bot - 1))&&(ijk != 0)){
            bots[ijk].neighbor.push_back(ijk+1);
            bots[ijk].neighbor.push_back(ijk-1);
        }
        else if(ijk == (gt_data.num_bot - 1)){
            bots[ijk].neighbor.push_back(ijk - 1);
        }
        else if(ijk == 0){
            bots[ijk].neighbor.push_back(ijk + 1);
        }


        //cout << "bots " << ijk << " Fs is " << bots[ijk].Fs << endl;
        //cout << "bots " << ijk << " indices are " << bots[ijk].Nm_ind << endl;
        //cout << "bots " << ijk << " mu_K is " << bots[ijk].mu_K << endl;
        //cout << "bots " << ijk << " Sigma_K is " << bots[ijk].Sigma_K << endl;
        //cout << "bots " << ijk << " w is " << bots[ijk].self_alpha << endl;
    }




    for(int ijk = 0; ijk < gt_data.num_bot; ijk++){
        packet.alpha_K = bots[ijk].alpha_K;
        packet.belta_K = bots[ijk].belta_K;
        packet.gamma_K = bots[ijk].gamma_K;
        bots[ijk].packets[ijk] = packet;
    }

    //g = g.transpose();//now is 2 x num_bot(3)


/*
 * initiate swarm status
 */


    int s_num = gt_data.Fss.rows();//945
    Eigen::MatrixXd s = gt_data.Xss.transpose();// 2 x 945

    //176 ~ 182 missing
    Eigen::VectorXd rms_stack = Eigen::VectorXd::Zero(it_num);
    Eigen::VectorXd var_stack = Eigen::VectorXd::Zero(it_num);

/*
 * initialize for consensus loop
 */

    int max_iter = 10;//consensus communication round

/*
 * first round communication
 */


    for(int ijk = 0; ijk < gt_data.num_bot; ijk++){
        bots = transmitPacket(bots,ijk);//update communication packets
    }

    vector<Eigen::MatrixXd> hist_alpha_K;
    vector<Eigen::MatrixXd> hist_belta_K;
    vector<Eigen::MatrixXd> hist_gamma_K;
    vector<Eigen::MatrixXd> hist_alpha_K_norm;
    vector<Eigen::MatrixXd> hist_belta_K_norm;
    vector<Eigen::MatrixXd> hist_gamma_K_norm;
    vector<Eigen::MatrixXd> hist_mu_K_norm;// all is 3 x 1, include 1000 x 3

    for(int i = 0; i < gt_data.num_bot; i++){
        hist_alpha_K.push_back(Eigen::MatrixXd::Zero(max_iter,gt_data.num_gau));
        hist_belta_K.push_back(Eigen::MatrixXd::Zero(max_iter,gt_data.num_gau));
        hist_gamma_K.push_back(Eigen::MatrixXd::Zero(max_iter,gt_data.num_gau));
        hist_alpha_K_norm.push_back(Eigen::MatrixXd::Zero(max_iter,gt_data.num_gau));
        hist_belta_K_norm.push_back(Eigen::MatrixXd::Zero(max_iter,gt_data.num_gau));
        hist_gamma_K_norm.push_back(Eigen::MatrixXd::Zero(max_iter,gt_data.num_gau));
        hist_mu_K_norm.push_back(Eigen::MatrixXd::Zero(max_iter,gt_data.num_gau));
    }

    Eigen::MatrixXd true_alpha_K(max_iter, model_rss.w.cols());
    for(int i = 0; i < max_iter; i++){
        true_alpha_K.row(i) = model_rss.w.row(0);
    }



    for(int it = 0; it < it_num; it++){
        bool loop_flag = true;
        int cur_iter = 1;
/*
 *  after consensus loop, updated variable:  1) bots.neighbor, 2) new local model, 3) bots.Nm_ind (after execution)
 * begin consensus process to refine local model for each robot
 */
        if(!stop_flag){
            while(loop_flag){ //for first round, use neighbors defined by default from the part of initialization above
                for(int ijk = 0; ijk < gt_data.num_bot; ijk++){
                    bots = transmitPacket(bots,ijk);//update communication packets
                    bots = updateBotComputations(bots, ijk, gt_data.Fss, eta);//then update self and consensus
                }

                for(int dijk = 0; dijk < gt_data.num_bot; dijk++){
                    hist_alpha_K[dijk].row(cur_iter) = bots[dijk].alpha_K;
                    hist_belta_K[dijk].row(cur_iter) = bots[dijk].belta_K;
                    hist_gamma_K[dijk].row(cur_iter) = bots[dijk].gamma_K;

                    hist_alpha_K_norm[dijk].row(cur_iter) = norm_prob(bots[dijk].alpha_K);
                    hist_belta_K_norm[dijk].row(cur_iter) = norm_prob(bots[dijk].belta_K);
                    hist_gamma_K_norm[dijk].row(cur_iter) = norm_prob(bots[dijk].gamma_K);

                    hist_mu_K_norm[dijk].row(cur_iter) = bots[dijk].mu_K;
                }
                cur_iter = cur_iter + 1;
                if(cur_iter >= max_iter){
                    cur_iter = cur_iter - 1;
                    break;
                }
            }
        }
    /*
     * Compute the Delaunay triangle information T for the current nodes
     */

    /*
     * For each sample point, find K, the index of the nearest generator. We do this efficiently by using the Delaunay
     * information with Matlab's DSEARCH command, rather than a brute force nearest neighbor computation.
     */

        Eigen::VectorXi k = powercellidx(g,s);//945 x 1

        Eigen::VectorXd pred_h = Eigen::VectorXd::Zero(gt_data.Fss.rows());//945 x 1
        Eigen::VectorXd pred_Var = Eigen::VectorXd::Zero(gt_data.Fss.rows());//945 x 1

        for(int iij = 0; iij < gt_data.num_bot; iij++){
            Eigen::VectorXi idx = get_label_idx(k,iij);
            Eigen::VectorXd pred_h_idx = Eigen::VectorXd::Zero(idx.size());
            Eigen::VectorXd pred_Var_idx = Eigen::VectorXd::Zero(idx.size());

            gmm_pred_wafr(extract_rows(gt_data.Xss, idx), extract_rows(gt_data.Fss, idx), bots[iij], gt_data, pred_h_idx, pred_Var_idx);
            pred_h = insert_vec(pred_h_idx, pred_h, idx);
            pred_Var = insert_vec(pred_Var_idx, pred_Var, idx);
        }

        Eigen::VectorXd est_mu = pred_h.array().abs();
        Eigen::VectorXd est_s2 = pred_Var.array().abs();
        Eigen::VectorXd phi_func = est_mu + beta * est_s2;

        Eigen::VectorXi idx_train = sort_unique_all_robots(bots, gt_data);
        Eigen::VectorXi idx_test = setdiff(idx_train, gt_data.Fss.rows());

        Eigen::VectorXd minus_tmp = extract_vec(pred_h, idx_test) - extract_vec(gt_data.Fss, idx_test);
        rms_stack(it) = sqrt(((minus_tmp.array() * minus_tmp.array()).sum())/idx_test.size());
        var_stack(it) = pred_Var.mean();

        Eigen::MatrixXd g_new = g;//3 x 2
        //cout << "rms " << it <<  " is " << rms_stack(it) << endl;
        //cout <<  "var " << it << " is " << var_stack(it) << endl;
        //cout << "g now is " << g_new << endl;

        Eigen::VectorXd m = Eigen::VectorXd::Zero(gt_data.num_bot);
        m = accumarray(k,phi_func,gt_data.num_bot);//this computes \sigma_{V_i} \phi(q) \dq for all voronoi cell V_i

        Eigen::VectorXd sumx = Eigen::VectorXd::Zero(gt_data.num_bot);
        Eigen::VectorXd sumy = Eigen::VectorXd::Zero(gt_data.num_bot);
        sumx = accumarray(k,gt_data.Xss.col(0).array()*phi_func.array(),gt_data.num_bot);
        sumy = accumarray(k,gt_data.Xss.col(1).array()*phi_func.array(),gt_data.num_bot);
        g_new.col(0) = sumx.array()/m.array();//get x coordinate for the new centroid
        g_new.col(1) = sumy.array()/m.array();//get y coordinate for the new centroid

        //cout << "g_new is " << g_new << endl;

        Eigen::MatrixXd g_actual = g;

        Eigen::VectorXd m_r = Eigen::VectorXd::Zero(gt_data.num_bot);
        Eigen::VectorXd Fss_vec = gt_data.Fss.col(0);
        m_r = accumarray(k,Fss_vec,gt_data.num_bot);//this computes \sigma_{V_i} \phi(q) \dq for all voronoi cell V_i

        Eigen::VectorXd sumx_r = Eigen::VectorXd::Zero(gt_data.num_bot);
        Eigen::VectorXd sumy_r = Eigen::VectorXd::Zero(gt_data.num_bot);
        sumx_r = accumarray(k,gt_data.Xss.col(0).array()*Fss_vec.array(),gt_data.num_bot);
        sumy_r = accumarray(k,gt_data.Xss.col(1).array()*Fss_vec.array(),gt_data.num_bot);
        g_actual.col(0) = sumx_r.array()/m_r.array();//get x coordinate for the actual centroid
        g_actual.col(1) = sumy_r.array()/m_r.array();//get y coordinate for the actual centroid

        //cout << "g_actual is " << g_actual << endl;

        g = g+kp*(g_new-g); //g_new
        //cout << "now g is " << g << endl;

        Eigen::VectorXi proj_g_idx = get_idx(g, gt_data.Xss);
        int stop_count = 0;
        for(int ijk = 0; ijk < gt_data.num_bot; ijk++){
            if(bots[ijk].Nm_ind(bots[ijk].Nm_ind.size()-1) != proj_g_idx(ijk)){
                bots[ijk].Nm_ind.conservativeResize(bots[ijk].Nm_ind.size()+1);
                bots[ijk].Nm_ind(bots[ijk].Nm_ind.size() - 1) = proj_g_idx(ijk);
                bots[ijk].Xs.conservativeResize(bots[ijk].Xs.rows()+1, bots[ijk].Xs.cols());
                bots[ijk].Xs.row(bots[ijk].Xs.rows()-1) = gt_data.Xss.row(proj_g_idx(ijk));
            }
            else{
                stop_count = stop_count + 1;
            }
        }

        if(stop_count == gt_data.num_bot){
            stop_flag = 1;
            cout << "all robots have arrived at the centroid of voronoi cells" << endl;
        }
    }
}


#endif
