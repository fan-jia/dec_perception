#ifndef _main_bot_distribute_
#define _main_bot_distribute_


#include "mixGaussEm_gmm.h"
using namespace std;

void main_bot_distribute(Eigen::VectorXd label_rss, Eigen::MatrixXd Xss, Eigen::MatrixXd Fss){
/*
 * Fss represents temperature (945 x 1), Xss represents coordination (945 x 2)
 */

    //Some parameters can be changed
    int unit_sam = 3;
    int num_gau = 3;
    int num_bot = 3;
    double map_x = 20;
    double map_y = 44;

    vector<Eigen::MatrixXd> pilot_Xs_stack;
    for(int i = 0; i < num_bot; i++){
        pilot_Xs_stack.push_back(Eigen::MatrixXd::Zero(unit_sam*num_gau+1, 2));
    }

    for(int ikoo = 0; ikoo < num_bot; ikoo++){
        Eigen::MatrixXd ind_ttmp = Eigen::MatrixXd::Zero(unit_sam, num_gau);
        for(int kik = 0; kik < num_gau; kik++){
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
            pilot_Xs_stack[ikoo].row(i) = Xss.row(v1(i));
        }
        Eigen::VectorXd last_row(2);
        last_row << ((double) rand() / (RAND_MAX)) * map_x, ((double) rand() / (RAND_MAX))*map_y;//replace by starting points in a smaller area
        pilot_Xs_stack[ikoo].row(pilot_Xs_stack[ikoo].rows()-1) = last_row;
    }

/*
 * Initialization
 */

    //if isempty(bots)  %nargin < 1
    /*
    vector<Eigen::VectorXd> tmp_init;
    for(int i = 0; i < num_bot; i++){
        tmp_init.push_back(Eigen::VectorXd::Zero(num_gau));
    }
    BOTS bots;
    bots.alpha_K = bots.belta_K = bots.gamma_K = bots.self_alpha = bots.self_belta = bots.self_gamma = tmp_init;
    bots.dot_alpha_K = bots.dot_belta_K = bots.dot_gamma_K = tmp_init;
    Eigen::MatrixXd g(num_bot,2);

    for(int ijk = 0; ijk < num_bot; ijk++){
        bots.Xs.push_back(pilot_Xs_stack[ijk]);
        bots.Nm_ind.push_back(get_idx(bots.Xs[ijk],Xss));
        g.row(ijk) = bots.Xs
    }
    */
    vector<BOTS> bots;
    BOTS bot;
    bot.alpha_K = Eigen::VectorXd::Zero(num_gau);
    bot.belta_K = bot.gamma_K = bot.self_alpha = bot.self_belta = bot.self_gamma = bot.alpha_K;
    bot.dot_alpha_K = bot.dot_belta_K = bot.dot_gamma_K = bot.alpha_K;
    Eigen::MatrixXd g(num_bot,2);

    for(int ijk = 0; ijk < num_bot; ijk++){
        bots.push_back(bot); //initialize
    }

    for(int ijk = 0; ijk < num_bot; ijk++){
        bots[ijk].Xs = pilot_Xs_stack[ijk];
        bots[ijk].Nm_ind = get_idx(bots[ijk].Xs,Xss);
        g.row(ijk) = bots[ijk].Xs.row(bots[ijk].Xs.rows()-1);//in coverage control, specify generator's positions (starting positions)
        for(int i = 0; i < bots[ijk].Nm_ind.size(); i++){
            bots[ijk].Fs.row(i) = Fss.row(bots[ijk].Nm_ind(i));//Fs:10 x 1
        }
    }


    Model model;
    for(int ijk = 0; ijk < num_bot; ijk++){
        Eigen::VectorXd label(bots[ijk].Fs.rows());
        mixGaussEm_gmm(bots[ijk].Fs.transpose(), num_gau, label, model);//Output is model, we don't care label




    }



}

#endif