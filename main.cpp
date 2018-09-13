#include <iostream>
#include "main_bot_distributed.h"
#include "load_data.h"
//#include "mixGaussEm_gmm.h"
using namespace std;

int main() {

    GTData gt_data;
    gt_data.num_gau = 3;
    gt_data.num_bot = 3;

    gt_data.Xss = Eigen::MatrixXd::Zero(945,2);
    gt_data.Fss = Eigen::MatrixXd::Zero(945,1);
    load_data(gt_data.Xss, gt_data.Fss);

    main_bot_distribute(gt_data);

    return 0;
}