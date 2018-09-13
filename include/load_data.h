#ifndef _load_data_
#define _load_data_

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <Eigen/Dense>

using namespace std;

void load_data(Eigen::MatrixXd &Xss, Eigen::MatrixXd &Fss){
    ifstream finFss("/home/fanjia/CLionProjects/distributed_perception/include/Fss.txt");

    if (!finFss.is_open())
    {
        cout << "open src File: Error opening file" << endl;
    }

    vector<double> Fss_vec;
    string line;

    while (getline(finFss, line))
    {
        istringstream sin(line);

        string field;
        double a;

        while (getline(sin, field, ','))
        {
            a = stod(field);
            Fss_vec.push_back(a);
        }
    }
    for(int i = 0; i < Fss_vec.size(); i ++){
        Fss(i, 0) = Fss_vec[i];
    }

    ifstream finXss("/home/fanjia/CLionProjects/distributed_perception/include/Xss.txt");

    if (!finXss.is_open())
    {
        cout << "open src File: Error opening file" << endl;
    }

    vector<double> Xss_x;
    vector<double> Xss_y;
    string line1;

    while (getline(finXss, line1))
    {
        istringstream sin(line1);

        string field;
        double a;
        int n = 0;

        while (getline(sin, field, ','))
        {
            a = stod(field);
            if(n == 0){
                Xss_x.push_back(a);
                n = n + 1;
            }
            else if(n == 1){
                Xss_y.push_back(a);
            }
        }
    }
    for(int i = 0; i < Xss_x.size(); i ++){
        Xss(i, 0) = Xss_x[i];
        Xss(i, 1) = Xss_y[i];
    }
}

#endif
