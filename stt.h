#ifndef STT__H
#define STT__H
#include<cmath>
#include<iostream>
#include<vector>
vector<double> random_sample(size_t k){
    vector<double>res;
    for(size_t i=0;i<k;i++)
        res.push_back((double)(rand()%200)/200);
    return res;
}

double minsd(vector<double>v,double s){
    double a = (double)v.size()/(double)(v.size()+1);
    double w = 0;
    double W = 0;
    for(size_t i=0;i<v.size();i++){
        w+=v[i];
        W+=v[i]*v[i];
    }
    double b = -2*w/(double)(v.size()+1);
    double c = W - w*w/(double)(v.size()+1)-s*s*(double)(v.size()+1);
    double x = -b/(2*a);
    if(b*b-4*a*c>=0)
        x = (-b+sqrt(b*b-4*a*c))/(2*a);
    return x;
}

double minmd(vector<double>v,double m){
    double res = 0;
    for(size_t i=0;i<v.size();i++)
        res += v[i];
    return (double)(v.size()+1)*m-res;
}

double sv(vector<double>v){
    double m = 0;
    for(size_t i=0;i<v.size();i++)
        m+=v[i];
    m = m/(double)v.size();
    double s = 0;
    for(size_t i=0;i<v.size();i++)
        s+=(m-v[i])*(m-v[i]);
    return sqrt(s/(double)v.size());
}

double mv(vector<double>v){
    double res = 0;
    for(size_t i=0;i<v.size();i++)
        res += v[i];
    return res/(double)v.size();
}
#endif

