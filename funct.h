#ifndef FUNCT_H
#define FUNCT_H
#include<cmath>
#include<vector>
#include<map>
using namespace std;
class FUNCT{
    public:
        virtual double f(double)=0;
        virtual double d(double)=0;
        virtual double inv(double)=0;
};
class Sigmoid:public FUNCT{
    public:
        Sigmoid(double a):FUNCT(),a(a){}
        double a;
        double f(double x){
            return 1/(1+exp(-x*a));
        }
        double d(double x){
            return a*(1/(cosh(a*x/2)*cosh(a*x/2)))/4;
        }
        double inv(double x){
            return log(x/(1-x));
        }
};
#endif

