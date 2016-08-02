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
class Phi:public FUNCT{
    public:
        
        Phi(double m,double s):FUNCT(),m(m),s(s){}
        
        double m;
        double s;
        double f(double x){
            return 0.5*(1+erf((x-m)/(s*sqrt(2))));
        }
        double d(double x){
            return exp(-(x-m)*(x-m)/(2*s*s))/(s*sqrt(2*M_PI));
        }
        double inv(double x){
            return 0;
        }
        
};
#endif

