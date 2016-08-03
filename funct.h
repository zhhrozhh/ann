#ifndef FUNCT_H
#define FUNCT_H
#include<cmath>
#include<vector>
#include<map>
#define sech(x) (1/cosh(x))
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
class Sigmoid_:public FUNCT{
    public:
        Sigmoid_():FUNCT(){}
        double f(double x){
            return 10/(1+exp(-x)) - 5;
        }
        double d(double x){
            return 2*exp(-0.2*x)/((exp(-0.2*x)+1)*(exp(-0.2*x)+1));
        }
        double inv(double x){
            return 0;
        }
};
class Tanh:public FUNCT{
    public:
        double f(double x){
            return 10*tanh(x);
        }
        double d(double x){
            return 10*sech(x)*sech(x);
        }
        double inv(double x){
            return 0;
        }
};
#endif

