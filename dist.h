#ifndef DISTRH
#define DISTRH
#include"bp3.h"
#include"stt.h"
vector<double>removePos(vector<double>&v,size_t pos){
    vector<double>res(v);
    res.erase(res.begin()+pos);
    return res;
}
class Ndist{
    public:
        Ndist(double mu,double sigma):mubp(5,8,1),sigmabp(5,8,1),mu(mu),sigma(sigma){
            s_ptr = 0;
            for(size_t i=0;i<6;i++)
                sample.push_back(0.01);
        }
        double mu;
        double sigma;
        vector<double>sample;
        size_t s_ptr;
        BP3 mubp;
        BP3 sigmabp;
        double get(){
            vector<double>v = removePos(sample,s_ptr);
            vector<double>Ym(1,minmd(v,mu));
            vector<double>Ys(1,minsd(v,sigma));

            mubp.train(sample,Ym);
            sigmabp.train(sample,Ys);

            double mr = mubp.val[mubp.result->index];
            double sr = sigma.val[sigma.result->index];

            sample[s_ptr] = mr;
            double rm = mv(sample);
            sample[s_ptr] = sr;
            double rs = sv(sample);

            double alpha = (rm-mu)*(rm-mu)/((rm-mu)*(rm-mu)+(rs-sigma)*(rs-sigma));
            sample[s_ptr] = alpha * mr + (1-alpha)*sr;
            
            s_ptr = (s_ptr+1)%6;
            return mv(sample);
        }

};

#endif

