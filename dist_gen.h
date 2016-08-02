#include"ann.h"
#include"funct.h"
#ifndef dbg
#define dbg cout<<__LINE__<<endl
#endif
class DistGen{
    public:
        DistGen(size_t k,double mu,double sigma):ann(ANN(4,3*k+12)),k(k),mu(mu),sigma(sigma){
            //k_sample=vector<double>(k,mu);
            for(size_t i=0;i<k;i++)
                k_sample.push_back((double)(rand()%200)/200);
            vector<size_t>layer;
            for(size_t i=0;i<k+4;i++)
                layer.push_back(i);
            ann.layerize(1,layer);
            layer.clear();
            for(size_t i=0;i<k+6;i++)
                layer.push_back(i+k+4);
            ann.layerize(2,layer);
            layer.clear();
            for(size_t i=0;i<k+1;i++)
                layer.push_back(i+2*k+10);
            ann.layerize(3,layer);
            ann.layerize(4,3*k+12-1,-1);
            ann.configBPLayer();
            ann.configBPBias();
            ann.eta = 10e3;
        }
        ANN ann;
        vector<double> k_sample;
        size_t sample_ptr;
        size_t k;
        double mu;
        double sigma;//0<mu-4sigma<mu+4sigma<1
        double m(){
            double res = 0;
            for(size_t i=0;i<k;i++)
                res += k_sample[i];
            return res/(double)k;
        }
        double s(){
            double avg = m();
            double res = 0;
            for(size_t i=0;i<k;i++)
                res += (k_sample[i]-avg)*(k_sample[i]-avg);
            return sqrt(res/(double)k);
        }
        double mk1(double xk){
            return (m()*k+xk)/(double)(k+1);
        }
        double sk1(double xk){
            double res = 0;
            for(size_t i=0;i<k_sample.size();i++)
                res += (k_sample[i]-mk1(xk))*(k_sample[i]-mk1(xk));
            res += (xk-mk1(xk))*(xk-mk1(xk));
            return sqrt(res/(double)(k+1));
        }
        double nd(){
            double ss = s();
            double mm = m();
            vector<double>inp;
            for(size_t i=0;i<k;i++)
                inp.push_back(k_sample[i]);
            inp.push_back(mu);
            inp.push_back(sigma);
            inp.push_back(mm);
            inp.push_back(ss);
            ann.ff(inp);
            vector<double>dEdy;
            double y = ann.val[ann.result->index];
            if(ss)
                dEdy.push_back(Phi(mu,sigma).d(y+0.2*sigma) - Phi(mu,sigma).d(y-0.2*sigma)-Phi(mm,ss).d(y+0.2*ss)+Phi(mm,ss).d(y-0.2*ss));
            else
                dEdy.push_back(10*((sk1(y)*sk1(y)-sigma*sigma)*(y-mk1(y))/(double)(k+1)  +  (mk1(y)-mu)*y/(k+1)));
            ann.bpWithDEDy(dEdy);
            k_sample[sample_ptr] = y;
            sample_ptr = (sample_ptr+1)%k;
            return y;
        }
};
