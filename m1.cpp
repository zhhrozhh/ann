#include"ann.h"
vector<double> random_sample(size_t k){
    vector<double>res;
    for(size_t i=0;i<k;i++)
        res.push_back((double)(rand()%200)/200);
    return res;
}
double minsd(vector<double>v,size_t pos,double s){
    double a = (double)(v.size()-1)/(double)v.size();
    double w = 0;
    double W = 0;
    for(size_t i=0;i<v.size();i++)
        if(i!=pos){
            w+=v[i];
            W+=v[i]*v[i];
        }
    double b = 2*w/(double)v.size();
    double c = W - w*w/(double)v.size()-s*s*(double)v.size();
    cout<<a<<" "<<b<<" "<<c<<endl;
    return b*b-4*a*c;
    
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
int main(){
    /*
    ANN ann(3,66);
    vector<size_t>layer;
    for(size_t i=0;i<30;i++)
        layer.push_back(i);
    ann.layerize(1,layer);
    layer.clear();
    for(size_t i=0;i<35;i++)
        layer.push_back(i+30);
    ann.layerize(2,layer);
    ann.layerize(3,65,-1);
    */
    for(size_t i=0;i<100;i++)
        cout<<minsd(random_sample(30),2,0.11)<<endl;
}
