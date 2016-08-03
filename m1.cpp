#include"ann.h"
#include"stt.h"
int main(){
    ANN muann(3,66);
    ANN sigmann(3,66);
    vector<size_t>layer;
    for(size_t i=0;i<30;i++)
        layer.push_back(i);
    muann.layerize(1,layer);
    sigmann.layerize(1,layer);
    layer.clear();
    for(size_t i=30;i<65;i++)
        layer.push_back(i);
    muann.layerize(2,layer);
    sigmann.layerize(2,layer);
    muann.layerize(3,65,-1);
    sigmann.layerize(3,65,-1);


    muann.configBPLayer();
    muann.configBPBias();

    sigmann.configBPLayer();
    sigmann.configBPBias();

    cout.precision(10);
    sigmann.eta = 0.18;
    for(size_t j=0;j<1;j++)
    for(size_t i=0;i<10;i++){
        vector<double>ss = random_sample(30);
        vector<double>Ys(1,minsd(ss,0.2));
        vector<double>Ym(1,minmd(ss,0.2));
        sigmann.train(ss,Ys);
        muann.train(ss,Ym);
    }
    
    for(size_t i=0;i<50;i++){
        vector<double>sample = random_sample(30);
        double ym = minmd(sample,0.2);
        vector<double>Ym(1,ym);
        double ys = minsd(sample,0.2);
        vector<double>Ys(1,ys);
        muann.train(sample,Ym);
        sigmann.train(sample,Ys);
        sample.push_back(muann.val[65]);
        cout<<"merror:"<<mv(sample) - 0.2<<endl;
        sample[30]=sigmann.val[65];
        cout<<"serror:"<<sv(sample)-0.2<<" hat:"<<sample[30]<<" sol:"<<ys<<endl;
        sample[30]=ys;
        cout<<"ss: "<<sv(sample)<<endl;

    }
}
