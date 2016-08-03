#include"ann.h"
#include"stt.h"
int main(){
    ANN muann(3,15);
    vector<size_t>layer;
    for(size_t i=0;i<9;i++)
        layer.push_back(i);
    muann.layerize(1,layer);
    layer.clear();
    for(size_t i=9;i<14;i++)
        layer.push_back(i);
    muann.layerize(2,layer);
    muann.layerize(3,14,-1);
    muann.configActf(new Tanh());
    muann.configBPLayer();
    muann.configBPBias();
    muann.eta = 0.01;
    cout.precision(5);
    //muann.printWeight();
    cout<<"================"<<endl;
    /*for(size_t i=0;i<2000;i++){
        vector<double>sample = random_sample(12);
        vector<double>solution(1,minmd(sample,0.7));
        muann.train(sample,solution);
   // muann.printWeight();
    }*/

    for(size_t i=0;i<200;i++){
        vector<double>sample = random_sample(12);
        vector<double>solution(1,minmd(sample,0.7));
        muann.train(sample,solution);
        sample.push_back(solution[0]);
        cout<<"stds: "<<mv(sample)<<"reals:";
        sample[12]=muann.val[muann.result->index];
        cout<<mv(sample)<<endl;
        cout<<"<"<<solution[0]<<","<<muann.val[muann.result->index]<<">"<<endl;

    
    }
    muann.printVal();

}
