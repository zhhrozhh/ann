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
    muann.configActf(new Sigmoid_());
    muann.configBPLayer();
    muann.configBPBias();
    muann.eta = 0.0065;
    cout.precision(5);
    //muann.printWeight();
    cout<<"================"<<endl;
    for(size_t i=0;i<20000;i++){
        vector<double>sample = random_sample(12);
        vector<double>solution(1,minmd(sample,0.7));
        if(solution[0]<5 and solution[0]>-5)
            muann.train(sample,solution,20);
   // muann.printWeight();
    }

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
