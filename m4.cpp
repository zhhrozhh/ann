#include"ann.h"
#include"stt.h"
int main(){
    ANN muann(3,21);
    vector<size_t>layer;
    for(size_t i=0;i<12;i++)
        layer.push_back(i);
    muann.layerize(1,layer);
    layer.clear();
    for(size_t i=12;i<20;i++)
        layer.push_back(i);
    muann.layerize(2,layer);
    muann.layerize(3,20,-1);
    muann.configActf(new Sigmoid_());
    muann.configBPLayer();
    muann.configBPBias();
    muann.eta = 1;
    cout.precision(5);
    //muann.printWeight();
    cout<<"================"<<endl;
    for(size_t i=0;i<2000;i++){
        vector<double>sample = random_sample(12);
        vector<double>solution(1,minsd(sample,0.18));
        muann.train(sample,solution);
        // muann.printWeight();
    }

    for(size_t i=0;i<200;i++){
        vector<double>sample = random_sample(12);
        vector<double>solution(1,minsd(sample,0.12));
        muann.train(sample,solution);
        sample.push_back(solution[0]);
        cout<<"stds: "<<sv(sample)<<"reals:";
        sample[12]=muann.val[muann.result->index];
        cout<<sv(sample)<<endl;
        cout<<"<"<<solution[0]<<","<<muann.val[muann.result->index]<<">"<<endl;


    }
    muann.printDval();

}
