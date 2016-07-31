#include"ANN.h"

int main(){
    ANN ann = ANN(4,4);
    //ann.layerize(1,0,-1);
    //ann.layerize(2,1,-1);
    //ann.layerize(3,2,-1);
    //ann.layerize(4,3,-1);
    //ann.configBPLayer();
    //ann.configBPBias();

    vector<double>x;
    vector<double>y;
    vector<double>xx;
    vector<double>yy;
    vector<double>xxx;
    vector<double>yyy;
    x.push_back(0.6);
    y.push_back(0.2);
    xx.push_back(0.17);
    yy.push_back(0.82);
    xxx.push_back(0.44);
    yyy.push_back(0.5);
    ann.load("ann.sav");
    ann.printVal();
    ann.printWeight();
    ann.printDval();
    
    /*for(size_t i=0;i<2000;i++){
        for(size_t j=0;j<200;j++){
        ann.train(x,y);
        ann.train(xx,yy);
        ann.train(xxx,yyy);
        }
    }
    ann.save("ann.sav");
    */
    ann.ff(x);
    ann.printVal();
    ann.ff(xx);
    ann.printVal();
    ann.ff(xxx);
    ann.printVal();
    /*for(map<string,double>::iterator it= ann.weight.begin();it!=ann.weight.end();it++)
        cout<<it->first<<":"<<it->second<<endl;
    ann.bpDEF(y);
    cout<<"====="<<endl;
    for(map<string,double>::iterator it= ann.weight.begin();it!=ann.weight.end();it++)
        cout<<it->first<<":"<<it->second<<endl;
    cout<<"====="<<endl;
    for(size_t i=0;i<5;i++)
        cout<<i<<"::"<<ann.dval[i]<<endl;
    for(size_t i=0;i<ann.layer.size();i++){
        for(size_t j=0;j<ann.layer[i].size();j++){
            cout<<"<"<<ann.layer[i][j]<<","<<ann.val[ann.layer[i][j]]<<">   ";
        }
        cout<<endl;
    }
    */
}
