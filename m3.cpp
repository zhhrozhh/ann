#include"dist_gen.h"
int main(){
    DistGen dg = DistGen(30,0.5,0.11);
    vector<double>inp;
    for(size_t i=0;i<34;i++){
        inp.push_back(0.5);
    }
    dg.ann.ff(inp);
    cout.precision(10);
    for(size_t i=0;i<6000;i++)
        for(size_t j=0;j<6000;j++)
            dg.nd();
    for(size_t i=0;i<100;i++)
        cout<<dg.nd()<<"  ";
    
    cout<<endl;
    //dg.ann.printDval();
    //dg.ann.printWeight();
    cout<<dg.m()<<" "<<dg.s()<<endl;
}
