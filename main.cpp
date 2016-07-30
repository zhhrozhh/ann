#include"ANN.h"
double mean(vector<double>v){
    double res = 0;
    for(size_t i=0;i<v.size();i++)
        res += v[i];
    return res/(double)v.size();
}
double f(double a,double b,double c,double d){
    return 1/(1+exp(-(a/2+b/3+c/4+d/5)));
}
int main(){
    ANN x = ANN(4,20);
    x.layerize(1,0,1,2,3,-1);
    x.layerize(2,4,5,6,7,8,9,10,11,12,-1);
    x.layerize(3,13,14,15,16,17,18,-1);
    x.layerize(4,19,-1);
    x.configBPLayer();
    x.configBPBias();


    vector<vector<double> >sa;
    vector<vector<double> >so;
    vector<double>X;
    vector<double>Y;
    X.push_back(0);
    X.push_back(0);
    X.push_back(0);
    X.push_back(1);
    Y.push_back(0);


    sa.push_back(X);
    so.push_back(Y);

    X.clear();
    Y.clear();

    X.push_back(0);
    X.push_back(0);
    X.push_back(1);
    X.push_back(0);
    Y.push_back(0.2);


    sa.push_back(X);
    so.push_back(Y);

    X.clear();
    Y.clear();

    X.push_back(0);
    X.push_back(0);
    X.push_back(1);
    X.push_back(1);
    Y.push_back(0.4);


    sa.push_back(X);
    so.push_back(Y);

    X.clear();
    Y.clear();

    X.push_back(0);
    X.push_back(1);
    X.push_back(0);
    X.push_back(0);
    Y.push_back(0.6);

    sa.push_back(X);
    so.push_back(Y);

    for(size_t i=0;i<300;i++){
        size_t ind = rand()%sa.size();
        x.train(sa[ind],so[ind],10);
    }
    for(size_t i=0;i<x.layer.size();i++){
        for(size_t j=0;j<x.layer[i].size();j++){
            cout<<"<"<<x.layer[i][j]<<","<<x.val[x.layer[i][j]]<<">   ";
        }
        cout<<endl;
    }
    vector<double>XX;
    XX.push_back(0);
    XX.push_back(1);
    XX.push_back(0);
    XX.push_back(0);
    x.ff(XX);
    cout<<x.val[x.result->index]<<endl;

}
