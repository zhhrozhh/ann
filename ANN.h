#ifndef ANN_H
#define ANN_H
#include<iostream>
#include<cmath>
#include<vector>
#include<map>
#include<string>
#include<stdarg.h>
#include"funct.h"
using namespace std;
class ANNode{
    public:
        ANNode(size_t i):index(i),actfunc(new Sigmoid(1)){}
        ANNode(size_t i,FUNCT*f):index(i),actfunc(f){}
        size_t index;
        vector<size_t>i;
        vector<size_t>o;
        FUNCT*actfunc;
};
#define XKEY(a,b) to_string(a)+","+to_string(b)
class ANN{
    public:
        ANN(size_t ln,size_t nn){
            actf = new Sigmoid(1);
            val = vector<double>(nn+ln,0);
            dval = vector<double>(nn+ln,0);
            for(size_t i=0;i<=ln;i++)
                layer.push_back(vector<size_t>());
            for(size_t i=0;i<nn+ln;i++)
                pool.push_back(new ANNode(i,actf));
            for(size_t i=0;i<ln;i++){
                layer[0].push_back(i+nn);
                val[i+nn] = 1;
            }
            result = pool[nn-1];
            eta = 0.08;
            bias = 0;
        }
        

        void printDval(){
            cout<<"dval:"<<endl;
            for(size_t i=1;i<layer.size();i++){
                cout<<"layer #"<<i<<": ";
                for(size_t j=0;j<layer[i].size();j++)
                    cout<<layer[i][j]<<":<"<<dval[layer[i][j]]<<">  ";
            }
            cout<<endl;
            if(bias){
                cout<<"bias: ";
                for(size_t i=0;i<layer[0].size();i++)
                    cout<<layer[0][i]<<":<"<<dval[layer[0][i]]<<">  ";
                cout<<endl;
            }
        }
        void printVal(){
            cout<<"val:"<<endl;
            for(size_t i=1;i<layer.size();i++){
                cout<<"layer #"<<i<<": ";
                for(size_t j=0;j<layer[i].size();j++)
                    cout<<layer[i][j]<<":<"<<val[layer[i][j]]<<">  ";
            }
            cout<<endl;
            if(bias){
                cout<<"bias: ";
                for(size_t i=0;i<layer[0].size();i++)
                    cout<<layer[0][i]<<":<"<<val[layer[0][i]]<<">  ";
                cout<<endl;
            }
        }
        void printWeight(){
            for(map<string,double>::iterator it = weight.begin();it!=weight.end();it++){
                cout<<"/"<<it->first<<"/ "<<"weight:"<<it->second<<",  dweight"<<dweight[it->first]<<endl;
            }
        }
        void configBPLayer(){
            for(size_t i=1;i<layer.size()-1;i++)
                group(layer[i],layer[i+1]);
        }
        void configBPBias(){
            for(size_t i=1;i<layer[0].size();i++)
                group(layer[0][i-1],layer[i+1]);
            bias = 1;
        }
        void ff(vector<double>&x){
            for(size_t i=0;i<x.size();i++)
                val[layer[1][i]] = x[i];
            for(size_t i=2;i<layer.size();i++){
                for(size_t j=0;j<layer[i].size();j++){
                    double res = 0;
                    ANNode*node = pool[layer[i][j]];
                    for(size_t k=0;k<node->i.size();k++){
                        res += val[node->i[k]]*weight[XKEY(node->i[k],node->index)];
                    }
                    val[layer[i][j]]=node->actfunc->f(res);
                }
            }
        }
        void bpDEF(vector<double>&y){
            for(size_t i=0;i<y.size();i++){
                ANNode*node = pool[layer[layer.size()-1][i]];
                dval[node->index] = val[node->index]-y[i];
            }
            for(size_t i=layer.size()-2;i+1!=0;i--){
                for(size_t j=0;j<layer[i].size();j++){
                    ANNode*node = pool[layer[i][j]];
                    double dEdg=0;
                    double dEdw=0;
                    for(size_t k=0;k<node->o.size();k++){
                        ANNode*ak = pool[node->o[k]];
                        dEdg+=dval[node->o[k]] * weight[XKEY(node->index,ak->index)] * ak->actfunc->d(ak->actfunc->inv(val[node->o[k]]));
                        dEdw=dval[node->o[k]]*ak->actfunc->d(ak->actfunc->inv(val[node->o[k]]))*val[node->index];
                        dweight[XKEY(node->index,node->o[k])]=dEdw;
                    }
                    for(size_t k=0;k<node->o.size();k++)
                        weight[XKEY(node->index,node->o[k])] -= eta*dweight[XKEY(node->index,node->o[k])];
                    dval[node->index]=dEdg;
                }
            }
        }
        void train(vector<double>&x,vector<double>&y){
            ff(x);
            bpDEF(y);
        }
        void train(vector<double>&x,vector<double>&y,size_t times){
            for(size_t i=0;i<times;i++){
                ff(x);
                bpDEF(y);
            }
        }
        void layerize(size_t n,vector<size_t>&v){
            for(size_t i=0;i<v.size();i++)
                layer[i].push_back(v[i]);
        }
        void layerize(size_t n,int a,...){
            layer[n]=vector<size_t>();
            layer[n].push_back(a);
            va_list vp;
            va_start(vp,a);
            while(1){
                int v = va_arg(vp,int);
                if(v==-1)break;
                layer[n].push_back(v);
            }
            va_end(vp);
        }
        void group(vector<size_t>&f,vector<size_t>&t){
            for(size_t i=0;i<f.size();i++)
                for(size_t j=0;j<t.size();j++){
                    pool[f[i]]->o.push_back(t[j]);
                    pool[t[j]]->i.push_back(f[i]);
                    weight[XKEY(f[i],t[j])] = 0.5;
                }
        }
        void group(size_t f,vector<size_t>&t){
            for(size_t i=0;i<t.size();i++){
                pool[f]->o.push_back(t[i]);
                pool[t[i]]->i.push_back(f);
                weight[XKEY(f,t[i])]=0.5;
            }
        }
        void save(string fn){
        
        }
        ANNode*result;
        FUNCT*actf;
        vector<ANNode*>pool;
        vector<double>val;
        vector<double>dval;
        vector<vector<size_t> >layer;
        map<string,double>weight;
        map<string,double>dweight;
        double eta;
        bool bias;
};

#endif

