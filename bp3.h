#ifndef BP3NNH
#define BP3NNH
#include"ann.h"
class BP3:public ANN{
    public:
        BP3(size_t in,size_t hide,size_t out):ANN(3,in+hide+out){
            vector<size_t>layer;
            for(size_t i=0;i<in;i++)
                layer.push_back(i);
            layerize(1,layer);
            layer.clear();
            for(size_t i=0;i<hide;i++)
                layer.push_back(in+i);
            layerize(2,layer);
            layer.clear();
            for(size_t i=0;i<out;i++)
                layer.push_back(in+hide+i);
            layerize(3,layer);
            configBPLayer();
            configBPBias();
        }
};


#endif

