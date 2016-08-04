#include"bp3.h"
#include"stt.h"
int main(){
    BP3 bp3(5,8,1);
    for(size_t i=0;i<100;i++){
        vector<double>sample = random_sample(5);
        vector<double>solution(1,minsd(sample,0.2));
        bp3.train(sample,solution,20);
    }
    bp3.ff(0.33,0.46,0.15,0.53,0.22);
    cout<<bp3.val[13]<<" "<<endl;
}
