#include <iostream>
#include <set>
#include <vector>
#include <algorithm>
#include <fstream>
#include <string>
#include <stdlib.h>
using namespace std;


/*
 we use an artificial data set to study PLA. The data set is in

 https://www.csie.ntu.edu.tw/~htlin/mooc/datasets/mlfound_math/hw1_15_train.dat

 Each line of the data set contains one (xn,yn) with xn∈ℝ4. The first 4 numbers of the line contains the components of xn orderly, the last number is yn.

 Please initialize your algorithm with w=0 and take sign(0) as −1.

 Implement a version of PLA by visiting examples in the naive cycle using the order of examples in the data set. Run the algorithm on the data set. What is the number of updates before the algorithm halts?
 */


int iteration=0;

int sign(vector<double> weight,vector<double> p){
    double temp=0;
    vector<double>::iterator i,j;
    for(i=weight.begin(),j=p.begin();i!=weight.end();i++,j++){
        temp+=(*i)*(*j);
    }
    if(temp>0){
        return 1;
    }else{
        return -1;
    }
}

void updateWeight(vector<double> &weight,vector<double> p,int s){
    vector<double>::iterator i,j;
    for(i=weight.begin(),j=p.begin();i!=weight.end();i++,j++){
        *i += (*j)*s;
    }
}

void PLA(vector<vector<double>> data,vector<double> &weight,vector<int> s){
    vector< vector<double> >::iterator i;
    vector<int>::iterator j;

    int end=0;
    while(1){
        for(i=data.begin(),j=s.begin();i!=data.end();i++,j++){
            if(sign(weight,*i) != *j){
                updateWeight(weight, *i,*j);
                iteration++;
                end=0;
            }
            end++;
        }
        if(end>s.size()){
            break;
        }else{
            i=data.begin();
            j=s.begin();
        }

    }
}


int main()
{
    int n,m;
    double temp;
    vector< vector<double> > datas;
    vector< vector<double> >::reverse_iterator j;
    vector<double> a;
    vector<double> weight;
    vector<int> s;

    cout<<"Please input how many n in one data?"<<endl;
    cout<<"data would be: n1, n2, n3, .... , label"<<endl;
    cin >> n;


    ifstream inputF;
    inputF.open("..\\data_practice\\PLA_train_data.txt",ios::in);
    if(!inputF){
        cout<<"read file error"<<endl;
        return 0;
    }

	while (!inputF.eof())
	{
        datas.push_back(a);
        j = datas.rbegin();
        (*j).push_back(1); //初始化data[0]=1
        for(int k=0;k<n;k++){
            inputF >> temp;
            (*j).push_back(temp);
        }
        inputF >> temp;
        s.push_back(temp);
    }
    inputF.close();


    weight.push_back(0); //初始化w[0]=0
    for(int i=0;i<n;i++){
        weight.push_back(0);
    }

    PLA(datas, weight,s);

	cout << "find a weight can linearly seperate datas" << endl;

    return 0;
}
