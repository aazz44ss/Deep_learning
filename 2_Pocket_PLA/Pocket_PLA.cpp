#include <iostream>
#include <set>
#include <vector>
#include <algorithm>
#include <fstream>
#include <string>
#include <stdlib.h>
using namespace std;


/*
we play with the pocket algorithm. Modify your PLA in Question 16 to visit examples purely randomly, and then add the "pocket" steps to the algorithm. We will use

https://www.csie.ntu.edu.tw/~htlin/mooc/datasets/mlfound_math/hw1_18_train.dat

as the training data set ��, and

https://www.csie.ntu.edu.tw/~htlin/mooc/datasets/mlfound_math/hw1_18_test.dat

as the test set for "verifying'' the g returned by your algorithm (see lecture 4 about verifying). The sets are of the same format as the previous one. Run the pocket algorithm with a total of 50 updates on �� , and verify the performance of wPOCKET using the test set. Please repeat your experiment for 2000 times, each with a different random seed. What is the average error rate on the test set?


data��400��

*/


/*
pocket���
�C��s�@��weight,�N�����f�U����weight���Ӧn
*/


int sign(vector<double> &weight, vector<double> &p) {
	double temp = 0;
	vector<double>::iterator i, j;
	for (i = weight.begin(), j = p.begin(); i != weight.end(); i++, j++) {
		temp += (*i)*(*j);
	}
	if (temp>0) {
		return 1;
	}
	else {
		return -1;
	}
}

void updateWeight(vector<double> &weight, vector<double> &p, int s) {
	vector<double>::iterator i, j;
	for (i = weight.begin(), j = p.begin(); i != weight.end(); i++, j++) {
			*i += (*j)*s;
	}
}

//�έpweight�U���h��error
int getError(vector<vector<double>> &datas, vector<double> &weight, vector<int> &s) {
	int errorNumber = 0;
	vector< vector<double> >::iterator i;
	vector<int>::iterator j;

	for (i = datas.begin(), j = s.begin(); i != datas.end(); i++, j++) {
		if (sign(weight, *i) != *j) {
			errorNumber++;
		}
	}
	return errorNumber;
}

vector<int> predict(vector<vector<double>> &datas, vector<double> &weight) {
	vector< vector<double> >::iterator i;
	vector<int> prediction;
	for (i = datas.begin(); i != datas.end(); i++) {
		if (sign(weight, *i) == 1) {
			prediction.push_back(1);
		}
		else {
			prediction.push_back(0);
		}
	}
	return prediction;
}


vector<double> pocketPLA(int pocketNumber, vector<vector<double>> &datas, vector<double> &weight, vector<int> &s) {

	vector< vector<double> >::iterator i;
	vector<int>::iterator j;
	vector<double> pocketWeight;
	int pocketError, currentError;
	pocketWeight = weight;
	pocketError = getError(datas, weight, s);

	int end = 0;
	for (int m = 0; m<pocketNumber; m++) {
		for (i = datas.begin(), j = s.begin(); i != datas.end(); i++, j++) {
			if (sign(weight, *i) != *j) {
				updateWeight(weight, *i, *j);
				
				end = 0;

				//���current�Mpocket���Ӥ���n
				currentError = getError(datas, weight, s);
				if (currentError < pocketError) {
					pocketError = currentError;
					pocketWeight = weight;
				}
			}
			end++;
		}
		if (end>s.size()) {
			break;
		}
		else {
			i = datas.begin();
			j = s.begin();
		}
	}
	return pocketWeight;
}


int main()
{
	int n, o, errorNumber;
	double temp;
	vector< vector<double> > datas;
	vector< vector<double> > testDatas;
	vector< vector<double> >::reverse_iterator j;
	vector<double> a;
	vector<double> weight;
	vector<double> pocketWeight;
	vector<int> s;//sign data
	vector<int> testSign;

	
	cout << "Please input how many n in one data?" << endl;
	cout << "data would be: n1, n2, n3, .... , label" << endl;
	cin >> n;
	cout << "Please input pocket numbers" << endl;
	cin >> o;
	
	ifstream inputTrain,inputTest;


	//read train data
	inputTrain.open("..\\data_practice\\Pocket_train_data.txt", ios::in);
	if (!inputTrain) {
		cout << "train data file error" << endl;
		return 0;
	}
	while (!inputTrain.eof())
	{
		datas.push_back(a);
		j = datas.rbegin();
		(*j).push_back(1); //initial data[0]=1 as threshold
		for (int k = 0; k<n; k++) {
			inputTrain >> temp;
			(*j).push_back(temp);
		}
		inputTrain >> temp;
		s.push_back(temp);
	}
	datas.pop_back(); //clear ghost data owing to '\n'
	s.pop_back();
	inputTrain.close();

	//initial weight
	weight.push_back(0); //initial w[0]=0 as threshold
	for (int i = 0; i<n; i++) {
		weight.push_back(0);
	}

	pocketWeight = pocketPLA(o, datas, weight, s);
	errorNumber = getError(datas, pocketWeight, s);


	//read test data
	inputTest.open("..\\data_practice\\Pocket_test_data.txt", ios::in);
	if (!inputTest) {
		cout << "test data file error" << endl;
		return 0;
	}
	while (!inputTest.eof())
	{
		testDatas.push_back(a);
		j = testDatas.rbegin();
		(*j).push_back(1); //initial data[0]=1 as threshold
		for (int k = 0; k<n; k++) {
			inputTest >> temp;
			(*j).push_back(temp);
		}
//		inputTest >> temp;      // If test data have y
//		testSign.push_back(temp);
	}
	testDatas.pop_back(); //clear ghost data owing to '\n'
//	testSign.pop_back();
	inputTest.close();



//	errorNumber = getError(testDatas, pocketWeight, testSign);
	vector<int> prediction;
	prediction = predict(testDatas, pocketWeight);

	return 0;
}
