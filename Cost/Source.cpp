#include<fstream>
#include<random>
#include<time.h>
#include<Windows.h>
#include<iostream>
using namespace std;

struct neuron {//������
	double value;
	double error;
	void act() {//������� ���������
		value = (1 / (1 + pow(2.71828, -value)));
	}
};

struct data_one {//������ ��� ��������
	double info[4096]; //"��������" ����������� 64�64
	char rresult; //rightresult, ����� ����� � ��������
};

struct network {
	int layers;//���-�� �����
	int* size;//���-�� �������� � ����
	neuron** neurons;//��������� ������ ��������
	double*** weights;//���� ��������([����][����� �������][����� ����� ������� �� ��������� �����])
};
network nn;

	double sigm_proizvodnaya(double x) {//����������� ������� ���������, ����������� ��� ������� ���������
		if ((fabs(x - 1) < 1e-9) || (fabs(x) < 1e-9)) return 0.0;
		double res = x * (1.0 - x);
		return res;
	}

	void setLayersNotStudy(int n, int* p, string filename) {//���� �� ����� ��������
		ifstream fin;
		fin.open(filename);//��������� ���� � ��������� ������ ����
		srand(time(0));
		nn.layers = n;
		nn.neurons = new neuron * [n];
		nn.weights = new double** [n - 1];
		nn.size = new int[n];
		for (int i = 0; i < n; i++) {
			nn.size[i] = p[i];
			nn.neurons[i] = new neuron[p[i]];
			if (i < n - 1) {
				nn.weights[i] = new double* [p[i]];
				for (int j = 0; j < p[i]; j++) {
					nn.weights[i][j] = new double[p[i + 1]];
					for (int k = 0; k < p[i + 1]; k++) {
						fin >> nn.weights[i][j][k];
					}
				}
			}
		}
	}

	void setLayers(int n, int* p) {//���� ����� ��������
		srand(time(0));
		nn.layers = n;
		nn.neurons = new neuron * [n];
		nn.weights = new double** [n - 1];
		nn.size = new int[n];
		for (int i = 0; i < n; i++) {
			nn.size[i] = p[i];
			nn.neurons[i] = new neuron[p[i]];
			if (i < n - 1) {
				nn.weights[i] = new double* [p[i]];
				for (int j = 0; j < p[i]; j++) {
					nn.weights[i][j] = new double[p[i + 1]];
					for (int k = 0; k < p[i + 1]; k++) {
						nn.weights[i][j][k] = ((rand() % 100)) * 0.01 / nn.size[i];//����������� ��������� ����
					}
				}
			}
		}
	}

	void set_input(double p[]) {//��������� ������� �������� ��� ��������� (�� 0 �� 255(������� ����� ����� � ����� ���������, � ����� ������- �����)) � ����������� �� �������
		for (int i = 0; i < nn.size[0]; i++) {
			nn.neurons[0][i].value = p[i];
		}
	}

	void LayersCleaner(int LayerNumber, int start, int stop) {//������� ����
		srand(time(0));
		for (int i = start; i < stop; i++) {
			nn.neurons[LayerNumber][i].value = 0;
		}
	}

	void ForwardFeeder(int LayerNumber, int start, int stop) {//���������� ������� ForwardFeed (������������� ���������, ����� ������� �������� ���������� �� ����� � ������ ��������
		for (int j = start; j < stop; j++) {
			for (int k = 0; k < nn.size[LayerNumber - 1]; k++) {
				nn.neurons[LayerNumber][j].value += nn.neurons[LayerNumber - 1][k].value * nn.weights[LayerNumber - 1][k][j];
			}
			nn.neurons[LayerNumber][j].act();
		}
	}

	double ForwardFeed() {//������������ � ��������
		setlocale(LC_ALL, "ru");
		for (int i = 1; i < nn.layers; i++) {
					LayersCleaner(i, 0, nn.size[i]);//������� ����
					ForwardFeeder(i, 0, nn.size[i]);//"���������" �������
		}
		double max = 0;
		double prediction = 0;
		for (int i = 0; i < nn.size[nn.layers - 1]; i++) {//����������� "�����������" ����� (�.�. � ����� ������ �������- ��� �� ��� ���� �����)

			if (nn.neurons[nn.layers - 1][i].value > max) {
				max = nn.neurons[nn.layers - 1][i].value;
				prediction = i;
			}
		}
		return prediction;
	}

	double ForwardFeed(string param) {//������������, ����� ���������� ����, ������� "�����" ���� �� �����, ���������� �� ���� ���������� �������
		setlocale(LC_ALL, "ru");
		for (int i = 1; i < nn.layers; i++) {
					LayersCleaner(i, 0, nn.size[i]);
					ForwardFeeder(i, 0, nn.size[i]);
		}
		double max = 0;
		double prediction = 0;
		for (int i = 0; i < nn.size[nn.layers - 1]; i++) {
			cout << char(i + 65) << " : " << nn.neurons[nn.layers - 1][i].value << "\n";
			if (nn.neurons[nn.layers - 1][i].value > max) {
				max = nn.neurons[nn.layers - 1][i].value;
				prediction = i;
			}
		}
		return prediction;
	}

	void BackPropogation(double prediction, double rresult, double lr) {//������� ��� ������ � �������� ��������� �� ����� �������� (��� ��������� ������) �� ������ ��������� ��������������� ������
		for (int i = nn.layers - 1; i > 0; i--) {//��� ���������� � �������� ��������, ��� ���������� ���������� ������
				if (i == nn.layers - 1) {
					for (int j = 0; j < nn.size[i]; j++) {
						if (j != int(rresult)) {
							nn.neurons[i][j].error = -pow((nn.neurons[i][j].value), 2);
						}
						else {
							nn.neurons[i][j].error = pow(1.0 - nn.neurons[i][j].value, 2);
						}
					}
				}
				else { //����� ��� �������� ���� ������� � ������� ��������, ��� ���� ������������ �������� ������
					for (int j = 0; j < nn.size[i]; j++) {
						double error = 0.0;
						for (int k = 0; k < nn.size[i + 1]; k++) {
							error += nn.neurons[i + 1][k].error * nn.weights[i][j][k];
						}
						nn.neurons[i][j].error = error;
					}
				}
		}
		for (int i = 0; i < nn.layers - 1; i++) {//������ �������� ������ ������ ���� ����

				for (int j = 0; j < nn.size[i]; j++) {
					for (int k = 0; k < nn.size[i + 1]; k++) {
						nn.weights[i][j][k] += lr * nn.neurons[i + 1][k].error * sigm_proizvodnaya(nn.neurons[i + 1][k].value) * nn.neurons[i][j].value;//����� ���������� ������ ���������� �������������� �����
					}
				}
		}
	}

	bool SaveWeights() {//���������� ����� ����� (������������ � �������� ��� ������)
		ofstream fout;
		fout.open("Weights.txt");
		for (int i = 0; i < nn.layers; i++) {
			if (i < nn.layers - 1) {
				for (int j = 0; j < nn.size[i]; j++) {
					for (int k = 0; k < nn.size[i + 1]; k++) {
						fout << nn.weights[i][j][k] << " ";
					}
				}
			}
		}
		fout.close();
		return 1;
	}





int main() {

	srand(time(0));
	setlocale(LC_ALL, "Russian");
	ifstream fin;
	const int l = 4;//layers-����
	const int input_l = 4096;//����������� 64�64
	int size[l] = { input_l, 256, 64, 26 }; //4096 ��������� �� ���-�� ����� � ��������=256, � ����� 256/4=64, � 26, �.�. ���� � ���������� �������� 26
	double input[input_l];//������ �������� ��� ��������
	char rresult; //right result
	double result; //���������, ����������� ��� �������� ���������
	double ra = 0; //right answer- ���-�� ��������� ����
	int maxra = 0; //max right answer. ������������ ���-�� ��������� ����
	int maxraepoch = 0;//����� ����� ��� �������� ��������� �����, ��� ������� ������������ ���-�� ����
	const int n = 129;
	bool to_study = 0;
	cout << "����������� ��������?";
	cin >> to_study;

	data_one* data = new data_one[n];//���� ����� "������" "�����������" ������

	if (to_study) {
		fin.open("Lib.txt");//��� � "�������"
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < input_l; j++) {
				fin >> data[i].info[j];//"������"
			}
			fin >> data[i].rresult;
			data[i].rresult -= 65;//� ����� ���� �����, ������� �������, ��� �� ������� ��� ������- ��� ��������� �������, ��� �� ������ ��� "����" � ��� ��������, ����� "������" �������. -65, �.�. � ������� ASCII � ����� � ������ 65
		}

		setLayers(l, size);//������������� ���� � ������ ����
		for (int e = 0; ra / n * 100 < 100; e++) { //e- epoch(�����). �� ����, ���� ���� ����� "����", ���� �������� ��������� �� ����� 100% � ���������� "�����������" ����

			ra = 0; //������ ��� �������� rightanswer ��� ����� ����� ��������

			for (int i = 0; i < n; i++) {

				for (int j = 0; j < input_l; j++) {//��������� ������� �������� ��� ���������(��� ��, ��� ����� �������, "������" � �������)
					input[j] = data[i].info[j];
				}
				rresult = data[i].rresult;//��������� � ������ ������ ��������
				set_input(input);
				result = ForwardFeed();//"������" �������
				if (result == rresult) {//���� "���������" ���� ��������� ���������, �� ����� �������
					cout << "������ ����� " << char(rresult + 65) << "\n";
					ra++;
				}
				else {
					BackPropogation(result, rresult, 0.5);//���� ���������� ������, �� ���������� ������������� �������� �����
				}
			}

			cout << "Right answers: " << ra / n * 100 << "% \t Max RA: " << double(maxra) / n * 100 << "%(epoch " << maxraepoch << " )" << "\n";
			if (ra > maxra) {
				maxra = ra;
				maxraepoch = e;//��� � ������� ������������ ���-�� ��������� ���� � �����, ����� ��� ���������
			}
			if (maxraepoch < e - 250) {
				maxra = 0;
			}
		}
		if (SaveWeights()) {//��������������� ����� � ������������ � ���������
			cout << "���� ���������!";
		}
	}
	else {//���� �� ����� ��������
		setLayersNotStudy(l, size, "Weights.txt");
	}
	fin.close();

	cout << "������ ����:(1/0) ";
	bool to_start_test = 0;
	cin >> to_start_test;
	char right_res;
	if (to_start_test) {
		fin.open("Test.txt");//��������� "��������� �����������" � ������ ��������
		for (int i = 0; i < input_l; i++) {
			fin >> input[i];
		}
		set_input(input);//��������� �������� "���������� �����������"
		result = ForwardFeed(string("show results"));//����� "���������" �������� ��������� ���������
		cout << "� ������, ��� ��� ����� " << char(result + 65) << "\n\n";
		cout << "� ����� ��� ����� �� ����� ����?...";
		cin >> right_res;
		if (right_res != result + 65) {//���� ��������� �� ������� �����, �� ������ ���� � ��������� ��
			cout << "������ ��������, ��������� ������!";
			BackPropogation(result, right_res - 65, 0.15);
			SaveWeights();
		}
	}

	return 0;
}