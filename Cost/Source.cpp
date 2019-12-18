#include<fstream>
#include<random>
#include<time.h>
#include<Windows.h>
#include<iostream>
using namespace std;

struct neuron {//нейрон
	double value;
	double error;
	void act() {//функция активации
		value = (1 / (1 + pow(2.71828, -value)));
	}
};

struct data_one {//данные для обучения
	double info[4096]; //"значения" изображение 64х64
	char rresult; //rightresult, нужно будет в обучении
};

struct network {
	int layers;//кол-во слоев
	int* size;//кол-во нейронов в слое
	neuron** neurons;//двумерный массив нейронов
	double*** weights;//веса нейронов([слой][номер нейрона][номер связи нейрона со следующим слоем])
};

network nn;

	double sigm_proizvodnaya(double x) {//производная функции активации, понадобится при ошмбках нейросети
		if ((fabs(x - 1) < 1e-9) || (fabs(x) < 1e-9)) return 0.0;
		double res = x * (1.0 - x);
		return res;
	}

	void setLayersNotStudy(int n, int* p, string filename) {//если не нужно обучение
		ifstream fin;
		fin.open(filename);//открываем файл и считываем оттуда веса
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

	void setLayers(int n, int* p) {//если нужно обучение
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
						nn.weights[i][j][k] = ((rand() % 100)) * 0.01 / nn.size[i];//присваиваем рандомные веса
					}
				}
			}
		}
	}

	void set_input(double p[]) {//принимает входные значения для нейросети (от 0 до 255(оттенки цвета лежат в таком диапозоне, в нашем случае- серый)) и присваивает их нейрону
		for (int i = 0; i < nn.size[0]; i++) {
			nn.neurons[0][i].value = p[i];
		}
	}

	void LayersCleaner(int LayerNumber, int stop) {//очищает слои
		srand(time(0));
		for (int i = 0; i < stop; i++) {
			nn.neurons[LayerNumber][i].value = 0;
		}
	}

	void ForwardFeeder(int LayerNumber, int stop) {//производит процесс ForwardFeed (разносидность нейросети, когда нейроны передают информацию от входа к выходу напрямую
		for (int j = 0; j < stop; j++) {
			for (int k = 0; k < nn.size[LayerNumber - 1]; k++) {
				nn.neurons[LayerNumber][j].value += nn.neurons[LayerNumber - 1][k].value * nn.weights[LayerNumber - 1][k][j];
			}
			nn.neurons[LayerNumber][j].act();
		}
	}

	double ForwardFeed() {//используется в обучении
		setlocale(LC_ALL, "ru");
		for (int i = 1; i < nn.layers; i++) {
					LayersCleaner(i, nn.size[i]);//очистка слоя
					ForwardFeeder(i, nn.size[i]);//"кормление" нейрона
		}
		double max = 0;
		double prediction = 0;
		for (int i = 0; i < nn.size[nn.layers - 1]; i++) {//высчитывает "вероятность" буквы (т.е. с каким шансом рисунок- это та или иная буква)

			if (nn.neurons[nn.layers - 1][i].value > max) {
				max = nn.neurons[nn.layers - 1][i].value;
				prediction = i;
			}
		}
		return prediction;
	}

	double ForwardFeed(string param) {//используется, когда начинается тест, выводит "шансы" букв на экран, аналогична по сути предыдущей функции
		setlocale(LC_ALL, "ru");
		for (int i = 1; i < nn.layers; i++) {
					LayersCleaner(i, nn.size[i]);
					ForwardFeeder(i, nn.size[i]);
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

	void BackPropogation(double prediction, double rresult, double lr) {//функция для работы с ошибками нейросети во время обучения (или неверного ответа) по методу обратного распространения ошибки
		for (int i = nn.layers - 1; i > 0; i--) {//все начинается с выходных нейронов, где происходит вычисление ошибки
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
				else { //далее это значение идет обратно к скрытым нейронам, где идет суммирование входящих ошибок
					for (int j = 0; j < nn.size[i]; j++) {
						double error = 0.0;
						for (int k = 0; k < nn.size[i + 1]; k++) {
							error += nn.neurons[i + 1][k].error * nn.weights[i][j][k];
						}
						nn.neurons[i][j].error = error;
					}
				}
		}
		for (int i = 0; i < nn.layers - 1; i++) {//каждый выходной нейрон меняет свои веса

				for (int j = 0; j < nn.size[i]; j++) {
					for (int k = 0; k < nn.size[i + 1]; k++) {
						nn.weights[i][j][k] += lr * nn.neurons[i + 1][k].error * sigm_proizvodnaya(nn.neurons[i + 1][k].value) * nn.neurons[i][j].value;//после вычисления ошибки происходит перевычисление весов
					}
				}
		}
	}

	bool SaveWeights() {//сохранение новых весов (используется в обучении или ошибке)
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
	const int l = 4;//layers-слой
	const int input_l = 4096;//изображение 64х64
	int size[l] = { input_l, 256, 64, 26 }; //4096 разделить на кол-во слоев в квадрате=256, а потом 256/4=64, а 26, т.к. букв в английском алфавите 26
	double input[input_l];//массив значений для нейронов
	char rresult; //right result
	double result; //результат, порлученный при обучении нейросети
	double ra = 0; //right answer- кол-во угаданных букв
	int maxra = 0; //max right answer. максимальное кол-во угаданных букв
	int maxraepoch = 0;//нужно будет для отслежки последней эпохи, где угадано максимальное кол-во букв
	const int n = 129;
	bool to_study = 0;
	cout << "Производить обучение?";
	cin >> to_study;

	data_one* data = new data_one[n];//сюда будем "пихать" "учительские" данные

	if (to_study) {
		fin.open("Lib.txt");//вот и "учитель"
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < input_l; j++) {
				fin >> data[i].info[j];//"пихаем"
			}
			fin >> data[i].rresult;
			data[i].rresult -= 65;//в конце есть буква, которая говорит, что за рисунок был считан- так нейросеть понмает, что ей только что "дали" и что выводить, когда "увидит" похожее. -65, т.к. в таблице ASCII у буквы А индекс 65
		}

		setLayers(l, size);//устанавливаем слои и нужные веса
		for (int e = 0; ra / n * 100 < 100; e++) { //e- epoch(эпоха). По сути, этот цикл будет "идти", пока точность нейросети не будет 100% в угадывании "учительских" букв

			ra = 0; //каждый раз обнуляем rightanswer для новой эпохи обучения

			for (int i = 0; i < n; i++) {

				for (int j = 0; j < input_l; j++) {//принимаем входные значения для нейросети(все то, что ранее считали, "кладем" в нейроны)
					input[j] = data[i].info[j];
				}
				rresult = data[i].rresult;//принимаем в массив верные значения
				set_input(input);
				result = ForwardFeed();//"кормим" нейроны
				if (result == rresult) {//если "кормление" дало эталонный результат, то буква угадана
					cout << "Угадал букву " << char(rresult + 65) << "\n";
					ra++;
				}
				else {
					BackPropogation(result, rresult, 0.5);//если получилась ошибка, то происходит корректировка значений весов
				}
			}

			cout << "Right answers: " << ra / n * 100 << "% \t Max RA: " << double(maxra) / n * 100 << "%(epoch " << maxraepoch << " )" << "\n";
			if (ra > maxra) {
				maxra = ra;
				maxraepoch = e;//тут и находим максимальное кол-во угаданных букв и эпоху, когда это произошло
			}
			if (maxraepoch < e - 250) {
				maxra = 0;
			}
		}
		if (SaveWeights()) {//переопределение весов в соответствии с обучением
			cout << "Веса сохранены!";
		}
	}
	else {//если не нужно обучение
		setLayersNotStudy(l, size, "Weights.txt");
	}
	fin.close();

	cout << "Начать тест:(1/0) ";
	bool to_start_test = 0;
	cin >> to_start_test;
	char right_res;
	if (to_start_test) {
		fin.open("Test.txt");//открываем "текстовое изображение" и отдаем нейронам
		for (int i = 0; i < input_l; i++) {
			fin >> input[i];
		}
		set_input(input);//принимаем значения "текстового изображения"
		result = ForwardFeed(string("show results"));//после "кормления" получаем возможный результат
		cout << "Я считаю, что это буква " << char(result + 65) << "\n\n";
		cout << "А какая это буква на самом деле?...";
		cin >> right_res;
		if (right_res != result + 65) {//если нейросеть не угадала букву, то меняем веса и сохраняем их
			cout << "Хорошо господин, исправляю ошибку!";
			BackPropogation(result, right_res - 65, 0.15);
			SaveWeights();
		}
	}
	return 0;
}
