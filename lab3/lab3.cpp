#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <iomanip>
#include <omp.h>
#include <chrono>
#include <ctime>    
#include <math.h>
#include <stdlib.h>



using namespace std;


vector<vector<double>> readVectorFromFile(string filePath){
    string rows;
    string row, value;
    ifstream file(filePath);
    vector<vector<double>> matrix;
    if(file.is_open()){
        getline(file, rows);
        while(getline(file, row)){   
            std::stringstream ss(row);
            vector<double> currentRowValues;
            while(getline(ss, value, ';')){
                currentRowValues.push_back(stod(value));
            }
            matrix.push_back(currentRowValues);
        }
    }
    else{
        cout<< filePath << " - Nie ma takiego pliku!" << endl;
        exit(1);
    }
    file.close();
    return matrix;
}

void findHalfPivot(vector<vector<double>> &matrix, int currentRow, int currentColumn){
    double max = fabs(matrix[currentRow][currentColumn]);
    int rowWithMax = currentRow;
    for(int i = currentRow; i < matrix.size(); i ++){
        if(fabs(matrix[i][currentColumn]) > max){
            max = matrix[i][currentColumn];
            rowWithMax = i;
        }
    }
    if(rowWithMax != currentRow){
        swap(matrix[currentRow], matrix[rowWithMax]);
    }
}

/*
Nie jest dla mnie zrozumiale, dlaczego dla niektorych danych uzycie tej funkcji sprawia, ze wyniki są inne. Czy to blad w implementacji, czy mojego zrozumienia
Prawdopodbnie w zlym momencie uwzgledniane jest zmienienie kolumn w algorytmie, ale nie potrafilem odnalezc poprawnego
*/
void findPivot(vector<vector<double>> &matrix, int currentRow, int currentColumn, vector<int> &changedColumns){ 
    double max = fabs(matrix[currentRow][currentColumn]);
    int rowWithMax = currentRow;
    int columnWithMax = currentColumn;
    for(int i = currentRow; i < matrix.size(); i ++){
        for(int j = currentColumn; j < matrix.size(); j ++){
            if(fabs(matrix[i][j]) > max){
                max = matrix[i][j];
                rowWithMax = i;
                columnWithMax = j;
            }
        }
    }
    if (max == 0) {
        cout<<"Znaleziony element o maksymalnym module jest równy 0 oznacza to, iż wyznacznik macierzy współczynników det 𝐴 = 0 i należy przerwać obliczenia." << endl;
        exit(1);
    }
    if(rowWithMax != currentRow){
        swap(matrix[currentRow], matrix[rowWithMax]);
        //  cout<< " Zamieniono wiersz " << currentRow << " z " << rowWithMax<< endl;
    }
    if(columnWithMax != currentColumn){
        for(int i = currentRow; i < matrix.size(); i ++){
            swap(matrix[i][currentColumn], matrix[i][columnWithMax]);
        }
        changedColumns.push_back(currentColumn);
        changedColumns.push_back(columnWithMax);
        // cout<< " Zamieniono kolumne " << currentColumn << " z " << columnWithMax<< endl;
        // cout<< " aktualny rozmiar changed " << changedColumns.size() << endl;

    }
}

void resolveChangedColumns(vector<double> &resultMatrix,vector<int> &changedColumns){
    int changedColumnsSize = changedColumns.size();
    // cout<< "rozmiar "<< changedColumnsSize - 1 << endl;
    for(int i = changedColumnsSize - 1; i >= 1; i = i-2){
        //   cout<<"Naprawiono "<< changedColumns[i] << " z " << changedColumns[i-1] << endl;
        swap(resultMatrix[changedColumns[i]], resultMatrix[changedColumns[i-1]]);
    }
}


void printMatrix(vector<vector<double>> matrix){
    cout<<"Drukowanie macierzy: " <<endl;
    for ( int i = 0; i < matrix.size(); i++){
        for (int j = 0; j < matrix[i].size(); j++)
        {
            cout<< matrix[i][j] << " \t";
        }
        cout<<endl;
    }
}

void gaussMethodPartOne(vector<vector<double>> &matrix, bool fullPivot, vector<int> &changedColumns, bool useThreads, int numberOfThreads){
    for(int r = 0; r < matrix.size() - 1; r++){
        // cout<<"drukowanie w gaus " << r << endl;
        // printMatrix(matrix);
        if(fullPivot) findPivot(matrix, r, r, changedColumns);
        else findHalfPivot(matrix, r, r);
        // cout<<"drukowanie w gaus po pivot " << r << endl;
        // printMatrix(matrix);
        #pragma omp parallel for collapse(2) if(useThreads) num_threads(numberOfThreads)
        for(int i = r + 1; i < matrix.size(); i++){
            for(int j = r + 1; j <= matrix.size(); j++){
                matrix[i][j] = matrix[i][j] - (matrix[i][r])/(matrix[r][r])*matrix[r][j];   
            }
        }

        // printMatrix(matrix);
        
    } 
}

vector<double> countValues(vector<vector<double>> &matrix){
    int valueN = matrix.size() - 1;
    vector<double> resultMatrix(valueN + 1, 0.0);
    resultMatrix[valueN] = matrix[valueN][valueN + 1]/matrix[valueN][valueN];
    for(int i = valueN - 1; i >= 0; i--){
        double s = 0;
        for(int r = i + 1; r <= valueN; r++){
            s += matrix[i][r] * resultMatrix[r];
            // cout <<" wartosc r " << r << endl;
        }
        // cout<<"wartosc i " << i << "wartosc xx " << matrix[i][valueN + 1] << " wartosc s " << s << " wartosc ii " << matrix[i][i] << endl;
        resultMatrix[i] = (matrix[i][valueN + 1] - s)/matrix[i][i];
    }
    return resultMatrix;
}


vector<double> gaussMethod(vector<vector<double>> &matrix, bool fullPivot, bool useThreads, int numberOfThreads){
    vector<int> changedColumns;
    gaussMethodPartOne(matrix, fullPivot, changedColumns, useThreads, numberOfThreads);
    vector<double> results = countValues(matrix);
    if(!changedColumns.empty()){
        cout<<" Naprawiono kolejnosc kolumn " << endl;
        resolveChangedColumns(results,  changedColumns);
    }
    for (std::vector<double>::const_iterator i = results.begin(); i != results.end(); ++i)
    std::cout << *i << ' ';
    cout<<endl;
    return results;
}

void saveResultsToFile(vector<double> resultMatrix, string filePath){
    ofstream file(filePath);
    if(file.is_open()){
        file << fixed << setprecision(6);
        file << resultMatrix.size() << endl;
        for ( int i = 0; i < resultMatrix.size(); i++){
                file<< resultMatrix[i];
                if(i != resultMatrix.size() - 1){
                    file << ";";
                }
        }
    }
    else{
        cout<< filePath << " - Problem z otworzeniem lub utworzeniem pliku" << endl;
        exit(1);
    }
    file.close();
}




void startSequenzAndParallelWithDefault(ofstream &dataLogger, vector<vector<double>> matrixA, int numberOfThreads, bool fullPivot)
{
    vector<vector<double>> copyMatrixA = matrixA;
    cout<<"Prosze czekac, obliczenia w trakcie"<<endl;
    double startOfCountingSequenz = omp_get_wtime();
    vector<double> results = gaussMethod(matrixA, fullPivot, false, numberOfThreads);
    double timeOfCountingSequenz = omp_get_wtime() - startOfCountingSequenz;
    double startOfCountingParallel = omp_get_wtime();
    gaussMethod(copyMatrixA, fullPivot, true, numberOfThreads);
    double timeOfCountingParallel = omp_get_wtime() - startOfCountingParallel;
    saveResultsToFile(results, "X_" + to_string(timeOfCountingSequenz) + "_"  + to_string(timeOfCountingParallel) +  ".csv" );
    time_t end_time = time(0);
    dataLogger << ctime(&end_time) << "Rozwiazywanie rownan z liczbą watkow czas szeregowo : " <<  to_string(timeOfCountingSequenz) << " czas rownolegle: " <<  to_string(timeOfCountingParallel) << endl;
    dataLogger << "Parametry zrownoleglenia:" << endl << "- liczba watkow " << numberOfThreads << endl;
    if(fullPivot) dataLogger << "Z pełnym wyborem elementu podstawowego" << endl;
    else dataLogger << "Z czesciowym wyborem elementu podstawowego" << endl;
    dataLogger << endl;
 }

int changeNumberOfThreads(){  // Brak walidacji
    int newNumber = 0;
    cout<<"Podaj nowa liczbe watkow"<<endl;
    cin>>newNumber;
    return newNumber;
}

int main(){
    ofstream dataLogger("Datalogger.txt");
    vector<vector<double>> matrixA = readVectorFromFile("A.csv");
    
    int menuOption = 0; 
    int numberOfThreads = 4;
    while(menuOption != 6)
    {
        cout<<"*******************MENU********************"<<endl;
        cout<<"1.Wersja sekwencyjna i rownolegla - czesciowy wybor elementu podstawowego"<<endl;
        cout<<"2.Wersja sekwencyjna i rownolegla - pelny wybor elementu podstawowego"<<endl;
        cout<<"3.Zmien liczbe watkow w uruchomieniu rownoleglym, aktualna: " << numberOfThreads << endl;
        
        cout<<"6.Zakoncz dzialanie programu"<<endl;
        cout<<"Wybieram : ";
        cin>>menuOption;
        system("cls");
        switch (menuOption)
        {
            case 1:
            {
                startSequenzAndParallelWithDefault(dataLogger,  matrixA, numberOfThreads, false);
                break;
            }
            case 2:
            {
                startSequenzAndParallelWithDefault(dataLogger,  matrixA, numberOfThreads, true);
                break;
            }    
            case 3:
            {
                numberOfThreads = changeNumberOfThreads();
                break;
            }
            case 6: 
                    break;  
            default:
                cout<<"Niepoprawny wybor" << endl;
                break;
        }
     }
  
    dataLogger.close();
    return 0;
}