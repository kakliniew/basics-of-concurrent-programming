#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <iomanip>
#include <omp.h>
#include <chrono>
#include <ctime>    
#include <stdlib.h>



using namespace std;

int setenv(const char *name, const char *value, int overwrite)  //funkcja z internetu zeby wprowadzic ciekawa zmiana taktyki schedule runtime
{
    int errcode = 0;
    if(!overwrite) {
        size_t envsize = 0;
        errcode = -1;
        if(errcode || envsize) return errcode;
    }
    return _putenv_s(name, value);
}

vector<vector<float>> readVectorFromFile(string filePath){
    string rows, cols;
    string row, value;
    ifstream file(filePath);
    vector<vector<float>> matrix;
    if(file.is_open()){
        getline(file, rows);
        getline(file, cols);
        while(getline(file, row)){   
            std::stringstream ss(row);
            vector<float> currentRowValues;
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

void printMatrix(vector<vector<float>> matrix){
    cout<<"Drukowanie macierzy: " <<endl;
    for ( int i = 0; i < matrix.size(); i++){
        for (int j = 0; j < matrix[i].size(); j++)
        {
            cout<< matrix[i][j] << " \t";
        }
        cout<<endl;
    }
}

vector<vector<float>> multiplyMatrixes(vector<vector<float>> matrixA, vector<vector<float>> matrixB, int numberOfThreads, int collapseValue, bool useThreads, int scheduleValue, bool useNestedThreads){
    int rowsResult = matrixA.size();
    int colsResult = matrixB[0].size();
    vector<vector<float>> resultMatrix(rowsResult, vector<float>(colsResult, 0.0)); // Przeniesienie inicjalizacji vektora przed funkcję(i pomiar czasu) i przekazanie stworzonego wektora przez referencję by skróciło czasy obliczen, jednak w przypadku rownoleglym jak i szeregowym, wiec bez znaczenia dla zadanego problemu.
    if(matrixA[0].size() == matrixB.size()){
        /* 
            Klauzula collapse pozwala na lepsze rozdystrybuowanie zadan na wątki. W przypadku gdy zadanie jest dzielone tylko na kolejne iteracje zewnętrznej petli, to "pojedyncze zadania" są spore i jesli jedno z nich
            zajmie wiecej czasu niz inne, to pozostale wątki bedą czekać na zakonczenie. Collapse pozwala na podzielenie "pojedynczych zadan" na jeszcze drobniejsze, dzięki czemu dużo latwiej wyrownac czas pracy kazdego z watkow. 
            Dodatkowo w przypadku, gdy zewnętrzna petla posiadałaby mniej iteracji niż jest watkow, to nadwyżka by sie marnowała. Najlepszym przykladem do zastosowania collapse jest zagniezdzona petla, gdzie kolejne wiersze mają rozne liczby obiektow.
            Niestety klauzula collapse nie pozwala na uzycie parametru, przez co konieczne bylo rozwazenie roznych przypadkow, przy pomocy if,else lub osobnych funkcji, co nie jest zbyt ladne. 

            Klauzula schedule pozwala wybranie taktyki przydzielania realizowanych iteracji dla kazdego z watkow. Dopasowany tryb do zadania może przyspieszyć obliczenia. Podzial static jest szybszy od dynamic, poniewaz w tym drugim potrzebne są zasoby dodatkowe i czas na przydzielenie dynamiczne.
            Jedną z zalet podobnie jak przy pomocy collapse mozemy uzyskac lepsze rownowazenie obciazenia na kazdy z watkow. 

            Zagniezdzone petle jest trochę niebezpieczne, bo tworzy numberOfThreads^2 watkow, przez co watkow moze byc duzo wiecej niz iteracji w petlach. Bezpieczniejsze rozwiazanie wydaje sie byc collapse, bo chyba zawiera zalety zagniezdzania petli, a nie zawiera wad.
        */
        if(useNestedThreads){
            #pragma omp parallel for num_threads(numberOfThreads) if (useThreads) collapse(1) schedule(runtime) // Nie jest konieczna klauzula shared(resultMatrix) poniewaz każdy z wątkow pracuje na "swojej komorce"
            for(int i = 0; i < matrixA.size(); i++)
            {
                #pragma omp parallel for
                for(int j = 0; j < matrixB[0].size(); j++){
                    float countedValue = 0;
                    for(int k = 0; k < matrixB.size(); k++)
                    {
                        countedValue += matrixA[i][k] * matrixB[k][j];
                    }
                    resultMatrix[i][j] = countedValue;  
                    /*
                        Pierwotna wersja programu zamiast przypisania wartości do komórki w wektorze posiadała resultMatrix.push_back(counterValue), jednak to rozwiazanie było wadliwe, 
                        bo wymagało synchronizacji wątków i uzycia "ordered". Takie podejscie sprawiało, ze wątki musialy czekac i czesto rownolegle uruchomienie nie bylo szybsze niz szeregowe.
                    */
                }
            }
        }else if (collapseValue == 2) {
            #pragma omp parallel for num_threads(numberOfThreads) if (useThreads) collapse(2) schedule(runtime) // Nie jest konieczna klauzula shared(resultMatrix) poniewaz każdy z wątkow pracuje na "swojej komorce"
             for(int i = 0; i < matrixA.size(); i++)
            {
                for(int j = 0; j < matrixB[0].size(); j++){
                    float countedValue = 0;
                    for(int k = 0; k < matrixB.size(); k++)
                    {
                        countedValue += matrixA[i][k] * matrixB[k][j];
                    }
                    resultMatrix[i][j] = countedValue;  
                    /*
                        Pierwotna wersja programu zamiast przypisania wartości do komórki w wektorze posiadała resultMatrix.push_back(counterValue), jednak to rozwiazanie było wadliwe, 
                        bo wymagało synchronizacji wątków i uzycia "ordered". Takie podejscie sprawiało, ze wątki musialy czekac i czesto rownolegle uruchomienie nie bylo szybsze niz szeregowe.
                    */
                }
            }
        } else {
            #pragma omp parallel for num_threads(numberOfThreads) if (useThreads) collapse(1) schedule(runtime) // Nie jest konieczna klauzula shared(resultMatrix) poniewaz każdy z wątkow pracuje na "swojej komorce"
            for(int i = 0; i < matrixA.size(); i++)
            {
                for(int j = 0; j < matrixB[0].size(); j++){
                    float countedValue = 0;
                    for(int k = 0; k < matrixB.size(); k++)
                    {
                        countedValue += matrixA[i][k] * matrixB[k][j];
                    }
                    resultMatrix[i][j] = countedValue;  
                    /*
                        Pierwotna wersja programu zamiast przypisania wartości do komórki w wektorze posiadała resultMatrix.push_back(counterValue), jednak to rozwiazanie było wadliwe, 
                        bo wymagało synchronizacji wątków i uzycia "ordered". Takie podejscie sprawiało, ze wątki musialy czekac i czesto rownolegle uruchomienie nie bylo szybsze niz szeregowe.
                    */
                }
            }

        }

        
    }
    else{
        cout << "Mnozenie macierzy jest mozliwe jesli macierz A ma tyle samo kolumn co macierz B ma wierszy." << endl;
        cout << "Macierz A kolumny: " << matrixA[0].size() << " / macierz B wiersze: " << matrixB.size() << endl;
        exit(1);
    }

    return resultMatrix; 
}

void saveToFile(vector<vector<float>> resultMatrix, string filePath){
    ofstream file(filePath);
    if(file.is_open()){
        file << fixed << setprecision(6);
        file << resultMatrix.size() << endl;
        file << resultMatrix[0].size() << endl;
        for ( int i = 0; i < resultMatrix.size(); i++){
            for (int j = 0; j < resultMatrix[i].size(); j++)
            {
                file<< resultMatrix[i][j];
                if(j != resultMatrix[i].size() - 1){
                    file << ";";
                }
            }
            file<<endl;
        }
    }
    else{
        cout<< filePath << " - Problem z otworzeniem lub utworzeniem pliku" << endl;
        exit(1);
    }
    file.close();
}


void startSequenzAndParallelWithDefault(ofstream &dataLogger, vector<vector<float>> matrixA, vector<vector<float>> matrixB, int numberOfThreads, int collapseValue, int scheduleValue, bool useNestedThreads)
{
    cout<<"Prosze czekac, obliczenia w trakcie"<<endl;
    double startOfCountingSequenz = omp_get_wtime();
    vector<vector<float>> multipliedMatrix = multiplyMatrixes(matrixA, matrixB, 1, false, collapseValue, scheduleValue, false);
    double timeOfCountingSequenz = omp_get_wtime() - startOfCountingSequenz;
    double startOfCountingParallel = omp_get_wtime();
    vector<vector<float>> multipliedMatrixButParallel = multiplyMatrixes(matrixA, matrixB, numberOfThreads, true, collapseValue, scheduleValue, useNestedThreads);
    double timeOfCountingParallel = omp_get_wtime() - startOfCountingParallel;
   // printMatrix(multipliedMatrix);
    saveToFile(multipliedMatrix, "C_" + to_string(timeOfCountingSequenz) + "_"  + to_string(timeOfCountingParallel) +  ".csv" );
    time_t end_time = time(0);
    dataLogger << ctime(&end_time) << "Mnożenie macierzy A[" << matrixA.size() << ", " << matrixA[0].size() << "] i B[" << matrixB.size() << ", " << matrixB[0].size() << "] czas szeregowo : " <<  to_string(timeOfCountingSequenz) << " czas rownolegle: " <<  to_string(timeOfCountingParallel) << endl;
    dataLogger << "Parametry zrownoleglenia:" << endl << "- liczba watkow " << numberOfThreads << endl << "- wartosc collapse " << collapseValue << endl;
    dataLogger << "- aktualna taktyka schedule " << getenv("OMP_SCHEDULE")<< endl << "- uzycie zagniezdzonej rownoleglosci " << useNestedThreads << endl;
}

int changeNumberOfThreads(){  // Brak walidacji
    int newNumber = 0;
    cout<<"Podaj nowa liczbe watkow"<<endl;
    cin>>newNumber;
    return newNumber;
}

int changeStaticValue(){  // Brak walidacji
    int newNumber = 0;
    cout<<"Podaj nowa liczbe, wartosc schedule"<<endl;
    cin>>newNumber;
    return newNumber;
}

void setScheduleTactic(){
    int menuOption;
    cout<<"Wybierz taktykę " << endl;
    cout<<"1. Guided " << endl;
    cout<<"2. Static " << endl; 
    cout<<"3. Dynamic "<<endl;
    cout<<"4. Auto "<<endl;
    cin >> menuOption;

    switch (menuOption)
        {
            case 1:
            {
                setenv("OMP_SCHEDULE", "guided", 1);
                break;
            }
            case 2:
            {
                setenv("OMP_SCHEDULE", "static", 1);
                break;
            }
            case 3:
            {
                setenv("OMP_SCHEDULE", "dynamic", 1);
                break;
            }
            case 4:
            {   
                setenv("OMP_SCHEDULE", "auto", 1);
                break;
            }
            default: 
                cout<<"Niepoprawny wybor" << endl;
                break;
        }
   
}

int main(){
    ofstream dataLogger("Datalogger.txt");
    vector<vector<float>> matrixA = readVectorFromFile("fA2.csv");
    vector<vector<float>> matrixB = readVectorFromFile("fB2.csv");
    int numberOfThreads = 4;
    int scheduleStaticValue = 5;
    string scheduleTactic("static, ");
    scheduleTactic.append(to_string(scheduleStaticValue));
    const char *formatCorrectScheduleTactic = scheduleTactic.c_str(); 
    setenv("OMP_SCHEDULE", formatCorrectScheduleTactic ,1);
    int menuOption = 0; 
    while(menuOption != 6)
    {
        cout<<"*******************MENU********************"<<endl;
        cout<<"1.Wersja sekwencyjna i rownolegla schedule static"<<endl;
        cout<<"2.Zmien liczbe watkow w uruchomieniu rownoleglym, aktualna: " << numberOfThreads << endl;
        cout<<"3.Uruchom przy pomocy petli zagniezdzonej"<< endl;
        cout<<"4.Zmien aktualna wartosc dla schedule static, aktualna: " << scheduleStaticValue << endl;
        cout<<"5.Zmien taktyke schedule i uruchom , aktualna: " << getenv("OMP_SCHEDULE") << endl; 
        cout<<"6.Zakoncz dzialanie programu"<<endl;
        cout<<"Wybieram : ";
        cin>>menuOption;
        system("cls");
        switch (menuOption)
        {
            case 1:
            {
                string scheduleTactic("static, ");
                scheduleTactic.append(to_string(scheduleStaticValue));
                const char *formatCorrectScheduleTactic = scheduleTactic.c_str();
                setenv("OMP_SCHEDULE", formatCorrectScheduleTactic ,1);
            
                startSequenzAndParallelWithDefault(dataLogger, matrixA, matrixB, numberOfThreads, 2, scheduleStaticValue, false);
                break;
            }
            case 2:
            {
                numberOfThreads = changeNumberOfThreads();
                break;
            }    
            case 3:
            {
                startSequenzAndParallelWithDefault(dataLogger, matrixA, matrixB, numberOfThreads, 1, scheduleStaticValue, true);
                break;
            }
            case 4:          
            {
                scheduleStaticValue = changeStaticValue();
                break;
            }         
            case 5:    
            {
                setScheduleTactic();
                startSequenzAndParallelWithDefault(dataLogger, matrixA, matrixB, numberOfThreads, 2, scheduleStaticValue, false);
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