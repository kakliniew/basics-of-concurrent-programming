#include <iostream>
#include <mpi.h>
#include <cmath>
#include <vector>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <stdlib.h>


/* 
Program stworzony z zamysłem "uniwersalności" tzn. powinien poradzić sobie z każdym rozmiarem macierzy i z kazdym rozmiarem world_size>= 1"
przyjmuje pliki A.csv i B.csv - wazne jest zeby byly w UTF-8, nie UTF-8 BOM
sprawdzone dla macierzy wynikowej do 466x466

*/

using namespace std;

vector<float> readVectorFromFile(string filePath, int &colsOf, int &rowsOf){
    string rows, cols;
    string row, value;
    
    ifstream file(filePath.c_str());
    vector<float> matrix;
    if(file.is_open()){
        getline(file, rows);
        getline(file, cols);
        while(getline(file, row)){   
            std::stringstream ss(row);
            while(getline(ss, value, ';')){
                matrix.push_back(stod(value.c_str()));
            }
        }
    }
    else{
        cout<< filePath << " - Nie ma takiego pliku!" << endl;
        exit(1);
    }
    colsOf = stoi(cols.c_str());
    rowsOf = stoi(rows.c_str());
    file.close();
    return matrix;
}

void printMatrix(vector<float> matrix, int rowsMatrix, int colsMatrix, string matrixName){
    cout<<"Drukowanie macierzy " << matrixName <<endl;
    for ( int i = 0; i < rowsMatrix; i++){
        for (int j = 0; j < colsMatrix; j++)
        {
            cout<< matrix[i * colsMatrix + j] << " \t";
        }
        cout<<endl;
    }
}

void saveToFile(vector<float> resultMatrix, int rowsMatrix, int colsMatrix, string filePath){
    ofstream file(filePath.c_str());
    if(file.is_open()){
        file << fixed << setprecision(4);
        file << rowsMatrix << endl;
        file << colsMatrix << endl;
        for ( int i = 0; i < rowsMatrix; i++){
            for (int j = 0; j < colsMatrix; j++)
            {
                file<< resultMatrix[i * colsMatrix + j];
                if(j != colsMatrix - 1){
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

vector<float> multiplyMatrixes(vector<float> matrixA, int colsA, int rowsA, vector<float> matrixB, int colsB, int rowsB, int world_rank, int world_size){
    int matrixSize = world_rank == world_size - 1 ? rowsA * colsB - (rowsA * colsB / world_size) * (world_size - 1) : rowsA * colsB / world_size;
    vector<float> partialResultMatrix(matrixSize); // Przeniesienie inicjalizacji vektora przed funkcję(i pomiar czasu) i przekazanie stworzonego wektora przez referencję by skróciło czasy obliczen, jednak w przypadku rownoleglym jak i szeregowym, wiec bez znaczenia dla zadanego problemu.
    if(colsA == rowsB){ 
        int startOfCounting = world_rank == world_size - 1 ? rowsA * colsB  - matrixSize : matrixSize*world_rank;
        int endOfCounting = world_rank == world_size - 1 ? rowsA * colsB : startOfCounting+ matrixSize;
        for(int i = startOfCounting; i < endOfCounting; i++){
            float countedValue = 0;
            for(int k = 0; k < rowsB; k++)
            {
                countedValue += matrixA[(i/colsB)*colsA + k % colsA ] * matrixB[i % colsB + k * colsB ];    
              
            }
            partialResultMatrix[i % matrixSize  ] = countedValue;  
        }
    }
    else{
        cout << "Mnozenie macierzy jest mozliwe jesli macierz A ma tyle samo kolumn co macierz B ma wierszy." << endl;
        cout << "Macierz A kolumny: " << colsA << " / macierz B wiersze: " << rowsB << endl;
        exit(1);
    }
    return partialResultMatrix; 
}


struct Arguments{
    int cols;
    int rows;
};


int main()
{

    MPI_Init(NULL, NULL);
    
    MPI_Datatype matrixInfo;
    int blocklen[6] = {1,1};
    MPI_Aint array_of_displacements[] = {offsetof(Arguments, cols), offsetof(Arguments, rows)};
    MPI_Datatype type[6] = {MPI_INT, MPI_INT};
    
    MPI_Type_create_struct(2, blocklen, array_of_displacements,type , &matrixInfo); 
    MPI_Type_commit(&matrixInfo);
    MPI_Status status;

    


    int world_size;
    int tag = 5;
    int world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
   
    double processResult = 0;
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;   
    double timeStart;
    MPI_Get_processor_name(processor_name, &name_len);
    Arguments matrixAInfos, matrixBInfos;
    

    
    vector<float> matrixA; 
    vector<float> matrixB; 
    vector<float> resultMatrix;
    MPI_Barrier(MPI_COMM_WORLD);
    if (world_rank == 0) {
        matrixAInfos.cols = 0;
        matrixAInfos.rows = 0; 
        matrixBInfos.cols = 0;
        matrixBInfos.rows = 0;
        matrixA = readVectorFromFile("A.csv", matrixAInfos.cols, matrixAInfos.rows);
       // printMatrix(matrixA, matrixAInfos.rows, matrixAInfos.cols, "A");
        matrixB = readVectorFromFile("B.csv", matrixBInfos.cols, matrixBInfos.rows);
      //  printMatrix(matrixB, matrixBInfos.rows, matrixBInfos.cols, "B");
       
        timeStart = MPI_Wtime();
        MPI_Bcast(&matrixBInfos, 1, matrixInfo, 0, MPI_COMM_WORLD);
        MPI_Bcast(&matrixB.front(), matrixBInfos.cols * matrixBInfos.rows, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&matrixAInfos, 1, matrixInfo, 0, MPI_COMM_WORLD);
        MPI_Bcast(&matrixA.front(), matrixAInfos.cols * matrixAInfos.rows, MPI_FLOAT, 0, MPI_COMM_WORLD);
        resultMatrix.resize(matrixAInfos.rows * matrixBInfos.cols); 

    } else {
        MPI_Bcast(&matrixBInfos, 1, matrixInfo, 0, MPI_COMM_WORLD);
        matrixB.resize(matrixBInfos.cols * matrixBInfos.rows);
        MPI_Bcast(&matrixB.front(), matrixBInfos.cols * matrixBInfos.rows, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&matrixAInfos, 1, matrixInfo, 0, MPI_COMM_WORLD);
        matrixA.resize(matrixAInfos.cols * matrixAInfos.rows);
        MPI_Bcast(&matrixA.front(), matrixAInfos.cols * matrixAInfos.rows, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }
    vector<float> partialResults = multiplyMatrixes(matrixA, matrixAInfos.cols, matrixAInfos.rows, matrixB, matrixBInfos.cols, matrixBInfos.rows, world_rank, world_size);
    
    int othersNumOfCount = matrixAInfos.rows * matrixBInfos.cols / world_size;
    MPI_Gather(partialResults.data(), othersNumOfCount, MPI_FLOAT, resultMatrix.data(), othersNumOfCount, MPI_FLOAT, 0 , MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    int lastNumOfCount = matrixAInfos.rows * matrixBInfos.cols - (matrixAInfos.rows * matrixBInfos.cols / world_size) * (world_size - 1);
    
    /* 
    Funkcją gather są zbierane policzone dane od kazdego z procesow. W przypadku, gdy zadania nie da się podzielic po rowno dla kazdego z procesow
    np 16/3 to ponizej jest realizowane przeslanie od procesu "ostatniego" narmiaru, ktory on liczyl. 
    */

    if(lastNumOfCount != othersNumOfCount && (world_rank == 0 || world_rank == world_size-1)){ 
        int excessSize = lastNumOfCount - othersNumOfCount;
        vector<float> excessLast(excessSize);
        if(world_rank == world_size - 1) {
            MPI_Send(&partialResults[partialResults.size() - excessSize], excessSize, MPI_FLOAT, 0, tag, MPI_COMM_WORLD);

        }
        else if (world_rank == 0){
            MPI_Recv(excessLast.data(), excessSize, MPI_FLOAT, world_size - 1, tag, MPI_COMM_WORLD, &status);
            resultMatrix.insert(resultMatrix.end() - excessSize, excessLast.begin(), excessLast.end());
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    
    if (world_rank == 0) {
        double timeEnd = MPI_Wtime();
        //printMatrix(resultMatrix, matrixAInfos.rows, matrixBInfos.cols, "wynikowej");
        double timeParallel = timeEnd - timeStart;
        cout << "Wynik obliczen rownolegle obliczono w czasie: " << timeParallel << " sekund" << endl;
        timeStart = MPI_Wtime();
        multiplyMatrixes(matrixA, matrixAInfos.cols, matrixAInfos.rows, matrixB, matrixBInfos.cols, matrixBInfos.rows, world_rank, 1);
        timeEnd = MPI_Wtime();
        cout << "Wynik obliczen szeregowo obliczono w czasie: " << timeEnd - timeStart << " sekund" << endl;
        saveToFile(resultMatrix, matrixAInfos.rows, matrixBInfos.cols, "C_" + to_string(timeEnd - timeStart) + "_" + to_string(timeParallel) + ".csv");
    }
    MPI_Type_free(&matrixInfo);
    MPI_Finalize();
}
