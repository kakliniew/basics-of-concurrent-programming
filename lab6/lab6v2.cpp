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

vector<float> readVectorFromFile(string filePath){
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

vector<float> multiplyMatrixes(vector<float> matrixA, vector<float> matrixB){
    vector<float> partialResultMatrix(400*100); 
    for(int i = 0; i < 400*100; i++){
        float countedValue = 0;
        for(int k = 0; k < 400; k++)
        {
            countedValue += matrixA[(i/400)*400 + k % 400 ] * matrixB[i % 400 + k * 400 ];    
            
        }
        partialResultMatrix[i % 400  ] = countedValue;  
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
    
    MPI_Status status;

    


    int world_size;
    int tag = 2;
    int world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
   

    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;   
    MPI_Get_processor_name(processor_name, &name_len);
    
    

    
    vector<float> matrixA;
    vector<float> matrixB; 
    vector<float> resultMatrix;
    vector<float> partOfA(400*100);
    MPI_Barrier(MPI_COMM_WORLD);
    if (world_rank == 0) {
         
        matrixA = readVectorFromFile("A.csv");

        matrixB = readVectorFromFile("B.csv");
    
        MPI_Bcast(&matrixB.front(), 400*400 , MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Scatter(&matrixA.front(), 400*100, MPI_FLOAT, &partOfA.front(), 400*100, MPI_FLOAT, 0, MPI_COMM_WORLD);
        resultMatrix.resize(400*400); 

    } else {
        
        matrixB.resize(400*400);
        MPI_Bcast(&matrixB.front(), 400*400 , MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Scatter(&matrixA.front(), 400*100, MPI_FLOAT, &partOfA.front(), 400*100, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }
    vector<float> partialResults = multiplyMatrixes(partOfA, matrixB);
    
    
    MPI_Gather(partialResults.data(), 400*100, MPI_FLOAT, &resultMatrix.front(), 400*100, MPI_FLOAT, 0 , MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
   
    
    
    if (world_rank == 0) {
        
    
        multiplyMatrixes(matrixA, matrixB);
        
        saveToFile(resultMatrix, 400, 400, "newfile.csv");
    }
    MPI_Finalize();
}
