#include <iostream>
#include <mpi.h>
#include <cmath>


using namespace std;

double countFx(const double varA,const double varB, const double varC, const double x)
{
   return varA * x * x + varB * x + varC;
}

double countRectangleMethod(const double varA, const double varB, const double varC, const int countOfSections, const double x1, const double x2, int world_size, int world_rank)
{   
    
    double result = 0.0; 
    double h = (x2 - x1) / countOfSections;
    int countOfSectionsForProcess = countOfSections/(world_size-1);
    int startOfCounting = (world_rank - 1) * countOfSectionsForProcess;
    int endOfCounting = world_rank == world_size - 1 ? countOfSections  : startOfCounting + countOfSectionsForProcess; 
    //cout<< "Obszar petli przejmowany: " << startOfCounting << " do " << endOfCounting << endl;
    for (int i = startOfCounting; i < endOfCounting ; i++){
        double startOfSection = x1 + i * h;
        double sectionResult = fabs(countFx(varA, varB, varC, startOfSection));
        if ( i != countOfSections - 1){
            sectionResult *=  h; 
        } else {
            sectionResult *=  (x2 - startOfSection); 
        }
        result += sectionResult;
    }
    return result;
}

struct Arguments{
    int numberOfParts;
    double a;
    double b;
    double c;
    double x1;
    double x2; 
};


int main()
{

    MPI_Init(NULL, NULL);
        
    

    MPI_Datatype transferObject;
    int blocklen[6] = {1,1,1,1,1,1};
    MPI_Aint array_of_displacements[] = {offsetof(Arguments, numberOfParts), offsetof(Arguments,a), offsetof(Arguments, b), offsetof(Arguments, c), offsetof(Arguments, x1), offsetof(Arguments, x2)};
    MPI_Datatype type[6] = {MPI_INT, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE};
    
    MPI_Type_create_struct(6, blocklen, array_of_displacements,type , &transferObject); 
    MPI_Type_commit(&transferObject);


    int world_size;
    int tag = 5;
    int world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    double result = 0;
    double processResult = 0;
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;   
    double time;
    MPI_Get_processor_name(processor_name, &name_len);
    Arguments argumenty;



    if (world_rank == 0) {
      //  cout << "Hello world from processor " << processor_name << ", ranking " << world_rank << " out of " << world_size << " processors\n";
        cout << "Podaj liczbę przedziałow " << endl;
        cin >> argumenty.numberOfParts;
        cout << "Podaj wspolczynniki Ax^2 + Bx + C (liczby dziesietne rozdzielone spacjami, kropka znak dziesietny np 2.04 3.2 1.3)" << endl;
        cin >> argumenty.a>>argumenty.b>>argumenty.c;
        cout << "Podaj granice całkowania x1, x2: " << endl; 
        cin >> argumenty.x1 >> argumenty.x2;
        time = MPI_Wtime();
        MPI_Bcast(&argumenty, 1, transferObject, 0, MPI_COMM_WORLD);
       
            
    } else {
        MPI_Bcast(&argumenty, 1, transferObject, 0, MPI_COMM_WORLD);
        processResult = countRectangleMethod(argumenty.a, argumenty.b, argumenty.c, argumenty.numberOfParts, argumenty.x1, argumenty.x2, world_size, world_rank);
        
       // cout << "Hello world from processor " << processor_name << ", ranking " << world_rank << " out of " << world_size << " processors, wynik: " << processResult << " result " << result << endl;
        
    }
    
    MPI_Reduce(&processResult, &result, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
   
    MPI_Barrier(MPI_COMM_WORLD);
    if (world_rank == 0) {
        time = MPI_Wtime() - time;
        cout << "Wynik obliczen rownolegle " << result << " obliczono w czasie: " << time << " sekund" << endl;
    
    
        time = MPI_Wtime();
        processResult = countRectangleMethod(argumenty.a, argumenty.b, argumenty.c, argumenty.numberOfParts, argumenty.x1, argumenty.x2, 2, 1);
        time = MPI_Wtime() - time;
        cout << "Wynik obliczen szeregowo" << result << " obliczono w czasie: " << time << " sekund" << endl;
    }
    MPI_Type_free(&transferObject);
    MPI_Finalize();
}

