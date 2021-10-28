#include <stdio.h>
#include <omp.h>
#include <iostream>

using namespace std;


double countFx(const double varA,const double varB, const double varC, const double x)
{
   return varA * x * x + varB * x + varC;
}

void countRectangleMethod(const double varA, const double varB, const double varC, const int countOfSections, const double x1, const double x2, const int numberOfThreads = 1, const bool useThreads = false)
{   
    double startOfCounting = omp_get_wtime();
    double result = 0.0; 
    double h = (x2 - x1) / countOfSections;
    #pragma omp parallel for reduction(+:result) num_threads(numberOfThreads) if (useThreads)
    for (int i = 0; i < countOfSections ; i++){
        double startOfSection = x1 + i * h;
        double sectionResult = countFx(varA, varB, varC, startOfSection);
        if ( i != countOfSections - 1){
            sectionResult *=  h; 
        } else {
            sectionResult *=  (x2 - startOfSection); 
        }
        result += sectionResult;
    }
    double timeOfCounting = omp_get_wtime() - startOfCounting;
    cout << "Obliczona wartosc calki: " << result << " przy pomocy " << numberOfThreads << " watkow" << endl << "Czas obliczen: " << timeOfCounting << endl;
}

void countRectangleMethodWithoutThreads(const double varA, const double varB, const double varC, const int countOfSections, const double x1, const double x2)
{   
    double startOfCounting = omp_get_wtime();
    double result = 0.0; 
    double h = (x2 - x1) / countOfSections;
    for (int i = 0; i < countOfSections ; i++){
        double startOfSection = i * h;
        double sectionResult = (varA * startOfSection * startOfSection + varB * startOfSection + varC);
        if ( i != countOfSections - 1){
            sectionResult *=  h; 
        } else {
            sectionResult *=  (x2 - startOfSection); 
        }
        result += sectionResult;
    }
    double timeOfCounting = omp_get_wtime() - startOfCounting;
    cout << "Obliczona wartosc calki: " << result << " przy pomocy 1 watku" << endl << "Czas obliczen: " << timeOfCounting << endl;

}


void countSimpsonMethod(const double varA, const double varB, const double varC, const int countOfSections, const double x1, const double x2, const int numberOfThreads = 1, const bool useThreads = false)
{
    double startOfCounting = omp_get_wtime();
    double firstSum = 0.0;
    double secondSum = 0.0;
    #pragma omp parallel for reduction(+:firstSum, secondSum) num_threads(numberOfThreads) if (useThreads)
    for(int i = 1; i <= countOfSections; i++)
    {   
        double x_i = x1 + (double)i/countOfSections * (x2 - x1);
        double m_i = ((x1 + (double)(i-1)/countOfSections * (x2 - x1)) + x_i)/2;
        if( i < countOfSections){
            firstSum += countFx(varA, varB, varC, x_i);
            secondSum += countFx(varA, varB, varC, m_i);
        }
        else {
            secondSum += countFx(varA, varB, varC, m_i);
        }
    }
    double result = (x2 - x1)/(6 * countOfSections) * (countFx(varA, varB, varC, x1) + countFx(varA, varB, varC, x2) + 2 * firstSum + 4 * secondSum);
    double timeOfCounting = omp_get_wtime() - startOfCounting;
    cout << "Obliczona wartosc calki: " << result << " przy pomocy " << numberOfThreads << " watkow" << endl << "Czas obliczen: " << timeOfCounting << endl;
}

int main()
{
    double varA, varB, varC, x1, x2;
    int countOfSections, numberOfThreads, metoda;
    cout << "Podaj wspolczynniki Ax^2 + Bx + C (liczby dziesietne rozdzielone spacjami np. 2.04 3.2 1.3)" << endl;
    cin >> varA >> varB >> varC; 
    cout << "Podaj granice calkowania x1, x2: " << endl;
    cin >> x1 >> x2; 
    cout << "Podaj liczbe przedzialow: " << endl;
    cin >> countOfSections;
    cout <<"Podaj liczbe watkow: "<< endl;
    cin >> numberOfThreads;
    cout << "Wybierz metode obliczen: " << endl;
    cout << "1. Metoda prostokatow" << endl;
    cout << "2. Metoda Simona" << endl;
    cin >> metoda;

    switch (metoda){
        case 1:
            countRectangleMethod(varA, varB, varC, countOfSections, x1, x2, numberOfThreads, true);
            countRectangleMethodWithoutThreads(varA, varB, varC, countOfSections, x1, x2);
            break; 
        case 2:
            countSimpsonMethod(varA, varB, varC, countOfSections, x1, x2, numberOfThreads, true);  
            countSimpsonMethod(varA, varB, varC, countOfSections, x1, x2);    
            break;
        default:
            cout << "Niepoprawny wybór " << endl;
    }

    return 0;
}

