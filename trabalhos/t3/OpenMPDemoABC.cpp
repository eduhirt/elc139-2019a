#include <iostream>
#include <string>
#include <omp.h>
#include <algorithm>

using namespace std;

class SharedArray{
    public:
        char* array;
        int index;
        int size;

        SharedArray(int n): size(n), index(0){
            array = new char[size];
            std::fill(array, array+size, '-');
        }

        ~SharedArray() {
            delete[] array;
        }

        std::string toString() {
            return std::string(array, size);
        }

        int countOccurrences(char c) {
            return std::count(array, array+size, c);
        }

        void addChar(char c) {
            #pragma omp critical
            {
                array[index] = c;
                spendSomeTime();
                index++; 
            }  
        }

        void addCharErr(char c) {
            array[index] = c;
            spendSomeTime();
            index++;
        }

        private:
            void spendSomeTime() {
                for (int i = 0; i < 10000; i++) {
                    for (int j = 0; j < 100; j++) {
                        // These loops shouldn't be removed by the compiler
                    }
                }
            } 
};

class ArrayFiller{
    private:
        int nThreads;
        int chunkSize;
        SharedArray* array;

    public:
        ArrayFiller(int nthreads, int chunk) : nThreads(nthreads), 
        chunkSize(chunk) {
            array = new SharedArray(nThreads * chunk);
        }

        //Schedule dynamic com chunk
        void scheduleDynamicChunk()
        {
            #pragma omp parallel num_threads(nThreads), shared(array)
            {    
                #pragma omp for schedule(dynamic, chunkSize)
                for (int i = 0; i < array->size; i++){
                    array->addChar('A' + omp_get_thread_num());
                    
                }
            }
        }

        //Schedule dynamic sem chunk

        void scheduleDynamic()
        {
            #pragma omp parallel for shared(array) schedule(dynamic)
                for (int i = 0; i < array->size; i++)
                    array->addChar('A' + omp_get_thread_num());
        }

        //Schedule static com chunk

        void scheduleStaticChunk()
        {
            #pragma omp parallel for shared(array) schedule(static, chunkSize)
                for (int i = 0; i < array->size; i++)
                    array->addChar('A' + omp_get_thread_num());
        }

        //Schedule static sem chunk

        void scheduleStatic()
        {
            #pragma omp parallel for shared(array) schedule(static)
                for (int i = 0; i < array->size; i++)
                    array->addChar('A' + omp_get_thread_num());
        }

        //Schedule guided com chunk

        void scheduleGuidedChunk()
        {
            #pragma omp parallel for shared(array) schedule(guided, chunkSize)
                for (int i = 0; i < array->size; i++)
                    array->addChar('A' + omp_get_thread_num());
        }

        //Schedule guided sem chunk

        void scheduleGuided()
        {
            #pragma omp parallel for shared(array) schedule(guided)
                for (int i = 0; i < array->size; i++)
                    array->addChar('A' + omp_get_thread_num());
        }

        //Schedule runtime
        void scheduleRunTime(){
            #pragma omp parallel for shared(array)
                for (int i = 0; i < array->size; i++)
                        array->addChar('A' + omp_get_thread_num());
        }

        //Schedule auto

        void scheduleAuto(){
            #pragma omp parallel for shared(array) schedule(auto)
                for (int i = 0; i < array->size; i++)
                        array->addChar('A' + omp_get_thread_num());
        }

        //Com erro

        void scheduleError(){
            #pragma omp parallel for shared(array) schedule(static)
                for (int i = 0; i < array->size; i++)
                        array->addCharErr('A' + omp_get_thread_num());
        }


        void printStats() {
            std::cout << array->toString() << std::endl;
            int count = 0;
            for (int i = 0; i < nThreads; ++i){  
                std::cout << (char) ('A'+i) << "=" << array->countOccurrences('A'+i) << " ";
                std::cout << std::endl;
                count += array->countOccurrences('A'+i);
            }
            std::cout << "Num total:" << count << std::endl;
        }
        
        ~ArrayFiller() {
            delete array;
        }    
};


int main(int argc, char const *argv[])
{
    int nthreads = 4;
    int chunk = 5;
    std::cout << "Dynamic com Chunk:" << std::endl;
    ArrayFiller a1(nthreads, chunk);
    a1.scheduleDynamicChunk();
    a1.printStats();
    std::cout << "----------------------" << std::endl;

    std::cout << "Dynamic sem Chunk:" << std::endl;
    ArrayFiller a2(nthreads, chunk);
    a2.scheduleDynamic();
    a2.printStats();
    std::cout << "----------------------" << std::endl;

    std::cout << "Static com Chunk:" << std::endl;
    ArrayFiller a3(nthreads, chunk);
    a3.scheduleStaticChunk();
    a3.printStats();
    std::cout << "----------------------" << std::endl;

    std::cout << "Static sem Chunk:" << std::endl;
    ArrayFiller a4(nthreads, chunk);
    a4.scheduleStatic();
    a4.printStats();
    std::cout << "----------------------" << std::endl;

    std::cout << "Guided com chunk:" << std::endl;
    ArrayFiller a5(nthreads, chunk);
    a5.scheduleGuidedChunk();
    a5.printStats();
    std::cout << "----------------------" << std::endl;

    std::cout << "Guided sem chunk:" << std::endl;
    ArrayFiller a6(nthreads, chunk);
    a6.scheduleGuided();
    a6.printStats();
    std::cout << "----------------------" << std::endl;

    std::cout << "RunTime:" << std::endl;
    ArrayFiller a7(nthreads, chunk);
    a7.scheduleRunTime();
    a7.printStats();
    std::cout << "----------------------" << std::endl;

    std::cout << "Auto:" << std::endl;
    ArrayFiller a8(nthreads, chunk);
    a8.scheduleAuto();
    a8.printStats();
    std::cout << "----------------------" << std::endl;

    std::cout << "Com Erro:" << std::endl;
    ArrayFiller a9(nthreads, chunk);
    a9.scheduleError();
    a9.printStats();
    std::cout << "----------------------" << std::endl;


    return 0;
}
