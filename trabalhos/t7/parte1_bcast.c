// Adaptado de www.mpitutorial.com
// Exemplo de implementação do MPI_Bcast usando MPI_Send e MPI_Recv

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char **argv)
{

  int myrank; // "rank" do processo
  int p;      // numero de processos
  int root;   // rank do processo root
  double tempo_inicial, tempo_final;


  // MPI_Init deve ser invocado antes de qualquer outra chamada MPI
  MPI_Init(&argc, &argv);
  // Descobre o "rank" do processo
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  // Descobre o numero de processos
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  int data; // dado a ser enviado
  root = 0; // define o rank do root

  if (myrank == root){
    data = 100;
    tempo_inicial = MPI_Wtime();
  }

  MPI_Bcast(&data, 1, MPI_INT, root, MPI_COMM_WORLD);
  printf("Processo %d recebendo dado %d do processo root\n", myrank, data);
  MPI_Barrier(MPI_COMM_WORLD);

  if (myrank == root){
    tempo_final = MPI_Wtime();
    printf("\n-----------------------------------------------\n");
    printf("Tempo: %f microssegundos", (tempo_final-tempo_inicial)*1000000);
    printf("\n-----------------------------------------------\n");
  }
  MPI_Finalize();
  return 0;
}