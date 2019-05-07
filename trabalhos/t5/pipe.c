
#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

int main(int argc, char **argv)
{
   int rank, tasks, source, dest, msg, tag = 0;
   MPI_Status status;

   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &tasks);

   if(rank == 0) {
      msg = 0;
      dest = 1;

      printf("Process #%d: sending %d\n", rank, msg);

      MPI_Send(&msg, 1, MPI_INT, dest, tag, MPI_COMM_WORLD);

   }else if(rank != tasks - 1){
      source = rank - 1;
      dest = rank + 1;
      MPI_Recv(&msg, 1, MPI_INT, source, tag, MPI_COMM_WORLD, &status);

      printf("\n---------------------\n");
      printf("Process #%d: received %d\n", rank, msg);

      msg = msg + 1;
      MPI_Send(&msg, 1, MPI_INT, dest, tag, MPI_COMM_WORLD);

      printf("Process #%d: sending %d\n", rank, msg);

   }else{
      source = rank - 1;
      MPI_Recv(&msg, 1, MPI_INT, source, tag, MPI_COMM_WORLD, &status);

      printf("\n---------------------\n");
      printf("Process #%d: received %d\n", rank, msg);
      printf("---------------------\n");
   }

   MPI_Finalize();
   return 0;
}