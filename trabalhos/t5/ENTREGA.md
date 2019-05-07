# T5: Primeiros passos com MPI

### ELC139-2019a - Programação Paralela

**Nome:** Eduardo Rafael Hirt
**Matrícula:** 201411329

# Parte 1

Ainda não terminada.

# Parte 2

Implementação de um programa MPI que transporte uma mensagem em um pipeline formado por processos de 0 a NP-1. <br/>
**Programa**: [pipe.c](pipe.c)

# Parte 3: Correções

[mpi_corrigido1.c](mpi_corrigido1.c): Valor das tags era igualado ao rank, o que gerava valores diferentes <br/>
[mpi_corrigido2.c](mpi_corrigido2.c): Faltava o MPI_Finalize() no final do código.