Eduardo Rafael Hirt - T2 - Programação Paralela 2019/1

# Parte I

1. Explique como se encontram implementadas as 4 etapas de projeto: particionamento, comunicação, aglomeração, mapeamento (use trechos de código para ilustrar a explicação).

#### Particionamento

O particionamento é feito dentro de cada thread criada pelo codigo:

```
for (k = 0; k < dotdata.repeat; k++) {
    mysum = 0.0;
    for (i = start; i < end ; i++)  {
        mysum += (a[i] * b[i]);
    }
}
```

pericionamento efetivo é feito dentro do for mais interno na função, onde cada thread pode executar livremente este trecho de código



#### Comunicação

A comunicação é feita no trecho abaixo:

```
pthread_mutex_lock (&mutexsum);
dotdata.c += mysum;
pthread_mutex_unlock (&mutexsum);
```

Aqui, sao somadas as parciais das threads


#### Aglomeração

A aglomeração e feita no codigo abaixo, onde ocorre a soma das varias multiplicações das threads:

```
for (k = 0; k < dotdata.repeat; k++) {
    mysum = 0.0;
    for (i = start; i < end ; i++)  {
        mysum += (a[i] * b[i]);
    }
}
```

#### Mapeamento

E um mapeamento estático, onde os dados de tamanho de thread são passadas como argumentos pelo usuario. E efetivamente feito pela funcao *dotprod_worker*, que seta o tamanho das threads criadas:

```
void *dotprod_worker(void *arg)
{
   int i, k;
   long offset = (long) arg;
   double *a = dotdata.a;
   double *b = dotdata.b;     
   int wsize = dotdata.wsize;
   int start = offset*wsize;
   int end = start + wsize;
   double mysum;
```



2. Considerando o tempo (em microssegundos) mostrado na saída do programa, qual foi a aceleração (speedup) com o uso de threads?

Speedup = TSeq / TPar
Speedup = 6809935 usec / 3528764 usec =~ 1.92
Aceleracao de aproximadamente 1.92




3. A aceleração se sustenta para outros tamanhos de vetores, números de threads e repetições? Para responder a essa questão, você terá que realizar diversas execuções, variando o tamanho do problema (tamanho dos vetores e número de repetições) e o número de threads (1, 2, 4, 8..., dependendo do número de núcleos). Cada caso deve ser executado várias vezes, para depois calcular-se um tempo de processamento médio para cada caso. Atenção aos fatores que podem interferir na confiabilidade da medição: uso compartilhado do computador, tempos muito pequenos, etc.

| Vetor    | Repetições | 1 Thread | 2 Threads | 4 Threads |
| -------- | ---------- | -------- | --------- | --------- |
| 2000000  | 2000       | 13339420 | 7044889   | 5685256   |
| 2000000  | 1000       | 6676072  | 3519265   | 2917797   |
| 1000000  | 1000       | 3593589  | 1757049   | 1472929   |
| 500000   | 1000       | 1684252  | 914815    | 718219    |
| 500000   | 2000       | 3675168  | 1760059   | 1473369   |


| Vetor    | Repetições | 1-2 Speedup        | 2-4 Speedup        |
| -------- | ---------- | ------------------ | ------------------ |
| 2000000  | 2000       | 1.89               | 1.21               |
| 2000000  | 1000       | 1.89               | 1.20               |
| 1000000  | 1000       | 2.04               | 1.19               |
| 500000   | 1000       | 1.84               | 1.27               |
| 500000   | 2000       | 2.08               | 1.19               |

Analisando as duas tabelas podemos notar que quanto maior o número de threads utilizadas, mais rápido é executado o código como um todo. Também conseguimos observar que o SpeedUp fica menor a cada adição de thread.



5. Explique as diferenças entre [pthreads_dotprod.c](pthreads_dotprod/pthreads_dotprod.c) e [pthreads_dotprod2.c](pthreads_dotprod/pthreads_dotprod2.c). Com as linhas removidas, o programa está correto?

O programa *pthreads_dotprod* utiliza exlusão mútua via mutex para o acesso à variável compartilhada, enquanto o *pthreads_dotprod2* deixa o acesso livre. Isto implica que o resultado da *pthreads_dotprod2* não será confiável, já que pode ocorrer leituras e gravações simultâneas de uma variável compartilhada.


