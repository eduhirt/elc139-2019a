# T8: Geração de Imagem em Paralelo com CUDA

### ELC139-2019a - Programação Paralela

**Nome:** Eduardo Rafael Hirt <br/>
**Matrícula:** 201411329

**Implementações:** [wavecuda1.cu](wave/wavecuda1.cu), [wavecuda2.cu](wave/wavecuda2.cu) <br/>
**Máquina:** Tesla T4 <br/>
**Slides** [slides](slides.pdf)

# Parte 1

Foi feita a paralelização do laço mais externo (`for (int frame = 0; frame < frames; frame++)`). A função resultante é dada abaixo:

~~~cpp
__global__ void calcularFrame(unsigned char* pic, int width)
{
    int frame = threadIdx.x;
    for (int row = 0; row < width; row++) {
        for (int col = 0; col < width; col++) {
            float fx = col - 1024/2;
            float fy = row - 1024/2;
            float d = sqrtf( fx * fx + fy * fy );
            unsigned char color = (unsigned char) (160.0f + 127.0f *
                                                cos(d/10.0f - frame/7.0f) /
                                                (d/50.0f + 1.0f));

            pic[frame * width * width + row * width + col] = (unsigned char) color;
        }
    }
    
}
~~~
Após isso, ambos os códigos, [wavecuda1.cu](wave/wavecuda1.cu) e [wave.cpp](wave/wave.cpp), foram executados nos Notebooks do Google Colab e os seguintes resultados foram obtidos:

**RESULTADOS**

| frame_width, num frames 	| Tempo sequencial 	| Tempo paralelo 	|
|-------------------------	|------------------	|----------------	|
| 1024, 100               	| 5.3447 s         	| 0.8369 s       	|
| 1024, 200               	| 10.2808 s        	| 0.8769 s       	|


Para conseguir observar melhor, executei com variações de ambos os parâmetros (frame_width e num_frames). Assim, conseguimos ver que o código paralelizado em GPU é muito mais rápido que o código sequencial executado em um processador.

| frame_width   | num_frames    | Wave          | Wavecuda1     |
|-------------  |------------   |-----------    |-----------    |
| 512           | 32            | 0.4635 s      | 0.4341 s      |
| 512           | 64            | 0.9146 s      | 0.4438 s      |
| 512           | 128           | 1.6287 s      | 0.4829 s      |
| 1024          | 32            | 1.6643 s      | 0.8002 s      |
| 1024          | 64            | 3.2539 s      | 0.8233 s      |
| 1024          | 128           | 6.6157 s      | 0.8214 s      |
| 1024          | 32            | 6.5379 s      | 2.2104 s      |
| 1024          | 64            | 12.9149 s     | 2.2477 s      |
| 1024          | 128           | 26.0005 s     | 2.2558 s      |


# Parte 2

Na segunda parte, foi utilizada a estratégia de blocos (baseada no [exemplo](https://devblogs.nvidia.com/unified-memory-cuda-beginners/) para otimizar o código. A implementação se encontra aqui: [wavecuda2.cu](wave/wavecuda2.cu). Os resultados foram os seguintes:

| frame_width   | num_frames    | Wave          | Wavecuda1     | Wavecuda2     |
|-------------  |------------   |-----------    |-----------    |-----------    |
| 512           | 32            | 0.4635 s      | 0.4341 s      | 0.4694 s      |
| 512           | 64            | 0.9146 s      | 0.4438 s      | 0.4598 s      |
| 512           | 128           | 1.6287 s      | 0.4829 s      | 0.4949 s      |
| 1024          | 32            | 1.6643 s      | 0.8002 s      | 0.8384 s      |
| 1024          | 64            | 3.2539 s      | 0.8233 s      | 0.8586 s      |
| 1024          | 128           | 6.6157 s      | 0.8214 s      | 0.8588 s      |
| 1024          | 32            | 6.5379 s      | 2.2104 s      | 2.2786 s      |
| 1024          | 64            | 12.9149 s     | 2.2477 s      | 2.3266 s      |
| 1024          | 128           | 26.0005 s     | 2.2558 s      | 2.3727 s      |

O tempo de execução aumentou se comparado ao Wavecuda1, imagino que seja pelo fato de o wavecuda1 ser mais paralelizado que o 2, que agrupa mais processamento em cada kernel.


# Referências 

- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)  
  Guia da NVIDIA sobre programação em CUDA.
  
- [Experiência com grids, blocks e threads em CUDA](https://colab.research.google.com/drive/1uSTM6C0p4n4aAuvFksplqFxa4NG87rMp)  
  Notebook no Google Colab com um programa que permite experimentos variando as dimensões de grids e blocos.

- [Unified Memory for CUDA Beginners](https://devblogs.nvidia.com/unified-memory-cuda-beginners/)