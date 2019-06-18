# T7: Avaliação de desempenho de programas MPI

### ELC139-2019a - Programação Paralela

**Nome:** Eduardo Rafael Hirt <br/>
**Matrícula:** 201411329

# Parte 1

Foi feita a paralelização do laço mais externo (`for (int frame = 0; frame < frames; frame++)`). A função resultante é dada abaixo:

```
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
```
Após isso, ambos os códigos, [wavecuda1.cu](/wave/wavecuda1.cu) e [wave.cpp](/wave/wave.cpp), foram executados nos Notebooks do Google Colab e os seguintes resultados foram obtidos:

**RESULTADOS**

| frame_width, num frames 	| Tempo sequencial 	| Tempo paralelo 	|
|-------------------------	|------------------	|----------------	|
| 1024, 100               	| 5.3447 s         	| 0.8369 s       	|
| 1024, 200               	| 10.2808 s        	| 0.8769 s       	|


Para conseguir observar melhor, executei com variações de ambos os parâmetros (frame_width e num_frames). Assim, conseguimos ver que o código paralelizado em GPU é muito mais rápido que o código sequencial executado em um processador.

**WAVE**

| frame_width 	| num_frames 	| Tempo de execução 	|
|-------------	|------------	|-------------------	|
| 512         	| 32         	| 0.4635 s          	|
| 512         	| 64         	| 0.9146 s          	|
| 512         	| 128        	| 1.6287 s          	|
| 1024        	| 32         	| 1.6643 s          	|
| 1024        	| 64         	| 3.2539 s          	|
| 1024        	| 128        	| 6.6157 s          	|
| 1024        	| 32         	| 6.5379 s          	|
| 1024        	| 64         	| 12.9149 s         	|
| 1024        	| 128        	| 26.0005 s         	|


**WAVECUDA1**

| frame_width 	| num_frames 	| Tempo de execução 	|
|-------------	|------------	|-------------------	|
| 512         	| 32         	| 0.4341 s          	|
| 512         	| 64         	| 0.4438 s          	|
| 512         	| 128        	| 0.4829 s          	|
| 1024        	| 32         	| 0.8002 s          	|
| 1024        	| 64         	| 0.8233 s          	|
| 1024        	| 128        	| 0.8214 s          	|
| 1024        	| 32         	| 2.2104 s          	|
| 1024        	| 64         	| 2.2477 s          	|
| 1024        	| 128        	| 2.2558 s          	|
