#include <cstdlib>
#include <sys/time.h>
#include <math.h>
#include "wave.h"

__global__ void calcularFrame(unsigned char* pic, int width)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int frame = 0; frame < frames; frame++) {
        for (int row = index; row < width; row += stride) {
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
    
}

int main(int argc, char *argv[])
{

    // check command line
    if (argc != 3) {fprintf(stderr, "usage: %s frame_width num_frames\n", argv[0]); exit(-1);}
    int width = atoi(argv[1]);
    if (width < 100) {fprintf(stderr, "error: frame_width must be at least 100\n"); exit(-1);}
    int frames = atoi(argv[2]);
    if (frames < 1) {fprintf(stderr, "error: num_frames must be at least 1\n"); exit(-1);}
    printf("computing %d frames of %d by %d picture\n", frames, width, width);

     // allocate picture array
    unsigned char* pic;

    // start time
    timeval start, end;
    gettimeofday(&start, NULL);

    cudaMallocManaged(&pic, frames * width * width * sizeof(char));

    int blockSize = 256;
    int numBlocks = (width + blockSize - 1) / blockSize;

    calcularFrame<<<numBlocks,blockSize>>>(pic,width,frames);

    // verify result by writing frames to BMP files
    if ((width <= 256) && (frames <= 100)) {
      for (int frame = 0; frame < frames; frame++) {
        char name[32];
        sprintf(name, "wave%d.bmp", frame + 1000);
        writeBMP(width, width, &pic[frame * width * width], name);
      }
    }

    cudaFree(pic);

    // end time
    gettimeofday(&end, NULL);
    double runtime = end.tv_sec + end.tv_usec / 1000000.0 - start.tv_sec - start.tv_usec / 1000000.0;
    printf("compute time: %.4f s\n", runtime);

    return 0;
}
