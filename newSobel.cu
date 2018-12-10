#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>

using namespace std;

// Struct: Color Photo Pixels
struct Color{
   unsigned char red;
   unsigned char green;
   unsigned char blue;
};

// Struct: BW Photo
struct BW{
   unsigned char pixel;
};

// Global Variables
char FileP6[3] = {"P6"};
char FileP5[3] = {"P5"};
FILE *SOBEL;
FILE *IMAGE;
FILE *FILTER_IMAGE;
#define MASTER 0
#define WIDTH 0
#define HEIGHT 1

void Write_Image (struct BW *FilterPhoto, int *size, int ColCode, FILE *FiltImage);
void ConvertBW (struct Color *ColorPhoto, struct BW *BWphoto, struct Color *Ctmp, 
                struct BW *BWtmp, int size);
void SobelX (struct BW *BWimg, struct BW *Sobel_Buff, int * size);
void CUDAsobel (struct BW *BWimg, struct BW *Sobel_Buff, int * size);
void errorCheck (int code, cudaError_t err);
__global__ void SobelKernel (struct BW *BWtmp, struct BW *Sobel_Buff, int * size);

int main(int argc, char const *argv[])
{
	char FileType[3];

	int size[2];
	int MaxColCode, /*numtasks, rank,*/ sendcount/*, recvcount, source*/ = 0;
	struct Color *ColorPhoto;
	struct Color *ColorPhotoTMP;
	struct BW *BWphoto; 
	struct BW *Sobel_Buff;
	struct BW *BWphotoTMP;


	// Opening Sobel file and Image file
	IMAGE = fopen(argv[1], "r");
	//SOBEL = fopen(argv[1], "r");
	FILTER_IMAGE = fopen(argv[2], "w");

	if ((IMAGE == NULL) | (FILTER_IMAGE == NULL)){
	  fprintf(stderr, "One of the files couldn't open please try again\n");
	  return 1;   
	}

	// Defining .ppm or .pmg file descripors
	fscanf(IMAGE,"%s", FileType);
	fscanf(IMAGE,"%i", &size[WIDTH]);
	fscanf(IMAGE,"%i", &size[HEIGHT]);
	fscanf(IMAGE,"%i", &MaxColCode);

	// Dynamic Allocation of structs
	ColorPhoto = (struct Color*)calloc((size[HEIGHT] * size[WIDTH]),sizeof(struct Color));
	BWphoto = (struct BW*)calloc((size[HEIGHT] * size[WIDTH]),sizeof(struct BW));
	BWphotoTMP = (struct BW*)calloc(sendcount,sizeof(struct BW));
	ColorPhotoTMP = (struct Color*)calloc(sendcount,sizeof(struct Color));
	Sobel_Buff = (struct BW*)calloc((size[HEIGHT] * size[WIDTH]),sizeof(struct BW));

	// Checking if file is .ppm or .pgm,
	// loading struct, 
	// and Converting to Black and White if .ppm
	if (strcmp(FileType,FileP6) == 0){
		int numPixels = size[0] * size[1];

        fread(ColorPhoto, sizeof(struct Color), (size[HEIGHT] * size[WIDTH]), IMAGE);
        cout << "read in image\n;";
		ConvertBW(ColorPhoto, BWphoto, ColorPhotoTMP, BWphotoTMP, numPixels);
		cout << "image Converted to BW\n";
//		SobelX(BWphoto, Sobel_Buff, size);
		cout << "image ran through CPU sobel\n";
		CUDAsobel(BWphoto, Sobel_Buff, size);
		cout << "image ran through CUDA sobel\n";
		Write_Image(Sobel_Buff, size, MaxColCode, FILTER_IMAGE);
	}

	return 0;
}

void ConvertBW (struct Color *ColorPhoto, struct BW *BWphoto, struct Color *Ctmp, 
                struct BW *BWtmp, int size)
{
   for (int i = 0; i < size; ++i){
      BWphoto[i].pixel = (ColorPhoto[i].red + ColorPhoto[i].green + ColorPhoto[i].blue) / 3;
   }
}

void Write_Image (struct BW *FilterPhoto, int *size, int ColCode, FILE *FiltImage)
{
	cout << "write image begin\n";
   fprintf(FiltImage, "P5\n");
   fprintf(FiltImage, "%i %i\n", size[WIDTH], size[HEIGHT]);
   fprintf(FiltImage, "%i\n", ColCode);
   cout << "about to write\n";
   fwrite(FilterPhoto, sizeof(struct BW), (size[HEIGHT] * size[WIDTH]) * sizeof(unsigned char), FiltImage);
   fclose(FiltImage);
}

void SobelX (struct BW *BWimg, struct BW *Sobel_Buff, int * size)
{
	int width = size[WIDTH];
	int height = size[HEIGHT];
	for (int i = 1; i < (width - 1); ++i){
		// cout << "sobel another row\n";
   		for (int j = 1; j < (height - 1); ++j){
   			// cout << endl << "i = " << i << "| j = " << j << endl;
   			/*we want to apply the convolution kernel:		-1 0 1
															-2 0 2
															-1 0 1
			to each pixels, except the very edges of the image
			then set the value of the pixel to the sum of each pixel in the
			3x3 area when multiplied by the coresponding value in the kernel
			*/
			//this is our sum
			int sobelSumX = 0;

			sobelSumX += BWimg[(j-1)*width+(i-1)].pixel * -1;
			sobelSumX += BWimg[(j-1)*width+(i+1)].pixel *  1;
			sobelSumX += BWimg[(j)*width + (i-1)].pixel * -2;
			sobelSumX += BWimg[(j)*width + (i+1)].pixel *  2;
			sobelSumX += BWimg[(j+1)*width+(i-1)].pixel * -1;
			sobelSumX += BWimg[(j+1)*width+(i+1)].pixel *  1;

			int sobelSumY = 0;

			sobelSumY += BWimg[(j-1)*width+(i-1)].pixel * -1;
			sobelSumY += BWimg[(j-1)*width+(i)].pixel   * -2;
			sobelSumY += BWimg[(j-1)*width+(i+1)].pixel * -1;
			sobelSumY += BWimg[(j+1)*width+(i-1)].pixel *  1;
			sobelSumY += BWimg[(j+1)*width+(i)].pixel   *  2;
			sobelSumY += BWimg[(j+1)*width+(i+1)].pixel *  1;

			//set the pixel in the output
			// cout << "setting pixel\n";
			double color = max(0.0, min((double)(sobelSumX+sobelSumY), 255.0));
			if(color > 60.0)
				Sobel_Buff[j * width + i].pixel = color;
			else 
				Sobel_Buff[j * width + i].pixel = 0;
			//max(0.0, min((double)sobelSum, 1.0));
			// cout << "done setting pixel\n";
		}
	}
}

void CUDAsobel (struct BW *BWimg, struct BW *Sobel_Buff, int * size)
{
	struct BW *cuda_BW;
	struct BW *cuda_sobel;
	int *cuda_size;
	size_t MEMsize = size[HEIGHT]*size[WIDTH]*sizeof(struct BW);

	// creating dynamic arrays
	errorCheck(1,cudaMalloc((void **)&cuda_BW,MEMsize));
	errorCheck(2,cudaMalloc((void **)&cuda_sobel,MEMsize));
	errorCheck(3,cudaMalloc((void **)&cuda_size,2*sizeof(int)));

	// copping memory to global memory
	errorCheck(4,cudaMemcpy(cuda_BW,BWimg,MEMsize,cudaMemcpyHostToDevice));
	errorCheck(5,cudaMemcpy(cuda_sobel,Sobel_Buff,MEMsize,cudaMemcpyHostToDevice));
	errorCheck(6,cudaMemcpy(cuda_size,size,2*sizeof(int),cudaMemcpyHostToDevice));

	// creating grid and block size
	dim3 dimblock(32,32,1);
	//WE MAY BE ALLOCATING TO MANY BLOCKS IN THE GRID. MEMSIZE IS THE SIZE OF THE 
	//WHOLE IMAGE
	dim3 dimgrid(ceil(MEMsize/(dimblock.x)),ceil(MEMsize/(dimblock.y)),1);

	// running kernel
	SobelKernel<<<dimgrid,dimblock>>>(cuda_BW,cuda_sobel,cuda_size);
	errorCheck(9,cudaThreadSynchronize());

	// getting back sobel buffer
	errorCheck(8,cudaMemcpy(Sobel_Buff,cuda_sobel,MEMsize,cudaMemcpyDeviceToHost));
}

void errorCheck (int code, cudaError_t err)
{
   if (err != cudaSuccess){
      printf("%d %s in %s at line %d\n\n", code, cudaGetErrorString(err),__FILE__,__LINE__);
      exit(EXIT_FAILURE);
   }
}

__global__ void SobelKernel (struct BW *BWimg, struct BW *Sobel_Buff, int * size)
{
   int bx = blockIdx.x; int by = blockIdx.y;
   int tx = threadIdx.x; int ty = threadIdx.y;
   int dx = blockDim.x; int dy = blockDim.y;

   int Row = dx*bx+tx;
   int Col = dy*by+ty;
   int MTXwidth = size[WIDTH];
   //printf("Col: %i\n", Col);

	if ((Col < size[WIDTH]) && (Col > 0) && (Row < size[HEIGHT]) && (Row > 0))
	{	
		//printf("ROW: %i COL: %i\n",Row,Col);

		int sobelSumX = 0;

		sobelSumX += BWimg[(Row-1)*MTXwidth+(Col-1)].pixel * -1;
		sobelSumX += BWimg[(Row-1)*MTXwidth+(Col+1)].pixel *  1;
		sobelSumX += BWimg[(Row)*MTXwidth + (Col-1)].pixel * -2;
		sobelSumX += BWimg[(Row)*MTXwidth + (Col+1)].pixel *  2;
		sobelSumX += BWimg[(Row+1)*MTXwidth+(Col-1)].pixel * -1;
		sobelSumX += BWimg[(Row+1)*MTXwidth+(Col+1)].pixel *  1;

		int sobelSumY = 0;

		sobelSumY += BWimg[(Row-1)*MTXwidth+(Col-1)].pixel * -1;
		sobelSumY += BWimg[(Row-1)*MTXwidth+(Col)].pixel   * -2;
		sobelSumY += BWimg[(Row-1)*MTXwidth+(Col+1)].pixel * -1;
		sobelSumY += BWimg[(Row+1)*MTXwidth+(Col-1)].pixel *  1;
		sobelSumY += BWimg[(Row+1)*MTXwidth+(Col)].pixel   *  2;
		sobelSumY += BWimg[(Row+1)*MTXwidth+(Col+1)].pixel *  1;

		double color = max(0.0, min((double)(sobelSumX+sobelSumY), 255.0));
		if(color > 60.0)
			Sobel_Buff[Row*MTXwidth+Col].pixel = color;
		else 
			Sobel_Buff[Row*MTXwidth+Col].pixel = 0;
	}
}