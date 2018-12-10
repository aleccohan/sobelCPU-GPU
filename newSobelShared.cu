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
__global__ void sharedSobelKernel (struct BW *BWimg, struct BW *Sobel_Buff, int * size, int rowSize);

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
	int rowSize = min(1024, size[WIDTH]);
	dim3 dimblock(rowSize,1,1);
	dim3 dimgrid(ceil(MEMsize/(dimblock.x)),1,1);
	//allocate enough shared memory to hold 3 rows of black and white pixels
	//add 2 to rowsize for the 3 pixels on the ends of the row
	int sharedSize = (rowSize+2) * 3 * sizeof(struct BW);
	// running kernel
	sharedSobelKernel<<<dimgrid,dimblock,sharedSize>>>(cuda_BW,cuda_sobel,cuda_size, rowSize+2);
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

/*Description: The GPU kernel that processes in input BW image into a sobel
 *		filtered BW image. The kernel divides the image into rows for processing
 *		and loads any relevant data into shared memory before performing the
 *		sobel opertation
 *Preconditions: The function requires a black and white image to process,
 *		and an empty BW image of the same size to output the filtered image to
 *		It also requires a int matrix with 2 elements "size" where 
 *			size[0] = width of the image
 *			size[1] = height of the image
 *Postconditions: Sobel buff is now a BW image that holds the filtered output
 *		from BWimg
*/
__global__ void sharedSobelKernel (struct BW *BWimg, struct BW *Sobel_Buff, int * size, int rowSize)
{
	extern __shared__ BW rows[];
	__shared__ BW * topRow;
	topRow = rows;
	__shared__ BW * middleRow;
	middleRow = rows + rowSize;
	__shared__ BW * bottomRow;
	bottomRow = rows + ((rowSize) * 2);

	int bx = blockIdx.x;
	//offset by 1 because of the extra column of pixels to the left of the row 
	//of threads.
	int tx = threadIdx.x + 1;
	int dx = blockDim.x;

	int MTXwidth = size[WIDTH];

	int pos = dx*bx+tx;
	int Row = floorf(pos/MTXwidth);
	int Col = pos%MTXwidth;
	//printf("Col: %i\n", Col);

	//fill shared mem arrays
	//each thread grabs 1 value from each row (above, on, below)
	if(Row != 0 && Row != size[HEIGHT]) {
		topRow[tx].pixel = BWimg[pos - MTXwidth].pixel;
		middleRow[tx].pixel = BWimg[pos].pixel;
		bottomRow[tx].pixel = BWimg[pos + MTXwidth].pixel;
	}
	if(tx == 0) {
		topRow[tx-1].pixel = BWimg[pos - MTXwidth-1].pixel;
		middleRow[tx-1].pixel = BWimg[pos-1].pixel;
		bottomRow[tx-1].pixel = BWimg[pos + MTXwidth-1].pixel;
	}
	if(tx == dx) {
		topRow[tx+1].pixel = BWimg[pos - MTXwidth+1].pixel;
		middleRow[tx+1].pixel = BWimg[pos+1].pixel;
		bottomRow[tx+1].pixel = BWimg[pos + MTXwidth+1].pixel;
	}
	
	//make sure all the threads have finished loading up the shared memory
	__syncthreads();

	if ((Col < size[WIDTH]) && (Col > 0) && (Row < size[HEIGHT]) && (Row > 0))
	{	
		//printf("ROW: %i COL: %i\n",Row,Col);

		int sobelSumX = 0;

		sobelSumX += topRow[tx-1].pixel * -1;		//BWimg[(Row-1)*MTXwidth+(Col-1)].pixel * -1;
		sobelSumX += topRow[tx+1].pixel * 1; 	//BWimg[(Row-1)*MTXwidth+(Col+1)].pixel *  1;
		sobelSumX += middleRow[tx-1].pixel * -2;	//BWimg[(Row)*MTXwidth + (Col-1)].pixel * -2;
		sobelSumX += middleRow[tx+1].pixel * 2;	//BWimg[(Row)*MTXwidth + (Col+1)].pixel *  2;
		sobelSumX += bottomRow[tx-1].pixel * -1; 	//BWimg[(Row+1)*MTXwidth+(Col-1)].pixel * -1;
		sobelSumX += bottomRow[tx+1].pixel * 1;	//BWimg[(Row+1)*MTXwidth+(Col+1)].pixel *  1;

		int sobelSumY = 0;

		sobelSumY += topRow[tx-1].pixel * -1;		//BWimg[(Row-1)*MTXwidth+(Col-1)].pixel * -1;
		sobelSumY += topRow[tx].pixel * -2;		//BWimg[(Row-1)*MTXwidth+(Col)].pixel   * -2;
		sobelSumY += topRow[tx+1].pixel * -1;		//BWimg[(Row-1)*MTXwidth+(Col+1)].pixel * -1;
		sobelSumY += bottomRow[tx-1].pixel * 1;	//BWimg[(Row+1)*MTXwidth+(Col-1)].pixel *  1;
		sobelSumY += bottomRow[tx].pixel * 2;		//BWimg[(Row+1)*MTXwidth+(Col)].pixel   *  2;
		sobelSumY += bottomRow[tx+1].pixel * 1;	//BWimg[(Row+1)*MTXwidth+(Col+1)].pixel *  1;

		double color = max(0.0, min((double)(sobelSumX+sobelSumY), 255.0));
		//color = (sobelSumX+sobelSumY)/2;
		//Sobel_Buff[Row*MTXwidth+Col].pixel = color;
		//stuff
		if(color > 60.0)
			Sobel_Buff[Row*MTXwidth+Col].pixel = color;
		else 
			Sobel_Buff[Row*MTXwidth+Col].pixel = 0;
		
	}


}

/*
__global__ void SobelKernel (struct BW *BWimg, struct BW *Sobel_Buff, int * size)
{
   int bx = blockIdx.x; int by = blockIdx.y;
   int tx = threadIdx.x; int ty = threadIdx.y;
   int dx = blockDim.x; int dy = blockDim.y;

   int pos = dx*bx+tx;
   int Row = floor(pos/width)
   int Col = pos%width;
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
*/