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

/* Function: Write_Image
 * Parameters: 
 * 	FilterPHoto - the struct of black and white pixels for the image that has 
 						  the sobel filter applied
 *		size - an array of integers that holds the width and height in pixels of the img
 *		ColCode - the number that signifies the highest pixel number for the file
 *		FiltImage - the output file variable for where to print to
 * Summary: 
 *		This function prints out the contents of the image, the pixel sizes, and the 
 *		P5 to the file so it will show up correctly
 */
void Write_Image (struct BW *FilterPhoto, int *size, int ColCode, FILE *FiltImage);

/* Function: ConvertBW
 * Parameters: 
 *		ColorPhoto - the struct that holds the color pixels input from the original image
 *		BWPhoto - the struct of black and white pixels where the function writes to
 *		Ctmp - temporary structure used for conversion 
 *		BWtmp - temporary structure used for conversion
 *		size - an array of integers that holds the width and height in pixels of the img
 * Summary: 
 *		This function takes the original picture's pixels and converts them to black 
 * 		and white to put them into the new BW struct
 */
void ConvertBW (struct Color *ColorPhoto, struct BW *BWphoto, struct Color *Ctmp, 
                struct BW *BWtmp, int size);

/* Function: SobelX
 * Parameters:
 		BWimg - the struct that holds the black and white version of the original image in pixels
 		Sobel_Buff - the struct that will hold the sobel filtered version of the black and white pixels
 		size - an array of integers that holds the width and height in pixels of the img
 * Summary: 
 *		This function is the old CPU version of the sobel filter
 */
void CUDAsobel (struct BW *BWimg, struct BW *Sobel_Buff, int * size);

/* Function: errorCheck
 * Parameters: 
 		code - an int to help represent where something went wrong with the program.
 		cudaError_t err - the string for the last error generated.
 * Summary: 
 		This function checks each pre kernel function call to see if they have executed correctly.
 */
void errorCheck (int code, cudaError_t err);

/* Funtion: sharedSobelKernel
 * Parameters:
 		*BWimg - the struct BW that holds all of the pixels for the black and white image. 
 		*Sobel_Buff - the struct BW that will eventually hold the completed sobel values.
 		rowSize - the size of each row of the given image.
 * Summary:
 		This function holds the kernel that uses shared memory to manipulate each row with the sobel filter and then sends back the 
 		struct Sobel_Buff holding the completed image after sobel manipulation.  
 */
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


/*Description: CUDAsobel is a function that sets up everything the kernel needs
 *		to run, and then tells the kernel to run.
 *Preconditions: The function requires a black and white image to process,
 *		and an empty BW image of the same size to output the filtered image to
 *		It also requires a int matrix with 2 elements "size" where 
 *			size[0] = width of the image
 *			size[1] = height of the image
 *Postconditions: Sobel buff is now a BW image that holds the filtered output
 *		from BWimg
*/
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

	// copying memory to global memory
	errorCheck(4,cudaMemcpy(cuda_BW,BWimg,MEMsize,cudaMemcpyHostToDevice));
	errorCheck(5,cudaMemcpy(cuda_sobel,Sobel_Buff,MEMsize,cudaMemcpyHostToDevice));
	errorCheck(6,cudaMemcpy(cuda_size,size,2*sizeof(int),cudaMemcpyHostToDevice));

	// creating grid and block size
	dim3 dimblock(32,32,1);
	dim3 dimgrid(ceil(size[HEIGHT]/(double)(dimblock.x)),ceil(size[WIDTH]/(double)(dimblock.y)),1);

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

/*Description: The GPU kernel that processes in input BW image into a sobel
 *		filtered BW image
 *Preconditions: The function requires a black and white image to process,
 *		and an empty BW image of the same size to output the filtered image to
 *		It also requires a int matrix with 2 elements "size" where 
 *			size[0] = width of the image
 *			size[1] = height of the image
 *Postconditions: Sobel buff is now a BW image that holds the filtered output
 *		from BWimg
*/
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