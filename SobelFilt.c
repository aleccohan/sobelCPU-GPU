2/* Name: Alec Cohan
 * Class: CSC4310
 * Date: 9-17-2016
 * Location: ~/csc4310/SobelFilt
 *
 * General Comment: This program takes in both a P6 .ppm file as well as a predetermined Sobel Filter
 * 					and converts the .ppm file first into black and white and then applies the sobel filter.
 */

#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Global Variables
char FileP6[3] = {"P6"};
char FileP5[3] = {"P5"};
FILE *SOBEL;
FILE *IMAGE;
FILE *FILTER_IMAGE;
#define MASTER 0
#define WIDTH 0
#define HEIGHT 1

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

// Function Prototypes 
void ConvertBW (struct Color *ColorPhoto, struct BW *BWphoto, struct Color *Ctmp, 
                struct BW *BWtmp, int src, int sndcnt, int rcvcnt);
void Write_Image (struct BW *FilterPhoto, int *size, int ColCode, FILE *FiltImage);
void Load_Sobel (int *SFilter, FILE *SOBEL);
void SobelFilter (struct BW *BWphoto, struct BW *BWtmp, struct BW *Sobel_Buff, int *SFilter, 
	               int *size, int src, int sndcnt, int rcvcnt);

int main(int argc, char *argv[])
{
   // Local Variables
   char FileType[3];
   int SFilterX[9];
   int SFilterY[9];
   int size[2];
   int MaxColCode, numtasks, rank, sendcount, recvcount, source = 0;
   struct Color *ColorPhoto;
   struct Color *ColorPhotoTMP;
   struct BW *BWphoto; 
   struct BW *BWphotoTMP;
   struct BW *Sobel_Buff;

   // If there are not enough arguments exit the program
   if (argc != 4){
      fprintf(stderr, "Not enough arguments, exiting program\n");
      return 1;
   }

   // Opening Sobel file and Image file
   IMAGE = fopen(argv[2], "r");
   SOBEL = fopen(argv[1], "r");
   FILTER_IMAGE = fopen(argv[3], "w");

   if (FILTER_IMAGE == NULL || SOBEL == NULL || FILTER_IMAGE == NULL){
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
   Sobel_Buff = (struct BW*)calloc(sendcount,sizeof(struct BW));

   // Loading Sobel Filter
   Load_Sobel(SFilterX, SOBEL);
   Load_Sobel(SFilterY, SOBEL);

   // Checking if file is .ppm or .pgm,
   // loading struct, 
   // and Converting to Black and White if .ppm
   if (strcmp(FileType,FileP6) == 0){
      // MASTER Reads in pixels of photograph
      if (rank == MASTER)
         fread(ColorPhoto, sizeof(struct Color), (size[HEIGHT] * size[WIDTH]), IMAGE);
      // Converting to Black and White
      ConvertBW(ColorPhoto, BWphoto, ColorPhotoTMP, BWphotoTMP, source, sendcount, recvcount);
      SobelFilter(BWphoto, BWphotoTMP, Sobel_Buff, SFilter, size, source, sendcount, recvcount);
      fclose(IMAGE);
      fclose(SOBEL);
   }
   else if (strcmp(FileType,FileP5) == 0){
   	// Reading in pixels from photograph
      if (rank == MASTER)
         fread(BWphoto, sizeof(struct BW), (size[HEIGHT] * size[WIDTH]), IMAGE);
      SobelFilter(BWphoto, BWphotoTMP, Sobel_Buff, SFilter, size, source, sendcount, recvcount);
      fclose(IMAGE);
      fclose(SOBEL);
   }
   else{
      fprintf(stderr, "File is neither .ppm or .pgm please try again\n");
      return 1;
   }

   // Printing BW Image
   if (rank == MASTER)
      Write_Image(BWphoto, size, MaxColCode, FILTER_IMAGE);

   // Freeing Memory
   free(ColorPhoto); 
   free(BWphoto); 
   free(ColorPhotoTMP); 
   free(BWphotoTMP);

}

/*
   Description:  This function takes in the color photo struct ColorPhoto and converts it to black and white

*/
void ConvertBW (struct Color *ColorPhoto, struct BW *BWphoto, struct Color *Ctmp, 
                struct BW *BWtmp, int src, int sndcnt, int rcvcnt)
{
   MPI_Scatter(ColorPhoto,sndcnt,MPI_UNSIGNED_CHAR,Ctmp,rcvcnt,MPI_UNSIGNED_CHAR,src,MPI_COMM_WORLD);
   for (int i = 0; i < rcvcnt; ++i){
      BWtmp[i].pixel = (Ctmp[i].red + Ctmp[i].green + Ctmp[i].blue) / 3;
   }
   MPI_Gather(BWtmp,rcvcnt/3,MPI_UNSIGNED_CHAR,BWphoto,sndcnt/3,MPI_UNSIGNED_CHAR,src,MPI_COMM_WORLD);
}

/*
   Description:  This function takes in the color photo struct ColorPhoto and 

*/
void Write_Image (struct BW *FilterPhoto, int *size, int ColCode, FILE *FiltImage)
{
   fprintf(FiltImage, "P5\n");
   fprintf(FiltImage, "%i %i\n", size[WIDTH], size[HEIGHT]);
   fprintf(FiltImage, "%i\n", ColCode);
   fwrite(FilterPhoto, sizeof(struct BW), (size[HEIGHT] * size[WIDTH]), FiltImage);
   fclose(FiltImage);
}

/*
   Description:  This function takes in the color photo struct ColorPhoto and 

*/
void Load_Sobel (int *SFilter, FILE *SOBEL)
{
	for (int i = 0; i < 9; i++){
		fscanf(SOBEL,"%i", &SFilter[i]);
	}
}

/*
   Description:  This function takes in the color photo struct ColorPhoto and 

*/
void SobelFilter (struct BW *BWphoto, struct BW *BWtmp, struct BW *Sobel_Buff, int *SFilter, 
	               int *size, int src, int sndcnt, int rcvcnt)
{
   MPI_Scatter(BWphoto,sndcnt/3,MPI_UNSIGNED_CHAR,BWtmp,rcvcnt/3,MPI_UNSIGNED_CHAR,src,MPI_COMM_WORLD);
   
   SobelX()

   
   MPI_Gather(Sobel_Buff,rcvcnt/3,MPI_UNSIGNED_CHAR,BWphoto,sndcnt/3,MPI_UNSIGNED_CHAR,src,MPI_COMM_WORLD);
}

void SobelX (struct BW *BWtmp, struct BW *Sobel_Buff, int *SFilter)
{
   for (int i = 1; i < ((size[HEIGHT]) - 1); ++i){
   	for (int j = 1; j < ((size[WIDTH]) - 1); ++j){
         (Sobel_Buff+(j*size[WIDTH]+i))->pixel += (BWtmp+((j-1)*size[WIDTH]+(i-1)))->pixel * SFilter[0];
         (Sobel_Buff+(j*size[WIDTH]+i))->pixel += (BWtmp+((j-1)*size[WIDTH]+(i)))->pixel * SFilter[1];
         (Sobel_Buff+(j*size[WIDTH]+i))->pixel += (BWtmp+((j-1)*size[WIDTH]+(i+1)))->pixel * SFilter[2];
         (Sobel_Buff+(j*size[WIDTH]+i))->pixel += (BWtmp+((j)*size[WIDTH]+(i-1)))->pixel * SFilter[3];
         (Sobel_Buff+(j*size[WIDTH]+i))->pixel += (BWtmp+((j)*size[WIDTH]+(i+1)))->pixel * SFilter[5];
         (Sobel_Buff+(j*size[WIDTH]+i))->pixel += (BWtmp+((j+1)*size[WIDTH]+(i-1)))->pixel * SFilter[6];
         (Sobel_Buff+(j*size[WIDTH]+i))->pixel += (BWtmp+((j+1)*size[WIDTH]+(i)))->pixel * SFilter[7];
         (Sobel_Buff+(j*size[WIDTH]+i))->pixel += (BWtmp+((j+1)*size[WIDTH]+(i+1)))->pixel * SFilter[8];
   	}
   }
}
