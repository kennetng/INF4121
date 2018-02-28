#include <stdio.h>
#include <string.h>      
#include <stdlib.h>
// make use of two functions from the simplejpeg library
typedef struct
{
  float** image_data;  /* a 2D array of floats */
  int m;               /* # pixels in y-direction */
  int n;               /* # pixels in x-direction */
} 
image;
void import_JPEG_file(const char *filename, unsigned char **image_chars, int *image_height, int *image_width, int *num_components);
void export_JPEG_file(const char *filename, unsigned char *image_chars, int image_height, int image_width, int num_components, int quality);

void allocate_image(image *u, int m, int n){
	u->m = m;
	u->n = n;
	u->image_data = (float **)malloc(n * sizeof(float*));
	u->image_data[0] = (float*)malloc(n * m * sizeof(float));

  	for(int i = 1; i < n;i++){
     	u->image_data[i] = &u->image_data[0][m * i];
  	}
}
void deallocate_image(image *u){
  	free(u->image_data[0]);
  	free(u->image_data);
}
void convert_jpeg_to_image(const unsigned char* image_chars, image *u){
  	int x = u->n;
 	int y = u->m;
 
  	int count = 0;
  	for(int i = 0; i<x; i++){
    	for(int j = 0; j<y; j++){
      	u->image_data[i][j] = image_chars[x*j+i];
    	}
  	}
}

void convert_image_to_jpeg(const image *u, unsigned char* image_chars){
  	int x = u->n;
  	int y = u->m;

  	int count = 0;
  	for(int i = 0; i<x; i++){
    	for(int j = 0; j<y; j++){
      	image_chars[x*j+i] = u->image_data[i][j];
    	}
  	}
}
void iso_diffusion_denoising(image *u, image *u_bar, float kappa, int iters){
  int x = u->n;
  int y = u->m;
  
  for(int i = 0; i<x; i++){
  	u_bar->image_data[i][0] = u->image_data[i][0];
    u_bar->image_data[i][y-1] = u->image_data[i][y-1];
  }  
  for(int i = 0; i<y;i++){
  	u_bar->image_data[0][i] = u->image_data[0][i];
    u_bar->image_data[x-1][i] = u->image_data[x-1][i];
  }
  

	for(int k = 0; k<iters; k++){
		for(int i = 1; i<x-1; i++){
      for(int j = 1; j<y-1; j++){
        
        u_bar->image_data[i][j] = u->image_data[i][j] + kappa * (u->image_data[i-1][j] + u->image_data[i][j-1] - 
             		                  4 * u->image_data[i][j] + u->image_data[i][j+1] + u->image_data[i+1][j]);
                                      
     	}
    }
    for (int i=1; i<x-1; i++){
      for (int j=0; j<y-1; j++){
        u->image_data[i][j] = u_bar->image_data[i][j];
      }
    }
  }
}
int main(int argc, char *argv[]){
  if(argc != 5){
    printf("./serial_main <kappa> <iters> <input_jpeg_filename> <output_jpeg_filename> \n");
    exit(0);
  }
  int m, n, c, iters;
  float kappa;
  image u, u_bar;
  unsigned char *image_chars;
  char *input_jpeg_filename, *output_jpeg_filename;
  //init
  kappa = atof(argv[1]);
  iters = atoll(argv[2]);
  input_jpeg_filename = argv[3];
  output_jpeg_filename = argv[4];

  /* read from command line: kappa, iters, input_jpeg_filename, output_jpeg_filename */
  /* ... */
  import_JPEG_file(input_jpeg_filename, &image_chars, &m, &n, &c);
  allocate_image (&u, m, n);
  allocate_image (&u_bar, m, n);
  convert_jpeg_to_image (image_chars, &u);
  iso_diffusion_denoising (&u, &u_bar, kappa, iters);
  convert_image_to_jpeg (&u_bar, image_chars);
  export_JPEG_file(output_jpeg_filename, image_chars, m, n, c, 75);
  deallocate_image (&u);
  deallocate_image (&u_bar);

  return 0; 
}