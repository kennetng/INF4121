#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

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
void iso_diffusion_denoising_parallel(image *u, image *u_bar, float kappa, int iters, int my_rank, int num_procs){
   int x = u->n;
   int y = u->m;

   int top = -1;
   int bottom = -1;

   int sizes[2] = {x, y};
   int subsizes[2] = {x, 1};
   int starts[2] = {0,y-1};
   int starts2[2] = {0,0};

   MPI_Status status;
   float *topX = malloc(sizeof(float) * x);
   float *bottomX  = malloc(sizeof(float) * x);
   MPI_Datatype newtype, newtype2;

   //Legger verdier i forste og siste rad
   for(int i = 0; i<x; i++){
      u_bar->image_data[i][0] = u->image_data[i][0];
      u_bar->image_data[i][y-1] = u->image_data[i][y-1];
   }


  //Legger verdier i forste og siste kolonne
   for(int i = 0; i<y;i++){
      u_bar->image_data[0][i] = u->image_data[0][i];
      u_bar->image_data[x-1][i] = u->image_data[x-1][i];
   }
  

   //Lager datatype for kommunikasjon
   if(my_rank != 0){
      top = 0;
      MPI_Type_create_subarray(2,sizes, subsizes, starts2, MPI_ORDER_C, MPI_FLOAT, &newtype2);   
      MPI_Type_commit(&newtype2);
   }
   if(my_rank != num_procs-1){
      bottom = 0;
      MPI_Type_create_subarray(2,sizes, subsizes, starts, MPI_ORDER_C, MPI_FLOAT, &newtype);
      MPI_Type_commit(&newtype);
   }

   for(int k = 0; k<iters; k++){
      //"Smoothing"
      for(int i = 1; i<x-1; i++){
         for(int j = 1; j<y-1; j++){
            u_bar->image_data[i][j] = u->image_data[i][j] + kappa * (u->image_data[i-1][j] + u->image_data[i][j-1] - 
                                  4 * u->image_data[i][j] + u->image_data[i][j+1] + u->image_data[i+1][j]);                         
         }
      }
      //"Smoother" bunn delen av bildet
      if(bottom == 0){

         MPI_Sendrecv(&u->image_data[0][y-1], 1, newtype, my_rank+1, my_rank+1,
                      &bottomX[0], x, MPI_FLOAT, my_rank+1, my_rank, MPI_COMM_WORLD,&status);

         for(int i = 1; i<x-1;i++){
                        u_bar->image_data[i][y-1] = u->image_data[i][y-1] + kappa * (u->image_data[i-1][y-1] + u->image_data[i][y-2] - 
                                  4 * u->image_data[i][y-1] + bottomX[i] + u->image_data[i+1][y-1]);  
                   
         }
      } 
      //"Smoother" top delen av bildet
      if(top == 0){

         MPI_Sendrecv(&u->image_data[0][0], 1, newtype2, my_rank-1, my_rank-1,
                      &topX[0], x, MPI_FLOAT, my_rank-1, my_rank, MPI_COMM_WORLD,&status);  
         for(int i = 1; i<x-1;i++){
            u_bar->image_data[i][0] = u->image_data[i][0] + kappa * (u->image_data[i-1][0] + topX[i] - 
                                 4 * u->image_data[i][0] + u->image_data[i][1] + u->image_data[i+1][0]);  
      
         }      
      }
      //Setter verdier inn fra u_bar til u

      for (int i=1; i<x-1; i++){
         for (int j=0; j<y; j++){
            u->image_data[i][j] = u_bar->image_data[i][j];
         }
      }
   }

   free(bottomX);
   free(topX);
}

int main(int argc, char *argv[]){

   int m, n, c, iters;
   int my_m, my_n, my_rank, num_procs;
   int antall_m, antall_n;
   float kappa;
   image u, u_bar;
   unsigned char *image_chars, *my_image_chars;
   char *input_jpeg_filename, *output_jpeg_filename;

   int *sendcounts;
   int *displs;
   int sum = 0;

   MPI_Status status;
   MPI_Init (&argc, &argv);
   MPI_Comm_rank (MPI_COMM_WORLD, &my_rank);
   MPI_Comm_size (MPI_COMM_WORLD, &num_procs);
  if(argc != 5){
      if (my_rank==0) {
         printf("mpirun -np x <threads> parallel_main <kappa> <iters> <input_jpeg_filename> <output_jpeg_filename> \n");
      }
      MPI_Finalize ();
      exit(0);
   }

   /* read from command line: kappa, iters, input_jpeg_filename, output_jpeg_filename */
   /* ... */
   kappa = atof(argv[1]);
   iters = atoll(argv[2]);
   input_jpeg_filename = argv[3];
   output_jpeg_filename = argv[4];

   if (my_rank==0) {
      import_JPEG_file(input_jpeg_filename, &image_chars, &m, &n, &c);
   }

  
   MPI_Bcast (&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
   MPI_Bcast (&n, 1, MPI_INT, 0, MPI_COMM_WORLD);  
    
   /* divide the m x n pixels evenly among the MPI processes */
   my_m = m/num_procs;
   my_n = n;
   //Siste prosessen tar resten 
  if(my_rank == num_procs-1){
      my_m = my_m + m % num_procs;
   }
   allocate_image (&u, my_m, my_n);
   allocate_image (&u_bar, my_m, my_n);

   sendcounts = malloc(sizeof(int)*num_procs);
   displs = malloc(sizeof(int)*num_procs);
   my_image_chars = malloc(my_m*my_n*sizeof(unsigned char));
   my_m = m/num_procs;

   //Lager plassering over hvor i arrayet det skal gis bort til hver prosess
   for(int i = 0; i<num_procs; i++){
      sendcounts[i] = my_m * my_n;
      if(num_procs-1 == i) {
         sendcounts[i] = (my_m + m % num_procs) * my_n;
      }
      displs[i] = sum;
      sum += sendcounts[i];
   }

   if(my_rank == num_procs-1){
      my_m = my_m + m % num_procs;
   }

   /* each process asks process 0 for a partitioned region */
   /* of image_chars and copy the values into u */
   /*  ...  */
   MPI_Scatterv(&image_chars[0], sendcounts, displs, MPI_UNSIGNED_CHAR, &my_image_chars[0], my_m * my_n, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
   convert_jpeg_to_image (my_image_chars, &u);
   iso_diffusion_denoising_parallel (&u, &u_bar, kappa, iters, my_rank, num_procs);
   /* each process sends its resulting content of u_bar to process 0 */
   /* process 0 receives from each process incoming values and */
   /* copy them into the designated region of struct whole_image */
   /*  ...  */
   convert_image_to_jpeg(&u_bar, my_image_chars); 
   MPI_Gatherv(&my_image_chars[0], my_m * my_n, MPI_UNSIGNED_CHAR, image_chars, sendcounts, displs, MPI_UNSIGNED_CHAR,0, MPI_COMM_WORLD);
   if (my_rank==0) {
      export_JPEG_file(output_jpeg_filename, image_chars, m, n, c, 75);
   }

   deallocate_image (&u);
   deallocate_image (&u_bar);
   free(sendcounts);
   free(displs);
   free(my_image_chars);
   MPI_Finalize ();

   return 0; 
}
