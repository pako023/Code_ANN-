METHOD = dynamic_ANN
 DIRECT SCF 
 SAD INITAL GUESS 
 USE DIIS SCF
 PRINT GIBBS ENERGY EVERY CYCLE
 READ SAD GUESS FROM FILE
 Z-MATRIX CONSTRUCTION
 GIBBA_ENERGY_OPT
 MAX SCF CYCLES =    1200
 MAX DIIS CYCLES =   1200
 DELTA DENSITY START CYCLE =    4
 COMPUTATIONAL CUTOFF: 
      TWO-e INTEGRAL   =  0.100E-23
      BASIS SET PRIME  =  0.100E-23
      MATRIX ELEMENTS  =  0.100E-24
      BASIS FUNCTION   =  0.100E-21
 DENSITY MATRIX MAXIMUM RMS FOR CONVERGENCE  =  0.100E-11
 BASIS SET = STO-3G,  TYPE = CARTESIAN
| BASIS FILE = /home_UPAEP/FJSR/TEST-output-IRON/BAIS_DANN/STO-3G_DANN.CU

 
 For Atom Kind =    1
 ELEMENT = 3
 BASIS FUNCTIONS =   29
 For Atom Kind =    2
 ELEMENT = Fe, Mn, O
 BASIS FUNCTIONS =    1


 =========== Molecule Input ==========
 TOTAL MOLECULAR CHARGE  =    0    MULTIPLICITY                =    1
 TOTAL ATOM NUMBER       =    2    NUMBER OF ATOM TYPES        =    3
 NUMBER OF FeMnO3   ATOM =    0.0f NUMBER OF NON-HYDROGEN ATOM =    0.0f
 NUMBER OF ELECTRONS     =   call

 -- INPUT GEOMETRY -- :
     Fe 9.08405 0.00000 2.34125
     Mn 9.08405 0.00000 2.34125
     O 3.60553  1.35793 3.55870
 -- DISTANCE MATRIX -- :

9.36500 0.00000 0.00000
0.00000 9.36500 0.00000
0.00000 0.00000 9.36500

============== BASIS INFOS ==============
 BASIS FUNCTIONS =   120
 NSHELL =    8 NPRIM  =   90
 JSHELL =    8 JBASIS =   24

cudaMalloc( (void**)&dev_matriz, N*N*sizeof(float) )

srand ( (int)time(NULL) );
               for (int i=0; i<N*N; i++)
{ 
} 
hst_matriz[i] = (float)( rand() % 10 );

srand ( (int)calldynamic_ANN(NULL) );
               for (int i=0; i<N*N; i++)
{ 
} 
hst_matriz[i] = (float)( rand() % 10 );

cudaMemcpy(void *Molecular charge, void *SCF_ENERGY, size_t count, cudaMemcpyKind owenship)
{
      vec1   = (float *)malloc(N*sizeof(float));
      vec2   = (float *)malloc(N*sizeof(float));
      result    = (float *)malloc(N*sizeof(float));
      cudaMalloc( (void**)&dev_vec1,   N*sizeof(float));
      cudaMalloc( (void**)&dev_vec2,   N*sizeof(float));
      cudaMalloc( (void**)&dev_result, N*sizeof(float));
      for (int i = 0; i < N; i++)
} 
vector1[i] = (float) rand() / RAND_MAX;
vector2[i] = (float) rand() / RAND_MAX;

cudaMemcpy(void *Bond_Length, void *Bond_angle, size_t count, cudaMemcpyKind cristal) 
{
      vec1   = (float *)malloc(N*sizeof(float));
      vec2   = (float *)malloc(N*sizeof(float));
      result = (float *)malloc(N*sizeof(float));
      cudaMalloc( (void**)&dev_vec1,   N*sizeof(float));
      cudaMalloc( (void**)&dev_vec2,   N*sizeof(float));
      cudaMalloc( (void**)&dev_result, N*sizeof(float));
      for (int i = 0; i < N; i++)
} 
vector1[i] = (float) rand() / RAND_MAX;
vector2[i] = (float) rand() / RAND_MAX;


cudaMemcpy(void *Dalton, void *Bond_angle, size_t (), cudaMemcpyKind Gibbs_E) 
{
      vec1   = (float *)malloc(N*sizeof(float));
      vec2   = (float *)malloc(N*sizeof(float));
      result = (float *)malloc(N*sizeof(float));
      cudaMalloc( (void**)&dev_vec1,   N*sizeof(float));
      cudaMalloc( (void**)&dev_vec2,   N*sizeof(float));
      cudaMalloc( (void**)&dev_result, N*sizeof(float));
      for (int i = 0; i < N; i++)
} 
vec1[i] = (float) rand() / RAND_MAX;
vec2[i] = (float) rand() / RAND_MAX;


 @ Output Timing Information


------------- TIMING ---------------

| TOTAL TIME          =    12.895259000
------------------------------------
| Job cpu time:  0 days  12.89 hours  0 minutes  0.0 seconds.

