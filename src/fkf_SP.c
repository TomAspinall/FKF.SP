#include <R.h>
#include <Rdefines.h>
#include <Rinternals.h>
#include <Rmath.h>
#include <R_ext/BLAS.h>
#include <R_ext/Lapack.h>

/* Macro to transform an index of a 2-dimensional array into an index of a vector */
#define IDX(i,j,dim0) (i) + (j) * (dim0)

/*******************************************/
/* Macros for debugging Kalman filter cfkf */
/*******************************************/
/*#define DEBUG_PRINT  */
//#define DEBUGME
//#define DEBUGMENA

//#define VERBOSE
//#define VERBOSEME
//define COUNTME

/* #define NA_DETECTION_DEBUG_PRINT */
/* #define NA_REDUCED_ARRAYS_DEBUG_PRINT */

/*******************************************/
/* Functions to print 2-dimensional arrays */
/*******************************************/
void print_array(double * data, int i, int j, const char * lab)
{
	int icnt, jcnt;
	Rprintf("\n'%s':\n", lab);
	for(icnt = 0; icnt < i; icnt++){
	for(jcnt = 0; jcnt < j; jcnt++){
	/* Rprintf("\nIDX(icnt, jcnt, j) = %d\n", IDX(icnt, jcnt, i)); */
		Rprintf("%3.6f   ", data[IDX(icnt, jcnt, i)]);
	}
	Rprintf("\n");
    }
}

void print_int_array(int * data, int i, int j, const char * lab)
{
    int icnt, jcnt;
    Rprintf("\n'%s':\n", lab);
    for(icnt = 0; icnt < i; icnt++){
	for(jcnt = 0; jcnt < j; jcnt++){
		Rprintf("%i   ", data[IDX(icnt, jcnt, i)]);
	}
	Rprintf("\n");
    }
}

/*****************************************/
/* Locate and count NA's in observations */
/*****************************************/
void locateNA(double *vec, int *NAindices, int *positions, int len)
{
    int j = 0;
    for(int i=0; i < len; i++)
    {
	if(ISNAN(vec[i]))
		NAindices[i] = 1;
	else{
	    NAindices[i] = 0;
	    positions[j] = i;
	    j++;
	}
    }
}

int numberofNA(double *vec, int *NAindices, int *positions, int len)
{
    locateNA(vec, NAindices, positions, len);
    
    int sum = 0;
    for(int i=0; i < len; i++)
	sum += NAindices[i];

    return sum;
}

/**************************************************************************/
/* In case numberofNA(yt[,i]) > 0 && < d: Create temporary reduced arrays */
/**************************************************************************/
void reduce_array(double *array_full, int dim0, int dim1,
		  double *array_reduced, int *pos, int len)
{
    for(int i=0; i < len; i++){
	for(int j=0; j < dim1; j++)
	    array_reduced[IDX(i,j,len)] = array_full[IDX(pos[i],j,dim0)];
    }
}

//And a temporary reduced vector of Measurement errors:
void reduce_GGt(double *array_full, int dim0,
		double *array_reduced, int *pos, int len)
{
//Reduced Dimensions of GGt
	for(int j=0; j < len; j++)
		array_reduced[IDX(j,0,len)] = array_full[IDX(pos[j],0,dim0)];
    
}

/***********************************************************************************/
/* ---------- ---------- ---------- Kalman filter ---------- ---------- ---------- */
/***********************************************************************************/
void cfkf_SP(
	/* inputs */
	int m, int d, int n,
	double * a0, double * P0,
	double * dt, int incdt,
	double * ct, int incct,
	double * Tt, int incTt,
	double * Zt, int incZt,
	double * HHt, int incHHt,
	double * GGt, int incGGt,
	double * yt,
	/* outputs */
	double * loglik)

{
  /*  Description: */
  
  /*  In what follows, m denotes the dimension of the state vector, */
  /*  d denotes the dimension of the observations, and n denotes the */
  /*  number of observations. */

  /*  The state space model consists of the transition equation */
  /*  and the measurement equation: */

  /*  Transition equation: */
  /*   alpha(t + 1) = d(t) + T(t) * alpha(t) + e(t) */

  /*  Measurement equation: */
  /*   y(t) = c(t) + Z(t) * a(t) + u(t) */

  /*  e(t) and u(t) are independent innovations with zero */
  /*  expectation and variance HH(t) and GG(t), respectively. */
  /*  Covariance between e(t) and u(t) is not supported. */

  /*  The deterministic parameters admit the following dimensions: */
  /*   alpha(t) in R^(m) */
  /*   d(t) in R^(m) */
  /*   T(t) in R^(m x m) */
  /*   HH(t) in R^(m x m)     */

  /*   y(t) in R^(d) */
  /*   c(t) in R^(d) */
  /*   Z(t) in R^(d x m) */
  /*   GG(t) in R^(d x d) */

  /*  If the parameters are constant, i.e. x(i) = x(j) for i not equal to j, */
  /*  incx should be zero. Otherwise incx must be one, which means */
  /*  if x(t) in R^(i x j), x is an array of dimension R^(i x j x n), */
  /*  where n is the number of observations. */

  /*  The outputs at, att, Pt, Ptt, vt, Ft, Kt, ans status admit */
  /*  the following dimensions and interpretations: */

  /*  at in R^(m x (n + 1)), at(i) = E(alpha(i) | y(1), ..., y(i - 1)), at(0) = a0 */
  /*  att in R^(m x n), att(i) = E(alpha(i) | y(1), ..., y(i))  */
  /*  Pt in R^(m x m x (n + 1)), Pt(i) = var(alpha(i) | y(1), ..., y(i - 1)), Pt(0) = P0 */
  /*  Ptt in R^(m x m x n), Ptt(i) = var(alpha(i) | y(1), ..., y(i)) */
  /*  vt in R^(d x n), measurement equation error term v(i) = y(i) - c(i) - Z(i) * a(i) */
  /*  Ft in R^(d x d x n) */
  /*  Kt in R^(m x d x n) */
  /*  status in Z^2 */

  /*  loglik denotes the log-likelihood */

  /*  cfkf uses 'dcopy', 'dgemm' and 'daxpy' from BLAS and 'dpotri' and */
  /*  'dpotrf' from LAPACK. */
  /*  cfkf stops if dpotri or dpotrf return with a non-zero exit status */
  /*  for the computation of the inverse of F(t). potrf is also used to */
  /*  compute the determinant of F(t). However, if the determinant can not be */
  /*  calculated, the loop wont break, but the log-likelihood 'loglik' will be NA. */

  int m_x_m = m * m;
  int m_x_d = m * d;
  int NAsum;
  int i = 0;

  /* integers and double precisions used in dcopy and dgemm */
  int intone = 1;
  double dblone = 1.0, dblminusone = -1.0, dblzero = 0.0;

  char *transpose = "T", *dont_transpose = "N";

  /* NA detection */
  int *NAindices = malloc(sizeof(int) * d);
  int *positions = malloc(sizeof(int) * d);

  /* Reduced arrays for case 2 */
  double *yt_temp  = malloc(sizeof(double) * (d - 1));
  double *ct_temp  = malloc(sizeof(double) * (d - 1));
  double *Zt_temp  = malloc(sizeof(double) * (d - 1) * m);
  double *GGt_temp = malloc(sizeof(double) * (d - 1));
  
  double *Zt_t   = malloc(sizeof(double) * (d * m));
  double *Zt_tSP = malloc(sizeof(double) * m);

//SEQUENTIAL PROCESSING DEFINED VARIABLES:

#ifdef DEBUGME
int SP_int = 3;
int i_int = 0;
 #endif
 
	int N_obs = 0;
	//Doubles for the SP iteration:
	double V;
	double Ft;
  	double tmpFt_inv;
	
	double *at = malloc(sizeof(double) * m); 
	double *Pt = malloc(sizeof(double) * m * m);
	double *Kt = malloc(sizeof(double) * m);
   
	//SP temporary arrays:
	double *tmpmxSP = (double *) Calloc(m, double);
	double *tmpmxm = (double *) Calloc(m_x_m, double);

	*loglik = 0;

  /* at = a0 */
  F77_NAME(dcopy)(&m, a0, &intone, at, &intone);

  /* Pt = P0 */
  F77_NAME(dcopy)(&m_x_m, P0, &intone, Pt, &intone);

  /****************************************************************/
  /* ---------- ---------- start recursions ---------- ---------- */
  /****************************************************************/
    while(i < n){
//    while(i < 2){

#ifdef VERBOSE
      Rprintf("\n\nLoop Nr.:  %d\n", i);
#endif


      /****************************************/
      /* check for NA's in observation yt[,i] */
      /****************************************/
      NAsum = numberofNA(&yt[d * i], NAindices, positions, d);

#ifdef DEBUGME
      Rprintf("\nNumber of NAs in iter %i: %i\n", i, NAsum);
#endif


      /*********************************************************************************/
      /* ---------- ---------- ---------- filter step ---------- ---------- ---------- */
      /*********************************************************************************/

      /***************************************************************************/
      /* ---------- case 1: no NA's: filtering using full information ---------- */
      /***************************************************************************/
      if(NAsum == 0)
      {
      	//Create Zt for time t
      	  F77_NAME(dcopy)(&m_x_d, &Zt[m_x_d * i * incZt], &intone, Zt_t, &intone);

	 	N_obs += d;

      /* Sequential Processing - perform univariate treatment of the multivariate series:*/
	     for(int SP=0; SP < d; SP++)
    {

#ifdef DEBUGME
Rprintf("SP = %i", SP);
#endif    
    //Get the specific values of Z for SP:
    for(int j = 0; j < m; j++)
    {
    Zt_tSP[j] = Zt_t[SP + j*d]; 	
	}
    	
    	//Step 1 - Measurement Error:
    	//Compute Vt[SP,i] = yt[SP,i] - ct[SP,i * incct] + Zt[SP,,i * incZt] %*% at[SP,i]
    	
    	//vt[SP,i] = yt[SP,i] - ct[SP,i * incct]
    	V = yt[SP + d * i] - ct[SP + d * i * incct];
    	
      #ifdef DEBUGME
      if(SP == SP_int)
      {
      	if(i == i_int){
      print_array(Zt_tSP, 1, m, "Zt[SP,]");
      Rprintf("\n V - Pre Mat Mult: %f", V);      		
		  }
	  }
      #endif

    	//vt[SP,i] = vt[SP,i] - Zt[SP,, i * incZt] %*% at[,i]
    		  F77_NAME(dgemm)(dont_transpose, dont_transpose, &intone,
			  &intone, &m, &dblminusone,
			  Zt_tSP, &intone,
			  at, &m,
			  &dblone, &V, &intone);			  
	
		
	#ifdef DEBUGME
      if(SP == SP_int)
      {
      	if(i == i_int){
      Rprintf("\n V - Post Mat Mult: %f", V);
   		
		  }
	  }
      #endif
				
		//Step 2 - Function of Covariance Matrix:
		//Compute Ft = Zt[SP,,i * incZt] %*% Pt %*% t(Zt[SP,,i * incZt]) + diag(GGt)[SP]
		
		//First, Let us calculate:
		//Pt %*% t(Zt[SP,,i * incZt])
		//because we use this result twice

	    F77_NAME(dgemm)(dont_transpose, transpose, &m,
	    &intone, &m, &dblone,
	    Pt, &m,
	    Zt_tSP, &intone,
	    &dblzero, tmpmxSP, &m);
		
		
	#ifdef DEBUGME
      if(SP == SP_int)
      {
      	if(i == i_int){
		print_array(tmpmxSP, m, 1, "tmpmxSP");
		  }
	  }
      #endif
		
		//Ft = GGt[SP]
		Ft = GGt[SP + (d * i * incGGt)];

	#ifdef DEBUGME
      if(SP == SP_int)
      {
      	if(i == i_int){
      Rprintf("\n Ft: %f \n", Ft);
		  }
	  }
      #endif
		
		//Ft = Zt[SP,,i*incZt] %*% tmpmxSP + Ft
		F77_NAME(dgemm)(dont_transpose, dont_transpose, &intone,
	    &intone, &m, &dblone,
	    Zt_tSP, &intone,
	    tmpmxSP, &m,
	    &dblone, &Ft, &intone);		

	#ifdef DEBUGME
      if(SP == SP_int)
      {
      	if(i == i_int){
	  Rprintf("\n New Ft: %f \n", Ft);
		  }
	  }
      #endif	   

    	//Step 3 - Calculate the Kalman Gain:
    	//Compute Kt = Pt %*% t(Zt[SP,,i * incZt]) %*% (1/Ft)
    	
    	//Inv Ft:
    	tmpFt_inv = 1 / Ft;
	#ifdef DEBUGME
      if(SP == SP_int)
      {
      	if(i == i_int){
	  Rprintf("\n Inverse Ft: %f \n", tmpFt_inv);
		  }
	  }
      #endif	   
		    	
    	//Kt is an m x 1 matrix
    	
    	//We already have tmpSPxm:    	
        //Kt = tmpmxSP %*% tmpFt_inv
	  F77_NAME(dgemm)(dont_transpose, dont_transpose, 
	                  &m, &intone, &intone, 
					  &dblone, tmpmxSP, &m,
			                   &tmpFt_inv, &intone,
			          &dblzero, Kt, &m);
					  
	#ifdef DEBUGME
      if(SP == SP_int)
      {
      	if(i == i_int){
      print_array(Kt, m, 1, "Kalman Gain");
		  }
	  }
      #endif	   

	#ifdef DEBUGME
      if(SP == SP_int)
      {
      	if(i == i_int){
      Rprintf("\n V - Post Mat Mult: %f", V);
   		
		  }
	  }
      #endif
  
		//Step 4 - Correct State Vector mean and Covariance:

       //Correction to att based upon prediction error:
       //att = Kt %*% V + att
	  F77_NAME(dgemm)(dont_transpose, dont_transpose, 
	                  &m, &intone, &intone, 
			  		  &dblone, Kt, &m,
			  				   &V, &intone,
			 	 	  &dblone, at, &m);

	#ifdef DEBUGME
      if(SP == SP_int)
      {
      	if(i == i_int){
      print_array(at, m, 1, "at Correction");
		  }
	  }
      #endif	   
    	
       //Correction to covariance based upon Kalman Gain:
       //ptt = ptt - ptt %*% t(Z[SP,,i * incZt]) %*% t(Ktt)
       //ptt = ptt - tempmxSP %*% t(Ktt)
	  F77_NAME(dgemm)(dont_transpose, transpose, 
	  				  &m,  &m, &intone, 
				&dblminusone,  tmpmxSP, &m,
		  	  					 Kt, &m,
		  	  			&dblone, Pt, &m);
      
	#ifdef DEBUGME
      if(SP == SP_int)
      {
      	if(i == i_int){
      print_array(Pt, m, m, "pt correction");      
		  }
	  }
      #endif	   
	
       //Step 5 - Update Log-Likelihood Score:
       *loglik -= 0.5 * (log(Ft) + (V * V * tmpFt_inv));

	#ifdef DEBUGME
      Rprintf("\n Log-Likelihood: %f \n", *loglik);
	#endif   
	   
	  }
}
	  else
	  {
	   // If NAsum == d, no filtering occurs
//      if(NAsum < d)
//      {
      			  	  


	  /*****************************************************************************/
	  /* ---------- case 2: some NA's: filter using partial information ---------- */
	  /*****************************************************************************/
	       int d_reduced = d - NAsum;
	 		   N_obs += d_reduced;
	    
	      reduce_array(&yt[d * i], d, 1, yt_temp, positions, d_reduced);
	      reduce_array(&ct[d * i * incct], d, 1, ct_temp, positions, d_reduced);
	      reduce_array(&Zt[m_x_d * i * incZt], d, m, Zt_temp, positions, d_reduced);
	      reduce_array(&GGt[d * i * incGGt], d, 1, GGt_temp, positions, d_reduced);
	      
//	      reduce_GGt(&GGt[d * i * incGGt], d, GGt_temp, positions, d_reduced);

//              print_array(GGt_temp, 1, d_reduced, "GGt_temp");
//print_array(&GGt[d*i*incGGt], 1, d, "GGt");

#ifdef DEBUGMENA
              print_int_array(positions, 1, d_reduced, "positions");
/*              print_array(&yt[d * 1], 1, d, "yt");
              print_array(&ct[d * i * incct], 1, d, "ct");
	      	  print_array(&Zt[m_x_d * i * incZt], d, m, "Zt");
              print_array(&GGt[d*i*incGGt], 1, d, "GGt"); */
              print_array(yt_temp, 1, d_reduced, "yt_temp");
              print_array(ct_temp, 1, d_reduced, "ct_temp");
	          print_array(Zt_temp, d_reduced, m, "Zt_temp");
              print_array(GGt_temp, 1, d_reduced, "GGt_temp");
              
#endif


		//Sequential Processing - Univariate Treatment of the Multivariate Series:
	     for(int SP=0; SP < d_reduced; SP++)
    {
    	
    	
    #ifdef DEBUGMENA
Rprintf("SP = %i", SP);
#endif    
	
    //Get the specific values of Z for SP:
    for(int j = 0; j < m; j++)
    {
    Zt_tSP[j] = Zt_temp[SP + j*d_reduced]; 	
	}

    #ifdef DEBUGMENA
	  print_array(Zt_tSP, 1, m, "Zt_tSP:");
#endif    


    	//Step 1 - Measurement Error:
    	//Compute Vt[SP,i] = yt[SP,i] - ct[SP,i * incct] + Zt[SP,,i * incZt] %*% at[SP,i]
    	
    	//vt[SP,i] = yt[SP,i] - ct[SP,i * incct]
    	V = yt_temp[SP] - ct_temp[SP];

	 #ifdef DEBUGMENA
	  Rprintf("\n Pre mat-mult V = %f", V);
      #endif
   
      
    	//vt[SP,i] = vt[SP,i] - Zt[SP,, i * incZt] %*% at[,i]
/*
    		  F77_NAME(dgemm)(dont_transpose, dont_transpose, &intone,
			  &intone, &m, &dblminusone,
			  &Zt_temp[SP], &intone,
			  at, &m,
			  &dblone, &V, &intone);		
*/
    		  F77_NAME(dgemm)(dont_transpose, dont_transpose, &intone,
			  &intone, &m, &dblminusone,
			  Zt_tSP, &intone,
			  at, &m,
			  &dblone, &V, &intone);		

		
	 #ifdef DEBUGMENA
	  Rprintf("\n Post mat-mult V = %f", V);
      #endif
		
		//Step 2 - Function of Covariance Matrix:
		//Compute Ft = Zt[SP,,i * incZt] %*% Pt %*% t(Zt[SP,,i * incZt]) + diag(GGt)[SP]
		
		//First, Let us calculate:
		//Pt %*% t(Zt[SP,,i * incZt])
		//because we use this result twice
/*		
	    F77_NAME(dgemm)(dont_transpose, transpose, &m,
	    &intone, &m, &dblone,
	    Pt, &m,
	    &Zt_temp[SP], &intone,
	    &dblzero, tmpmxSP, &m);
*/

		F77_NAME(dgemm)(dont_transpose, transpose, &m,
	    &intone, &m, &dblone,
	    Pt, &m,
	    Zt_tSP, &intone,
	    &dblzero, tmpmxSP, &m);
		
		
	 #ifdef DEBUGMENA
	  print_array(tmpmxSP, m, 1, "tmpmxSP:");	  
      #endif		
		
		//Ft = GGt[SP]
		Ft = GGt_temp[SP];
		
	  #ifdef DEBUGMENA
      Rprintf("\n Ft: %f \n", Ft);
      #endif
/*
		//Ft = Zt[SP,,i*incZt] %*% tmpmxSP + Ft
		F77_NAME(dgemm)(dont_transpose, dont_transpose, &intone,
	    &intone, &m, &dblone,
	    &Zt_temp[SP + m], &intone,
	    tmpmxSP, &m,
	    &dblone, &Ft, &intone);		
	    
	    */
	    //Ft = Zt[SP,,i*incZt] %*% tmpmxSP + Ft
		F77_NAME(dgemm)(dont_transpose, dont_transpose, &intone,
	    &intone, &m, &dblone,
	    Zt_tSP, &intone,
	    tmpmxSP, &m,
	    &dblone, &Ft, &intone);		
	   //is Pt & Ft indexed correctly here? Can we just provide the object pointer itself?
	   //is Zt indexed correctly here?

    	//Step 3 - Calculate the Kalman Gain:
    	//Compute Kt = Pt %*% t(Zt[SP,,i * incZt]) %*% (1/Ft)
    	
      #ifdef DEBUGMENA
	  Rprintf("\n New Ft: %f \n", Ft);
      #endif

    	//Inv Ft:
    	tmpFt_inv = 1 / Ft;
    	
      #ifdef DEBUGMENA
	  Rprintf("\n Inverse Ft: %f \n", tmpFt_inv);
      #endif

    	
    	//Kt is an m x 1 matrix
    	
		//We already have tmpSPxm:    	
        //Kt = tmpmxSP %*% tmpFt_inv
	  F77_NAME(dgemm)(dont_transpose, dont_transpose, &m,
			  &intone, &intone, &dblone,
			  tmpmxSP, &m,
			  &tmpFt_inv, &intone,
			  &dblzero, Kt, &m);
	
	  #ifdef DEBUGMENA
      print_array(Kt, m, 1, "Kalman Gain");
      #endif
		  
		//Step 4 - Correct State Vector mean and Covariance:

       //Correction to att based upon prediction error:
       //att = Kt %*% V + att
	  F77_NAME(dgemm)(dont_transpose, dont_transpose, &m,
			  &intone, &intone, &dblone,
			  Kt, &m,
			  &V, &intone,
			  &dblone, at, &m);
    	
       //Correction to covariance based upon Kalman Gain:
       //ptt = ptt - ptt %*% t(Z[SP,,i * incZt]) %*% t(Ktt)
       //ptt = ptt - tempmxSP %*% t(Ktt)
	  F77_NAME(dgemm)(dont_transpose, transpose, &m,
			  &m, &intone, &dblminusone,
			  tmpmxSP, &m,
		  	  Kt, &m,
		  	  &dblone, Pt, &m);
		  	  
	  #ifdef DEBUGMENA
//      print_array(&Pt, m, m, "Pt");
      #endif		  	  
	
       //Step 5 - Update Log-Likelihood Score:
       *loglik -= 0.5 * (log(Ft) + V * V * tmpFt_inv);
       //Increment number of observations for the Log-likelihood at the end:
    }
      #ifdef VERBOSE
	  Rprintf("\n SP Iteration Completed Successfully.\n");
      #endif
//	}
}
      /**************************************************************************************/
      /*  ---------- ---------- ---------- prediction step ---------- ---------- ---------- */
      /**************************************************************************************/

      /* ---------------------------------------------------------------------- */
      /* at[,i + 1] = dt[,i * incdt] + Tt[,,i * incTt] %*% att[,i]                   */
      /* ---------------------------------------------------------------------- */

#ifdef DEBUGME
      print_array(at, 1, m, "at:");
#endif

      //tmpmxm = Tt[,,i * incTt] %*% att[,i]
            F77_NAME(dgemm)(dont_transpose, dont_transpose, &m,
		      &intone, &m, &dblone,
		      &Tt[m_x_m * i * incTt], &m,
		      at, &m,
		      &dblzero, tmpmxSP, &m);
      
      /* at[,i + 1] = dt[,i] */
		F77_NAME(dcopy)(&m, &dt[m * i * incdt], &intone, at, &intone);
		F77_NAME(daxpy)(&m, &dblone, tmpmxSP, &intone, at, &intone);
      
#ifdef DEBUGME
      print_array(at, 1, m, "atp1:");
      print_array(Pt, m, m, "Pt:");      
#endif
 
      /* --------------------------------------------------------------------------------- */
      /* Pt[,,i + 1] = Tt[,,i * incTt] %*% Ptt[,,i] %*% t(Tt[,,i * incTt]) + HHt[,,i * incHHt] */
      /* --------------------------------------------------------------------------------- */

      /* tmpmxm = Ptt[,,i] %*% t(Tt[,,i * incTt]) */
      F77_NAME(dgemm)(dont_transpose, transpose, &m,
		      &m, &m, &dblone,
		      Pt, &m,
		      &Tt[m_x_m * i * incTt], &m,
		      &dblzero, tmpmxm, &m);

      /* Pt[,,i + 1] = HHt[,,i * incHHt] */      
//     F77_NAME(dcopy)(&m_x_m, &HHt[m_x_m * i * incHHt], &intone, &Pt[m_x_m * (i + 1)], &intone);
     F77_NAME(dcopy)(&m_x_m, &HHt[m_x_m * i * incHHt], &intone, Pt, &intone);

#ifdef DEBUG_PRINT
      print_array(&HHt[m_x_m * i * incHHt], m, m, "HHt:");
#endif

      /* Pt[,,i + 1] = Tt[,,i * incTt] %*% tmpmxm + Pt[,,i + 1] */
      F77_NAME(dgemm)(dont_transpose, dont_transpose, &m,
		      &m, &m, &dblone,
		      &Tt[m_x_m * i * incTt], &m,
		      tmpmxm, &m,
		      &dblone, Pt, &m);
		      
		      
#ifdef DEBUGME
      print_array(Pt, m, m, "Ptp1:");      
#endif
		             
#ifdef DEBUG_PRINT
//      Rprintf(loglik)
      print_array(&at, 1, m, "at:");
      print_array(&Pt, m, m, "Pt:");
      Rprintf("\n---------- iteration nr. %i ----------\n", i+1);
#endif

    //end iteration
      i++;
  }
  /**************************************************************/
  /* ---------- ---------- end recursions ---------- ---------- */
  /**************************************************************/

	  #ifdef DEBUGME
      Rprintf("\n Log-Likelihood: %f \n", *loglik);
      #endif

	    
  //Update the final Log-Likelihood Score:
    *loglik -= 0.5 * N_obs * log(2 * PI);
    //Have we defined PI correctly here?

//We will need some kind of statement for these:
//  status[0] = potri_info;
// status[1] = potrf_info;


//Free the vectors / matrices:
free(NAindices);
free(positions);

free(yt_temp);
free(ct_temp);
free(Zt_temp);
free(GGt_temp);

free(Zt_t);
free(Zt_tSP);

free(at);
free(Pt);
free(Kt);

#ifdef VERBOSEME
      Rprintf("\n---------- Recursion Complete ----------\n");
#endif


}

//Function Name:
SEXP fkf_SP(SEXP a0, SEXP P0, SEXP dt, SEXP ct, SEXP Tt,
	 SEXP Zt, SEXP HHt, SEXP GGt, SEXP yt)

{
  int m = length(a0);
  int d = INTEGER(GET_DIM(yt))[0];
  int n = INTEGER(GET_DIM(yt))[1];

//  SEXP loglik, status, ans, ans_names, class_name;
  SEXP loglik;
  PROTECT(loglik = NEW_NUMERIC(1));

  cfkf_SP(m, d, n,
       NUMERIC_POINTER(a0), NUMERIC_POINTER(P0),
       NUMERIC_POINTER(dt), INTEGER(GET_DIM(dt))[1] == n,
       NUMERIC_POINTER(ct), INTEGER(GET_DIM(ct))[1] == n,
       NUMERIC_POINTER(Tt), INTEGER(GET_DIM(Tt))[2] == n,
       NUMERIC_POINTER(Zt), INTEGER(GET_DIM(Zt))[2] == n,
       NUMERIC_POINTER(HHt), INTEGER(GET_DIM(HHt))[2] == n,
       NUMERIC_POINTER(GGt), INTEGER(GET_DIM(GGt))[1] == n,
       NUMERIC_POINTER(yt),
       NUMERIC_POINTER(loglik));

  UNPROTECT(1);
  return(loglik);
}





