#define USE_FC_LEN_T
#include <R.h>
#include <Rdefines.h>
#include <Rinternals.h>
#include <Rmath.h>
#include <R_ext/BLAS.h>
#include <R_ext/Lapack.h>
#ifndef FCONE
# define FCONE
#endif

/* Macro to transform an index of a 2-dimensional array into an index of a vector */
#define IDX(i,j,dim0) (i) + (j) * (dim0)

/*#define DEBUG_PRINT*/
//#define DEBUGME

/* Print arrays */
void print_array(double * data, int i, int j, const char * lab)
{
	int icnt, jcnt;
	Rprintf("\n'%s':\n", lab);
	for(icnt = 0; icnt < i; icnt++){
	for(jcnt = 0; jcnt < j; jcnt++){
		Rprintf("%3.6f   ", data[IDX(icnt, jcnt, i)]);
	}
	Rprintf("\n");
    }
}

/* Print observation numbers */
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

/* Locate NA's in observations at time t*/
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

/* Number of NA's in observations at time t*/
int numberofNA(double *vec, int *NAindices, int *positions, int len)
{
    locateNA(vec, NAindices, positions, len);
    int sum = 0;
    for(int i=0; i < len; i++)
	sum += NAindices[i];
    return sum;
}

/* Temporary reduced arrays when missing obverations are present */
void reduce_array(double *array_full, int dim0, int dim1,
		double *array_reduced, int *pos, int len)
{
    for(int i=0; i < len; i++){
	for(int j=0; j < dim1; j++)
	    array_reduced[IDX(i,j,len)] = array_full[IDX(pos[i],j,dim0)];
    }
}

/* Temporary reduced vector of measurement error */
void reduce_GGt(double *array_full, int dim0,
		double *array_reduced, int *pos, int len)
{
	//Reduced Dimensions of GGt
	for(int j=0; j < len; j++)
		array_reduced[IDX(j,0,len)] = array_full[IDX(pos[j],0,dim0)];
}

/************************************************************************************/
/* ---------- ---------- Kalman filter: Verbose Sequential Processing ---------- ---*/
/************************************************************************************/
// This function is the verbose version. It returns Filtered values.
// The Filtered values it returns are:
// - The log Likelihood
// - at, att
// - pt, ptt
// The inverse of ft,i
// These are necessary for the Kalman smoother.
void cfkf_SP_verbose(
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
	/* output */
	double * loglik,
	double * at_output, double * att_output,
	double * pt_output, double * ptt_output,
	double * Ftinv_output,
	double * vt_output,
	double * Kt_output)
{
/*  Notation: */

//m: the dimension of the state vector
//d: the dimension of observations
//n: the total number of observations

/*  loglik is the output of the function and denotes the log-likelihood */
/*  cfkf uses 'dcopy', 'dgemm' and 'daxpy' from BLAS */

int m_x_m = m * m;
int m_x_d = m * d;
int NAsum;
int t = 0;

//integers and double precisions used in dcopy and dgemm
int intone = 1;
double dblone = 1.0, dblminusone = -1.0, dblzero = 0.0;
//To transpose or not transpose matrix
char *transpose = "T", *dont_transpose = "N";

/* NA detection */
int *NAindices = malloc(sizeof(int) * d);
int *positions = malloc(sizeof(int) * d);

/* Reduced arrays when NA's at time t */
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
// double V;
double Ft;
// double tmpFtinv;

double *at = malloc(sizeof(double) * m); 
double *Pt = malloc(sizeof(double) * m * m);
// double *Kt = malloc(sizeof(double) * m);

//SP temporary arrays:
double *tmpmxSP = (double *) Calloc(m, double);
double *tmpmxm = (double *) Calloc(m_x_m, double);

*loglik = 0;

// intone = 1;

/* at = a0 */
F77_NAME(dcopy)(&m, a0, &intone, at, &intone);

/* Pt = P0 */
F77_NAME(dcopy)(&m_x_m, P0, &intone, Pt, &intone);


//att
F77_NAME(dcopy)(&m, at, &intone, &at_output[m * t + m], &intone);
F77_NAME(dcopy)(&m, Pt, &intone, &pt_output[m * t + m], &intone);	

/*Initial State Outputs:*/
F77_NAME(dcopy)(&m_x_m, Pt, &intone, &pt_output[0], &intone);
F77_NAME(dcopy)(&m, at, &intone, &at_output[0], &intone);

/*********************************************************************************/
/* ---------- ----------------- Begin Kalman Filter ----------------- ---------- */
/*********************************************************************************/
  
//Recursion across all time steps:
while(t < n){

//How many NA's at time t?
NAsum = numberofNA(&yt[d * t], NAindices, positions, d);

#ifdef DEBUGME
Rprintf("\nNumber of NAs in iter %i: %i\n", t, NAsum);
#endif

/*********************************************************************************/
/* ---------- ---------- ---------- filter step ---------- ---------- ---------- */
/*********************************************************************************/
	
	/*****************************************/
	/* ---------- case 1: no NA's:---------- */
	/*****************************************/
    if(NAsum == 0)
    {
  	//Create Zt for time t
  	F77_NAME(dcopy)(&m_x_d, &Zt[m_x_d * t * incZt], &intone, Zt_t, &intone);
  	        	
  	//Increment number of observations:
 	N_obs += d;

	//Sequential Processing - Univariate Treatment of the Multivariate Series:
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
    	//Compute Vt[SP,t] = yt[SP,t] - ct[SP,t * incct] + Zt[SP,,t * incZt] %*% at[SP,t]
    	
    	//vt[SP,t] = yt[SP,t] - ct[SP,t * incct]
    	// V = yt[SP + d * t] - ct[SP + d * t * incct];
		vt_output[SP + d * t] = yt[SP + d * t] - ct[SP + d * t * incct];
    	
	    #ifdef DEBUGME
	    if(SP == SP_int)
	    {
	      	if(t == i_int){
	    print_array(Zt_tSP, 1, m, "Zt[SP,]");
	    Rprintf("\n V - Pre Mat Mult: %f", V);      		
			}
		}
	    #endif

    	//vt[SP,t] = vt[SP,t] - Zt[SP,, t * incZt] %*% at[,t]
    	// F77_NAME(dgemm)(dont_transpose, dont_transpose, &intone,
		// 	&intone, &m, &dblminusone,
		// 	Zt_tSP, &intone,
		// 	at, &m,
		// 	&dblone, &V, &intone FCONE FCONE);			  
    	F77_NAME(dgemm)(dont_transpose, dont_transpose, &intone,
			&intone, &m, &dblminusone,
			Zt_tSP, &intone,
			at, &m,
			&dblone, &vt_output[SP + d * t], &intone FCONE FCONE);			  
		
		#ifdef DEBUGME
		if(SP == SP_int)
		{
			if(t == i_int){
			Rprintf("\n V - Post Mat Mult: %f", V);
			}
		}
	    #endif
				
		//Step 2 - Function of Covariance Matrix:
		//Compute Ft = Zt[SP,,t * incZt] %*% Pt %*% t(Zt[SP,,t * incZt]) + diag(GGt)[SP]
		
		//First, Let us calculate:
		//Pt %*% t(Zt[SP,,t * incZt])
		//because we use this result twice

	    F77_NAME(dgemm)(dont_transpose, transpose, &m,
	    	&intone, &m, &dblone,
	    	Pt, &m,
	    	Zt_tSP, &intone,
	    	&dblzero, tmpmxSP, &m FCONE FCONE);
		
		
		#ifdef DEBUGME
	    if(SP == SP_int)
	    {
	      	if(t == i_int){
			print_array(tmpmxSP, m, 1, "tmpmxSP");
			}
		}
	    #endif
		
		//Ft = GGt[SP]
		Ft = GGt[SP + (d * t * incGGt)];

		#ifdef DEBUGME
		if(SP == SP_int)
	    {
	      	if(t == i_int){
	      	Rprintf("\n Ft: %f \n", Ft);
			}
		}
	    #endif
		
		//Ft = Zt[SP,,t*incZt] %*% tmpmxSP + Ft
		F77_NAME(dgemm)(dont_transpose, dont_transpose, &intone,
	    	&intone, &m, &dblone,
	    	Zt_tSP, &intone,
	    	tmpmxSP, &m,
	    	&dblone, &Ft, &intone FCONE FCONE);		

		#ifdef DEBUGME
	    if(SP == SP_int)
	    {
	      	if(t == i_int){
			Rprintf("\n New Ft: %f \n", Ft);
			}
		}
	    #endif	   

    	//Step 3 - Calculate the Kalman Gain:
    	//Compute Kt = Pt %*% t(Zt[SP,,i * incZt]) %*% (1/Ft)
    	
    	//Inv Ft:
    	// tmpFtinv = 
		Ftinv_output[SP + d * t] = 1 / Ft;

    	
		#ifdef DEBUGME
	    if(SP == SP_int)
	    {
	      	if(t == i_int){
			Rprintf("\n Inverse Ft: %f \n", tmpFtinv);
			}
		}
	    #endif	   
		    	
    	//Kt is an m x 1 matrix
    	
    	//We already have tmpSPxm:    	
        //Kt = tmpmxSP %*% tmpFtinv
		// F77_NAME(dgemm)(dont_transpose, dont_transpose, 
		// 	&m, &intone, &intone, 
		// 	&dblone, tmpmxSP, &m,
		// 	&Ftinv_output[SP + d * t], &intone,
		// 	&dblzero, Kt, &m FCONE FCONE);
		F77_NAME(dgemm)(dont_transpose, dont_transpose, 
			&m, &intone, &intone, 
			&dblone, tmpmxSP, &m,
			&Ftinv_output[SP + d * t], &intone,
			&dblzero, &Kt_output[m * t], &m FCONE FCONE);




		#ifdef DEBUGME
	    if(SP == SP_int)
	    {
	    	if(t == i_int){
	    	print_array(Kt, m, 1, "Kalman Gain");
			}
		}
	    #endif	   

		#ifdef DEBUGME
	    if(SP == SP_int)
	    {
	      	if(t == i_int){
	    	Rprintf("\n V - Post Mat Mult: %f", V);
			}
		}
	    #endif
  
		//Step 4 - Correct State Vector mean and Covariance:

       //Correction to att based upon prediction error:
       //att = Kt %*% V + att
		// F77_NAME(dgemm)(dont_transpose, dont_transpose, 
		// 	&m, &intone, &intone, 
		// 	&dblone, &Kt_output[m*t], &m,
		// 	&V, &intone,
		// 	&dblone, at, &m FCONE FCONE);
		F77_NAME(dgemm)(dont_transpose, dont_transpose, 
			&m, &intone, &intone, 
			&dblone, &Kt_output[m*t], &m,
			&vt_output[SP + d * t], &intone,
			&dblone, at, &m FCONE FCONE);

		#ifdef DEBUGME
		if(SP == SP_int)
		{
			if(t == i_int){
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
			&Kt_output[m*t], &m,
			&dblone, Pt, &m FCONE FCONE);
      
		#ifdef DEBUGME
		if(SP == SP_int)
		{
			if(t == i_int){
			print_array(Pt, m, m, "pt correction");      
			}
		}
		#endif	   
	
		//Step 5 - Update Log-Likelihood Score:
		*loglik -= 0.5 * (log(Ft) + (vt_output[SP + d * t] * vt_output[SP + d * t] * Ftinv_output[SP + d * t]));

		#ifdef DEBUGME
		Rprintf("\n Log-Likelihood: %f \n", *loglik);
		#endif   
	   
	  }
}
	/*******************************************/
	/* ---------- case 2: some NA's ---------- */
	/*******************************************/
	else
	{
	int d_reduced = d - NAsum;
	N_obs += d_reduced;
    
    //Temporary, reduced arrays:
    reduce_array(&yt[d * t], d, 1, yt_temp, positions, d_reduced);
    reduce_array(&ct[d * t * incct], d, 1, ct_temp, positions, d_reduced);
    reduce_array(&Zt[m_x_d * t * incZt], d, m, Zt_temp, positions, d_reduced);
    reduce_array(&GGt[d * t * incGGt], d, 1, GGt_temp, positions, d_reduced);

	#ifdef DEBUGME
	print_int_array(positions, 1, d_reduced, "positions");
	print_array(yt_temp, 1, d_reduced, "yt_temp");
	print_array(ct_temp, 1, d_reduced, "ct_temp");
	print_array(Zt_temp, d_reduced, m, "Zt_temp");
	print_array(GGt_temp, 1, d_reduced, "GGt_temp");              
	#endif

	//Sequential Processing - Univariate Treatment of the Multivariate Series:
    for(int SP=0; SP < d_reduced; SP++)
    {    	
    	
		#ifdef DEBUGME
		Rprintf("SP = %i", SP);
		#endif    
	
	    //Get the specific values of Z for SP:
	    for(int j = 0; j < m; j++)
	    {
	    Zt_tSP[j] = Zt_temp[SP + j*d_reduced]; 	
		}
	
		#ifdef DEBUGME
		print_array(Zt_tSP, 1, m, "Zt_tSP:");
		#endif    

		//Step 1 - Measurement Error:
		//Compute Vt[SP,t] = yt[SP,t] - ct[SP,t * incct] + Zt[SP,,t * incZt] %*% at[SP,t]
		
		//vt[SP,t] = yt[SP,t] - ct[SP,t * incct]
		// V = yt_temp[SP] - ct_temp[SP];
		vt_output[SP + d * t] = yt_temp[SP] - ct_temp[SP];

		#ifdef DEBUGME
		Rprintf("\n Pre mat-mult V = %f", V);
		#endif

    	//vt[SP,t] = vt[SP,t] - Zt[SP,, t * incZt] %*% at[,t]
		// F77_NAME(dgemm)(dont_transpose, dont_transpose, &intone,
		// 	&intone, &m, &dblminusone,
		// 	Zt_tSP, &intone,
		// 	at, &m,
		// 	&dblone, &V, &intone FCONE FCONE);		
		F77_NAME(dgemm)(dont_transpose, dont_transpose, &intone,
			&intone, &m, &dblminusone,
			Zt_tSP, &intone,
			at, &m,
			&dblone, &vt_output[SP + d * t], &intone FCONE FCONE);		

		#ifdef DEBUGME
		Rprintf("\n Post mat-mult V = %f", V);
		#endif
		
		//Step 2 - Function of Covariance Matrix:
		//Compute Ft = Zt[SP,,t * incZt] %*% Pt %*% t(Zt[SP,,t * incZt]) + diag(GGt)[SP]
		
		//First, Let us calculate:
		//Pt %*% t(Zt[SP,,t * incZt])
		//because we use this result twice

		F77_NAME(dgemm)(dont_transpose, transpose, &m,
			&intone, &m, &dblone,
			Pt, &m,
			Zt_tSP, &intone,
			&dblzero, tmpmxSP, &m FCONE FCONE);
		
		#ifdef DEBUGME
		print_array(tmpmxSP, m, 1, "tmpmxSP:");	  
		#endif		
		
		//Ft = GGt[SP]
		Ft = GGt_temp[SP];
		
		#ifdef DEBUGME
		Rprintf("\n Ft: %f \n", Ft);
		#endif

	    //Ft = Zt[SP,,i*incZt] %*% tmpmxSP + Ft
		F77_NAME(dgemm)(dont_transpose, dont_transpose, &intone,
		    &intone, &m, &dblone,
		    Zt_tSP, &intone,
	    	tmpmxSP, &m,
	    	&dblone, &Ft, &intone FCONE FCONE);		

    	//Step 3 - Calculate the Kalman Gain:
    	//Compute Kt = Pt %*% t(Zt[SP,,i * incZt]) %*% (1/Ft)
    	
		#ifdef DEBUGME
		Rprintf("\n New Ft: %f \n", Ft);
		#endif
		
		//Inv Ft:
		// tmpFtinv = 1 / Ft;
		Ftinv_output[SP + d * t] = 1 / Ft;


		#ifdef DEBUGME
		Rprintf("\n Inverse Ft: %f \n", tmpFtinv);
		#endif
		
    	//Kt is an m x 1 matrix
    	
		//We already have tmpSPxm:    	
        //Kt = tmpmxSP %*% tmpFtinv
		// F77_NAME(dgemm)(dont_transpose, dont_transpose, &m,
		// 	&intone, &intone, &dblone,
		// 	tmpmxSP, &m,
		// 	&Ftinv_output[SP + d * t], &intone,
		// 	&dblzero, Kt, &m FCONE FCONE);
		F77_NAME(dgemm)(dont_transpose, dont_transpose, 
			&m, &intone, &intone, 
			&dblone, tmpmxSP, &m,
			&Ftinv_output[SP + d * t], &intone,
			&dblzero, &Kt_output[m * t], &m FCONE FCONE);
	
		  
		//Step 4 - Correct State Vector mean and Covariance:
		
		//Correction to att based upon prediction error:
		//att = Kt %*% V + att
		// F77_NAME(dgemm)(dont_transpose, dont_transpose, &m,
		// 	&intone, &intone, &dblone,
		// 	&Kt_output[m*t], &m,
		// 	&V, &intone,
		// 	&dblone, at, &m FCONE FCONE);
		F77_NAME(dgemm)(dont_transpose, dont_transpose, &m,
			&intone, &intone, &dblone,
			&Kt_output[m*t], &m,
			&vt_output[SP + d * t], &intone,
			&dblone, at, &m FCONE FCONE);
    	
		//Correction to covariance based upon Kalman Gain:
		//ptt = ptt - ptt %*% t(Z[SP,,i * incZt]) %*% t(Ktt)
		//ptt = ptt - tempmxSP %*% t(Ktt)
		F77_NAME(dgemm)(dont_transpose, transpose, &m,
			&m, &intone, &dblminusone,
			tmpmxSP, &m,
			&Kt_output[m*t], &m,
			&dblone, Pt, &m FCONE FCONE);
		  	  	
		//Step 5 - Update Log-Likelihood Score:
		*loglik -= 0.5 * (log(Ft) + vt_output[SP + d * t] * vt_output[SP + d * t] * Ftinv_output[SP + d * t]);
		//Increment number of observations for the Log-likelihood at the end:
    }
	#ifdef DEBUGME
	Rprintf("\n SP Iteration Completed Successfully.\n");
	#endif
}
/*********************************************************************************/
/*  ---------- ---------- ------- prediction step -------- ---------- ---------- */
/*********************************************************************************/
	
	/* ---------------------------------------------------------------------- */
	/* at[,t + 1] = dt[,t * incdt] + Tt[,,t * incTt] %*% att[,t]              */
	/* ---------------------------------------------------------------------- */

	#ifdef DEBUGME
	print_array(at, 1, m, "at:");
	#endif

	//tmpmxm = Tt[,,i * incTt] %*% att[,i]
	F77_NAME(dgemm)(dont_transpose, dont_transpose, &m,
		&intone, &m, &dblone,
		&Tt[m_x_m * t * incTt], &m,
		at, &m,
		&dblzero, tmpmxSP, &m FCONE FCONE);
    
	//att
	F77_NAME(dcopy)(&m, at, &intone, &att_output[m * t], &intone);
	F77_NAME(dcopy)(&m, Pt, &intone, &ptt_output[m * t], &intone);	
	
	  
	/* at[,t + 1] = dt[,t] + at[,t] */
	F77_NAME(dcopy)(&m, &dt[m * t * incdt], &intone, at, &intone);
	F77_NAME(daxpy)(&m, &dblone, tmpmxSP, &intone, at, &intone);
    
	  
	#ifdef DEBUGME
	print_array(at, 1, m, "atp1:");
	print_array(Pt, m, m, "Pt:");      
	#endif
 
	/* ------------------------------------------------------------------------------------- */
	/* Pt[,,t + 1] = Tt[,,t * incTt] %*% Ptt[,,t] %*% t(Tt[,,t * incTt]) + HHt[,,t * incHHt] */
	/* ------------------------------------------------------------------------------------- */

	/* tmpmxm = Ptt[,,i] %*% t(Tt[,,i * incTt]) */
	F77_NAME(dgemm)(dont_transpose, transpose, &m,
		&m, &m, &dblone,
		Pt, &m,
		&Tt[m_x_m * t * incTt], &m,
		&dblzero, tmpmxm, &m FCONE FCONE);

	/* Pt[,,i + 1] = HHt[,,i * incHHt] */      
	F77_NAME(dcopy)(&m_x_m, &HHt[m_x_m * t * incHHt], &intone, Pt, &intone);

	#ifdef DEBUGME
	print_array(&HHt[m_x_m * t * incHHt], m, m, "HHt:");
	#endif

	/* Pt[,,i + 1] = Tt[,,i * incTt] %*% tmpmxm + Pt[,,i + 1] */
	F77_NAME(dgemm)(dont_transpose, dont_transpose, &m,
		&m, &m, &dblone,
		&Tt[m_x_m * t * incTt], &m,
		tmpmxm, &m,
		&dblone, Pt, &m FCONE FCONE);      
		      
	#ifdef DEBUGME
	print_array(Pt, m, m, "Ptp1:");      
	#endif
		             
	#ifdef DEBUGME
	print_array(&at, 1, m, "at:");
	print_array(&Pt, m, m, "Pt:");
	Rprintf("\n---------- iteration nr. %i ----------\n", i+1);
	#endif

	//att
	F77_NAME(dcopy)(&m, at, &intone, &at_output[m * t + m], &intone);
	F77_NAME(dcopy)(&m, Pt, &intone, &pt_output[m * t + m], &intone);	


    //end iteration
    t++;
}
/**************************************************************/
/* ---------- ---------- end recursions ---------- ---------- */
/**************************************************************/

#ifdef DEBUGME
Rprintf("\n Log-Likelihood: %f \n", *loglik);
#endif


//Update the final Log-Likelihood Score:
*loglik -= 0.5 * N_obs * log(2 * PI);

//Memory clean - free vectors / matrices:
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
// free(Kt);

#ifdef DEBUGME
Rprintf("\n---------- Recursion Complete ----------\n");
#endif


}
/*********************************************************************************/
/* ---------- ------------------ End Kalman Filter ------------------ ---------- */
/*********************************************************************************/
/************************************************************************************/
/* ---------- ---------- Kalman filter: Sequential Processing ---------- ---------- */
/************************************************************************************/
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
	/* output */
	double * loglik)
{
/*  Notation: */

//m: the dimension of the state vector
//d: the dimension of observations
//n: the total number of observations

/*  loglik is the output of the function and denotes the log-likelihood */
/*  cfkf uses 'dcopy', 'dgemm' and 'daxpy' from BLAS */

int m_x_m = m * m;
int m_x_d = m * d;
int NAsum;
int t = 0;

//integers and double precisions used in dcopy and dgemm
int intone = 1;
double dblone = 1.0, dblminusone = -1.0, dblzero = 0.0;
//To transpose or not transpose matrix
char *transpose = "T", *dont_transpose = "N";

/* NA detection */
int *NAindices = malloc(sizeof(int) * d);
int *positions = malloc(sizeof(int) * d);

/* Reduced arrays when NA's at time t */
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
double tmpFtinv;

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

/*********************************************************************************/
/* ---------- ----------------- Begin Kalman Filter ----------------- ---------- */
/*********************************************************************************/
  
//Recursion across all time steps:
while(t < n){

//How many NA's at time t?
NAsum = numberofNA(&yt[d * t], NAindices, positions, d);

#ifdef DEBUGME
Rprintf("\nNumber of NAs in iter %i: %i\n", t, NAsum);
#endif

/*********************************************************************************/
/* ---------- ---------- ---------- filter step ---------- ---------- ---------- */
/*********************************************************************************/
	
	/*****************************************/
	/* ---------- case 1: no NA's:---------- */
	/*****************************************/
    if(NAsum == 0)
    {
  	//Create Zt for time t
  	F77_NAME(dcopy)(&m_x_d, &Zt[m_x_d * t * incZt], &intone, Zt_t, &intone);
  	        	
  	//Increment number of observations:
 	N_obs += d;

	//Sequential Processing - Univariate Treatment of the Multivariate Series:
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
    	//Compute Vt[SP,t] = yt[SP,t] - ct[SP,t * incct] + Zt[SP,,t * incZt] %*% at[SP,t]
    	
    	//vt[SP,t] = yt[SP,t] - ct[SP,t * incct]
    	V = yt[SP + d * t] - ct[SP + d * t * incct];
    	
	    #ifdef DEBUGME
	    if(SP == SP_int)
	    {
	      	if(t == i_int){
	    print_array(Zt_tSP, 1, m, "Zt[SP,]");
	    Rprintf("\n V - Pre Mat Mult: %f", V);      		
			}
		}
	    #endif

    	//vt[SP,t] = vt[SP,t] - Zt[SP,, t * incZt] %*% at[,t]
    	F77_NAME(dgemm)(dont_transpose, dont_transpose, &intone,
			&intone, &m, &dblminusone,
			Zt_tSP, &intone,
			at, &m,
			&dblone, &V, &intone FCONE FCONE);			  
		
		#ifdef DEBUGME
		if(SP == SP_int)
		{
			if(t == i_int){
			Rprintf("\n V - Post Mat Mult: %f", V);
			}
		}
	    #endif
				
		//Step 2 - Function of Covariance Matrix:
		//Compute Ft = Zt[SP,,t * incZt] %*% Pt %*% t(Zt[SP,,t * incZt]) + diag(GGt)[SP]
		
		//First, Let us calculate:
		//Pt %*% t(Zt[SP,,t * incZt])
		//because we use this result twice

	    F77_NAME(dgemm)(dont_transpose, transpose, &m,
	    	&intone, &m, &dblone,
	    	Pt, &m,
	    	Zt_tSP, &intone,
	    	&dblzero, tmpmxSP, &m FCONE FCONE);
		
		
		#ifdef DEBUGME
	    if(SP == SP_int)
	    {
	      	if(t == i_int){
			print_array(tmpmxSP, m, 1, "tmpmxSP");
			}
		}
	    #endif
		
		//Ft = GGt[SP]
		Ft = GGt[SP + (d * t * incGGt)];

		#ifdef DEBUGME
		if(SP == SP_int)
	    {
	      	if(t == i_int){
	      	Rprintf("\n Ft: %f \n", Ft);
			}
		}
	    #endif
		
		//Ft = Zt[SP,,t*incZt] %*% tmpmxSP + Ft
		F77_NAME(dgemm)(dont_transpose, dont_transpose, &intone,
	    	&intone, &m, &dblone,
	    	Zt_tSP, &intone,
	    	tmpmxSP, &m,
	    	&dblone, &Ft, &intone FCONE FCONE);		

		#ifdef DEBUGME
	    if(SP == SP_int)
	    {
	      	if(t == i_int){
			Rprintf("\n New Ft: %f \n", Ft);
			}
		}
	    #endif	   

    	//Step 3 - Calculate the Kalman Gain:
    	//Compute Kt = Pt %*% t(Zt[SP,,i * incZt]) %*% (1/Ft)
    	
    	//Inv Ft:
    	tmpFtinv = 1 / Ft;
    	
		#ifdef DEBUGME
	    if(SP == SP_int)
	    {
	      	if(t == i_int){
			Rprintf("\n Inverse Ft: %f \n", tmpFtinv);
			}
		}
	    #endif	   
		    	
    	//Kt is an m x 1 matrix
    	
    	//We already have tmpSPxm:    	
        //Kt = tmpmxSP %*% tmpFtinv
		F77_NAME(dgemm)(dont_transpose, dont_transpose, 
			&m, &intone, &intone, 
			&dblone, tmpmxSP, &m,
			&tmpFtinv, &intone,
			&dblzero, Kt, &m FCONE FCONE);
					  
		#ifdef DEBUGME
	    if(SP == SP_int)
	    {
	    	if(t == i_int){
	    	print_array(Kt, m, 1, "Kalman Gain");
			}
		}
	    #endif	   

		#ifdef DEBUGME
	    if(SP == SP_int)
	    {
	      	if(t == i_int){
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
			&dblone, at, &m FCONE FCONE);

		#ifdef DEBUGME
		if(SP == SP_int)
		{
			if(t == i_int){
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
			&dblone, Pt, &m FCONE FCONE);
      
		#ifdef DEBUGME
		if(SP == SP_int)
		{
			if(t == i_int){
			print_array(Pt, m, m, "pt correction");      
			}
		}
		#endif	   
	
		//Step 5 - Update Log-Likelihood Score:
		*loglik -= 0.5 * (log(Ft) + (V * V * tmpFtinv));

		#ifdef DEBUGME
		Rprintf("\n Log-Likelihood: %f \n", *loglik);
		#endif   
	   
	  }
}
	/*******************************************/
	/* ---------- case 2: some NA's ---------- */
	/*******************************************/
	else
	{
	int d_reduced = d - NAsum;
	N_obs += d_reduced;
    
    //Temporary, reduced arrays:
    reduce_array(&yt[d * t], d, 1, yt_temp, positions, d_reduced);
    reduce_array(&ct[d * t * incct], d, 1, ct_temp, positions, d_reduced);
    reduce_array(&Zt[m_x_d * t * incZt], d, m, Zt_temp, positions, d_reduced);
    reduce_array(&GGt[d * t * incGGt], d, 1, GGt_temp, positions, d_reduced);

	#ifdef DEBUGME
	print_int_array(positions, 1, d_reduced, "positions");
	print_array(yt_temp, 1, d_reduced, "yt_temp");
	print_array(ct_temp, 1, d_reduced, "ct_temp");
	print_array(Zt_temp, d_reduced, m, "Zt_temp");
	print_array(GGt_temp, 1, d_reduced, "GGt_temp");              
	#endif

	//Sequential Processing - Univariate Treatment of the Multivariate Series:
    for(int SP=0; SP < d_reduced; SP++)
    {    	
    	
		#ifdef DEBUGME
		Rprintf("SP = %i", SP);
		#endif    
	
	    //Get the specific values of Z for SP:
	    for(int j = 0; j < m; j++)
	    {
	    Zt_tSP[j] = Zt_temp[SP + j*d_reduced]; 	
		}
	
		#ifdef DEBUGME
		print_array(Zt_tSP, 1, m, "Zt_tSP:");
		#endif    

		//Step 1 - Measurement Error:
		//Compute Vt[SP,t] = yt[SP,t] - ct[SP,t * incct] + Zt[SP,,t * incZt] %*% at[SP,t]
		
		//vt[SP,t] = yt[SP,t] - ct[SP,t * incct]
		V = yt_temp[SP] - ct_temp[SP];
		
		#ifdef DEBUGME
		Rprintf("\n Pre mat-mult V = %f", V);
		#endif

    	//vt[SP,t] = vt[SP,t] - Zt[SP,, t * incZt] %*% at[,t]
		F77_NAME(dgemm)(dont_transpose, dont_transpose, &intone,
			&intone, &m, &dblminusone,
			Zt_tSP, &intone,
			at, &m,
			&dblone, &V, &intone FCONE FCONE);		
		
		#ifdef DEBUGME
		Rprintf("\n Post mat-mult V = %f", V);
		#endif
		
		//Step 2 - Function of Covariance Matrix:
		//Compute Ft = Zt[SP,,t * incZt] %*% Pt %*% t(Zt[SP,,t * incZt]) + diag(GGt)[SP]
		
		//First, Let us calculate:
		//Pt %*% t(Zt[SP,,t * incZt])
		//because we use this result twice

		F77_NAME(dgemm)(dont_transpose, transpose, &m,
			&intone, &m, &dblone,
			Pt, &m,
			Zt_tSP, &intone,
			&dblzero, tmpmxSP, &m FCONE FCONE);
		
		#ifdef DEBUGME
		print_array(tmpmxSP, m, 1, "tmpmxSP:");	  
		#endif		
		
		//Ft = GGt[SP]
		Ft = GGt_temp[SP];
		
		#ifdef DEBUGME
		Rprintf("\n Ft: %f \n", Ft);
		#endif

	    //Ft = Zt[SP,,i*incZt] %*% tmpmxSP + Ft
		F77_NAME(dgemm)(dont_transpose, dont_transpose, &intone,
		    &intone, &m, &dblone,
		    Zt_tSP, &intone,
	    	tmpmxSP, &m,
	    	&dblone, &Ft, &intone FCONE FCONE);		

    	//Step 3 - Calculate the Kalman Gain:
    	//Compute Kt = Pt %*% t(Zt[SP,,i * incZt]) %*% (1/Ft)
    	
		#ifdef DEBUGME
		Rprintf("\n New Ft: %f \n", Ft);
		#endif
		
		//Inv Ft:
		tmpFtinv = 1 / Ft;
    	
		#ifdef DEBUGME
		Rprintf("\n Inverse Ft: %f \n", tmpFtinv);
		#endif
		
    	//Kt is an m x 1 matrix
    	
		//We already have tmpSPxm:    	
        //Kt = tmpmxSP %*% tmpFtinv
		F77_NAME(dgemm)(dont_transpose, dont_transpose, &m,
			&intone, &intone, &dblone,
			tmpmxSP, &m,
			&tmpFtinv, &intone,
			&dblzero, Kt, &m FCONE FCONE);
	
		  
		//Step 4 - Correct State Vector mean and Covariance:
		
		//Correction to att based upon prediction error:
		//att = Kt %*% V + att
		F77_NAME(dgemm)(dont_transpose, dont_transpose, &m,
			&intone, &intone, &dblone,
			Kt, &m,
			&V, &intone,
			&dblone, at, &m FCONE FCONE);
    	
		//Correction to covariance based upon Kalman Gain:
		//ptt = ptt - ptt %*% t(Z[SP,,i * incZt]) %*% t(Ktt)
		//ptt = ptt - tempmxSP %*% t(Ktt)
		F77_NAME(dgemm)(dont_transpose, transpose, &m,
			&m, &intone, &dblminusone,
			tmpmxSP, &m,
			Kt, &m,
			&dblone, Pt, &m FCONE FCONE);
		  	  	
		//Step 5 - Update Log-Likelihood Score:
		*loglik -= 0.5 * (log(Ft) + V * V * tmpFtinv);
		//Increment number of observations for the Log-likelihood at the end:
    }
	#ifdef DEBUGME
	Rprintf("\n SP Iteration Completed Successfully.\n");
	#endif
}
/*********************************************************************************/
/*  ---------- ---------- ------- prediction step -------- ---------- ---------- */
/*********************************************************************************/
	
	/* ---------------------------------------------------------------------- */
	/* at[,t + 1] = dt[,t * incdt] + Tt[,,t * incTt] %*% att[,t]              */
	/* ---------------------------------------------------------------------- */

	#ifdef DEBUGME
	print_array(at, 1, m, "at:");
	#endif

	//tmpmxm = Tt[,,i * incTt] %*% att[,i]
	F77_NAME(dgemm)(dont_transpose, dont_transpose, &m,
		&intone, &m, &dblone,
		&Tt[m_x_m * t * incTt], &m,
		at, &m,
		&dblzero, tmpmxSP, &m FCONE FCONE);
      
	/* at[,t + 1] = dt[,t] */
	F77_NAME(dcopy)(&m, &dt[m * t * incdt], &intone, at, &intone);
	F77_NAME(daxpy)(&m, &dblone, tmpmxSP, &intone, at, &intone);
      
	#ifdef DEBUGME
	print_array(at, 1, m, "atp1:");
	print_array(Pt, m, m, "Pt:");      
	#endif
 
	/* ------------------------------------------------------------------------------------- */
	/* Pt[,,t + 1] = Tt[,,t * incTt] %*% Ptt[,,t] %*% t(Tt[,,t * incTt]) + HHt[,,t * incHHt] */
	/* ------------------------------------------------------------------------------------- */

	/* tmpmxm = Ptt[,,i] %*% t(Tt[,,i * incTt]) */
	F77_NAME(dgemm)(dont_transpose, transpose, &m,
		&m, &m, &dblone,
		Pt, &m,
		&Tt[m_x_m * t * incTt], &m,
		&dblzero, tmpmxm, &m FCONE FCONE);

	/* Pt[,,i + 1] = HHt[,,i * incHHt] */      
	F77_NAME(dcopy)(&m_x_m, &HHt[m_x_m * t * incHHt], &intone, Pt, &intone);

	#ifdef DEBUGME
	print_array(&HHt[m_x_m * t * incHHt], m, m, "HHt:");
	#endif

	/* Pt[,,i + 1] = Tt[,,i * incTt] %*% tmpmxm + Pt[,,i + 1] */
	F77_NAME(dgemm)(dont_transpose, dont_transpose, &m,
		&m, &m, &dblone,
		&Tt[m_x_m * t * incTt], &m,
		tmpmxm, &m,
		&dblone, Pt, &m FCONE FCONE);      
		      
	#ifdef DEBUGME
	print_array(Pt, m, m, "Ptp1:");      
	#endif
		             
	#ifdef DEBUGME
	print_array(&at, 1, m, "at:");
	print_array(&Pt, m, m, "Pt:");
	Rprintf("\n---------- iteration nr. %i ----------\n", i+1);
	#endif

    //end iteration
    t++;
}
/**************************************************************/
/* ---------- ---------- end recursions ---------- ---------- */
/**************************************************************/

#ifdef DEBUGME
Rprintf("\n Log-Likelihood: %f \n", *loglik);
#endif


//Update the final Log-Likelihood Score:
*loglik -= 0.5 * N_obs * log(2 * PI);

//Memory clean - free vectors / matrices:
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

#ifdef DEBUGME
Rprintf("\n---------- Recursion Complete ----------\n");
#endif


}
/*********************************************************************************/
/* ---------- ------------------ End Kalman Filter ------------------ ---------- */
/*********************************************************************************/


/************************************************************************************/
/* ---------- --------- Kalman Smoothing Through Sequential Processing --------- ---*/
/************************************************************************************/
// This function performs Kalman smoothing. It iterates backwards through t.
// void cfks_SP(/* Inputs */
// 	int m, int d, int n,
// 	double * yt,
// 	double * Zt, int incZt,
//     double * vt,
//     double * Tt, int incTt,
//     double * Kt,
//     double * Ftinv,
//     double * at,
//     double * Pt
// /* No outputs? Returns ahatt in at and Vt in Pt */
// )
// {

//     int m_x_m = m * m;
//     // int d_x_d = d * d;
//     int m_x_d = m * d;

//     int NAsum;
//     // Kalman smoothing iterates backwards:
//     int t = n - 1;


//     /* integers and double precisions used in dcopy and dgemm */
//     int intone = 1;
//     double dblone = 1.0, dblminusone = -1.0, dblzero = 0.0;
//     char *transpose = "T", *dont_transpose = "N";

//     /* temporary arrays */
//     double *tmpmxd = (double *) Calloc(m_x_d, double);
//     double *tmpmxm = (double *) Calloc(m_x_m, double);
//     double *tmpPt = (double *) Calloc(m_x_m, double);
//     double *tmpN = (double *) Calloc(m_x_m, double);
//     // double *tmpL = (double *) Calloc(m_x_m, double);

//     /* temporary vecs */
//     double *tmpr = (double *) Calloc(m, double);

//     /* recursion parameters */
//     double *N = (double *) Calloc(m_x_m,double);
//     double *r = (double *) Calloc(m,double);

//     double *L = (double *) Calloc(m_x_m, double);

//     /* NA detection */
//     int NAsum;
//     int *NAindices = malloc(sizeof(int) * d);
//     int *positions = malloc(sizeof(int) * d);

//     /* create reduced arrays for case 3 (see below) */
//     double *Zt_temp = malloc(sizeof(double) * (d - 1) * m);
//     double *vt_temp = malloc(sizeof(double) * (d - 1));
//     double *Ftinv_temp = malloc(sizeof(double) * (d - 1) * (d - 1));
//     double *Kt_temp = malloc(sizeof(double) * (d - 1) * m);

//     /* ---------- Begin iterations --------------*/
//     while(t>-1){

//         // The transitions from t to t-1 are identical to a standard Kalman smoother:
//         /* at[,i] = at[,i] + Pt[,,i] %*% r[,i-1] */
//         F77_NAME(dgemm)(dont_transpose, dont_transpose, &m,
//                 &intone, &m, &dblone,
//                 &Pt[m_x_m * t], &m,
//                 r, &m,
//                 &dblone, &at[m*t], &m FCONE FCONE);

// 		// /* tmpmxm = Pt[,,i] %*% tmpN */
// 		// F77_NAME(dgemm)(dont_transpose, dont_transpose, &m,
// 		// 		&m, &m, &dblone,
// 		// 		&Pt[m_x_m * i], &m, N, &m,
// 		// 		&dblzero, tmpmxm, &m FCONE FCONE);

// 		// /* Pt[,,i] = Pt[,,i] - tmpmxm%*% Pt[,,i] */
// 		// F77_NAME(dcopy)(&m_x_m, &Pt[m_x_m * i], &intone, tmpPt, &intone);
// 		// F77_NAME(dgemm)(dont_transpose, dont_transpose, &m, &m, &m, &dblminusone,
// 		// 		tmpmxm, &m, tmpPt, &m,
// 		// 		&dblone, &Pt[m_x_m *i], &m FCONE FCONE);


//         // How many NA's are in observation yt[,i] ?
//         NAsum = numberofNA(&yt[d*t], NAindices, positions, d);
        
// 		/*****************************************/
// 		/* ---------- case 1: no NA's:---------- */
// 		/*****************************************/
// 		if(NAsum == 0)
// 		{
// 			//Sequential Processing - Univariate Treatment of the Multivariate Series:
// 			for(int SP=d; SP > 0; SP--)
// 			{
	
// 			}


// }




/*********************************************************************************/
/* ---------- --------------- Convert between R and C  -------------- ---------- */
/*********************************************************************************/
SEXP fkf_SP_verbose(SEXP a0, SEXP P0, SEXP dt, SEXP ct, SEXP Tt,
	SEXP Zt, SEXP HHt, SEXP GGt, SEXP yt)
{

// Dimensions required for the called cfkf_SP_verbose function:
	int m = length(a0);
	int d = INTEGER(GET_DIM(yt))[0];
	int n = INTEGER(GET_DIM(yt))[1];
	
  	int m_x_n = m * n;
  	int m_x_m_x_n = m * m * n;
	int d_x_n = d * n;
  	double dbl_NA = NA_REAL;
    int intzero = 0, intone = 1;
// Symbolic expression - R data types. Essentially :
	
	SEXP loglik, at_output, att_output, pt_output, ptt_output, Ftinv_output,vt_output,Kt_output,ans, ans_names;
	SEXP dim_at, dim_att, dim_Pt, dim_Ptt, dim_Kt,dim_Ftinv,class_name;
	
  /* Allocate memory for objects to be returned and set values to NA. */
	PROTECT(loglik = NEW_NUMERIC(1));
	PROTECT(att_output = NEW_NUMERIC(m * n));
  	PROTECT(at_output = NEW_NUMERIC(m * (n + 1)));
  	PROTECT(ptt_output = NEW_NUMERIC(m * m * n));
  	PROTECT(pt_output = NEW_NUMERIC(m * m * (n + 1)));
  	PROTECT(Ftinv_output = NEW_NUMERIC(d * n));
  	PROTECT(vt_output = NEW_NUMERIC(d * n));
  	PROTECT(Kt_output = NEW_NUMERIC(m * n));


    //Set dimensions
    F77_NAME(dcopy)(&m_x_n, &dbl_NA, &intzero, NUMERIC_POINTER(att_output), &intone);
  	F77_NAME(dcopy)(&m_x_n, &dbl_NA, &intzero, NUMERIC_POINTER(at_output), &intone);
  	F77_NAME(dcopy)(&m_x_n, &dbl_NA, &intzero, NUMERIC_POINTER(Kt_output), &intone);
  	F77_NAME(dcopy)(&m_x_m_x_n, &dbl_NA, &intzero, NUMERIC_POINTER(ptt_output), &intone);
  	F77_NAME(dcopy)(&m_x_m_x_n, &dbl_NA, &intzero, NUMERIC_POINTER(pt_output), &intone);
  	F77_NAME(dcopy)(&d_x_n, &dbl_NA, &intzero, NUMERIC_POINTER(vt_output), &intone);
  	F77_NAME(dcopy)(&d_x_n, &dbl_NA, &intzero, NUMERIC_POINTER(Ftinv_output), &intone);


	cfkf_SP_verbose(m, d, n,
	// Input
		NUMERIC_POINTER(a0), NUMERIC_POINTER(P0),
		NUMERIC_POINTER(dt), INTEGER(GET_DIM(dt))[1] == n,
		NUMERIC_POINTER(ct), INTEGER(GET_DIM(ct))[1] == n,
		NUMERIC_POINTER(Tt), INTEGER(GET_DIM(Tt))[2] == n,
		NUMERIC_POINTER(Zt), INTEGER(GET_DIM(Zt))[2] == n,
		NUMERIC_POINTER(HHt), INTEGER(GET_DIM(HHt))[2] == n,
		NUMERIC_POINTER(GGt), INTEGER(GET_DIM(GGt))[1] == n,
	// Output
		NUMERIC_POINTER(yt),
		NUMERIC_POINTER(loglik),
		NUMERIC_POINTER(at_output),
		NUMERIC_POINTER(att_output),
		NUMERIC_POINTER(pt_output),
		NUMERIC_POINTER(ptt_output),
		NUMERIC_POINTER(Ftinv_output),
		NUMERIC_POINTER(vt_output),
		NUMERIC_POINTER(Kt_output)		
		);

  /* Produce named return list */
  PROTECT(ans = NEW_LIST(8));
  PROTECT(ans_names = NEW_CHARACTER(8));
  SET_STRING_ELT(ans_names, 0, mkChar("att"));
  SET_STRING_ELT(ans_names, 1, mkChar("at"));
  SET_STRING_ELT(ans_names, 2, mkChar("Ptt"));
  SET_STRING_ELT(ans_names, 3, mkChar("Pt"));
  SET_STRING_ELT(ans_names, 4, mkChar("Ftinv"));
  SET_STRING_ELT(ans_names, 5, mkChar("vt"));
  SET_STRING_ELT(ans_names, 6, mkChar("Kt"));
  SET_STRING_ELT(ans_names, 7, mkChar("logLik"));


  setAttrib(ans, R_NamesSymbol, ans_names);
  // Set matrix dimensions:
  PROTECT(dim_at = NEW_INTEGER(2));
  PROTECT(dim_att = NEW_INTEGER(2));

  PROTECT(dim_Kt = NEW_INTEGER(2));


  INTEGER(dim_at)[0] = m;
  INTEGER(dim_at)[1] = n + 1;

  INTEGER(dim_att)[0] = m;
  INTEGER(dim_att)[1] = n;

  INTEGER(dim_Kt)[0] = m;
  INTEGER(dim_Kt)[1] = n;

  setAttrib(at_output, R_DimSymbol, dim_at);
  setAttrib(att_output, R_DimSymbol, dim_att);
  setAttrib(Kt_output, R_DimSymbol, dim_Kt);

  /* Set array dimensions */
  PROTECT(dim_Pt = NEW_INTEGER(3));
  PROTECT(dim_Ptt = NEW_INTEGER(3));

  INTEGER(dim_Pt)[0] = m;
  INTEGER(dim_Pt)[1] = m;
  INTEGER(dim_Pt)[2] = n + 1;

  INTEGER(dim_Ptt)[0] = m;
  INTEGER(dim_Ptt)[1] = m;
  INTEGER(dim_Ptt)[2] = n;

  setAttrib(pt_output, R_DimSymbol, dim_Pt);
  setAttrib(ptt_output, R_DimSymbol, dim_Ptt);

  /* Set array dimensions */
  PROTECT(dim_Ftinv = NEW_INTEGER(2));
  INTEGER(dim_Ftinv)[0] = d;
  INTEGER(dim_Ftinv)[1] = n;

  setAttrib(Ftinv_output, R_DimSymbol, dim_Ftinv);
  setAttrib(vt_output, R_DimSymbol, dim_Ftinv);



  /* Fill the list */
  SET_VECTOR_ELT(ans, 0, att_output);
  SET_VECTOR_ELT(ans, 1, at_output);
  SET_VECTOR_ELT(ans, 2, ptt_output);
  SET_VECTOR_ELT(ans, 3, pt_output);
  SET_VECTOR_ELT(ans, 4, Ftinv_output);
  SET_VECTOR_ELT(ans, 5, vt_output);
  SET_VECTOR_ELT(ans, 6, Kt_output);
  SET_VECTOR_ELT(ans, 7, loglik);

  /* Set the class to 'fkf' */
  PROTECT(class_name = NEW_CHARACTER(1));
  SET_STRING_ELT(class_name, 0, mkChar("fkf.SP_verbose"));
  classgets(ans, class_name);

  UNPROTECT(17);
  return(ans);	
}

SEXP fkf_SP(SEXP a0, SEXP P0, SEXP dt, SEXP ct, SEXP Tt,
	SEXP Zt, SEXP HHt, SEXP GGt, SEXP yt)
{

// Dimensions required for the called cfkf_SP function:
	int m = length(a0);
	int d = INTEGER(GET_DIM(yt))[0];
	int n = INTEGER(GET_DIM(yt))[1];
	
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
