#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <complex.h>
#include <fftw3.h>
#include <cstdint>
#include <dlfcn.h>

#include "SAR_RDA.hpp"
#include "dash.h"

#define PROGPATH "./input/"
#define RAWDATA PROGPATH "rawdata_rda.txt"
#define OUTPUT "SAR_RDA-output.txt"

/***************************/
float *ta;
float *trng;

float *g;
float *g2;

float *S1;

float *src;
float *H;

float *sac;
float *s0;
/***************************/

float *_ta;
float *_trng;

float *_g;
float *_g2;

float *_S1;

float *_src;
float *_H;


float R0;
float Ka;

float *_sac;
float *_s0;

int Nslow;
int Nfast;
float v;
float Xmin;
float Xmax;
float Yc;
float Y0;
float Tr;
float Kr;
float h;
float lambda;
float c;
float Rmin, Rmax;

fftwf_complex *in_x1, *out_x1, *in_x2, *out_x2, *in_x3, *out_x3, *in_x4, *out_x4, *in_x5, *out_x5;
fftwf_plan p1, p2, p3, p4, p5;

fftwf_complex *_in_x1, *_out_x1, *_in_x2, *_out_x2, *_in_x3, *_out_x3, *_in_x4, *_out_x4, *_in_x5, *_out_x5;
fftwf_plan _p1, _p2, _p3, _p4, _p5;

// Pointer to use to hold the shared object file handle
void *dlhandle;
//void (*dash_fft_func)(float**, float**, size_t*, bool*);

void (*fft_accel_func)(dash_cmplx_flt_type**, dash_cmplx_flt_type**, size_t*, bool*, uint8_t);
void (*ifft_accel_func)(dash_cmplx_flt_type**, dash_cmplx_flt_type**, size_t*, bool*, uint8_t);

__attribute__((__visibility__("default"))) thread_local unsigned int __CEDR_CLUSTER_IDX__ = 0;

void __attribute__((constructor)) setup(void) {	
  printf("[SAR_RDA] intializing variables\n");
  
  c = 3e8;
  Nslow = 256;
  Nfast = 512;
  v = 150;
  Xmin = 0;
  Xmax = 50;
  Yc = 10000;
  Y0 = 500;
  Tr = 2.5e-6;
  Kr = 2e13;
  h = 5000;
  lambda = 0.0566;
 
/*******************************************************/ 
  ta = (float *) malloc(Nslow * sizeof(float));
  trng = (float *) malloc(Nfast * sizeof(float));
	
  g = (float *) malloc(2 * Nfast * sizeof(float));
  g2 = (float *) malloc(2 * Nfast * sizeof(float));
  
  S1 = (float *) malloc(2 * Nfast * Nslow * sizeof(float)); 
  
  H = (float *) malloc(2 * Nslow * sizeof(float));
  
  src = (float *) malloc(2 * Nfast * Nslow * sizeof(float)); 
  
  s0 = (float *) malloc(2 * Nslow * Nfast * sizeof(float)); //SAR_LFM_2 & SAR_node_1
  sac = (float *) malloc(Nslow * Nfast * sizeof(float));  
/*******************************************************/

  _ta = (float *) malloc(Nslow * sizeof(float));
  _trng = (float *) malloc(Nfast * sizeof(float));
	
  _g = (float *) malloc(2 * Nfast * sizeof(float));
  _g2 = (float *) malloc(2 * Nfast * sizeof(float));
  
  _S1 = (float *) malloc(2 * Nfast * Nslow * sizeof(float)); 
  
  _H = (float *) malloc(2 * Nslow * sizeof(float));
  
  _src = (float *) malloc(2 * Nfast * Nslow * sizeof(float));  
  
  _s0 = (float *) malloc(2 * Nslow * Nfast * sizeof(float)); //SAR_LFM_2 & SAR_node_1
  _sac = (float *) malloc(Nslow * Nfast * sizeof(float));
/*******************************************************/ 
  
  in_x1 = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * Nfast);
  out_x1 = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * Nfast);
  in_x2 = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * Nfast);
  out_x2 = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * Nfast);
  in_x3 = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * Nfast);
  out_x3 = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * Nfast);

  in_x4 = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * Nslow);
  out_x4 = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * Nslow);
  in_x5 = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * Nslow);
  out_x5 = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * Nslow);

  p1 = fftwf_plan_dft_1d(Nfast, in_x1, out_x1, FFTW_FORWARD, FFTW_ESTIMATE);
  p2 = fftwf_plan_dft_1d(Nfast, in_x2, out_x2, FFTW_FORWARD, FFTW_ESTIMATE);
  p3 = fftwf_plan_dft_1d(Nfast, in_x3, out_x3, FFTW_BACKWARD, FFTW_ESTIMATE);
  p4 = fftwf_plan_dft_1d(Nslow, in_x4, out_x4, FFTW_FORWARD, FFTW_ESTIMATE);
  p5 = fftwf_plan_dft_1d(Nslow, in_x5, out_x5, FFTW_BACKWARD, FFTW_ESTIMATE);

  _in_x1 = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * Nfast);
  _out_x1 = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * Nfast);
  _in_x2 = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * Nfast);
  _out_x2 = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * Nfast);
  _in_x3 = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * Nfast);
  _out_x3 = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * Nfast);

  _in_x4 = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * Nslow);
  _out_x4 = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * Nslow);
  _in_x5 = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * Nslow);
  _out_x5 = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * Nslow);

  _p1 = fftwf_plan_dft_1d(Nfast, _in_x1, _out_x1, FFTW_FORWARD, FFTW_ESTIMATE);
  _p2 = fftwf_plan_dft_1d(Nfast, _in_x2, _out_x2, FFTW_FORWARD, FFTW_ESTIMATE);
  _p3 = fftwf_plan_dft_1d(Nfast, _in_x3, _out_x3, FFTW_BACKWARD, FFTW_ESTIMATE);
  _p4 = fftwf_plan_dft_1d(Nslow, _in_x4, _out_x4, FFTW_FORWARD, FFTW_ESTIMATE);
  _p5 = fftwf_plan_dft_1d(Nslow, _in_x5, _out_x5, FFTW_BACKWARD, FFTW_ESTIMATE);

  Rmin = sqrt((Yc - Y0) * (Yc - Y0) + h * h);
  Rmax = sqrt((Yc + Y0) * (Yc + Y0) + h * h);
  
  
  ///////////////accelerator////////////////////////////////
	dlhandle = dlopen("./libdash-rt.so", RTLD_LAZY);
	if (dlhandle == nullptr) {
		printf("Unable to open FFT shared object!\n");
	} else {
		fft_accel_func = (void(*)(dash_cmplx_flt_type**, dash_cmplx_flt_type**, size_t*, bool*, uint8_t)) dlsym(dlhandle, "DASH_FFT_flt_fft");
		if (fft_accel_func == nullptr) {
			printf("Unable to get function handle for FFT accelerator function!\n");
		}
		ifft_accel_func = (void(*)(dash_cmplx_flt_type**, dash_cmplx_flt_type**, size_t*, bool*, uint8_t)) dlsym(dlhandle, "DASH_FFT_flt_fft");
		if (ifft_accel_func == nullptr) {
			printf("Unable to get function handle for IFFT accelerator function!\n");
		}
	}
  
  printf("[SAR_RDA] intialization done\n");
}

void __attribute__((destructor)) clean_app(void){
	printf("[SAR_RDA] destroying variables\n");
/*******************************************************/  	
	free(ta);
	free(trng);
	
	free(g);
	free(g2);
	
	free(S1);
	
	free(H);
	
	free(src);
	
	free(sac);
	free(s0);
/*******************************************************/  

	free(_ta);
	free(_trng);
	
	free(_g);
	free(_g2);
	
	free(_S1);
	
	free(_H);
	
	free(_src);
	
	free(_sac);
	free(_s0);

	fftwf_destroy_plan(p1);
	fftwf_destroy_plan(p2);
	fftwf_destroy_plan(p3);
	fftwf_destroy_plan(p4);
	fftwf_destroy_plan(p5);
	fftwf_destroy_plan(_p1);
	fftwf_destroy_plan(_p2);
	fftwf_destroy_plan(_p3);
	fftwf_destroy_plan(_p4);
	fftwf_destroy_plan(_p5);

	fftwf_free(in_x1);
	fftwf_free(out_x1);
	fftwf_free(in_x2);
	fftwf_free(out_x2);
	fftwf_free(in_x3);
	fftwf_free(out_x3);
	fftwf_free(in_x4);
	fftwf_free(out_x4);
	fftwf_free(in_x5);
	fftwf_free(out_x5);

	fftwf_free(_in_x1);
	fftwf_free(_out_x1);
	fftwf_free(_in_x2);
	fftwf_free(_out_x2);
	fftwf_free(_in_x3);
	fftwf_free(_out_x3);
	fftwf_free(_in_x4);
	fftwf_free(_out_x4);
	fftwf_free(_in_x5);
	fftwf_free(_out_x5);

	if (dlhandle != nullptr) {
		dlclose(dlhandle);
	}

/*******************************************************/  
	
	printf("[SAR_RDA] destruction done\n");
}

/* Function Declarations */
//void swap(float *, float *);
//void fftshift(float *, float);

void fftwf_fft(float *input_array, fftwf_complex *in, fftwf_complex *out, float *output_array, size_t n_elements, fftwf_plan p) {
    for(size_t i = 0; i < 2*n_elements; i+=2)
    {
		in[i/2][0] = input_array[i];
        in[i/2][1] = input_array[i+1];
	}
    fftwf_execute(p);
    for(size_t i = 0; i < 2*n_elements; i+=2)
    {
        output_array[i] = (float) out[i/2][0];
        output_array[i+1] = (float) out[i/2][1];
    }
}


/**************** Kernels ****************/
void swap(float *v1, float *v2) {
  float tmp = *v1;
  *v1 = *v2;
  *v2 = tmp;
}

void fftshift(float *data, float count) {
  int k;
  int c = (float)floor((float)count / 2);
  // For odd and for even numbers of element use different algorithm
  if ((int)count % 2 == 0) {
    for (k = 0; k < 2 * c; k += 2) {
      swap(&data[k], &data[k + 2 * c]);
      swap(&data[k + 1], &data[k + 1 + 2 * c]);
    }
  } else {
    float tmp1 = data[0];
    float tmp2 = data[1];
    for (k = 0; k < 2 * c; k += 2) {
      data[k] = data[2 * c + k + 2];
      data[k + 1] = data[2 * c + k + 3];
      data[2 * c + k + 2] = data[k + 2];
      data[2 * c + k + 3] = data[k + 3];
    }
    data[2 * c] = tmp1;
    data[2 * c] = tmp2;
  }
}

/**************** Tasks **********************/
extern "C" void SAR_node_head(void){
}

extern "C" void SAR_LFM_1(void){
  trng[0] = 0; 
  static bool select1 = true;
  
  if(select1){
	  // Create range vector
	  for (int i = 1; i < Nfast; i++) {
		trng[i] = trng[i - 1] + (2 * Rmax / c + Tr - 2 * Rmin / c) / (Nfast - 1);
	  } 
	  
	  for (int i = 0; i < 2 * Nfast; i += 2) {
		if (trng[i / 2] > -Tr / 2 && trng[i / 2] < Tr / 2) {
		  g[i] = cos(M_PI * Kr * trng[i / 2] * trng[i / 2]);
		  g[i + 1] = -sin(M_PI * Kr * trng[i / 2] * trng[i / 2]);
		} else {
		  g[i] = 0;
		  g[i + 1] = 0;
		}
		
	  }
	  //  KERN_ENTER(make_label("FFT[1D][%d][complex][float64][forward]", Nfast));
		  fftwf_fft(g, in_x1, out_x1, g2, Nfast, p1);
		  //gsl_fft(g, g2, Nfast);
	  //  KERN_EXIT(make_label("FFT[1D][%d][complex][float64][forward]", Nfast));	  
  } else {
	    // Create range vector
	  for (int i = 1; i < Nfast; i++) {
		_trng[i] = _trng[i - 1] + (2 * Rmax / c + Tr - 2 * Rmin / c) / (Nfast - 1);
	  } 
	  
	  for (int i = 0; i < 2 * Nfast; i += 2) {
		if (_trng[i / 2] > -Tr / 2 && _trng[i / 2] < Tr / 2) {
		  _g[i] = cos(M_PI * Kr * _trng[i / 2] * _trng[i / 2]);
		  _g[i + 1] = -sin(M_PI * Kr * _trng[i / 2] * _trng[i / 2]);
		} else {
		  _g[i] = 0;
		  _g[i + 1] = 0;
		}
	  }
	  
	  //  KERN_ENTER(make_label("FFT[1D][%d][complex][float64][forward]", Nfast));
		  fftwf_fft(_g, _in_x1, _out_x1, _g2, Nfast, _p1);
		  //gsl_fft(_g, _g2, Nfast);
	  //  KERN_EXIT(make_label("FFT[1D][%d][complex][float64][forward]", Nfast));
  }
  select1 = !select1;
 
}

extern "C" void SAR_LFM_2(void){
  float R0;
  float Ka;
  
  int i;
  FILE *fp;
  FILE *_fp;
  
  static bool select2 = true;
  
  R0 = sqrt(Yc * Yc + h * h);
  Ka = 2 * v * v / lambda / R0;
  ta[0] = 0;
  
  /* Read in raw radar data */
  if(select2){
	  fp = fopen(RAWDATA, "r");
	  for (i = 0; i < 2 * Nslow * Nfast; i++) {
		fscanf(fp, "%f", &s0[i]);
	  }	 
	   fclose(fp);	
	  // Create azimuth vector
	  for (i = 1; i < Nslow; i++) {
		ta[i] = ta[i - 1] + (Xmax - Xmin) / v / (Nslow - 1);
	  }
	  
	   // Azimuth Compression
	  for (i = 0; i < 2 * Nslow; i += 2) {
		if (ta[i / 2] > -Tr / 2 * (Xmax - Xmin) / v / (2 * Rmax / c + Tr - 2 * Rmin / c) &&
			ta[i / 2] < Tr / 2 * (Xmax - Xmin) / v / (2 * Rmax / c + Tr - 2 * Rmin / c)) {
		  H[i] = cos(M_PI * Ka * ta[i / 2] * ta[i / 2]);
		  H[i + 1] = sin(M_PI * Ka * ta[i / 2] * ta[i / 2]);
		} else {
		  H[i] = 0;
		  H[i + 1] = 0;
		}
	  }
	   
  } else {
	  _fp = fopen(RAWDATA, "r");
	  for (i = 0; i < 2 * Nslow * Nfast; i++) {
		fscanf(_fp, "%f", &_s0[i]);
	  }  
	    fclose(_fp);
		  // Create azimuth vector
	  for (i = 1; i < Nslow; i++) {
		_ta[i] = _ta[i - 1] + (Xmax - Xmin) / v / (Nslow - 1);
	  }
	  
	   // Azimuth Compression
	  for (i = 0; i < 2 * Nslow; i += 2) {
		if (_ta[i / 2] > -Tr / 2 * (Xmax - Xmin) / v / (2 * Rmax / c + Tr - 2 * Rmin / c) &&
			_ta[i / 2] < Tr / 2 * (Xmax - Xmin) / v / (2 * Rmax / c + Tr - 2 * Rmin / c)) {
		  _H[i] = cos(M_PI * Ka * _ta[i / 2] * _ta[i / 2]);
		  _H[i + 1] = sin(M_PI * Ka * _ta[i / 2] * _ta[i / 2]);
		} else {
		  _H[i] = 0;
		  _H[i + 1] = 0;
		}
	  }
  }

  select2 = !select2;
} 

extern "C" void SAR_node_1_cpu(void){
  printf("[SAR_RDA] SAR_node_1 execution begun\n");
  float *fft_arr;
  float *temp;
  float *temp2;
  float *temp3;	
  
  int i, j;
  
  static bool select3 = true;
	
  fft_arr = (float*) malloc(2 * Nfast * sizeof(float));
  temp = (float*) malloc(2 * Nfast * sizeof(float));
  temp2 = (float*) malloc(2 * Nfast * sizeof(float));
  temp3 = (float*) malloc(2 * Nfast * sizeof(float));	
	
  if(select3){
	  for (i = 0; i < Nslow; i++) {
		for (j = 0; j < 2 * Nfast; j++) {
		  fft_arr[j] = s0[j + i * 2 * Nfast];
		}
		//KERN_ENTER(make_label("FFT[1D][%d][complex][float64][forward]", Nfast));
		fftwf_fft(fft_arr, in_x2, out_x2, temp, Nfast, p2);
		//gsl_fft(fft_arr, temp, Nfast);
		//KERN_EXIT(make_label("FFT[1D][%d][complex][float64][forward]", Nfast));
		fftshift(temp, Nfast);
		//KERN_ENTER(make_label("ZIP[multiply][%d][float64][complex]", Nfast));
		for (j = 0; j < 2 * Nfast; j += 2) {
		  temp2[j] = temp[j] * g2[j] - temp[j + 1] * g2[j + 1];
		  temp2[j + 1] = temp[j + 1] * g2[j] + temp[j] * g2[j + 1];
		}
		//KERN_EXIT(make_label("ZIP[multiply][%d][float64][complex]", Nfast));
		//KERN_ENTER(make_label("FFT[1D][%d][complex][float64][backward]", Nfast));
		fftwf_fft(temp2, in_x3, out_x3, temp3, Nfast, p3);
		//gsl_ifft(temp2, temp3, Nfast);
		//KERN_EXIT(make_label("FFT[1D][%d][complex][float64][backward]", Nfast));
		for (j = 0; j < 2 * Nfast; j += 2) {
		  src[j * Nslow + 2 * i] = temp3[j];
		  src[j * Nslow + 2 * i + 1] = temp3[j + 1];
		}
	  }
  } else {
	  for (i = 0; i < Nslow; i++) {
		for (j = 0; j < 2 * Nfast; j++) {
		  fft_arr[j] = _s0[j + i * 2 * Nfast];
		}
		//KERN_ENTER(make_label("FFT[1D][%d][complex][float64][forward]", Nfast));
		fftwf_fft(fft_arr, _in_x2, _out_x2, temp, Nfast, _p2);
		//gsl_fft(fft_arr, temp, Nfast);
		//KERN_EXIT(make_label("FFT[1D][%d][complex][float64][forward]", Nfast));
		fftshift(temp, Nfast);
		//KERN_ENTER(make_label("ZIP[multiply][%d][float64][complex]", Nfast));
		for (j = 0; j < 2 * Nfast; j += 2) {
		  temp2[j] = temp[j] * _g2[j] - temp[j + 1] * _g2[j + 1];
		  temp2[j + 1] = temp[j + 1] * _g2[j] + temp[j] * _g2[j + 1];
		}
		//KERN_EXIT(make_label("ZIP[multiply][%d][float64][complex]", Nfast));
		//KERN_ENTER(make_label("FFT[1D][%d][complex][float64][backward]", Nfast));
		fftwf_fft(temp2, _in_x3, _out_x3, temp3, Nfast, _p3);
		//gsl_ifft(temp2, temp3, Nfast);
		//KERN_EXIT(make_label("FFT[1D][%d][complex][float64][backward]", Nfast));
		for (j = 0; j < 2 * Nfast; j += 2) {
		  _src[j * Nslow + 2 * i] = temp3[j];
		  _src[j * Nslow + 2 * i + 1] = temp3[j + 1];
		}
	  }	  
  }
 
  
  free(fft_arr);
  free(temp);
  free(temp2);
  free(temp3);
  select3 = !select3;  
  printf("[SAR_RDA] SAR_node_1 done\n");

}

extern "C" void SAR_node_1_acc(void){
  printf("[SAR_RDA] SAR_node_1 execution begun\n");
  float *fft_arr;
  float *temp;
  float *temp2;
  float *temp3;	
  
  int i, j;
  
  static bool select3 = true;
  size_t size = Nfast;
  bool fwdFFT = true;
  bool invFFT = false;
	
  fft_arr = (float*) malloc(2 * Nfast * sizeof(float));
  temp = (float*) malloc(2 * Nfast * sizeof(float));
  temp2 = (float*) malloc(2 * Nfast * sizeof(float));
  temp3 = (float*) malloc(2 * Nfast * sizeof(float));	
	
  if(select3){
	  for (i = 0; i < Nslow; i++) {
		for (j = 0; j < 2 * Nfast; j++) {
		  fft_arr[j] = s0[j + i * 2 * Nfast];
		}
		//KERN_ENTER(make_label("FFT[1D][%d][complex][float64][forward]", Nfast));
		//gsl_fft(fft_arr, temp, Nfast);
		(*fft_accel_func)(((dash_cmplx_flt_type**)&fft_arr), ((dash_cmplx_flt_type**)&temp), &size, &fwdFFT, __CEDR_CLUSTER_IDX__);
		//KERN_EXIT(make_label("FFT[1D][%d][complex][float64][forward]", Nfast));
		fftshift(temp, Nfast);
		//KERN_ENTER(make_label("ZIP[multiply][%d][float64][complex]", Nfast));
		for (j = 0; j < 2 * Nfast; j += 2) {
		  temp2[j] = temp[j] * g2[j] - temp[j + 1] * g2[j + 1];
		  temp2[j + 1] = temp[j + 1] * g2[j] + temp[j] * g2[j + 1];
		}
		//KERN_EXIT(make_label("ZIP[multiply][%d][float64][complex]", Nfast));
		//KERN_ENTER(make_label("FFT[1D][%d][complex][float64][backward]", Nfast));
		//gsl_ifft(temp2, temp3, Nfast);
		(*fft_accel_func)(((dash_cmplx_flt_type**)&temp2), ((dash_cmplx_flt_type**)&temp3), &size, &invFFT, __CEDR_CLUSTER_IDX__);
		//KERN_EXIT(make_label("FFT[1D][%d][complex][float64][backward]", Nfast));
		for (j = 0; j < 2 * Nfast; j += 2) {
		  src[j * Nslow + 2 * i] = temp3[j];
		  src[j * Nslow + 2 * i + 1] = temp3[j + 1];
		}
	  }
  } else {
	  for (i = 0; i < Nslow; i++) {
		for (j = 0; j < 2 * Nfast; j++) {
		  fft_arr[j] = _s0[j + i * 2 * Nfast];
		}
		//KERN_ENTER(make_label("FFT[1D][%d][complex][float64][forward]", Nfast));
		(*fft_accel_func)(((dash_cmplx_flt_type**)&fft_arr), ((dash_cmplx_flt_type**)&temp), &size, &fwdFFT, __CEDR_CLUSTER_IDX__);
		//KERN_EXIT(make_label("FFT[1D][%d][complex][float64][forward]", Nfast));
		fftshift(temp, Nfast);
		//KERN_ENTER(make_label("ZIP[multiply][%d][float64][complex]", Nfast));
		for (j = 0; j < 2 * Nfast; j += 2) {
		  temp2[j] = temp[j] * _g2[j] - temp[j + 1] * _g2[j + 1];
		  temp2[j + 1] = temp[j + 1] * _g2[j] + temp[j] * _g2[j + 1];
		}
		//KERN_EXIT(make_label("ZIP[multiply][%d][float64][complex]", Nfast));
		//KERN_ENTER(make_label("FFT[1D][%d][complex][float64][backward]", Nfast));
		//gsl_ifft(temp2, temp3, Nfast);
		(*fft_accel_func)(((dash_cmplx_flt_type**)&temp2), ((dash_cmplx_flt_type**)&temp3), &size, &invFFT, __CEDR_CLUSTER_IDX__);
		//KERN_EXIT(make_label("FFT[1D][%d][complex][float64][backward]", Nfast));
		for (j = 0; j < 2 * Nfast; j += 2) {
		  _src[j * Nslow + 2 * i] = temp3[j];
		  _src[j * Nslow + 2 * i + 1] = temp3[j + 1];
		}
	  }	  
  }
 
  
  free(fft_arr);
  free(temp);
  free(temp2);
  free(temp3);
  select3 = !select3;  
  printf("[SAR_RDA] SAR_node_1 done\n");

}

extern "C" void SAR_node_2_cpu(void){
  float *temp4;
  float *fft_arr_2;
  int i, j;
  
  static bool select4 = true;
  
  fft_arr_2 = (float*) malloc(2 * Nslow * sizeof(float));
  temp4 = (float*) malloc(2 * Nslow * sizeof(float));
  
  if(select4){
	  // Azimuth FFT
	  for (i = 0; i < Nfast; i++) {
		for (j = 0; j < 2 * Nslow; j += 2) {
		  fft_arr_2[j] = src[j + i * 2 * Nslow];
		  fft_arr_2[j + 1] = src[j + 1 + i * 2 * Nslow];
		}
		//KERN_ENTER(make_label("FFT[1D][%d][complex][float64][forward]", Nslow));
		fftwf_fft(fft_arr_2, in_x4, out_x4, temp4, Nslow, p4);
		//gsl_fft(fft_arr_2, temp4, Nslow);
		//KERN_EXIT(make_label("FFT[1D][%d][complex][float64][forward]", Nslow));
		fftshift(temp4, Nslow);
		for (j = 0; j < 2 * Nslow; j += 2) {
		  S1[j + i * 2 * Nslow] = temp4[j];
		  S1[j + 1 + i * 2 * Nslow] = temp4[j + 1];
		}
	  }	  
  } else {
	  // Azimuth FFT
	  for (i = 0; i < Nfast; i++) {
		for (j = 0; j < 2 * Nslow; j += 2) {
		  fft_arr_2[j] = _src[j + i * 2 * Nslow];
		  fft_arr_2[j + 1] = _src[j + 1 + i * 2 * Nslow];
		}
		//KERN_ENTER(make_label("FFT[1D][%d][complex][float64][forward]", Nslow));
		fftwf_fft(fft_arr_2, _in_x4, _out_x4, temp4, Nslow, _p4);
		//gsl_fft(fft_arr_2, temp4, Nslow);
		//KERN_EXIT(make_label("FFT[1D][%d][complex][float64][forward]", Nslow));
		fftshift(temp4, Nslow);
		for (j = 0; j < 2 * Nslow; j += 2) {
		  _S1[j + i * 2 * Nslow] = temp4[j];
		  _S1[j + 1 + i * 2 * Nslow] = temp4[j + 1];
		}
	  }
  }
  
  
  free(fft_arr_2);
  free(temp4);
  select4 = !select4;
}

extern "C" void SAR_node_2_acc(void){
  float *temp4;
  float *fft_arr_2;
  int i, j;
  size_t len = 256;
  bool isFwd = true;
  
  static bool select4 = true;
  
  fft_arr_2 = (float*) malloc(2 * Nslow * sizeof(float));
  temp4 = (float*) malloc(2 * Nslow * sizeof(float));
  
  if(select4){
	  // Azimuth FFT
	  for (i = 0; i < Nfast; i++) {
		for (j = 0; j < 2 * Nslow; j += 2) {
		  fft_arr_2[j] = src[j + i * 2 * Nslow];
		  fft_arr_2[j + 1] = src[j + 1 + i * 2 * Nslow];
		}
		//KERN_ENTER(make_label("FFT[1D][%d][complex][float64][forward]", Nslow));
		(*fft_accel_func)(((dash_cmplx_flt_type**)&fft_arr_2), ((dash_cmplx_flt_type**)&temp4), &len, &isFwd, __CEDR_CLUSTER_IDX__);
		//KERN_EXIT(make_label("FFT[1D][%d][complex][float64][forward]", Nslow));
		fftshift(temp4, Nslow);
		for (j = 0; j < 2 * Nslow; j += 2) {
		  S1[j + i * 2 * Nslow] = temp4[j];
		  S1[j + 1 + i * 2 * Nslow] = temp4[j + 1];
		}
	  }
  } else {
	  // Azimuth FFT
	  for (i = 0; i < Nfast; i++) {
		for (j = 0; j < 2 * Nslow; j += 2) {
		  fft_arr_2[j] = _src[j + i * 2 * Nslow];
		  fft_arr_2[j + 1] = _src[j + 1 + i * 2 * Nslow];
		}
		//KERN_ENTER(make_label("FFT[1D][%d][complex][float64][forward]", Nslow));
		(*fft_accel_func)(((dash_cmplx_flt_type**)&fft_arr_2), ((dash_cmplx_flt_type**)&temp4), &len, &isFwd, __CEDR_CLUSTER_IDX__);
		//KERN_EXIT(make_label("FFT[1D][%d][complex][float64][forward]", Nslow));
		fftshift(temp4, Nslow);
		for (j = 0; j < 2 * Nslow; j += 2) {
		  _S1[j + i * 2 * Nslow] = temp4[j];
		  _S1[j + 1 + i * 2 * Nslow] = temp4[j + 1];
		}
	  }
  }
  
  free(fft_arr_2);
  free(temp4);
  select4 = !select4;
}

extern "C" void SAR_node_3_cpu(void){
  float *fft_arr_4;
  float *temp8;
  float *temp9;	
  
  int i, j;
  
  static bool select5 = true;
  
  fft_arr_4 = (float*) malloc(2 * Nslow * sizeof(float));
  temp8 = (float*) malloc(2 * Nslow * sizeof(float));
  temp9 = (float*) malloc(2 * Nslow * sizeof(float));

  if(select5){
	  // ZIP & IFFT
	  for (i = 0; i < Nfast; i++) {
		for (j = 0; j < 2 * Nslow; j++) {
		  temp8[j] = S1[j + i * 2 * Nslow];
		}
		//KERN_ENTER(make_label("ZIP[multiply][%d][float64][complex]", Nslow));
		for (j = 0; j < 2 * Nslow; j += 2) {
		  fft_arr_4[j] = temp8[j] * H[j] - temp8[j + 1] * H[j + 1];
		  fft_arr_4[j + 1] = temp8[j + 1] * H[j] + temp8[j] * H[j + 1];
		}
		//KERN_EXIT(make_label("ZIP[multiply][%d][float64][complex]", Nslow));
		//KERN_ENTER(make_label("FFT[1D][%d][complex][float64][backward]", Nslow));
		fftwf_fft(fft_arr_4, in_x5, out_x5, temp9, Nslow, p5);
		//gsl_ifft(fft_arr_4, temp9, Nslow);
		//KERN_EXIT(make_label("FFT[1D][%d][complex][float64][backward]", Nslow));
		fftshift(temp9, Nslow);
		for (j = 0; j < Nslow; j++) {
		  sac[i + j * Nfast] = sqrt(temp9[2 * j] * temp9[2 * j] + temp9[2 * j + 1] * temp9[2 * j + 1]);
		}
	  }  
  } else {
	  // ZIP & IFFT
	  for (i = 0; i < Nfast; i++) {
		for (j = 0; j < 2 * Nslow; j++) {
		  temp8[j] = _S1[j + i * 2 * Nslow];
		}
		//KERN_ENTER(make_label("ZIP[multiply][%d][float64][complex]", Nslow));
		for (j = 0; j < 2 * Nslow; j += 2) {
		  fft_arr_4[j] = temp8[j] * H[j] - temp8[j + 1] * H[j + 1];
		  fft_arr_4[j + 1] = temp8[j + 1] * H[j] + temp8[j] * H[j + 1];
		}
		//KERN_EXIT(make_label("ZIP[multiply][%d][float64][complex]", Nslow));
		//KERN_ENTER(make_label("FFT[1D][%d][complex][float64][backward]", Nslow));
		fftwf_fft(fft_arr_4, _in_x5, _out_x5, temp9, Nslow, _p5);
		//gsl_ifft(fft_arr_4, temp9, Nslow);
		//KERN_EXIT(make_label("FFT[1D][%d][complex][float64][backward]", Nslow));
		fftshift(temp9, Nslow);
		for (j = 0; j < Nslow; j++) {
		  _sac[i + j * Nfast] = sqrt(temp9[2 * j] * temp9[2 * j] + temp9[2 * j + 1] * temp9[2 * j + 1]);
		}
	  }		
  }	
  free(fft_arr_4);
  free(temp8);
  free(temp9);
  select5 = !select5;
}

extern "C" void SAR_node_3_acc(void){
  float *fft_arr_4;
  float *temp8;
  float *temp9;	
  
  int i, j;

  size_t len = 256;
  bool isFwd = false;
  
  static bool select5 = true;
  
  fft_arr_4 = (float*) malloc(2 * Nslow * sizeof(float));
  temp8 = (float*) malloc(2 * Nslow * sizeof(float));
  temp9 = (float*) malloc(2 * Nslow * sizeof(float));

  if(select5){
	  // ZIP & IFFT
	  for (i = 0; i < Nfast; i++) {
		for (j = 0; j < 2 * Nslow; j++) {
		  temp8[j] = S1[j + i * 2 * Nslow];
		}
		//KERN_ENTER(make_label("ZIP[multiply][%d][float64][complex]", Nslow));
		for (j = 0; j < 2 * Nslow; j += 2) {
		  fft_arr_4[j] = temp8[j] * H[j] - temp8[j + 1] * H[j + 1];
		  fft_arr_4[j + 1] = temp8[j + 1] * H[j] + temp8[j] * H[j + 1];
		}
		//KERN_EXIT(make_label("ZIP[multiply][%d][float64][complex]", Nslow));
		//KERN_ENTER(make_label("FFT[1D][%d][complex][float64][backward]", Nslow));
		(*fft_accel_func)(((dash_cmplx_flt_type**)&fft_arr_4), ((dash_cmplx_flt_type**)&temp9), &len, &isFwd, __CEDR_CLUSTER_IDX__);
		//KERN_EXIT(make_label("FFT[1D][%d][complex][float64][backward]", Nslow));
		fftshift(temp9, Nslow);
		for (j = 0; j < Nslow; j++) {
		  sac[i + j * Nfast] = sqrt(temp9[2 * j] * temp9[2 * j] + temp9[2 * j + 1] * temp9[2 * j + 1]);
		}
	  }  
  } else {
	  // ZIP & IFFT
	  for (i = 0; i < Nfast; i++) {
		for (j = 0; j < 2 * Nslow; j++) {
		  temp8[j] = _S1[j + i * 2 * Nslow];
		}
		//KERN_ENTER(make_label("ZIP[multiply][%d][float64][complex]", Nslow));
		for (j = 0; j < 2 * Nslow; j += 2) {
		  fft_arr_4[j] = temp8[j] * H[j] - temp8[j + 1] * H[j + 1];
		  fft_arr_4[j + 1] = temp8[j + 1] * H[j] + temp8[j] * H[j + 1];
		}
		//KERN_EXIT(make_label("ZIP[multiply][%d][float64][complex]", Nslow));
		//KERN_ENTER(make_label("FFT[1D][%d][complex][float64][backward]", Nslow));
		(*fft_accel_func)(((dash_cmplx_flt_type**)&fft_arr_4), ((dash_cmplx_flt_type**)&temp9), &len, &isFwd, __CEDR_CLUSTER_IDX__);
		//KERN_EXIT(make_label("FFT[1D][%d][complex][float64][backward]", Nslow));
		fftshift(temp9, Nslow);
		for (j = 0; j < Nslow; j++) {
		  _sac[i + j * Nfast] = sqrt(temp9[2 * j] * temp9[2 * j] + temp9[2 * j + 1] * temp9[2 * j + 1]);
		}
	  }		
  }	
  free(fft_arr_4);
  free(temp8);
  free(temp9);
  select5 = !select5;
}

extern "C" void SAR_node_4(void){
  /* Write out image */
  FILE *fp1;
  FILE *_fp1;
  
  static bool select6 = true;
  
  if(select6){
	  fp1 = fopen(OUTPUT, "w");
	  for (int i = 0; i < Nslow; i++) {
		for (int j = 0; j < Nfast; j++) {
		  fprintf(fp1, "%lf ", sac[j + i * Nfast]);
		}
		fprintf(fp1, "\n");
		fflush(fp1);
	  }
	  fclose(fp1);	  
  } else {
	  _fp1 = fopen(OUTPUT, "w");
	  for (int i = 0; i < Nslow; i++) {
		for (int j = 0; j < Nfast; j++) {
		  fprintf(_fp1, "%lf ", _sac[j + i * Nfast]);
		}
		fprintf(_fp1, "\n");
		fflush(_fp1);
	  }
	  fclose(_fp1);	  
  }
  select6 = !select6;	
}


int main(int argc, char *argv[]) {
}

