#include "lab3_cuda.h"
#include<math.h>

#define Size_of_1_Block 16
#define TOLERANCE 0.001
#define JACOBI_UPDATE_TOLERANCE 0.001

double **S;
int state;
double **E;


int cmpfunc (const void * a, const void * b){
    double t1 = *(double*)a;
    double t2 = *(double*)b;
    if (t1 < t2) {
        return 1;
    }
    else if (t1 > t2) {
        return -1;
    }
    else {
        return 0;
    }
}

double *e;
int *ind;
bool *changed;
int N_J;


__global__ void gpu_matrix_transpose(double *mat, double *trans, unsigned int r, unsigned int c){
    unsigned int idx = blockIdx.x*blockDim.x;
    idx += threadIdx.x;
    unsigned int idy = blockIdx.y*blockDim.y;
    idy += threadIdx.y;
    if(idx < c && idy < r){
        int t1 = idx*r;
        t1+=idy;
        int t2 = idy*c;
        t2+=idx;
        trans[t1] = mat[t2];
    }
}

__global__ void mat_mul_cuda( int m, int n, int k, double *m1, double *m2, double *m3){
    int c = blockIdx.x*blockDim.x;
    c += threadIdx.x;
    int r = blockIdx.y*blockDim.y;
    r += threadIdx.y;
    double s = 0;
    if(c < k && r < m){
        int i = 0;
        while(i<n){
            int t1=r*n;
            t1+=i;
            int t2 = i*k;
            t2 += c;
            s += m1[t1]*m2[t2];
            i++;
        }
        m3[r*k+c] = s;
    }
}

int maxind(int k){
    int m = k+1;
    for (int i = k+2; i < N_J; i++){
        double f1 = fabs(S[k][i]); 
        double f2 = fabs(S[k][m]); 
        if ( f1 > f2 ){
            m = i;
        }
    }

    return m;
}

void update(int k, double t) {
    double ek_prev = e[k];
    

    e[k] = e[k] + t;
    
    if (e[k] < 0) {
        e[k] = 0;
    }

    if (changed[k] && fabs(ek_prev - e[k]) < JACOBI_UPDATE_TOLERANCE) {
        state = state - 1;
        changed[k] = false;
    }
    else if ((! changed[k]) && fabs(ek_prev - e[k]) > JACOBI_UPDATE_TOLERANCE) {
        state = state + 1;
        changed[k] = true;
    }
}

void init_jacobi(double ** input_matrix) {
    S = input_matrix;

    state = N_J;
    changed = (bool*)malloc(sizeof(bool)*N_J);
    
    E = (double**)malloc(__SIZEOF_POINTER__*N_J);
    ind = (int*)malloc(__SIZEOF_INT__*N_J);
    
    e = (double*)malloc(__SIZEOF_DOUBLE__*N_J);

    for (int i = 0; i < N_J; ++i) {
        E[i] = (double*)malloc(__SIZEOF_DOUBLE__*N_J);
        for (int j = 0; j < N_J; ++j) {
            E[i][j] = 0;
        }
    }
    for (int i=0; i < N_J; i++){
        changed[i] = true;
        e[i]       = S[i][i];
        ind[i]     = maxind(i);
        E[i][i] = 1;
    }
}

void Jacobi(double **input_matrix, int n, double **eigenvalues, double ***eigenvectors) {
    N_J = n;

    
    init_jacobi(input_matrix);
    int cc=0;
    
    while(state != 0){
        cc++;
        int m = 0;

        for (int k=1; k<N_J-1; k++){
            double t1 = fabs(S[k][ind[k]]);
            double t2 = fabs(S[m][ind[m]]);
            if ( t1 > t2 ){
                m = k;
            }
        }
        
        int l = ind[m];
        double p = S[m][l];
        double y = (e[l] - e[m]);
        y = y / 2.0;
        double d = fabs(y); 
        d += sqrt(p*p + y*y);
        double r = p*p;
        r += d*d;
        r = sqrt(r);
        double s = p / r;
        int k = m;
        
        double c = d / r;
        double t = (p*p) / d;

        if (y < 0.0) { s = -s; t = -t; }
        update(k, -t);

        S[k][l] = 0.0;
        
        double mat2;
        double mat1;
        int i=0;
        update(l, t);
        while(i<k){
            mat1 = S[i][l];
            mat2 = S[i][k];
            S[i][l] = s*mat2 + c*mat1;
            S[i][k] = c*mat2 - s*mat1;
            i++;
        }
        i = k+1;
        while(i<l){
            mat1 = S[i][l];
            mat2 = S[k][i];
            S[k][i] = c*mat2 - s*mat1;
            S[i][l] = s*mat2 + c*mat1;
            i++;
        }
        i= l+1;
        while(i<N_J){
            mat1 = S[l][i];
            mat2 = S[k][i];
            S[k][i] = c*mat2 - s*mat1;
            S[l][i] = s*mat2 + c*mat1;
            i++;
        }
        i=0;
        while(i<N_J){
            mat1 = E[i][l];
            mat2 = E[i][k];
            E[i][k] = c*mat2 - s*mat1;
            E[i][l] = s*mat2 + c*mat1;
            i++;
        }
        ind[k] = maxind(k);
        ind[l] = maxind(l);
    }
    printf("%d\n", cc);
    *eigenvalues = e;
    *eigenvectors = E;
}

void set_array_value(double* matrix, int row, int column, int num_col, double val) {
    matrix[ column+row*num_col ] = val;
}

double get_array_value(double* matrix, int row, int column, int num_col) {
    // cout << column + row*num_col<< endl;
    return matrix[ column+row*num_col ];
}

void mat_mul(double* mat1, double* mat2, double* result, int r_n, int r_m, int r_p) {
    for (int i = 0; i < r_n; ++i) {
        for (int j = 0; j < r_p; ++j) {
            double temp_sum = 0;
            for (int k = 0; k < r_m; ++k) {
                double a_ik = get_array_value(mat1, i, k, r_m);
                double b_kj = get_array_value(mat2, k, j, r_p);
                // cout << "mul" << a_ik << b_kj << endl;
                temp_sum += a_ik*b_kj;
            }
            // cout << "writing" << temp_sum << endl;
            set_array_value(result, i, j, r_p, temp_sum);
        }
    }
}


void SVD_and_PCA (int M, int N, double* D, double** U, double** SIGMA, double** V_T, double** D_HAT, int *K,int retention) {
    // write your code here
    double **prod;
    prod = (double**)malloc(sizeof(double*)*N);
    
    double *result;
    for (int i=0; i<N; i++) {
        prod[i] = (double*)malloc(sizeof(double)*N);
    }
    
    int original_size = M*N*sizeof(double);
    int n_cross_n = N*N*sizeof(double);
    result = (double*)malloc(n_cross_n);
    
    double *eigen;
    double **eigen_vec;
    double eigen_values[N];
    
    dim3 dimGrid((N+Size_of_1_Block-1)/Size_of_1_Block, (M+Size_of_1_Block-1)/Size_of_1_Block);
    dim3 dimGrid1((N+Size_of_1_Block-1)/Size_of_1_Block, (N+Size_of_1_Block-1)/Size_of_1_Block);
    dim3 dimBlock(Size_of_1_Block,Size_of_1_Block);

    int index[N];
    bool done[N];


    // CUDA declarations
    double *cuda_dt;
    cudaMalloc((void **)&cuda_dt, original_size);
    
    double *cuda_d;
    cudaMalloc((void **)&cuda_d, original_size);
    cudaMemcpy(cuda_d, D, original_size, cudaMemcpyHostToDevice);
    
    double *cuda_result;
    cudaMalloc((void **)&cuda_result, n_cross_n);

    gpu_matrix_transpose<<<dimGrid, dimBlock>>>(cuda_d,cuda_dt,M,N);
    cudaDeviceSynchronize();
    double* temp_u = (double*)malloc(n_cross_n);
    
    mat_mul_cuda<<<dimGrid1, dimBlock>>>(N,M,N,cuda_dt,cuda_d,cuda_result);
    cudaDeviceSynchronize();   
    cudaFree(cuda_dt);

    cudaMemcpy(result, cuda_result, n_cross_n, cudaMemcpyDeviceToHost);
    cudaFree(cuda_result);


    for(int i=0; i<N; i++){
        for(int j=0; j<N; j++){
            int t1 = i*N;
            t1 += j;
            prod[i][j] = result[t1];
        }
    }


    Jacobi(prod,N,&eigen,&eigen_vec);

    // for(int i=0; i<N; i++){
    // }

    for(int i=0;i<N;i++){
        done[i] = false;
        index[i] = -1;
        eigen_values[i] = eigen[i];
    }
    double** v_T = (double**)malloc(sizeof(double*)*N);
    
    double sigma[N],sigma_inv[N];

    
    double* u_T = (double*)malloc(original_size);
    double** v = (double**)malloc(sizeof(double*)*N);
    qsort(eigen_values,N,sizeof(eigen[0]),cmpfunc);
    
    for(int i=0;i<N;i++){
        v_T[i] = (double*)malloc(sizeof(double)*N);
    }

    for (int i = 0; i < N; ++i) {
        v[i] = (double*)malloc(sizeof(double)*N);
    }
    

    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++){
            if(done[j]==false){
                double temp = fabs(eigen_values[i] - eigen[j]);
                double thresh = 0.00001; 
                if( temp < thresh ){
                    index[i] = j;
                    done[j] = true;
                    break;
                }
            }
        }
    }

    for (int i = 0; i < N; ++i)
    {
        sigma[i] = sqrt(eigen_values[i]);
        if(sigma[i] > 0.00001) {
            sigma_inv[i] = 1/sqrt(eigen_values[i]);
        }
        else {
            sigma_inv[i] = 0;
        }
    }

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            v_T[i][j] = eigen_vec[j][index[i]];
        }
    }

    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++){
            v[j][i] = v_T[i][j];
        }
    }
    
    double *c_tempu;
    cudaMalloc((void **)&c_tempu, n_cross_n);

    double *c_u;
    cudaMalloc((void **)&c_u, original_size);

    double *c_ut;

    dim3 dimGrid2((N+Size_of_1_Block-1)/Size_of_1_Block, (M+Size_of_1_Block-1)/Size_of_1_Block);

    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++){
            int t1 = i*N;
            t1 += j;
            temp_u[t1] = v[i][j]*sigma_inv[j];
        }
    }

    cudaMalloc((void **)&c_ut, original_size);
    
    *U = (double*)malloc(sizeof(double)*N*N);
    *V_T = (double*)malloc(sizeof(double)*M*M);
    
    cudaMemcpy(c_tempu, temp_u, n_cross_n, cudaMemcpyHostToDevice);
    
    mat_mul_cuda<<<dimGrid2, dimBlock>>>(M, N, N, cuda_d, c_tempu, c_u);
    
    for(int i=0;i<N;i++){
        free(v_T[i]);
        free(eigen_vec[i]);
    }
    
    cudaDeviceSynchronize();    
    cudaFree(c_tempu);

    
    
    *SIGMA = (double*)malloc(sizeof(double)*N);
    for(int i=0;i<N;i++){
        *(*SIGMA + i) = sigma[i];
    }
    
    gpu_matrix_transpose<<<dimGrid, dimBlock>>>(c_u,c_ut,M,N);
    cudaDeviceSynchronize();
    cudaFree(c_u);


    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++){
            int t1 = i*N;
            t1 += j;
            *(*U + t1) = v[i][j];
        }
    }

    cudaMemcpy(u_T, c_ut, original_size, cudaMemcpyDeviceToHost);
    cudaFree(c_ut);
    
    free(temp_u);
    free(v_T);
    
    

    for(int i=0;i<M;i++){
        for(int j=0;j<M;j++){
            if(i<N){
                int t1 = i*M;
                t1 += j;

                *(*V_T + t1) = u_T[t1];
            }
            else{
                int t2 = i*M;
                t2 += j;

                *(*V_T + t2) = 0;
            }
        }
    }


    free(eigen_vec);
    free(u_T);
    
    for(int i=0;i<N;i++){
        sigma[i] = sigma[i]*sigma[i];
    }

    double tot = 0;
    for(int i=0;i<N;i++){
     tot += sigma[i];
    }

    double ret = (double)retention/(double)100;
    double reten =0;
    int k=N;
    
    for(int i=0;i<N;i++){
        if(reten>=ret){
            k = i;
            break;  
        }
        reten += sigma[i]/tot;
    }

    K[0] = k;
    dim3 dimGrid3((k+Size_of_1_Block-1)/Size_of_1_Block, (M+Size_of_1_Block-1)/Size_of_1_Block);

    double *W = (double*)malloc(sizeof(double)*(N*k));
    
    for(int i=0;i<N;i++){
        for(int j=0;j<k;j++){
            int t1 = i*k;
            t1 += j;
            W[t1] = v[i][j];
        }
    }
    
    double *c_w;
    cudaMalloc((void **)&c_w, N*k*__SIZEOF_DOUBLE__);

    *D_HAT = (double*)malloc(sizeof(double)*(M*k));
    
    double *c_dhat;
    cudaMalloc((void **)&c_dhat, M*k*__SIZEOF_DOUBLE__);
    cudaMemcpy(c_w, W, N*k*__SIZEOF_DOUBLE__, cudaMemcpyHostToDevice);
    
    mat_mul_cuda<<<dimGrid3, dimBlock>>>(M,N,k,cuda_d,c_w,c_dhat);
    cudaDeviceSynchronize();
    cudaFree(c_w);
    cudaFree(cuda_d);
    

    for(int i=0;i<N;i++){
        free(v[i]);
    }
    free(v);

    cudaMemcpy(*D_HAT, c_dhat, sizeof(double)*(M*k), cudaMemcpyDeviceToHost);
    cudaFree(c_dhat);
    
    

}
