

/* ****************** */
/* Include packages   */
/* ****************** */
#include <math.h>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <algorithm>
#include <random>

//include Eigen
#include "eigen3/Eigen/Dense"
#include "eigen3/Eigen/Sparse"
#include "eigen3/Eigen/SparseLU"
#include "eigen3/Eigen/SparseQR"
#include "eigen3/Eigen/SparseCholesky"
#include "eigen3/Eigen/IterativeLinearSolvers"
typedef Eigen::SparseMatrix<double > SpMat;
typedef Eigen::Triplet<double> Trip;
#include <typeinfo>

//include MEX related files
#include "mex.h"
#include "matrix.h"

/* ******************** */
/* State Variable Class */
/* ******************** */

Eigen::MatrixXd empty;
Eigen::ArrayXd emptyAry;
class stateVars {
    
public:
    Eigen::MatrixXd stateMat; //matrix to store state variables
    Eigen::MatrixXd stateMatNorm; //matrix to store normalized state variables [-1,1]
    Eigen::ArrayXd increVec; //vector to record steps
    Eigen::ArrayXd dVec; //vector to record steps
    int N; // num of dimensions
    int S; // number of rows for the grid
    Eigen::ArrayXd upperLims;
    Eigen::ArrayXd lowerLims;
    Eigen::ArrayXd gridSizes;

    stateVars (Eigen::ArrayXd, Eigen::ArrayXd, Eigen::ArrayXd); //constructors with arrays of upper/lower bounds and gridsizes 
    stateVars (Eigen::MatrixXd); //constructors by loading in data

};


stateVars::stateVars (Eigen::ArrayXd upper, Eigen::ArrayXd lower, Eigen::ArrayXd gridSizes) {
    
    upperLims = upper;
    lowerLims = lower;
    N = upperLims.size();
    S = gridSizes.prod();
    stateMat.resize(S,N);
    dVec.resize(N);
    increVec.resize(N);
    increVec(0) = 1;
        
    //fill in the state object; similar to the ndgrid function in MATLAB
    
    for (int n = 0; n < N; ++n) {
            
        if (n != 0) {
            increVec(n) = gridSizes(n - 1) * increVec(n - 1);
        }
        dVec(n) = (upper(n) - lower(n)) / (gridSizes(n) - 1);
        
        for (int i = 0; i < S; ++i) {
            stateMat(i,n) = lower(n) + dVec(n) * ( int(i /  increVec(n) ) % int( gridSizes(n) ) );
        }
            
    }
    
}

stateVars::stateVars (Eigen::MatrixXd preLoad) {

    //fill in stateMat based on data loaded
    N = preLoad.cols();
    S = preLoad.rows();
    stateMat.resize(S,N);
    dVec.resize(N); dVec.setZero();
    increVec.resize(N); increVec.setZero();
    upperLims.resize(N); lowerLims.resize(N);
    for (int j = 0; j < preLoad.cols(); ++j) {
        upperLims(j) = preLoad.col(j).maxCoeff();
        lowerLims(j) = preLoad.col(j).minCoeff();
    }
    
    stateMat = preLoad;
    
    //figure out dVec and increVec
    for (int i = 1; i < S; ++i) {
        for (int n = 0; n < N; ++n ) {
            double diff = stateMat(i,n) - stateMat(i-1,n);
            if (diff > 0 && dVec(n) == 0 && increVec(n) == 0) {
                dVec(n) = diff;
                increVec(n) = i;
            }
        }
        
    }
    


}

struct bc {
    double a0;
    double a0S;
    bool natural;
    Eigen::ArrayXd level;
    Eigen::ArrayXd first;
    Eigen::ArrayXd second;
    
    bc(int d) {
        level.resize(d); first.resize(d); second.resize(d);
    }
};

struct elas {
    Eigen::MatrixXd elas1sc;
    Eigen::MatrixXd elas1c; //exposure elas
    Eigen::MatrixXd elas2sc;
    Eigen::MatrixXd elas2c; //exposure elas
    Eigen::MatrixXd elas1p; //price elas
    Eigen::MatrixXd elas2p;  //price elas
    
    elas(int T, int S) {
        elas1sc.resize(S,T); elas1c.resize(S,T); elas1p.resize(S,T);
        elas2sc.resize(S,T); elas2c.resize(S,T); elas2p.resize(S,T);
    }
};




class linearSysVars {
    
public:
    double dt;
    std::vector<Trip> matList;
    SpMat L;  //matrix for elasticities
    SpMat statMat; //matrix to compute stationary density
    Eigen::MatrixXd phi; //the known vector
    Eigen::SparseLU<SpMat > solver; //solver for elasticities

    
    std::vector<int> atBounds;
    std::vector<int> corners;
    
    Eigen::MatrixXd muM;
    Eigen::MatrixXd sigmaM;
    Eigen::MatrixXd muX;
    std::vector<Eigen::MatrixXd> sigmaX;
    
    Eigen::VectorXd evalues; //store eigenvalues
    Eigen::MatrixXd evectors; //store eigenvalues
    int eigenInd;
    
    //solvedMatrix
    Eigen::MatrixXd phiAll;

    //member functions
    
    //constructor
    linearSysVars(stateVars &, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, std::vector<Eigen::MatrixXd>, const bc &, double);
    
    //function to construt matrix
    
    void constructMat(Eigen::VectorXd, Eigen::MatrixXd, Eigen::MatrixXd, std::vector<Eigen::MatrixXd>, stateVars &, const bc &, double, bool changed = false);
    
    //solve
    void solveT(int, const bc &, stateVars &, Eigen::MatrixXd, bool);
    
    //construct matrix for stationary density
    void constructStatMat( stateVars &);
    
};

linearSysVars::linearSysVars(stateVars & state_vars, Eigen::MatrixXd muC, Eigen::MatrixXd sigmaC, Eigen::MatrixXd muXInput, std::vector<Eigen::MatrixXd> sigmaXInput, const bc & bc, double dtInput) {
    
    //muC: Sx1 vector; sigmaC: S x (num of shocks); muX: S x N; sigmaX: a vector that contains n matrices of dims S x (num of shocks)
    
    L.resize(state_vars.S,state_vars.S);
    
    //store data
    dt = dtInput;
    muM = muC;
    sigmaM = sigmaC;
    muX = muXInput;
    sigmaX = sigmaXInput;

    //compute coefs for matrix construction
    Eigen::VectorXd levelCoefs; levelCoefs.resize(state_vars.S); levelCoefs =  -1.0 / dt +  muM.array() + 0.5 * sigmaM.rowwise().norm().array().pow(2);
    
    Eigen::MatrixXd firstCoefs; firstCoefs.resize(state_vars.S, state_vars.N);
    
    for (int n = 0; n < state_vars.N; ++n ) {
        firstCoefs.col(n) = muX.col(n) + sigmaX[n].cwiseProduct(sigmaM).rowwise().sum();
    }
    


    Eigen::MatrixXd secondCoefs; secondCoefs.resize(state_vars.S, state_vars.N);
    for (int n = 0; n < state_vars.N; ++n ) {
        secondCoefs.col(n) = 0.5 * sigmaX[n].rowwise().norm().array().pow(2);
    }
    
    std::vector<Eigen::MatrixXd> sigmaXCoef;
    sigmaXCoef = sigmaX;

    //construct matrix
    mexPrintf("...Constructing matrix... \n");
    mexEvalString("drawnow;");
    constructMat(levelCoefs, firstCoefs, secondCoefs, sigmaXCoef, state_vars, bc, dt);

    mexPrintf("...Finished constructing matrix... \n");
    mexEvalString("drawnow;");
    //factorize matrix
    mexPrintf("...Factorizing matrix... \n");
    mexEvalString("drawnow;");

    solver.analyzePattern(L);
    solver.factorize(L);
    mexPrintf("...Finished factorizing matrix... \n");
    mexEvalString("drawnow;");
    if(solver.info()!=Eigen::Success) {
        // decomposition failed
        mexErrMsgIdAndTxt("MATLAB:decompNoPardiso:failed","LU decomposition failed.");
    }
    

}



void linearSysVars::constructMat(Eigen::VectorXd levelCoefs, Eigen::MatrixXd firstCoefs, Eigen::MatrixXd secondCoefs, std::vector<Eigen::MatrixXd> sigmaXCoef, stateVars & state_vars, const bc & bc, double dt, bool changed) {
    matList.clear(); Eigen::ArrayXd atBoundsInd; atBoundsInd.resize(state_vars.N);
    //construct matrix
    for (int i = 0; i < state_vars.S; ++i) {

        bool atBound = false;
        bool upBound = false;
        bool lowBound = false;
        bool corner = false;
        double totalBound = 1.0;
        double totalChange = 1.0 / state_vars.dVec((state_vars.N - 1));
        
        //check boundaries
        
        for (int n = (state_vars.N - 1); n >=0; --n ) {
            
            atBoundsInd(n) = -1;
            //check if it is at boundary
            if ( std::abs(state_vars.stateMat(i,n) - state_vars.upperLims(n)) < state_vars.dVec(n) / 2.0 || std::abs(state_vars.stateMat(i,n) - state_vars.lowerLims(n)) < state_vars.dVec(n) / 2.0 ) {
                atBound = true;
                //check if it's at one of the corners
                for (int n_sub = n-1; n_sub >=0; --n_sub) {
                    if ( std::abs(state_vars.stateMat(i,n_sub) - state_vars.upperLims(n_sub)) < state_vars.dVec(n_sub) / 2.0 || std::abs(state_vars.stateMat(i,n_sub) - state_vars.lowerLims(n_sub)) < state_vars.dVec(n_sub) / 2.0 ) {
                        if (! (bc.natural) ) {
                            //if not using natural bdries, set as specified 
                            totalChange = totalChange + 1.0/state_vars.dVec(n_sub);
                            totalBound = totalBound + 1.0;
                        }
                        corner = true;
                        
                    }
                }
            }
            
            //check whether it's at upper or lower boundary
            if ( std::abs(state_vars.stateMat(i,n) - state_vars.upperLims(n)) < state_vars.dVec(n) / 2.0  ) {  //upper boundary
                atBoundsInd(n) = 1;
                if ( (!bc.natural) && (!corner) ) {
                    //if not using natural bdries, set as specified
                    matList.push_back(Trip(i, i, bc.level(n) + bc.first(n)/state_vars.dVec(n) + bc.second(n) / pow(state_vars.dVec(n), 2) ));
                    matList.push_back(Trip(i, i - state_vars.increVec(n), -bc.first(n)/state_vars.dVec(n) - 2 * bc.second(n) / pow(state_vars.dVec(n), 2) ));
                    matList.push_back(Trip(i, i - 2*state_vars.increVec(n), bc.second(n) / pow(state_vars.dVec(n), 2) ));
                } else if ( bc.natural ) {
                    //using natural bdries
                    matList.push_back(Trip(i, i, firstCoefs(i)/state_vars.dVec(n) + secondCoefs(i) / pow(state_vars.dVec(n), 2) ));
                    matList.push_back(Trip(i, i - state_vars.increVec(n), -firstCoefs(i)/state_vars.dVec(n) - 2 * secondCoefs(i) / pow(state_vars.dVec(n), 2) ));
                    matList.push_back(Trip(i, i - 2*state_vars.increVec(n), secondCoefs(i) / pow(state_vars.dVec(n), 2) ));
                }
            } else if ( std::abs(state_vars.stateMat(i,n) - state_vars.lowerLims(n)) < state_vars.dVec(n) / 2.0 ) { //lower boundary
                atBoundsInd(n) = 1;
                if ( (!bc.natural) && (!corner) ) {
                    //if not using natural bdries, set as specified
                    matList.push_back(Trip(i, i, bc.level(n) - bc.first(n)/state_vars.dVec(n) + bc.second(n) / pow(state_vars.dVec(n), 2) ));
                    matList.push_back(Trip(i, i + state_vars.increVec(n), bc.first(n)/state_vars.dVec(n) - 2 * bc.second(n) / pow(state_vars.dVec(n), 2) ));
                    matList.push_back(Trip(i, i + 2*state_vars.increVec(n), bc.second(n) / pow(state_vars.dVec(n), 2) ));
                
                } else if ( bc.natural ) {   

                    matList.push_back(Trip(i, i, - firstCoefs(i)/state_vars.dVec(n) + secondCoefs(i) / pow(state_vars.dVec(n), 2) ));
                    matList.push_back(Trip(i, i + state_vars.increVec(n), firstCoefs(i)/state_vars.dVec(n) - 2 * secondCoefs(i) / pow(state_vars.dVec(n), 2) ));
                    matList.push_back(Trip(i, i + 2*state_vars.increVec(n), secondCoefs(i) / pow(state_vars.dVec(n), 2) ));
                }
            }
            
            
            
        }
        
        
        
        //if the index is at one of the corners:
        if (corner) {
            if (! (bc.natural)) {
                matList.push_back(Trip(i,i, -1.0 ));
                
                for (int n = (state_vars.N - 1); n >=0; --n ) {
                    
                    //check whether it's at upper or lower boundary
                    if ( std::abs(state_vars.stateMat(i,n) - state_vars.upperLims(n)) < state_vars.dVec(n) / 2.0  ) {  //upper boundary
                        matList.push_back(Trip(i, i - state_vars.increVec(n), 1.0 / state_vars.N ) );
                        
                    } else if ( std::abs(state_vars.stateMat(i,n) - state_vars.lowerLims(n)) < state_vars.dVec(n) / 2.0 ) { //lower boundary
                        matList.push_back(Trip(i, i + state_vars.increVec(n), 1.0 / state_vars.N ) );
                    }
                    
                }
            }
            corners.push_back(i);
        }
        
        
        //handle the known vector
        if (atBound) {
            atBounds.push_back(i);
        }
        
        if ( ( !atBound && !bc.natural ) ) {
            matList.push_back(Trip(i,i,levelCoefs(i) ));
        } else if ( bc.natural ) {
            matList.push_back(Trip(i,i,levelCoefs(i) ));
        }
        
        for (int n = (state_vars.N - 1); n >=0; --n ) {
            if ( ( !atBound && !bc.natural ) || (bc.natural && (atBoundsInd(n) < 0 ) ) ) {
                //first derivative
                double firstCoef = firstCoefs(i,n);
                
                if (firstCoef != 0.0) {
                    
                    matList.push_back(Trip(i,i, ( -firstCoef * ( firstCoef > 0) + firstCoef * ( firstCoef < 0) ) / state_vars.dVec(n)  ));
                    matList.push_back(Trip(i,i + state_vars.increVec(n), firstCoef * ( firstCoef > 0) / state_vars.dVec(n) ));
                    matList.push_back(Trip(i,i - state_vars.increVec(n), - firstCoef * ( firstCoef < 0) / state_vars.dVec(n) ));
                }
                
                //second derivative
                double secondCoef = secondCoefs(i,n);
                if (secondCoef != 0.0) {
                    matList.push_back(Trip(i, i, -2 * secondCoef / ( pow(state_vars.dVec(n), 2) ) ));
                    matList.push_back(Trip(i, i + state_vars.increVec(n), secondCoef / ( pow(state_vars.dVec(n), 2) ) ));
                    matList.push_back(Trip(i, i - state_vars.increVec(n), secondCoef / ( pow(state_vars.dVec(n), 2) ) ));
                }
                
                //cross partials
                for (int n_sub = n-1; n_sub >=0; --n_sub) {
                    
                    double crossCoef = (sigmaXCoef[n].row(i).array() * sigmaXCoef[n_sub].row(i).array()).sum();
                    if (crossCoef != 0.0) {
                        matList.push_back(Trip(i, i + state_vars.increVec(n) * (1 + (-1) *   ( std::abs(state_vars.stateMat(i,n) - state_vars.upperLims(n)) < state_vars.dVec(n) / 2.0 ) + (1) *  ( std::abs(state_vars.stateMat(i,n) - state_vars.lowerLims(n)) < state_vars.dVec(n) / 2.0 ) ) 
                        + state_vars.increVec(n_sub) * (1 + (-1) *  ( std::abs(state_vars.stateMat(i,n_sub) - state_vars.upperLims(n_sub)) < state_vars.dVec(n_sub) / 2.0 ) + (1) *  ( std::abs(state_vars.stateMat(i,n_sub) - state_vars.lowerLims(n_sub)) < state_vars.dVec(n_sub) / 2.0 ) ), crossCoef / (4 * state_vars.dVec(n) * state_vars.dVec(n_sub)) ));
                        matList.push_back(Trip(i, i - state_vars.increVec(n) * (1 + (-1) *   ( std::abs(state_vars.stateMat(i,n) - state_vars.upperLims(n)) < state_vars.dVec(n) / 2.0 ) + (1) *  ( std::abs(state_vars.stateMat(i,n) - state_vars.lowerLims(n)) < state_vars.dVec(n) / 2.0 ) ) 
                        - state_vars.increVec(n_sub) * (1 + (-1) *  ( std::abs(state_vars.stateMat(i,n_sub) - state_vars.upperLims(n_sub)) < state_vars.dVec(n_sub) / 2.0 ) + (1) *  ( std::abs(state_vars.stateMat(i,n_sub) - state_vars.lowerLims(n_sub)) < state_vars.dVec(n_sub) / 2.0 )), crossCoef / (4 * state_vars.dVec(n) * state_vars.dVec(n_sub)) ));
                        matList.push_back(Trip(i, i + state_vars.increVec(n) * (1 + (-1) *   ( std::abs(state_vars.stateMat(i,n) - state_vars.upperLims(n)) < state_vars.dVec(n) / 2.0 ) + (1) *  ( std::abs(state_vars.stateMat(i,n) - state_vars.lowerLims(n)) < state_vars.dVec(n) / 2.0 ) ) 
                        - state_vars.increVec(n_sub) * (1 + (-1) *  ( std::abs(state_vars.stateMat(i,n_sub) - state_vars.upperLims(n_sub)) < state_vars.dVec(n_sub) / 2.0 ) + (1) *  ( std::abs(state_vars.stateMat(i,n_sub) - state_vars.lowerLims(n_sub)) < state_vars.dVec(n_sub) / 2.0 )), -crossCoef / (4 * state_vars.dVec(n) * state_vars.dVec(n_sub)) ));
                        matList.push_back(Trip(i, i - state_vars.increVec(n) * (1 + (-1) *   ( std::abs(state_vars.stateMat(i,n) - state_vars.upperLims(n)) < state_vars.dVec(n) / 2.0 ) + (1) *  (state_vars.stateMat(i,n) - state_vars.lowerLims(n) < state_vars.dVec(n) / 2.0 ) ) 
                        + state_vars.increVec(n_sub) * (1 + (-1) *  ( std::abs(state_vars.stateMat(i,n_sub) - state_vars.upperLims(n_sub)) < state_vars.dVec(n_sub) / 2.0 ) + (1) *  ( std::abs(state_vars.stateMat(i,n_sub) - state_vars.lowerLims(n_sub)) < state_vars.dVec(n_sub) / 2.0 )), -crossCoef / (4 * state_vars.dVec(n) * state_vars.dVec(n_sub)) ));
                    }
                    
                }
            }
            
        }
        

    }
    
    
    //construct matrix
    L.setFromTriplets(matList.begin(), matList.end());
    L = L * dt;
    L.makeCompressed();


}

void linearSysVars::solveT(int T, const bc & bc, stateVars & state_vars, Eigen::MatrixXd phi0, bool changed = false) {
    phiAll.resize(state_vars.S, T);
    for (int t = 0; t < T; t++) {
    
        if (t == 0) {
            //first iteration: interpolate; no need to solve
            phiAll.col(t) = phi0.array();
            

        } else if (t > 0){
            phiAll.col(t) = - phiAll.col(t-1);
            
            //take care of the known vector for boundaries
            if (! (bc.natural)) {

                for (int i = 0; i < atBounds.size(); ++i) {
                    phiAll(atBounds[i],t) =  -bc.a0;
                }
            }

            //solve
            phiAll.col(t) = solver.solve(phiAll.col(t));
            
            //adjust corners
                
            for (int i = 0; i < corners.size(); ++i) {
            
                phiAll(corners[i],t) = 0.0;

            }
        }
    }


}


void
mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[])
{
    int        i;

    
    /* Translating state space inputs into C++ EIGEN */
    
    /* Create state space */

    int nRows = mxGetM(prhs[0]); int nCols = mxGetN(prhs[0]);
    Eigen::Map<Eigen::MatrixXd> preLoadMat((double *)mxGetPr(prhs[0]),nRows,nCols);
    
    stateVars stateSpace(preLoadMat);    

    
    /* Load in drift and vol of consumption */
    mxArray *mxValue; 
    mxValue = mxGetField(prhs[1], 0, "muC");      
    nRows = mxGetM(mxValue); nCols = mxGetN(mxValue);
    Eigen::Map<Eigen::MatrixXd> muC((double *)mxGetPr(mxValue),nRows,nCols);
    
    mxValue = mxGetField(prhs[1], 0, "sigmaC");      
    nRows = mxGetM(mxValue); nCols = mxGetN(mxValue);
    Eigen::Map<Eigen::MatrixXd> sigmaC((double *)mxGetPr(mxValue),nRows,nCols);

    
    /* Load in drifts and vols of state variables */
    
    mxValue = mxGetField(prhs[1], 0, "muX");      
    nRows = mxGetM(mxValue); nCols = mxGetN(mxValue);
    Eigen::Map<Eigen::MatrixXd> muX((double *)mxGetPr(mxValue),nRows,nCols);

    std::vector<Eigen::MatrixXd> sigmaXVec; //create vector to hold vols
    
    mxValue = mxGetField(prhs[1], 0, "sigmaX"); 
    double  *p;
    mxArray *cellElement;  // Typo: * was missing, thanks James
    const mwSize *dims; 
    dims = mxGetDimensions( mxValue );
    for (int j = 0; j < dims[1]; j++) {
      cellElement = mxGetCell( mxValue,j);
      nRows = mxGetM(cellElement); nCols = mxGetN(cellElement);
      p = mxGetPr(cellElement);
      
      Eigen::Map<Eigen::MatrixXd> sigmaX(p,nRows,nCols);
      sigmaXVec.push_back(sigmaX);
    }

    /* Get boundary conditions */
    
    bc bc(stateSpace.N); //Create struct to hold boundary conditions
    mxValue = mxGetField(prhs[2], 0, "natural"); mxLogical *natural;
    natural = mxGetLogicals(mxValue);

    
    bc.natural = *natural;
    
    double *value; 
    if (! (bc.natural) ) {
        for(int j = 0; j < stateSpace.N; j++) {
            mxValue = mxGetField(prhs[2], 0, "level");
            value = mxGetPr(mxValue);
            bc.level(j) = value[j];
            mxValue = mxGetField(prhs[2], 0, "first");
            value = mxGetPr(mxValue);
            bc.first(j) = value[j];
            mxValue = mxGetField(prhs[2], 0, "second");
            value = mxGetPr(mxValue);
            bc.second(j) = value[j];
        }

    }

    mxValue = mxGetField(prhs[2], 0, "a0");
    value = mxGetPr(mxValue);
    bc.a0 = value[0];

    /* Construct linear system */
    mxValue = mxGetField(prhs[1], 0, "dt"); 
    double dtInput = mxGetPr(mxValue)[0];  //Load in dt;
    
    linearSysVars linearSys_vars(stateSpace, muC, sigmaC, muX, sigmaXVec, bc, dtInput);
        
    /***************************************/
    /* Solve the system and send to MATLAB */  
    /***************************************/
    mxValue = mxGetField(prhs[1], 0, "T"); 
    int T = mxGetPr(mxValue)[0];  //Load in T (total time);

    mxValue = mxGetField(prhs[1], 0, "RHS");      
    nRows = mxGetM(mxValue); nCols = mxGetN(mxValue);
    Eigen::Map<Eigen::MatrixXd> phi0((double *)mxGetPr(mxValue),nRows,nCols);
    
    /* Send solution to MATLAB */
    mwSize rows = stateSpace.S;
    mwSize cols = T;
    
    mexPrintf("...Solving for expectation(phi)... \n");
    mexEvalString("drawnow;");
    for (int j = 0; j < nCols; j++) {
        linearSys_vars.solveT(T, bc, stateSpace, phi0.col(j).array() );
        plhs[j] = mxCreateDoubleMatrix(rows, cols, mxREAL); // Create MATLAB array of same size
        Eigen::Map<Eigen::MatrixXd> map(mxGetPr(plhs[j]), rows, cols); // Map the array
        map = linearSys_vars.phiAll; // Copy
    }
    mexPrintf("...Finished solving for expectation(phi)... \n");
    mexEvalString("drawnow;");
    
    /* Send state space to MATLAB */
    plhs[nCols] = mxCreateDoubleMatrix(1, stateSpace.N, mxREAL); // Create MATLAB array of same size
    Eigen::Map<Eigen::MatrixXd> map1(mxGetPr(plhs[nCols]), 1, stateSpace.N); // Map the array
    map1 = stateSpace.dVec; // Copy
    
    plhs[nCols + 1] = mxCreateDoubleMatrix(1, stateSpace.N, mxREAL); // Create MATLAB array of same size
    Eigen::Map<Eigen::MatrixXd> map2(mxGetPr(plhs[nCols + 1]), 1, stateSpace.N); // Map the array
    map2 = stateSpace.increVec; // Copy
    

}

