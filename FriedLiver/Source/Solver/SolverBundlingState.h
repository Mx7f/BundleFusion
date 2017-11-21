#pragma once

#ifndef _SOLVER_STATE_
#define _SOLVER_STATE_

#include <cuda_runtime.h> 
#include "../SiftGPU/SIFTImageManager.h"
#include "../CUDACacheUtil.h"
#include "SolverBundlingParameters.h"

struct SolverInput
{	
	EntryJ* d_correspondences;
	int* d_variablesToCorrespondences;
	int* d_numEntriesPerRow;

	unsigned int numberOfCorrespondences;
	unsigned int numberOfImages;

	unsigned int maxNumberOfImages;
	unsigned int maxCorrPerImage;

	const int* d_validImages;
	const CUDACachedFrame* d_cacheFrames;
	unsigned int denseDepthWidth;
	unsigned int denseDepthHeight;
	float4 intrinsics;				//TODO constant buffer for this + siftimagemanger stuff?
	unsigned int maxNumDenseImPairs;
	float2 colorFocalLength; //color camera params (actually same as depthIntrinsics...)

	const float* weightsSparse;
	const float* weightsDenseDepth;
	const float* weightsDenseColor;
};

// State of the GN Solver
struct SolverState
{
	float3*	d_deltaRot;					// Current linear update to be computed
	float3*	d_deltaTrans;				// Current linear update to be computed
	
	float3* d_xRot;						// Current state
	float3* d_xTrans;					// Current state

	float3*	d_rRot;						// Residuum // jtf
	float3*	d_rTrans;					// Residuum // jtf
	
	float3*	d_zRot;						// Preconditioned residuum
	float3*	d_zTrans;					// Preconditioned residuum
	
	float3*	d_pRot;						// Decent direction
	float3*	d_pTrans;					// Decent direction
	
	float3*	d_Jp;						// Cache values after J

	float3*	d_Ap_XRot;					// Cache values for next kernel call after A = J^T x J x p
	float3*	d_Ap_XTrans;				// Cache values for next kernel call after A = J^T x J x p

	float*	d_scanAlpha;				// Tmp memory for alpha scan

	float*	d_rDotzOld;					// Old nominator (denominator) of alpha (beta)
	
	float3*	d_precondionerRot;			// Preconditioner for linear system
	float3*	d_precondionerTrans;		// Preconditioner for linear system

	float*	d_sumResidual;				// sum of the squared residuals //debug

	//float* d_residuals; // debugging
	//float* d_sumLinResidual; // debugging // helper to compute linear residual

	int* d_countHighResidual;

	__host__ float getSumResidual() const {
		float residual;
		cudaMemcpy(&residual, d_sumResidual, sizeof(float), cudaMemcpyDeviceToHost);
		return residual;
	}

	// for dense depth term
	float* d_denseJtJ;
	float* d_denseJtr;
	float* d_denseCorrCounts;

	float4x4* d_xTransforms;
	float4x4* d_xTransformInverses;

	uint2* d_denseOverlappingImages;
	int* d_numDenseOverlappingImages;

	//!!!DEBUGGING
	int* d_corrCount;
	int* d_corrCountColor;
	float* d_sumResidualColor;
};

struct SolverStateAnalysis
{
	// residual pruning
	int*	d_maxResidualIndex;
	float*	d_maxResidual;

	int*	h_maxResidualIndex;
	float*	h_maxResidual;
};

template <typename T>
void writePOD(std::ofstream& s, const T  v) {
    s.write((const char*)&v, sizeof(T));
}

template <typename T>
void writeArray(std::ofstream& s, const T* v, const uint count) {
    s.write((const char*)&count, sizeof(uint));
    s.write((const char*)v, sizeof(T)*count);
}

template <typename T>
void writeCudaArray(std::ofstream& s, const T* d_v, const uint count) {
    std::vector<T> vec;
    vec.resize(count);
    MLIB_CUDA_SAFE_CALL(cudaMemcpy(vec.data(), d_v, sizeof(T)*count, cudaMemcpyDeviceToHost));
    writeArray<T>(s, vec.data(), count);
}

template <typename T>
void readPOD(std::ifstream& s, T* v) {
    s.read((char*)v, sizeof(T));
}

template <typename T>
void readArray(std::ifstream& s, T* v, uint* count) {
    s.read((char*)count, sizeof(uint));
    s.read((char*)v, sizeof(T)*(*count));
}

template <typename T>
void readCudaArray(std::ifstream& s, T* d_v, uint* count) {
    std::vector<T> vec;
    s.read((char*)count, sizeof(uint));
    vec.resize(*count);
    s.read((char*)vec.data(), sizeof(T)*(*count));
    MLIB_CUDA_SAFE_CALL(cudaMemcpy(d_v, vec.data(), sizeof(T)*(*count), cudaMemcpyHostToDevice));
}

struct SolverInputPOD {
    uint numCorr;
    uint numIm;
    uint maxIm;
    uint maxCorr;
    uint denseW;
    uint denseH;
    float4 intrinsics;
    uint maxNumDenseImPairs;
    float2 colorFocalLength;
    SolverInputPOD() {}
    SolverInputPOD(const SolverInput& input) {
        numCorr = input.numberOfCorrespondences;
        numIm = input.numberOfImages;
        maxIm = input.maxNumberOfImages;
        maxCorr = input.maxCorrPerImage;
        denseW = input.denseDepthWidth;
        denseH = input.denseDepthWidth;
        intrinsics = input.intrinsics;
        maxNumDenseImPairs = input.maxNumDenseImPairs;
        colorFocalLength = input.colorFocalLength;
    }
    void transferValues(SolverInput& input) {
        input.numberOfCorrespondences = numCorr;
        input.numberOfImages = numIm;
        input.maxNumberOfImages = maxIm;
        input.maxCorrPerImage = maxCorr;
        input.denseDepthWidth = denseW;
        input.denseDepthWidth = denseH;
        input.intrinsics = intrinsics;
        input.maxNumDenseImPairs = maxNumDenseImPairs;
        input.colorFocalLength = colorFocalLength;
    }
};

static void saveAllStateToFile(std::string filename, const SolverInput& input, const SolverState& state, const SolverParameters& parameters) {
    std::ofstream out(filename, std::ios::binary);
    if (!out.is_open()) {
        std::cout << "Error opening " << filename << " for write" << std::endl;
        return;
    }
    writePOD<SolverParameters>(out, parameters);
    SolverInputPOD inputPOD(input);
    writePOD<SolverInputPOD>(out, inputPOD);
    writeCudaArray<EntryJ>(out, input.d_correspondences, inputPOD.numCorr);
    writeCudaArray<int>(out, input.d_variablesToCorrespondences, inputPOD.numIm*inputPOD.maxCorr);
    writeCudaArray<int>(out, input.d_numEntriesPerRow, inputPOD.numIm);
    writeCudaArray<int>(out, input.d_validImages, inputPOD.numIm);
    int hasCache = input.d_cacheFrames ? 1 : 0;
    writePOD<int>(out, hasCache);
    if (hasCache) {
        std::vector<CUDACachedFrame> cacheFrames;
        cacheFrames.resize(inputPOD.numIm);
        printf("%p\n", input.d_cacheFrames);
        MLIB_CUDA_SAFE_CALL(cudaMemcpy(cacheFrames.data(), input.d_cacheFrames, sizeof(CUDACachedFrame)*inputPOD.numIm, cudaMemcpyDeviceToHost));
        for (auto f : cacheFrames) {
            writeCudaArray<float>(out, f.d_depthDownsampled, inputPOD.numIm);
            writeCudaArray<float4>(out, f.d_cameraposDownsampled, inputPOD.numIm);
            writeCudaArray<float>(out, f.d_intensityDownsampled, inputPOD.numIm); //this could be packed with intensityDerivaties to a float4 dunno about the read there
            writeCudaArray<float2>(out, f.d_intensityDerivsDownsampled, inputPOD.numIm); //TODO could have energy over intensity gradient instead of intensity
            writeCudaArray<float4>(out, f.d_normalsDownsampled, inputPOD.numIm);
        }
    }
    writeArray<float>(out, input.weightsSparse, parameters.nNonLinearIterations);
    writeArray<float>(out, input.weightsDenseDepth, parameters.nNonLinearIterations);
    writeArray<float>(out, input.weightsDenseColor, parameters.nNonLinearIterations);
    writeCudaArray<float3>(out, state.d_xRot, inputPOD.numIm);
    writeCudaArray<float3>(out, state.d_xTrans, inputPOD.numIm);

    out.close();
}

static void loadAllStateFromFile(std::string filename, SolverInput& input, SolverState& state, SolverParameters& parameters) {
    std::ifstream inp(filename, std::ios::binary);
    if (!inp.is_open()) {
        std::cout << "Error opening " << filename << " for read" << std::endl;
        return;
    }
    readPOD<SolverParameters>(inp, &parameters);
    SolverInputPOD inputPOD;
    readPOD<SolverInputPOD>(inp, &inputPOD);
    inputPOD.transferValues(input);
    readCudaArray<EntryJ>(inp, input.d_correspondences, &inputPOD.numCorr);

    uint variableToCorrCount = inputPOD.numIm*inputPOD.maxCorr;
    readCudaArray<int>(inp, input.d_variablesToCorrespondences, &variableToCorrCount);
    readCudaArray<int>(inp, input.d_numEntriesPerRow, &inputPOD.numIm);
    readCudaArray<int>(inp, (int*)input.d_validImages, &inputPOD.numIm);

    int hasCache = 0;
    readPOD<int>(inp, &hasCache);
    if (hasCache) {
        std::vector<CUDACachedFrame> cacheFrames;
        cacheFrames.resize(inputPOD.numIm);
        MLIB_CUDA_SAFE_CALL(cudaMemcpy(cacheFrames.data(), input.d_cacheFrames, sizeof(CUDACachedFrame)*inputPOD.numIm, cudaMemcpyDeviceToHost));
        for (auto f : cacheFrames) {
            readCudaArray<float>(inp, f.d_depthDownsampled, &inputPOD.numIm);
            readCudaArray<float4>(inp, f.d_cameraposDownsampled, &inputPOD.numIm);
            readCudaArray<float>(inp, f.d_intensityDownsampled, &inputPOD.numIm);
            readCudaArray<float2>(inp, f.d_intensityDerivsDownsampled, &inputPOD.numIm);
            readCudaArray<float4>(inp, f.d_normalsDownsampled, &inputPOD.numIm);
        }
    }
    else {
        input.d_cacheFrames = nullptr;
    }
    readArray<float>(inp, (float*)input.weightsSparse, &parameters.nNonLinearIterations);
    readArray<float>(inp, (float*)input.weightsDenseDepth, &parameters.nNonLinearIterations);
    readArray<float>(inp, (float*)input.weightsDenseColor, &parameters.nNonLinearIterations);
    readCudaArray<float3>(inp, state.d_xRot, &inputPOD.numIm);
    readCudaArray<float3>(inp, state.d_xTrans, &inputPOD.numIm);

    inp.close();
}

#endif
