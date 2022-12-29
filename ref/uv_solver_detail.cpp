#include "uv_solver.h"
#include "uv_solver.cuh"

namespace tops
{
    TOPS_HOST_ACCESSABLE UVSolver::UVSolver(
        int device,
        size_t nhist,
        const std::shared_ptr<WorkspaceT> &workspace) : StreamEventHelper(device),
                                                        CGHelper(device),
                                                        nhist_(nhist),
                                                        workspace_(workspace),
                                                        coef_(0, (nhist + 1), device),
                                                        data_(0, (nhist + 1), device)
    {
#ifndef TOPS_USING_LIB_SOLVER_FALLBACK_LAPACKE
        set_device();
        dapi_checkCudaErrors(cusolverDnCreate(&cusolver_));
        dapi_checkCudaErrors(cusolverDnSetStream(cusolver_, stream_));
        dapi_checkCudaErrors(cusolverDnDDgesv_bufferSize(cusolver_, nhist_, 1, nullptr, nhist_, nullptr, nullptr, nhist_, nullptr, nhist_, nullptr, &working_size_));
        if (working_size_ > workspace_->size_in_bytes())
        {
            error_message("UVSolver Error: not enough workspace.");
        }
#else
        // fall back to lapacke on host.
        // no initialization required
#endif
    }

    TOPS_HOST_ACCESSABLE UVSolver::~UVSolver()
    {
#ifndef TOPS_USING_LIB_SOLVER_FALLBACK_LAPACKE
        set_device();
        dapi_checkCudaErrors(cusolverDnDestroy(cusolver_));
#else
        // fall back to lapacke on host.
        // no cleanning required
#endif
    }

    TOPS_HOST_ACCESSABLE double * UVSolver::to_raw_data(DualHistory &dh) { return dh.get_uv(); }
    TOPS_HOST_ACCESSABLE double UVSolver::to_raw_data(double &val) { return val; }
    TOPS_HOST_ACCESSABLE dapi_cudaStream_t UVSolver::get_stream(DualHistory &dh){return dh.stream_;}
    TOPS_HOST_ACCESSABLE dapi_cudaStream_t UVSolver::get_stream(double &val){ return nullptr; }
}