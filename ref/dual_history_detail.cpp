#include "dual_history.h"
#include "dual_history.cuh"

#include "uv_solver.h"

namespace tops
{
    TOPS_HOST_ACCESSABLE DualHistory::DualHistory(
        int device,
        size_t single_size,
        size_t nhist,
        const std::shared_ptr<WorkspaceT> &workspace) : StreamEventHelper(device),
                                                        CGHelper(device),
                                                        single_size_(single_size),
                                                        nhist_(nhist),
                                                        workspace_(workspace),
                                                        data_(0., (nhist + 1) * (nhist + 1), device),
                                                        uv_(0., (nhist + 1) * (nhist), device),
                                                        hist_(0., single_size * nhist, device),
                                                        hist_diff_(0., single_size * nhist, device),
                                                        field_(hist_.pointer(), single_size, nhist),
                                                        field_diff_(hist_diff_.pointer(), single_size, nhist),
                                                        is_valid_data_(-1, nhist, device) /*-1 means at the beginning all record is invalid*/,
                                                        is_valid_(is_valid_data_.pointer(), 1, nhist),
                                                        old_old_(data_.pointer()),
                                                        new_old_(data_.pointer() + nhist * nhist),
                                                        new_new_(data_.pointer() + nhist * (nhist + 1)),
                                                        u_(uv_.pointer()),
                                                        v_(uv_.pointer() + nhist * nhist)
    {
        set_device();
        dapi_checkCudaErrors(dapi_cublasCreate_v2(&dapi_cublashandle));
        dapi_checkCudaErrors(dapi_cublasSetStream_v2(dapi_cublashandle, stream_));
    }

    TOPS_HOST_ACCESSABLE DualHistory::~DualHistory()
    {
        set_device();
        dapi_checkCudaErrors(dapi_cublasDestroy_v2(dapi_cublashandle));
    }

    TOPS_HOST_ACCESSABLE dapi_cudaEvent_t DualHistory::update_inner_and_calc_uv(Field &field)
    {
        set_device();
        this_wait_other(field.stream_);
        calc_new_old_impl(field.field_diff_);
        calc_uv_impl(field.field_diff_);
        other_wait_this(field.stream_);
        return record_event();
    }
    TOPS_HOST_ACCESSABLE dapi_cudaEvent_t DualHistory::update_inner_and_calc_uv_invalid()
    {
        set_device();
        calc_uv_invalid_impl();
        return record_event();
    }

    TOPS_HOST_ACCESSABLE dapi_cudaEvent_t DualHistory::mix_and_push(Field &field, UVSolver &uv_solver, double acceptance)
    {
        set_device();
        this_wait_other(field.stream_);
        this_wait_other(uv_solver.stream_);
        mix_impl(field.field_, field.field_diff_, uv_solver.coef_, acceptance);
        push_impl();
        other_wait_this(field.stream_);
        other_wait_this(uv_solver.stream_);
        return record_event();
    }

    TOPS_HOST_ACCESSABLE dapi_cudaEvent_t DualHistory::mix_simple_and_push(Field &field, double acceptance)
    {
        set_device();
        this_wait_other(field.stream_);
        mix_simple_impl(field.field_, field.field_diff_, acceptance);
        push_impl();
        other_wait_this(field.stream_);
        return record_event();
    }

    TOPS_HOST_ACCESSABLE dapi_cudaEvent_t DualHistory::duplicate_and_push_invalid(Field &field)
    {
        set_device();
        this_wait_other(field.stream_);
        duplicate_impl(field.field_);
        push_invalid_impl();
        other_wait_this(field.stream_);
        return record_event();
    }

    TOPS_HOST_ACCESSABLE void DualHistory::calc_new_old_impl(double *new_field_diff)
    {
        if (field_diff_.available() == 0)
            return;
        double alpha = 1.;
        double beta = 0.;
        dapi_checkCudaErrors(dapi_cublasDgemv_v2(
            dapi_cublashandle, DAPI_CUBLAS_OP_T,
            single_size_, (int)(field_diff_.available()),
            &alpha,
            field_diff_.begin(), single_size_,
            new_field_diff, 1,
            &beta,
            new_old_, 1));
        return;
    }

    TOPS_HOST_ACCESSABLE void DualHistory::calc_uv_impl(double *new_field_diff)
    {
        this_wait_other(workspace_->stream_);
        dapi_checkCudaErrors(launch_kernel_on_stream(
            kernel::andersonDualHistoryCalcUV<threads_per_block_>,
            dim_grid_, dim_block_, 0,
            single_size_,
            nhist_,
            new_field_diff,
            data_,
            uv_,
            is_valid_data_,
            *workspace_,
            field_.available(),
            field_.pos()));
        other_wait_this(workspace_->stream_);
        return;
    }

    TOPS_HOST_ACCESSABLE void DualHistory::calc_uv_invalid_impl()
    {
        this_wait_other(workspace_->stream_);
        dapi_checkCudaErrors(launch_kernel_on_stream(
            kernel::andersonDualHistoryCalcUVInvalid,
            dim_grid_, dim_block_, 0,
            nhist_,
            data_,
            uv_,
            field_.pos()));
        other_wait_this(workspace_->stream_);
        return;
    }

    TOPS_HOST_ACCESSABLE void DualHistory::mix_impl(double *new_field, const double *new_field_diff, const double *coef, double acceptance)
    {
        dapi_checkCudaErrors(launch_kernel_on_stream(
            kernel::andersonDualHistoryMix,
            dim_grid_, dim_block_, (nhist_ + 1) * sizeof(double),
            single_size_, nhist_,
            hist_, hist_diff_,
            new_field, new_field_diff,
            coef,
            is_valid_data_,
            field_.available(),
            field_.pos(),
            acceptance));
        return;
    }

    TOPS_HOST_ACCESSABLE void DualHistory::mix_simple_impl(double *new_field, const double *new_field_diff, double acceptance)
    {
        dapi_checkCudaErrors(launch_kernel_on_stream(
            kernel::andersonDualHistoryMixSimple,
            dim_grid_, dim_block_, 0,
            single_size_,
            hist_, hist_diff_,
            new_field, new_field_diff,
            is_valid_data_,
            field_.pos(),
            acceptance));
        return;
    }

    TOPS_HOST_ACCESSABLE void DualHistory::duplicate_impl(const  double *new_field)
    {
        dapi_checkCudaErrors(launch_kernel_on_stream(
            kernel::andersonDualHistoryDuplicate,
            dim_grid_, dim_block_, 0,
            single_size_,
            hist_, hist_diff_,
            new_field,
            is_valid_data_,
            field_.pos()));
        return;
    }

    TOPS_HOST_ACCESSABLE void DualHistory::push_impl()
    {
        field_.push();
        field_diff_.push();
        valid_sequence_count_ += 1;
        if (valid_sequence_count_ > nhist_)
            valid_sequence_count_ = nhist_;
    }

    TOPS_HOST_ACCESSABLE void DualHistory::push_invalid_impl()
    {
        field_.push();
        field_diff_.push();
        valid_sequence_count_ = 0;
    }
}