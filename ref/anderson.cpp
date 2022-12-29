#include "device/config.h"

#include "anderson.h"

#include "dual_history.h"
#include "uv_solver.h"

namespace tops
{
    AndersonMixing::AndersonMixing(
        int device,
        size_t field_hist_size,
        size_t hist_capacity,
        const std::shared_ptr<tops::Workspace<double>> &workspace) : StreamEventHelper(device),
                                                                     dh_field_(std::make_unique<DualHistory>(device, field_hist_size, hist_capacity, workspace)),
                                                                     dh_cellpara_(std::make_unique<DualHistory>(device, 6, hist_capacity, workspace)),
                                                                     solver_(std::make_unique<UVSolver>(device, hist_capacity, workspace))
    {
    }

    AndersonMixing::~AndersonMixing()
    {
    }

    void AndersonMixing::schedule_iteration(size_t itr_number, double incompressibility, double field_error)
    {
        switch (variable_cell_switch_)
        {
        case AndersonMixing::IterationSwitch::FORCED_ON:
            if_mixing_cell_para_ = true;
            break;
        case AndersonMixing::IterationSwitch::FORCED_OFF:
            if_mixing_cell_para_ = false;
            break;
        case AndersonMixing::IterationSwitch::AUTO:
            if_mixing_cell_para_ =
                (itr_number > variable_cell_step_ && incompressibility < variable_cell_incompressibility_ && field_error < variable_cell_field_error_) ? true : false;
            break;
        default:
            break;
        }
        switch (anderson_mixing_switch_)
        {
        case AndersonMixing::IterationSwitch::FORCED_ON:
            if_anderson_mixing_ = true;
            break;
        case AndersonMixing::IterationSwitch::FORCED_OFF:
            if_anderson_mixing_ = false;
            break;
        case AndersonMixing::IterationSwitch::AUTO:
            if_anderson_mixing_ =
                (itr_number > anderson_mixing_step_ && incompressibility < anderson_mixing_incompressibility_ && field_error < anderson_mixing_field_error_) ? true : false;
            break;
        default:
            break;
        }
        return;
    }

    dapi_cudaEvent_t AndersonMixing::mixing_first(Field &field, Cellpara &cellpara)
    {
        set_device();
        this_wait_other(field.stream_, cellpara.stream_, dh_field_->stream_, dh_cellpara_->stream_, solver_->stream_);

        // note that following 4 values should be identical.
        size_t number_avaliable_field = dh_field_->get_valid_sequence_count();

        auto min_func = [](size_t a, size_t b)
        { return a < b ? a : b; };
        size_t N_hist = min_func(number_avaliable_field, anderson_mixing_max_using_history_);
        if (!if_anderson_mixing_)
            N_hist = 0; // note that in here N_hist is smaller than the original TOPS definition by 1!!!!!!!!!!!!!!!!!!!!!!!!!!

        // too less avaliable histories
        // call simple mixing directily
        if ((N_hist + 1) <= anderson_min_avaliable_history_ || (N_hist + 1) < 2) // note that in here N_hist is smaller than the original TOPS definition by 1!!!!!!!!!!!!!!!!!!!!!!!!!!
        {
            // std::cout << "Nhist = " << N_hist << ", Simple mixing" << std::endl;
            dh_field_->update_inner_and_calc_uv(field);
            dh_field_->mix_simple_and_push(field, simple_mixing_field_acceptance_);

            dh_cellpara_->update_inner_and_calc_uv_invalid();
            dh_cellpara_->duplicate_and_push_invalid(cellpara);
        }
        else
        {
            // std::cout << "Anderson mixing" << std::endl;

            dh_field_->update_inner_and_calc_uv(field);
            dh_cellpara_->update_inner_and_calc_uv_invalid();

            solver_->solve(N_hist, *dh_field_, 1.0); // note that in here N_hist is smaller than the original TOPS definition by 1!!But here the solver use correct N_hist
            dh_field_->mix_and_push(field, *solver_, (1. - pow(0.9, N_hist + 1)) * anderson_mixing_field_acceptance_); // note that in here N_hist is smaller than the original TOPS definition by 1!!!!!!!!!!!!!!!!!!!!!!!!!!
            dh_cellpara_->duplicate_and_push_invalid(cellpara);
        }

        other_wait_this(field.stream_, cellpara.stream_, dh_field_->stream_, dh_cellpara_->stream_, solver_->stream_);
        return record_event();
    }

    dapi_cudaEvent_t AndersonMixing::mixing_full(Field &field, Cellpara &cellpara)
    {
        set_device();
        this_wait_other(field.stream_, cellpara.stream_, dh_field_->stream_, dh_cellpara_->stream_, solver_->stream_);

        size_t number_avaliable_field = dh_field_->get_valid_sequence_count();
        size_t number_avaliable_cellpara = dh_cellpara_->get_valid_sequence_count();

        auto min_func = [](size_t a, size_t b)
        { return a < b ? a : b; };
        size_t N_hist = min_func(number_avaliable_field, number_avaliable_cellpara);
        N_hist = min_func(N_hist, anderson_mixing_max_using_history_);
        if (!if_anderson_mixing_)
            N_hist = 0; // note that in here N_hist is smaller than the original TOPS definition by 1!!!!!!!!!!!!!!!!!!!!!!!!!!

        // too less avaliable histories
        // call simple mixing directily
        if ((N_hist + 1) <= anderson_min_avaliable_history_ || (N_hist + 1) < 2) // note that in here N_hist is smaller than the original TOPS definition by 1!!!!!!!!!!!!!!!!!!!!!!!!!!
        {
            // std::cout << "Nhist = " << N_hist << ", Simple mixing" << std::endl;

            dh_field_->update_inner_and_calc_uv(field);
            dh_field_->mix_simple_and_push(field, simple_mixing_field_acceptance_);

            dh_cellpara_->update_inner_and_calc_uv(cellpara);
            dh_cellpara_->mix_simple_and_push(cellpara, 0.0);
            // the mix routine of DualHistory does not support weights.
            // so the simple mixing of cellpara need to be done on host, and then be copied to device.
            // note that the device routines should still be called, since it is not stateless.
            // the DualHistory need to record the inner products of raw difference
            double cellpara_diff_temp[6];
            for (int itr = 0; itr < 6; ++itr)
            {
                cellpara_diff_temp[itr] = variable_cell_acceptance_[itr] * cellpara.field_diff_host_[itr];
            }
            transform_cellpara(cellpara_diff_temp);
            for (size_t itr = 0; itr < 6; ++itr)
            {
                cellpara.field_host_[itr] += cellpara_diff_temp[itr];
            }
            cellpara.sync_to_device();
        }
        else
        {
            // std::cout << "Anderson mixing" << std::endl;

            dh_field_->update_inner_and_calc_uv(field);
            dh_cellpara_->update_inner_and_calc_uv(cellpara);

            solver_->solve(N_hist, *dh_field_, 1.0, *dh_cellpara_, anderson_mixing_cell_weight_[0]); // note that in here N_hist is smaller than the original TOPS definition by 1!!But here the solver use correct N_hist

            dh_field_->mix_and_push(field, *solver_, (1. - pow(0.9, N_hist + 1)) * anderson_mixing_field_acceptance_);         // note that in here N_hist is smaller than the original TOPS definition by 1!!!!!!!!!!!!!!!!!!!!!!!!!!
            dh_cellpara_->mix_and_push(cellpara, *solver_, (1. - pow(0.9, N_hist + 1)) * anderson_mixing_cell_acceptance_[0]); // note that in here N_hist is smaller than the original TOPS definition by 1!!!!!!!!!!!!!!!!!!!!!!!!!!

            // the mix routine of DualHistory does not support transform.
            // so copy down data to host, transform, and copy back

            double cellpara_diff_old_temp[6];
            double cellpara_diff_temp[6];
            // std::cout << "old cellpara is ";
            for (int itr = 0; itr < 6; ++itr)
            {
                cellpara_diff_old_temp[itr] = cellpara.field_host_[itr];
                // std::cout << cellpara_diff_old_temp[itr] << " ";
            }
            // std::cout << "\n";

            cellpara.sync_from_device();

            // std::cout << "new cellpara by AM is ";
            for (int itr = 0; itr < 6; ++itr)
            {
                cellpara_diff_temp[itr] = cellpara.field_host_[itr] - cellpara_diff_old_temp[itr];
                // std::cout << cellpara.field_host_[itr] << " ";
            }
            // std::cout << "\n";

            transform_cellpara(cellpara_diff_temp);
            // std::cout << "transformed delta is ";
            for (size_t itr = 0; itr < 6; ++itr)
            {
                cellpara.field_host_[itr] = cellpara_diff_old_temp[itr] + cellpara_diff_temp[itr];
                // std::cout << cellpara_diff_temp[itr] << " ";
            }
            // std::cout << "\n";
            
            cellpara.sync_to_device();
        }

        other_wait_this(field.stream_, cellpara.stream_, dh_field_->stream_, dh_cellpara_->stream_, solver_->stream_);
        return record_event();
    }

    void AndersonMixing::transform_cellpara(double *cellpara) const
    {
        double new_cellpara[6];
        for (size_t itr_i = 0; itr_i < 6; ++itr_i)
        {
            new_cellpara[itr_i] = 0;
            for (size_t itr_j = 0; itr_j < 6; ++itr_j)
            {
                new_cellpara[itr_i] += variable_cell_length_transform_[itr_i * 6 + itr_j] * cellpara[itr_j];
            }
        }
        for (size_t itr_i = 0; itr_i < 6; ++itr_i)
        {
            cellpara[itr_i] = new_cellpara[itr_i];
        }
    }

}