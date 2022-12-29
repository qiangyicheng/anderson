#pragma once

#include <memory>
#include <vector>
#include <stdexcept>
#include <type_traits>

#include "cblas.h"

#include "qutility/history.h"
#include "qutility/array_wrapper/array_wrapper_cpu.h"

#include "helper_macros.h"

namespace anderson
{
    namespace host
    {
        namespace detail
        {
            template <typename ValT = double, typename MaskValT = int>
            inline void calc_uv(
                size_t single_size, size_t nhist,
                const ValT *new_diff,
                ValT *inner, ValT *UV, const MaskValT *mask,
                size_t N_ava, size_t pos_push)
            {
                ValT diff_square = 0;
#pragma omp parallel for ANDERSON_OMP_SIMD_ALIGNED(new_diff) ANDERSON_OMP_SUM(diff_square)
                for (size_t itr = 0; itr < single_size; ++itr)
                {
                    diff_square += new_diff[itr] * new_diff[itr];
                }

                inner[nhist * (nhist + 1)] = diff_square;

                for (size_t itr_hist_i = 0; itr_hist_i < nhist; ++itr_hist_i)
                {
                    ValT new_new = inner[nhist * (nhist + 1)];
                    ValT new_old = inner[nhist * nhist + itr_hist_i];
                    UV[nhist * nhist + itr_hist_i] = itr_hist_i < N_ava ? (new_new - new_old) : 0;

                    for (size_t itr_hist_j = 0; itr_hist_j < nhist; ++itr_hist_j)
                    {
                        ValT new_old_2 = inner[nhist * nhist + itr_hist_j];
                        UV[itr_hist_i * nhist + itr_hist_j] =
                            (itr_hist_i < N_ava &&
                             itr_hist_j < N_ava &&
                             mask[itr_hist_i] >= 0 &&
                             mask[itr_hist_j] >= 0)
                                ? (inner[itr_hist_i * nhist + itr_hist_j] + new_new - new_old - new_old_2)
                                : (itr_hist_j == itr_hist_i) * 1.;
                        if (itr_hist_i == pos_push && itr_hist_j == pos_push)
                        {
                            inner[itr_hist_i * nhist + itr_hist_j] = new_new;
                        }
                        else if (itr_hist_i == pos_push)
                        {
                            inner[itr_hist_i * nhist + itr_hist_j] = new_old_2;
                        }
                        else if (itr_hist_j == pos_push)
                        {
                            inner[itr_hist_i * nhist + itr_hist_j] = new_old;
                        }
                    }
                }
            }

            template <typename ValT = double>
            void calc_uv_invalid(
                size_t nhist,
                ValT *inner, ValT *UV,
                size_t pos_push)
            {
                {
                    for (size_t itr_hist_i = 0; itr_hist_i < nhist; ++itr_hist_i)
                    {
                        inner[nhist * nhist + itr_hist_i] = 0;
                        UV[nhist * nhist + itr_hist_i] = 0;

                        for (size_t itr_hist_j = 0; itr_hist_j < nhist; ++itr_hist_j)
                        {
                            UV[itr_hist_i * nhist + itr_hist_j] = (itr_hist_j == itr_hist_i) * 1.;
                            if (itr_hist_i == pos_push || itr_hist_j == pos_push)
                            {
                                inner[itr_hist_i * nhist + itr_hist_j] = 0;
                            }
                        }
                    }
                }
            }

            template <typename ValT = double, typename MaskValT = int>
            void mix(
                size_t single_size, size_t nhist,
                ValT *hist, ValT *hist_diff,
                ValT *new_field, const ValT *new_field_diff,
                const ValT *coef,
                MaskValT *mask,
                size_t N_ava, size_t pos_push,
                ValT acceptance)
            {
                ValT *hist_to_store_ptr = hist + pos_push * single_size;
                ValT *hist_diff_to_store_ptr = hist_diff + pos_push * single_size;

#pragma omp parallel for ANDERSON_OMP_SIMD_ALIGNED(hist_to_store_ptr, hist_diff_to_store_ptr, new_field, new_field_diff, hist, hist_diff)
                for (size_t itr = 0; itr < single_size; ++itr)
                {
                    ValT save_val_field = new_field[itr];
                    ValT save_val_field_diff = new_field_diff[itr];
                    ValT new_val_field = coef[nhist] * save_val_field;
                    ValT new_val_field_diff = coef[nhist] * save_val_field_diff;
                    for (size_t itr_ava = 0; itr_ava < N_ava; ++itr_ava)
                    {
                        new_val_field += coef[itr_ava] * hist[itr_ava * single_size + itr];
                        new_val_field_diff += coef[itr_ava] * hist_diff[itr_ava * single_size + itr];
                    }
                    hist_to_store_ptr[itr] = save_val_field;
                    hist_diff_to_store_ptr[itr] = save_val_field_diff;
                    new_field[itr] = new_val_field + acceptance * new_val_field_diff;
                }

                mask[pos_push] = N_ava;
            }

            template <typename ValT = double, typename MaskValT = int>
            void mix_simple(
                size_t single_size,
                ValT *hist, ValT *hist_diff,
                ValT *new_field, const ValT *new_field_diff,
                MaskValT *mask,
                size_t pos_push,
                ValT acceptance)
            {
                ValT *hist_to_store_ptr = hist + pos_push * single_size;
                ValT *hist_diff_to_store_ptr = hist_diff + pos_push * single_size;

#pragma omp parallel for ANDERSON_OMP_SIMD_ALIGNED(hist_to_store_ptr, hist_diff_to_store_ptr, new_field, new_field_diff)
                for (size_t itr = 0; itr < single_size; ++itr)
                {
                    hist_to_store_ptr[itr] = new_field[itr];
                    hist_diff_to_store_ptr[itr] = new_field_diff[itr];
                    new_field[itr] += acceptance * new_field_diff[itr];
                }
                mask[pos_push] = 0;
            }

            template <typename ValT = double, typename MaskValT = int>
            inline void duplicate(
                size_t single_size,
                ValT *hist, ValT *hist_diff,
                const ValT *new_field, const ValT *new_field_diff,
                MaskValT *mask,
                size_t pos_push,
                bool if_valid)
            {
                ValT *hist_to_store_ptr = hist + pos_push * single_size;
                ValT *hist_diff_to_store_ptr = hist_diff + pos_push * single_size;
#pragma omp parallel
                {

#pragma omp for ANDERSON_OMP_SIMD_ALIGNED(hist_to_store_ptr, new_field) nowait
                    for (size_t itr = 0; itr < single_size; ++itr)
                    {
                        hist_to_store_ptr[itr] = new_field[itr];
                    }
                    if (if_valid)
                    {
#pragma omp for ANDERSON_OMP_SIMD_ALIGNED(hist_diff_to_store_ptr, new_field_diff) nowait
                        for (size_t itr = 0; itr < single_size; ++itr)
                        {
                            hist_diff_to_store_ptr[itr] = new_field_diff[itr];
                        }
                    }
                    else
                    {
#pragma omp for ANDERSON_OMP_SIMD_ALIGNED(hist_diff_to_store_ptr) nowait
                        for (size_t itr = 0; itr < single_size; ++itr)
                        {
                            hist_diff_to_store_ptr[itr] = 0;
                        }
                    }
                }

                mask[pos_push] = if_valid ? 0 : -1;

                return;
            }
        }

        template <typename ValT, typename MaskValT>
        class UVSolver;

        template <typename ValT = double, typename MaskValT = int>
        class DualHistory
        {
        public:
            using ArrayT = qutility::array_wrapper::ArrayDDR<ValT>;
            using HistoryT = qutility::history::DHistory<ValT>;

            using MaskArrayT = qutility::array_wrapper::ArrayDDR<MaskValT>;
            using MaskHistoryT = qutility::history::DHistory<MaskValT>;

            DualHistory() = delete;
            DualHistory(size_t single_size, size_t nhist);
            ANDERSON_CLASS_CTORS_ALL_DEFAULT(DualHistory);

            inline size_t get_valid_sequence_count() { return valid_sequence_count_; }
            inline size_t get_available_count() { return field_.available(); }
            inline size_t get_current_pos() { return field_.pos(); }
            inline ValT *get_uv() { return uv_; }

            DualHistory &update_inner_and_calc_uv(const ValT *new_field_diff);
            DualHistory &update_inner_and_calc_uv_invalid();
            DualHistory &mix_and_push(ValT *new_field, const ValT *new_field_diff, const ValT *mixing_coef, ValT acceptance, bool if_valid = true);
            DualHistory &mix_simple_and_push(ValT *new_field, const ValT *new_field_diff, ValT acceptance, bool if_valid = true);
            DualHistory &duplicate_and_push(const ValT *new_field, const ValT *new_field_diff, bool if_valid = false);

            const size_t single_size_;
            const size_t nhist_;

        private:
            friend class UVSolver<ValT, MaskValT>;

            ArrayT hist_;
            ArrayT hist_diff_;

            ArrayT data_;
            ArrayT uv_;

            HistoryT field_;
            HistoryT field_diff_;

            MaskArrayT is_valid_data_;
            MaskHistoryT is_valid_;

            size_t valid_sequence_count_ = 0;

            // derived ptrs
            ValT *const old_old_;
            ValT *const new_old_;
            ValT *const new_new_;
            ValT *const u_;
            ValT *const v_;

            void calc_new_old_impl(const ValT *new_field_diff);
            void calc_uv_impl(const ValT *new_field_diff);
            void calc_uv_invalid_impl();
            void mix_impl(ValT *new_field, const ValT *new_field_diff, const ValT *coef, ValT acceptance);
            void mix_simple_impl(ValT *new_field, const ValT *new_field_diff, ValT acceptance);
            void duplicate_impl(const ValT *new_field, const ValT *new_field_diff, bool if_valid);
            void push_impl(bool if_valid);
            void push_valid_impl();
            void push_invalid_impl();
        };

        template <typename ValT, typename MaskValT>
        DualHistory<ValT, MaskValT>::DualHistory(
            size_t single_size,
            size_t nhist) : single_size_(single_size),
                            nhist_(nhist),
                            data_(0., (nhist + 1) * (nhist + 1)),
                            uv_(0., (nhist + 1) * (nhist)),
                            hist_(0., single_size * nhist),
                            hist_diff_(0., single_size * nhist),
                            field_(hist_.pointer(), single_size, nhist),
                            field_diff_(hist_diff_.pointer(), single_size, nhist),
                            is_valid_data_(-1, nhist) /*-1 means at the beginning all record is invalid*/,
                            is_valid_(is_valid_data_.pointer(), 1, nhist),
                            old_old_(data_.pointer()),
                            new_old_(data_.pointer() + nhist * nhist),
                            new_new_(data_.pointer() + nhist * (nhist + 1)),
                            u_(uv_.pointer()),
                            v_(uv_.pointer() + nhist * nhist)
        {
        }

        template <typename ValT, typename MaskValT>
        DualHistory<ValT, MaskValT> &DualHistory<ValT, MaskValT>::update_inner_and_calc_uv(const ValT *new_field_diff)
        {
            calc_new_old_impl(new_field_diff);
            calc_uv_impl(new_field_diff);
            return *this;
        }
        template <typename ValT, typename MaskValT>
        DualHistory<ValT, MaskValT> &DualHistory<ValT, MaskValT>::update_inner_and_calc_uv_invalid()
        {

            calc_uv_invalid_impl();
            return *this;
        }

        template <typename ValT, typename MaskValT>
        DualHistory<ValT, MaskValT> &DualHistory<ValT, MaskValT>::mix_and_push(ValT *new_field, const ValT *new_field_diff, const ValT *mixing_coef, ValT acceptance, bool if_valid)
        {
            if (!if_valid)
            {
                throw std::logic_error("anderson mixing is only possible when current field residues is valid");
            }
            mix_impl(new_field, new_field_diff, mixing_coef, acceptance);
            push_impl(if_valid);
            return *this;
        }

        template <typename ValT, typename MaskValT>
        DualHistory<ValT, MaskValT> &DualHistory<ValT, MaskValT>::mix_simple_and_push(ValT *new_field, const ValT *new_field_diff, ValT acceptance, bool if_valid)
        {
            if (!if_valid)
            {
                throw std::logic_error("simple mixing is only possible when current field residues is valid");
            }
            mix_simple_impl(new_field, new_field_diff, acceptance);
            push_impl(if_valid);
            return *this;
        }

        template <typename ValT, typename MaskValT>
        DualHistory<ValT, MaskValT> &DualHistory<ValT, MaskValT>::duplicate_and_push(const ValT *new_field, const ValT *new_field_diff, bool if_valid)
        {
            duplicate_impl(new_field, new_field_diff, if_valid);
            push_impl(if_valid);
            return *this;
        }

        template <typename ValT, typename MaskValT>
        void DualHistory<ValT, MaskValT>::calc_new_old_impl(const ValT *new_field_diff)
        {
            if (field_diff_.available() == 0)
                return;
            ValT alpha = 1.;
            ValT beta = 0.;

            static_assert(std::is_same<double, ValT>::value || std::is_same<float, ValT>::value, "requested ValT not implemented");

            if constexpr (std::is_same<double, ValT>::value)
            {
                cblas_dgemv(CblasColMajor, CblasTrans,
                            single_size_, field_diff_.available(),
                            alpha,
                            field_diff_.begin(), single_size_,
                            new_field_diff, 1,
                            beta,
                            new_old_, 1);
            }
            else if constexpr (std::is_same<float, ValT>::value)
            {
                cblas_sgemv(CblasColMajor, CblasTrans,
                            single_size_, field_diff_.available(),
                            alpha,
                            field_diff_.begin(), single_size_,
                            new_field_diff, 1,
                            beta,
                            new_old_, 1);
            }
            return;
        }

        template <typename ValT, typename MaskValT>
        void DualHistory<ValT, MaskValT>::calc_uv_impl(const ValT *new_field_diff)
        {
            detail::calc_uv<ValT, MaskValT>(
                single_size_,
                nhist_,
                new_field_diff,
                data_,
                uv_,
                is_valid_data_,
                field_.available(),
                field_.pos());

            return;
        }

        template <typename ValT, typename MaskValT>
        void DualHistory<ValT, MaskValT>::calc_uv_invalid_impl()
        {
            detail::calc_uv_invalid<ValT>(
                nhist_,
                data_,
                uv_,
                field_.pos());

            return;
        }

        template <typename ValT, typename MaskValT>
        void DualHistory<ValT, MaskValT>::mix_impl(ValT *new_field, const ValT *new_field_diff, const ValT *coef, ValT acceptance)
        {
            detail::mix<ValT, MaskValT>(single_size_, nhist_,
                                        hist_, hist_diff_,
                                        new_field, new_field_diff,
                                        coef,
                                        is_valid_data_,
                                        field_.available(),
                                        field_.pos(),
                                        acceptance);
            return;
        }

        template <typename ValT, typename MaskValT>
        void DualHistory<ValT, MaskValT>::mix_simple_impl(ValT *new_field, const ValT *new_field_diff, ValT acceptance)
        {
            detail::mix_simple<ValT, MaskValT>(single_size_,
                                               hist_, hist_diff_,
                                               new_field, new_field_diff,
                                               is_valid_data_,
                                               field_.pos(),
                                               acceptance);
            return;
        }

        template <typename ValT, typename MaskValT>
        void DualHistory<ValT, MaskValT>::duplicate_impl(const ValT *new_field, const ValT *new_field_diff, bool if_valid)
        {
            detail::duplicate<ValT, MaskValT>(
                single_size_,
                hist_, hist_diff_,
                new_field, new_field_diff,
                is_valid_data_,
                field_.pos(),
                if_valid);
            return;
        }

        template <typename ValT, typename MaskValT>
        void DualHistory<ValT, MaskValT>::push_impl(bool if_valid)
        {
            if (if_valid)
            {
                push_valid_impl();
            }
            else
            {
                push_invalid_impl();
            }
        }

        template <typename ValT, typename MaskValT>
        void DualHistory<ValT, MaskValT>::push_valid_impl()
        {
            field_.push();
            field_diff_.push();
            valid_sequence_count_ += 1;
            if (valid_sequence_count_ > nhist_)
                valid_sequence_count_ = nhist_;
        }

        template <typename ValT, typename MaskValT>
        void DualHistory<ValT, MaskValT>::push_invalid_impl()
        {
            field_.push();
            field_diff_.push();
            valid_sequence_count_ = 0;
        }

    }
}