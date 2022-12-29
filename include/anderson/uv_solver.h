#pragma once

#include <memory>
#include <vector>
#include <stdexcept>
#include <type_traits>

#include "lapacke.h"

#include "qutility/traits.h"
#include "qutility/array_wrapper/array_wrapper_cpu.h"

#include "helper_macros.h"
#include "dual_history.h"

namespace anderson
{
    namespace host
    {
        namespace detail
        {
            inline double weightedAdd(size_t index) { return 0; }

            template <typename ValT, typename... PointerWeightTypePair>
            inline double weightedAdd(size_t index, ValT *FirstPtr, ValT FirstWeight, PointerWeightTypePair... Rest)
            {
                using qutility::traits::is_type_T;
                static_assert(
                    qutility::traits::is_correct_list_of_n<2, is_type_T<ValT *>, is_type_T<ValT>, ValT *, ValT, PointerWeightTypePair...>::value,
                    "Parameters must be pairs of double* and double");
                return FirstWeight * FirstPtr[index] + weightedAdd(index, Rest...);
            }

            template <typename ValT, typename PtrT, typename WeightT, typename... PointerWeightTypePair>
            inline void combineUV(size_t n_hist, PtrT FirstPtr, WeightT FirstWeight, PointerWeightTypePair... Rest)
            {
                static_assert(std::is_same<typename std::remove_cvref<PtrT>::type, ValT *>::value);
                static_assert(std::is_same<typename std::remove_cvref<WeightT>::type, ValT>::value);
                for (size_t itr = 0; itr < n_hist * (n_hist + 1); ++itr)
                {
                    FirstPtr[itr] *= FirstWeight;
                    FirstPtr[itr] += weightedAdd(itr, Rest...);
                }
            }

            inline bool isTooOld(size_t nhist, size_t pos, size_t pos_push, size_t max_age)
            {
                return (pos < pos_push ? pos_push - pos : (pos_push + nhist) - pos) > max_age;
            }

            template <typename ValT>
            inline void maskOldUV(size_t nhist, ValT *UV, size_t pos_push, size_t max_age)
            {
                for (size_t itr_i = 0; itr_i < nhist; ++itr_i)
                {
                    if (isTooOld(nhist, itr_i, pos_push, max_age))
                        UV[nhist * nhist + itr_i] = 0;
                    for (size_t itr_j = 0; itr_j < nhist; ++itr_j)
                    {
                        if (isTooOld(nhist, itr_j, pos_push, max_age) || isTooOld(nhist, itr_i, pos_push, max_age))
                        {
                            UV[itr_i * nhist + itr_j] = (itr_i == itr_j) * 1.;
                        }
                    }
                }
            }

            template <typename ValT>
            inline void modifyCoef(size_t nhist, ValT *coef)
            {
                double data = 0;
                for (size_t itr = 0; itr < nhist; ++itr)
                    data += coef[itr];
                coef[nhist] = 1 - data;
            }
        }

        template <typename ValT = double, typename MaskValT = int>
        class UVSolver
        {
        public:
            using ArrayT = qutility::array_wrapper::DArrayDDR<ValT>;
            using MaskArrayT = qutility::array_wrapper::DArrayDDR<MaskValT>;
            using DHT = DualHistory<ValT, MaskValT>;

            UVSolver() = delete;
            UVSolver(size_t nhist);
            ANDERSON_CLASS_CTORS_ALL_DEFAULT(UVSolver);

            template <typename... Ts>
            UVSolver &solve(
                size_t using_history_count_suggestion,
                DHT &first_dh, ValT first_weight, Ts &&...Args)
            {
                static_assert(qutility::traits::is_correct_list_of_n<
                              2, qutility::traits::is_type_T<DHT>, qutility::traits::is_type_T<ValT>, DHT, ValT, std::remove_reference_t<Ts>...>::value);

                size_t using_history_count = get_using_history_count(using_history_count_suggestion, first_dh, first_weight, Args...);
                size_t pos = first_dh.get_current_pos();
                solve_impl(using_history_count, pos, to_raw_data(first_dh), to_raw_data(first_weight), to_raw_data(Args)...);

                return *this;
            }

            const ValT *get_coef() { return coef_; }

        private:
            friend class DualHistory<ValT, MaskValT>;
            ;
            const size_t nhist_;
            ArrayT coef_;
            qutility::array_wrapper::ArrayDDR<int> data_;

            size_t get_using_history_count(size_t count_suggestion) { return std::min(count_suggestion, nhist_); }
            template <typename First, typename... Rest>
            size_t get_using_history_count(size_t count_suggestion, First &&first, Rest &&...rest)
            {
                if constexpr (std::is_same<std::remove_reference_t<First>, DHT>::value)
                {
                    size_t val_rest = get_using_history_count(count_suggestion, rest...);
                    size_t val_first_1 = first.get_valid_sequence_count();
                    size_t val_first_2 = first.get_available_count();
                    return std::min(std::min(val_first_1, val_first_2), val_rest);
                }
                else
                {
                    return get_using_history_count(count_suggestion, rest...);
                }
                return 0;
            }

            inline ValT *to_raw_data(DHT &dh) { return dh.get_uv(); }
            inline ValT to_raw_data(ValT &val) { return val; }

            template <typename... Ts>
            void solve_impl(size_t using_history_count, size_t current_pos, double *first_uv, double first_weight, Ts &&...Args)
            {
                static_assert(qutility::traits::is_correct_list_of_n<
                              2, qutility::traits::is_type_T<ValT *>, qutility::traits::is_type_T<ValT>, double *, double, std::remove_reference_t<Ts>...>::value);

                detail::combineUV<ValT>(nhist_, first_uv, first_weight, Args...);
                detail::maskOldUV(nhist_, first_uv, current_pos, using_history_count);

                ValT *U = first_uv;
                ValT *V = U + nhist_ * nhist_;

                // fallback to host host routines

                using qutility::array_wrapper::DArrayDDR;
                DArrayDDR<int> ipiv_host(nhist_);
                if constexpr (std::is_same<ValT, double>::value)
                {
                    auto info = LAPACKE_dgesv(LAPACK_COL_MAJOR, nhist_, 1, U, nhist_, ipiv_host.pointer(), V, nhist_);
                }
                else if constexpr (std::is_same<ValT, float>::value)
                {
                    auto info = LAPACKE_sgesv(LAPACK_COL_MAJOR, nhist_, 1, U, nhist_, ipiv_host.pointer(), V, nhist_);
                }
                // copy results in V to coef_
                for (size_t itr = 0; itr < nhist_ + 1; ++itr)
                {
                    coef_[itr] = V[itr];
                }
                detail::modifyCoef(nhist_, coef_.pointer());
            }
        };

        template <typename ValT, typename MaskValT>
        UVSolver<ValT, MaskValT>::UVSolver(size_t nhist) : nhist_(nhist),
                                                           coef_(0, (nhist + 1)),
                                                           data_(0, (nhist + 1))
        {
        }

    }
}