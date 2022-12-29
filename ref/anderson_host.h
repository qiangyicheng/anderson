#pragma once

#include <memory>
#include <cmath>

#include "qutility/array_wrapper/array_wrapper_cpu.h"
#include "qutility/history.h"

namespace anderson
{
	using HistoryT = qutility::history::DHistory<double>;

	// note that since there're many variables should be set before anderson mixing
	// can run, all these variables are simply exposed.
	// normally, a helper function should be added to transfer these parameters
	// from the TOPSInfo class into this, since this class considers a general cases,
	// more internal parameters would be added if necessary
	class AndersonMixing
	{
	public:
		enum IterationSwitch
		{
			FORCED_ON = 0,
			FORCED_OFF = 1,
			AUTO = 2
		};

		AndersonMixing(
			const std::shared_ptr<const TOPSMemory> &tops_memory) : tops_memory_(tops_memory) {}

		// this function can be used to determine whether the stress is needed
		// or the calculation can be skipped to speedup the calculation
		// note that this if the "fake" stress is used, one should clear the
		// history of the stress, since the "fake" stress is not correct
		void schedule_iteration(size_t itr_number, double incompressibility, double field_error);
		bool if_mixing_cell_para() const { return if_mixing_cell_para_; }
		bool if_anderson_mixing() const { return if_anderson_mixing_; }

		// note that variable_cell_incompressibility_ actually afects how residues
		// are generated, so it is not in this class
		double simple_mixing_field_acceptance_ = 0.05;

		// options for auto cell optimization
		IterationSwitch variable_cell_switch_ = IterationSwitch::AUTO;
		size_t variable_cell_step_ = 20;
		double variable_cell_incompressibility_ = 0.05;
		double variable_cell_field_error_ = 0.02;
		double variable_cell_acceptance_[6] = {0.01, 0.01, 0.01, 0.001, 0.001, 0.001};
		double variable_cell_length_transform_[36] = {
			1, 0, 0, 0, 0, 0,
			0, 1, 0, 0, 0, 0,
			0, 0, 1, 0, 0, 0,
			0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0};

		// operations for anderson mixing
		IterationSwitch anderson_mixing_switch_ = IterationSwitch::AUTO;
		size_t anderson_mixing_max_using_history_ = 20;
		size_t anderson_mixing_step_ = 50;
		double anderson_mixing_incompressibility_ = 0.02;
		double anderson_mixing_rescaler_ = 0.1;
		double anderson_mixing_coef_vector_max_module_ = 10;
		double anderson_mixing_field_error_ = 0.01;
		double anderson_mixing_cell_acceptance_[6] = {0.1, 0.1, 0.1, 0.01, 0.01, 0.01};
		double anderson_mixing_field_acceptance_ = 1.0;
		double anderson_mixing_cell_weight_[6] = {1., 1., 1., 1., 1., 1.};

		size_t anderson_min_avaliable_history_ = 2;

		void mixing(History &hist_field, const History &hist_field_diff, History &hist_cellpara, const History &hist_cellpara_diff) const;

	private:
		void simple_mixing_field(History &hist_field, const History &hist_field_diff) const;
		void simple_mixing_cellpara(History &hist_cellpara, const History &hist_cellpara_diff) const;
		static void repeat_cellpara(History &hist_cellpara);
		void create_and_set_UV_zeros(double *&U, double *&V, size_t N_hist) const;
		void destory_UV(double *&U, double *&V) const;
		static void accummelate_field_diff_to_UV(const History &hist_field_diff, double *const &U, double *const &V, size_t N_hist);
		static void accummelate_cellpara_diff_to_UV(const History &hist_cellpara_diff, double *const &U, double *const &V, const double *weight, size_t N_hist);
		void solve_UV(double *const &U, double *const &V, size_t N_hist) const;
		void append_new_field_according_to_V(History &hist_field, const History &hist_field_diff, const double *const &V, size_t N_hist) const;
		void append_new_cellpara_according_to_V(History &hist_cellpara, const History &hist_cellpara_diff, const double *const &V, size_t N_hist) const;
		void transform_cellpara(double *cellpara, size_t N_cellpara = 6) const;

		void mixing_full(History &hist_field, const History &hist_field_diff, History &hist_cellpara, const History &hist_cellpara_diff) const;
		void mixing_first(History &hist_field, const History &hist_field_diff, History &hist_cellpara, const History &hist_cellpara_diff) const;

		const std::shared_ptr<const TOPSMemory> tops_memory_;

		bool if_mixing_cell_para_ = true;
		bool if_anderson_mixing_ = true;
	};

}