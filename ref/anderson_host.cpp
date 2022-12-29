#include "anderson/anderson_host.h"

double* History::pos_move_next()
{
	if (number_ava_ == 0) {
		number_ava_ = 1;
		return space_;
	}
	else {
		number_ava_ = number_ava_ == number_max_history_ ? number_max_history_ : (number_ava_ + 1);
		if (current_pos_ < number_max_history_ - 1) {
			++current_pos_;
			return space_ + current_pos_ * size_history_;
		}
		else {
			current_pos_ = 0;
			return space_;
		}
	}
}

double* History::at(intptr_t pos) const
{
	for (; pos < 0; pos += static_cast<intptr_t>(number_max_history_)) {}
	size_t temp = current_pos_ + pos;
	temp = temp % number_max_history_;
	return space_ + temp * size_history_;
}

void History::reinterprete_at(intptr_t pos, double** ptr) const
{
	for (; pos < 0; pos += static_cast<intptr_t>(number_max_history_)) {}
	size_t temp = current_pos_ + pos;
	temp = temp % number_max_history_;
	auto required_pos = space_ + temp * size_history_;

	for (size_t index = 0; index < number_part_; ++index) {
		ptr[index] = required_pos + index * size_single_;
	}
	return;
}

void AndersonMixing::simple_mixing_field(History& hist_field, const History& hist_field_diff) const
{
	constexpr size_t alignment = MemspaceAlignment::value;
	const size_t size_history_field = hist_field.size_history_;
	__assume(size_history_field % (alignment / 8) == 0);

	if (hist_field.get_number_avaliable() < 1) {
		std::cout << "ERROR: No field has been found in history" << std::endl;
		exit(1);
	}
	if (hist_field.get_number_avaliable() < 1) {
		std::cout << "ERROR: No field difference has been found in history" << std::endl;
		exit(1);
	}

	double* field_diff_ptr0 = hist_field_diff.pos_current_ptr();
	double* field_ptr0 = hist_field.pos_current_ptr();
	double* new_field_ptr = hist_field.pos_move_next();

#pragma omp parallel for simd aligned(new_field_ptr,field_ptr0,field_diff_ptr0:alignment) TOPS_OMP_SIMDLEN_SCHEDULE
	for (size_t itr = 0; itr < size_history_field; ++itr) {
		new_field_ptr[itr] = field_ptr0[itr] + simple_mixing_field_acceptance_ * field_diff_ptr0[itr];
	}
}

void AndersonMixing::simple_mixing_cellpara(History& hist_cellpara, const History& hist_cellpara_diff) const
{
	const size_t size_history_cellpara = hist_cellpara.size_history_;
	if (size_history_cellpara > 6) {
		std::cout << "ERROR: This is not a supported cell parameter history" << std::endl;
		exit(1);
	}
	if (hist_cellpara.get_number_avaliable() < 1) {
		std::cout << "ERROR: No cell parameter has been found in history" << std::endl;
		exit(1);
	}
	if (hist_cellpara_diff.get_number_avaliable() < 1) {
		std::cout << "ERROR: No cell parameter difference has been found in history" << std::endl;
		exit(1);
	}

	double* cellpara_diff_ptr0 = hist_cellpara_diff.pos_current_ptr();
	double* cellpara_ptr0 = hist_cellpara.pos_current_ptr();
	double* new_cellpara_ptr = hist_cellpara.pos_move_next();

	double cellpara_diff_temp[6];
	for (size_t itr = 0; itr < size_history_cellpara; ++itr) {
		cellpara_diff_temp[itr] = variable_cell_acceptance_[itr] * cellpara_diff_ptr0[itr];
	}
	transform_cellpara(cellpara_diff_temp, size_history_cellpara);
	for (size_t itr = 0; itr < size_history_cellpara; ++itr) {
		new_cellpara_ptr[itr] = cellpara_ptr0[itr] + cellpara_diff_temp[itr];
	}
}

void AndersonMixing::repeat_cellpara(History& hist_cellpara)
{
	const size_t size_history_cellpara = hist_cellpara.size_history_;
	if (size_history_cellpara > 6) {
		std::cout << "ERROR: This is not a supported cell parameter history" << std::endl;
		exit(1);
	}
	if (hist_cellpara.get_number_avaliable() < 1) {
		std::cout << "ERROR: No cell parameter has been found in history" << std::endl;
		exit(1);
	}
	double* cellpara_ptr0 = hist_cellpara.pos_current_ptr();
	double* new_cellpara_ptr = hist_cellpara.pos_move_next();

	for (size_t itr = 0; itr < size_history_cellpara; ++itr) {
		new_cellpara_ptr[itr] = cellpara_ptr0[itr];
	}
}

void AndersonMixing::create_and_set_UV_zeros(double*& U, double*& V, size_t N_hist) const
{
	tops_memory_->anderson_memory_.make_matrix(&U, (N_hist - 1) * (N_hist - 1));
	tops_memory_->anderson_memory_.make_matrix(&V, (N_hist - 1));
	for (size_t itr = 0; itr < (N_hist - 1) * (N_hist - 1); ++itr) U[itr] = 0;
	for (size_t itr = 0; itr < (N_hist - 1); ++itr) V[itr] = 0;
}

void AndersonMixing::destory_UV(double*& U, double*& V) const
{
	tops_memory_->anderson_memory_.destory_matrix(&U);
	tops_memory_->anderson_memory_.destory_matrix(&V);
}

void AndersonMixing::accummelate_field_diff_to_UV(const History& hist_field_diff, double* const& U, double* const& V, size_t N_hist)
{
	constexpr size_t alignment = MemspaceAlignment::value;
	const size_t size_history_field = hist_field_diff.size_history_;
	__assume(size_history_field % (alignment / 8) == 0);
	if (N_hist > hist_field_diff.get_number_avaliable()) {
		std::cout << "ERROR: " << N_hist << " field historied are required while only " << hist_field_diff.get_number_avaliable() << " is present" << std::endl;
		exit(1);
	}

	double* field_diff_ptr0 = hist_field_diff.pos_current_ptr();
#pragma omp parallel
	{
		for (size_t m = 0; m < N_hist - 1; ++m) {
			double* field_diff_ptrm = hist_field_diff.at(-static_cast<intptr_t>(m) - 1);
			for (size_t n = 0; n < N_hist - 1; ++n) {
				double* field_diff_ptrn = hist_field_diff.at(-static_cast<intptr_t>(n) - 1);
				double U_temp = 0;
#pragma omp for simd aligned(field_diff_ptr0,field_diff_ptrm,field_diff_ptrn:alignment) TOPS_OMP_SIMDLEN_SCHEDULE nowait
				for (size_t itr = 0; itr < size_history_field; ++itr) {
					U_temp +=
						(field_diff_ptr0[itr] - field_diff_ptrm[itr]) *
						(field_diff_ptr0[itr] - field_diff_ptrn[itr]);
				}
#pragma omp atomic update
				U[m * (N_hist - 1) + n] += U_temp;
			}

			double V_temp = 0;
#pragma omp for simd aligned(field_diff_ptr0,field_diff_ptrm:alignment) TOPS_OMP_SIMDLEN_SCHEDULE nowait
			for (size_t itr = 0; itr < size_history_field; ++itr) {
				V_temp += (field_diff_ptr0[itr] - field_diff_ptrm[itr]) * field_diff_ptr0[itr];
			}
#pragma omp atomic update
			V[m] += V_temp;

		}
	}
}

void AndersonMixing::accummelate_cellpara_diff_to_UV(const History& hist_cellpara_diff, double* const& U, double* const& V, const double* weight, size_t N_hist)
{
	const size_t size_history_cellpara = hist_cellpara_diff.size_history_;
	if (size_history_cellpara > 6) {
		std::cout << "ERROR: This is not a supported cell parameter history" << std::endl;
		exit(1);
	}
	if (N_hist > hist_cellpara_diff.get_number_avaliable()) {
		std::cout << "ERROR: " << N_hist << " cell parameter historied are required while only " << hist_cellpara_diff.get_number_avaliable() << " is present" << std::endl;
		exit(1);
	}

	double* cellpara_diff_ptr0 = hist_cellpara_diff.pos_current_ptr();
#pragma omp parallel for TOPS_OMP_SCHEDULE
	for (size_t m = 0; m < N_hist - 1; ++m) {
		double* cellpara_diff_ptrm = hist_cellpara_diff.at(-static_cast<intptr_t>(m) - 1);
		for (size_t n = 0; n < N_hist - 1; ++n) {
			double* cellpara_diff_ptrn = hist_cellpara_diff.at(-static_cast<intptr_t>(n) - 1);
			for (size_t itr = 0; itr < size_history_cellpara; ++itr) {
				U[m * (N_hist - 1) + n] +=
					(cellpara_diff_ptr0[itr] - cellpara_diff_ptrm[itr]) *
					(cellpara_diff_ptr0[itr] - cellpara_diff_ptrn[itr]) * weight[itr];
			}
		}
		for (size_t itr = 0; itr < size_history_cellpara; ++itr) {
			V[m] += (cellpara_diff_ptr0[itr] - cellpara_diff_ptrm[itr]) * cellpara_diff_ptr0[itr] * weight[itr];
		}
	}
}

void AndersonMixing::solve_UV(double* const& U, double* const& V, size_t N_hist) const
{
	lapack_int* ipiv;
	tops_memory_->anderson_memory_.make_matrix(&ipiv, (N_hist - 1));
	auto info = LAPACKE_dgesv(LAPACK_ROW_MAJOR, N_hist - 1, 1, U, N_hist - 1, ipiv, V, 1);
	tops_memory_->anderson_memory_.destory_matrix(&ipiv);
	if (info != 0) {
		std::cout << "ERROR: An unrecoverable error encountered when solving matrix reverse during Anderson Mixing" << std::endl;
		exit(1);
	}
	double max = 0;
	for (size_t itr = 0; itr < N_hist - 1; ++itr) {
		max = max > fabs(V[itr]) ? max : fabs(V[itr]);
	}
	if (max > anderson_mixing_coef_vector_max_module_)
		for (size_t itr = 0; itr < N_hist - 1; ++itr) {
			V[itr] *= (anderson_mixing_coef_vector_max_module_ / max);
		}
	for (size_t itr = 0; itr < N_hist - 1; ++itr) {
		V[itr] *= anderson_mixing_rescaler_;
	}
}

void AndersonMixing::append_new_field_according_to_V(History& hist_field, const History& hist_field_diff, const double* const& V, size_t N_hist) const
{
	constexpr size_t alignment = MemspaceAlignment::value;
	const size_t size_history_field = hist_field.size_history_;
	__assume(size_history_field % (alignment / 8) == 0);

	// field part
	double* temp_field, * temp_field_diff;
	tops_memory_->anderson_memory_.make_matrix(&temp_field, size_history_field);
	tops_memory_->anderson_memory_.make_matrix(&temp_field_diff, size_history_field);

	double* field_ptr0 = hist_field.pos_current_ptr();
	double* field_diff_ptr0 = hist_field_diff.pos_current_ptr();
	//new field is added
	double* new_field_ptr = hist_field.pos_move_next();
	double increase_ratio = (1. - pow(0.9, N_hist)) * anderson_mixing_field_acceptance_;
#pragma omp parallel
	{
#pragma omp for simd aligned(temp_field,field_ptr0,temp_field_diff,field_diff_ptr0:alignment) TOPS_OMP_SIMDLEN_SCHEDULE nowait
		for (size_t itr = 0; itr < size_history_field; ++itr) {
			temp_field[itr] = field_ptr0[itr];
			temp_field_diff[itr] = field_diff_ptr0[itr];
		}
		for (size_t n = 0; n < N_hist - 1; ++n) {
			double* field_ptrn = hist_field.at(-static_cast<intptr_t>(n) - 2);//n-2 since the new field has already been appended
			double* field_diff_ptrn = hist_field_diff.at(-static_cast<intptr_t>(n) - 1);
			double coef_temp = V[n];
#pragma omp for simd aligned(temp_field,temp_field_diff,field_ptrn,field_ptr0,field_diff_ptrn,field_diff_ptr0:alignment) TOPS_OMP_SIMDLEN_SCHEDULE nowait
			for (size_t itr = 0; itr < size_history_field; ++itr) {
				temp_field[itr] += coef_temp * (field_ptrn[itr] - field_ptr0[itr]);
				temp_field_diff[itr] += coef_temp * (field_diff_ptrn[itr] - field_diff_ptr0[itr]);
			}
		}
#pragma omp for simd aligned(new_field_ptr,temp_field,temp_field_diff:alignment) TOPS_OMP_SIMDLEN_SCHEDULE
		for (size_t itr = 0; itr < size_history_field; ++itr) {
			new_field_ptr[itr] = temp_field[itr] + increase_ratio * temp_field_diff[itr];
		}
	}
	tops_memory_->anderson_memory_.destory_matrix(&temp_field);
	tops_memory_->anderson_memory_.destory_matrix(&temp_field_diff);
}

void AndersonMixing::append_new_cellpara_according_to_V(History& hist_cellpara, const History& hist_cellpara_diff, const double* const& V, size_t N_hist) const
{
	const size_t size_history_cellpara = hist_cellpara_diff.size_history_;
	double* temp_cellpara = new double[size_history_cellpara];
	double* temp_cellpara_diff = new double[size_history_cellpara];

	double* cellpara_ptr0 = hist_cellpara.pos_current_ptr();
	double* cellpara_diff_ptr0 = hist_cellpara_diff.pos_current_ptr();
	double* new_cellpara_ptr = hist_cellpara.pos_move_next();

	for (size_t itr = 0; itr < size_history_cellpara; ++itr) {
		temp_cellpara[itr] = cellpara_ptr0[itr];
		temp_cellpara_diff[itr] = cellpara_diff_ptr0[itr];
	}
	for (size_t n = 0; n < N_hist - 1; ++n) {
		double* cellpara_ptrn = hist_cellpara.at(-static_cast<intptr_t>(n) - 2);//n-2 since the new field has already been appended
		double* cellpara_diff_ptrn = hist_cellpara_diff.at(-static_cast<intptr_t>(n) - 1);
		double coef_temp = V[n];
		for (size_t itr = 0; itr < size_history_cellpara; ++itr) {
			temp_cellpara[itr] += coef_temp * (cellpara_ptrn[itr] - cellpara_ptr0[itr]);
			temp_cellpara_diff[itr] += coef_temp * (cellpara_diff_ptrn[itr] - cellpara_diff_ptr0[itr]);
		}
	}
	double increase_ratio = 1. - pow(0.9, N_hist);
	double cellpara_diff_temp[6];
	for (size_t itr = 0; itr < size_history_cellpara; ++itr) {
		cellpara_diff_temp[itr] = temp_cellpara[itr] + anderson_mixing_cell_acceptance_[itr] * increase_ratio * temp_cellpara_diff[itr] - cellpara_ptr0[itr];
	}
	transform_cellpara(cellpara_diff_temp, size_history_cellpara);
	for (size_t itr = 0; itr < size_history_cellpara; ++itr) {
		new_cellpara_ptr[itr] = cellpara_ptr0[itr] + cellpara_diff_temp[itr];
	}
	delete[] temp_cellpara;
	delete[] temp_cellpara_diff;
}

void AndersonMixing::transform_cellpara(double* cellpara, size_t N_cellpara) const
{
	double new_cellpara[6];
	for (size_t itr_i = 0; itr_i < N_cellpara; ++itr_i) {
		new_cellpara[itr_i] = 0;
		for (size_t itr_j = 0; itr_j < N_cellpara; ++itr_j) {
			new_cellpara[itr_i] += variable_cell_length_transform_[itr_i * 6 + itr_j] * cellpara[itr_j];
		}
	}
	for (size_t itr_i = 0; itr_i < N_cellpara; ++itr_i) {
		cellpara[itr_i] = new_cellpara[itr_i];
	}
}

void AndersonMixing::mixing_full(History& hist_field, const History& hist_field_diff, History& hist_cellpara, const History& hist_cellpara_diff)const
{
	size_t number_avaliable_field = hist_field.get_number_avaliable();
	size_t number_avaliable_field_diff = hist_field_diff.get_number_avaliable();
	size_t number_avaliable_cellpara = hist_cellpara.get_number_avaliable();
	size_t number_avaliable_cellpara_diff = hist_cellpara_diff.get_number_avaliable();
	if (number_avaliable_cellpara_diff == 0) {
		mixing_first(hist_field, hist_field_diff, hist_cellpara, hist_cellpara_diff);
		return;
	}

	auto min_func = [](size_t a, size_t b) {return a < b ? a : b; };
	size_t N_hist = min_func(number_avaliable_field, number_avaliable_field_diff);
	N_hist = min_func(N_hist, number_avaliable_cellpara);
	N_hist = min_func(N_hist, number_avaliable_cellpara_diff);
	N_hist = min_func(N_hist, anderson_mixing_max_using_history_);
	if (!if_anderson_mixing_) N_hist = 1;

	size_t size_single_field = hist_field.size_history_;
	size_t size_single_cellpara = hist_cellpara.size_history_;

	//too less avaliable histories
	//call simple mixing directily
	if (N_hist <= anderson_min_avaliable_history_ || N_hist < 2) {
		simple_mixing_field(hist_field, hist_field_diff);
		simple_mixing_cellpara(hist_cellpara, hist_cellpara_diff);
		return;
	}

	double* U, * V;
	create_and_set_UV_zeros(U, V, N_hist);
	accummelate_field_diff_to_UV(hist_field_diff, U, V, N_hist);
	accummelate_cellpara_diff_to_UV(hist_cellpara_diff, U, V, anderson_mixing_cell_weight_, N_hist);
	solve_UV(U, V, N_hist);

	append_new_field_according_to_V(hist_field, hist_field_diff, V, N_hist);
	append_new_cellpara_according_to_V(hist_cellpara, hist_cellpara_diff, V, N_hist);

	destory_UV(U, V);
}

void AndersonMixing::mixing_first(History& hist_field, const History& hist_field_diff, History& hist_cellpara, const History& hist_cellpara_diff)const
{
	size_t number_avaliable_field = hist_field.get_number_avaliable();
	size_t number_avaliable_field_diff = hist_field_diff.get_number_avaliable();

	auto min_func = [](size_t a, size_t b) {return a < b ? a : b; };
	size_t N_hist = min_func(number_avaliable_field, number_avaliable_field_diff);
	N_hist = min_func(N_hist, anderson_mixing_max_using_history_);
	if (!if_anderson_mixing_) N_hist = 1;

	size_t size_single_field = hist_field.size_history_;

	//too less avaliable histories
	//call simple mixing directily
	if (N_hist <= anderson_min_avaliable_history_ || N_hist < 2) {
		simple_mixing_field(hist_field, hist_field_diff);
		repeat_cellpara(hist_cellpara);
		return;
	}

	double* U, * V;
	create_and_set_UV_zeros(U, V, N_hist);
	accummelate_field_diff_to_UV(hist_field_diff, U, V, N_hist);
	solve_UV(U, V, N_hist);

	append_new_field_according_to_V(hist_field, hist_field_diff, V, N_hist);
	repeat_cellpara(hist_cellpara);

	destory_UV(U, V);
}

void AndersonMixing::schedule_iteration(size_t itr_number, double incompressibility, double field_error)
{
	switch (variable_cell_switch_)
	{
	case AndersonMixing::FORCED_ON:
		if_mixing_cell_para_ = true;
		break;
	case AndersonMixing::FORCED_OFF:
		if_mixing_cell_para_ = false;
		break;
	case AndersonMixing::AUTO:
		if_mixing_cell_para_ =
			(itr_number > variable_cell_step_ && incompressibility < variable_cell_incompressibility_ && field_error < variable_cell_field_error_) ?
			true : false;
		break;
	default:
		break;
	}
	switch (anderson_mixing_switch_)
	{
	case AndersonMixing::FORCED_ON:
		if_anderson_mixing_ = true;
		break;
	case AndersonMixing::FORCED_OFF:
		if_anderson_mixing_ = false;
		break;
	case AndersonMixing::AUTO:
		if_anderson_mixing_ =
			(itr_number > anderson_mixing_step_ && incompressibility < anderson_mixing_incompressibility_ && field_error < anderson_mixing_field_error_) ?
			true : false;
		break;
	default:
		break;
	}
	return;
}

void AndersonMixing::mixing(History& hist_field, const History& hist_field_diff, History& hist_cellpara, const History& hist_cellpara_diff) const {
	switch (if_mixing_cell_para_)
	{
	case true:
		mixing_full(hist_field, hist_field_diff, hist_cellpara, hist_cellpara_diff);
		break;
	case false:
		mixing_first(hist_field, hist_field_diff, hist_cellpara, hist_cellpara_diff);
		break;
	}
}