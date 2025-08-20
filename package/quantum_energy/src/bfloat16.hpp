#pragma once

#include <omp.h>


	struct bfloat16 {
		unsigned short int data;
		public:
		bfloat16(){
			data = 0;
		}
		bfloat16(double dbl_val){
			*this = dbl_val;
		}
		//cast to float
		operator float() const {
			unsigned int proc = data<<16;
			return *reinterpret_cast<float*>(&proc);
		}
		//assign to bfloat16
		template <class T>
		bfloat16& operator =(T T_val){
			*this = float(T_val);
			return *this;
		}
		//add bfloat16
		template <class T>
		bfloat16& operator +=(T T_val){
			*this = float(*this) + float(T_val);
			return *this;
		}
		//sub bfloat16
		template <class T>
		bfloat16& operator -=(T T_val){
			*this = float(*this) - float(T_val);
			return *this;
		}
		//div bfloat16
		template <class T>
		bfloat16& operator /=(T T_val){
			*this = float(*this) / float(T_val);
			return *this;
		}
		//mul bfloat16
		template <class T>
		bfloat16& operator *=(T T_val){
			*this = float(*this) * float(T_val);
			return *this;
		}

		bool operator<(const bfloat16& rhs) const {
			return float(*this) < float(rhs);
		}
	};

	template <class T>
	bool operator<(const T& lhs, const bfloat16& rhs) {
		return float(lhs) < float(rhs);
	}
	
	template <class T>
	bool operator<(const bfloat16& lhs, const T& rhs) {
		return float(lhs) < float(rhs);
	}

	float max(const bfloat16 &lhs, const bfloat16 &rhs) {
		return float(lhs) > float(rhs) ? float(lhs) : float(rhs);
	}

	//assign to bfloat16
	template <>
	bfloat16& bfloat16::operator=<float>(float float_val){
		data = (*reinterpret_cast<unsigned int *>(&float_val))>>16;
		return *this;
	}


#ifndef _MSC_VER
#pragma omp declare reduction(+: bfloat16: omp_out += omp_in)
#pragma omp declare reduction(max: bfloat16: omp_out = (omp_in<omp_out) ? omp_out : omp_in)
#pragma omp declare reduction(min: bfloat16: omp_out = (omp_out<omp_in) ? omp_out : omp_in)
#endif
