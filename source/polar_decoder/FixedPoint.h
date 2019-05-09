#ifndef FIXEDPOINT_H
#define FIXEDPOINT_H

#include <algorithm>
#include <limits>
#include <cstdint>
#include <cmath>

namespace FixedPoint {

    template<unsigned NUMBER_BITS, unsigned NUMBER_FRACTIONAL_BITS>
    class Type {
    public:
        static constexpr unsigned k_numberBits = NUMBER_BITS;
        static constexpr unsigned k_numberFractionalBits = NUMBER_FRACTIONAL_BITS;

        static constexpr int8_t k_max = (int8_t) ((1 << (NUMBER_BITS - 1)) - 1);
        static constexpr int8_t k_min = (int8_t) -(1 << (NUMBER_BITS - 1));
    };

    template<typename TYPE>
    class Data {
    private:
        int8_t data = 0;

    public:
        Data() = default;

        explicit Data(const int8_t input) {
            this->data = input;
        }

        explicit Data(const float input) {
            const float result = std::round(input * (1 << TYPE::k_numberFractionalBits));

            if (result > TYPE::k_max)
                this->data = TYPE::k_max;
            else if (result < TYPE::k_min)
                this->data = TYPE::k_min;
            else
                this->data = (int8_t) result;
        }

        explicit operator float() const {
            return this->data / (float) (1 << TYPE::k_numberFractionalBits);
        }

        int8_t getRawData() const {
            return data;
        }

        template<typename T>
        friend Data<T> operator+(const Data<T> &lhs, const Data<T> &rhs);

        template<typename LHS_T, typename RHS_T>
        friend Data<LHS_T> operator*(const Data<LHS_T> &lhs, const Data<RHS_T> &rhs);

        template<typename T>
        friend bool operator<(const Data<T> &lhs, const Data<T> &rhs);

        template<typename T>
        friend bool operator>(const Data<T> &lhs, const Data<T> &rhs);

        template<typename T>
        friend bool operator==(const Data<T> &lhs, const Data<T> &rhs);
    };

    template<typename T>
    Data<T> operator+(const Data<T> &lhs, const Data<T> &rhs) {
        if (rhs.data > 0 && lhs.data > T::k_max - rhs.data)
            return Data<T>(T::k_max);
        else if (rhs.data < 0 && lhs.data < T::k_min - rhs.data)
            return Data<T>(T::k_min);
        else
            return Data<T>((int8_t) (lhs.data + rhs.data));
    }

    template<typename LHS_T, typename RHS_T>
    Data<LHS_T> operator*(const Data<LHS_T> &lhs, const Data<RHS_T> &rhs) {
        auto lhsExpanded = (int16_t) lhs.data;
        auto rhsExpanded = (int16_t) rhs.data;

        if (RHS_T::k_numberFractionalBits > LHS_T::k_numberFractionalBits) {
            constexpr auto k_pointAdjustment = RHS_T::k_numberFractionalBits - LHS_T::k_numberFractionalBits;

            lhsExpanded <<= k_pointAdjustment;
        }

        if (RHS_T::k_numberFractionalBits < LHS_T::k_numberFractionalBits) {
            constexpr auto k_pointAdjustment = LHS_T::k_numberFractionalBits - RHS_T::k_numberFractionalBits;

            rhsExpanded <<= k_pointAdjustment;
        }

        const auto mul = lhsExpanded * rhsExpanded;

        constexpr auto k_mask = (((uint16_t) 1 << LHS_T::k_numberBits) - 1);

        auto scaleBack = [&](const auto input, const auto scale) {
            if(input < 0) {
                const auto result = (int8_t) ((((input / (1 << scale)) - 1) / 2) & k_mask);
                return Data<LHS_T>((int8_t) (result | ~k_mask));
            }
            else {
                const auto result = (int8_t) ((((input / (1 << scale)) + 1) / 2) & k_mask);
                return Data<LHS_T>(result);
            }
        };

        if (RHS_T::k_numberFractionalBits > LHS_T::k_numberFractionalBits) {
            constexpr auto k_backScale = 2 * RHS_T::k_numberFractionalBits - LHS_T::k_numberFractionalBits - 1;
            return scaleBack(mul, k_backScale);
        } else {
            constexpr auto k_backScale = 2 * LHS_T::k_numberFractionalBits - LHS_T::k_numberFractionalBits - 1;
            return scaleBack(mul, k_backScale);
        }

    }

    template<typename T>
    bool operator<(const Data<T> &lhs, const Data<T> &rhs) {
        return lhs.data < rhs.data;
    }

    template<typename T>
    bool operator>(const Data<T> &lhs, const Data<T> &rhs) {
        return lhs.data > rhs.data;
    }

    template<typename T>
    bool operator==(const Data<T> &lhs, const Data<T> &rhs) {
        return lhs.data == rhs.data;
    }
}

using Q8s7 = FixedPoint::Data<FixedPoint::Type<8, 7>>;
using Q8s6 = FixedPoint::Data<FixedPoint::Type<8, 6>>;
using Q8s5 = FixedPoint::Data<FixedPoint::Type<8, 5>>;
using Q8s4 = FixedPoint::Data<FixedPoint::Type<8, 4>>;
using Q8s3 = FixedPoint::Data<FixedPoint::Type<8, 3>>;
using Q8s2 = FixedPoint::Data<FixedPoint::Type<8, 2>>;
using Q8s1 = FixedPoint::Data<FixedPoint::Type<8, 1>>;

using Q7s6 = FixedPoint::Data<FixedPoint::Type<7, 6>>;
using Q7s5 = FixedPoint::Data<FixedPoint::Type<7, 5>>;
using Q7s4 = FixedPoint::Data<FixedPoint::Type<7, 4>>;
using Q7s3 = FixedPoint::Data<FixedPoint::Type<7, 3>>;
using Q7s2 = FixedPoint::Data<FixedPoint::Type<7, 2>>;
using Q7s1 = FixedPoint::Data<FixedPoint::Type<7, 1>>;

using Q6s5 = FixedPoint::Data<FixedPoint::Type<6, 5>>;
using Q6s4 = FixedPoint::Data<FixedPoint::Type<6, 4>>;
using Q6s3 = FixedPoint::Data<FixedPoint::Type<6, 3>>;
using Q6s2 = FixedPoint::Data<FixedPoint::Type<6, 2>>;
using Q6s1 = FixedPoint::Data<FixedPoint::Type<6, 1>>;

using Q5s4 = FixedPoint::Data<FixedPoint::Type<5, 4>>;
using Q5s3 = FixedPoint::Data<FixedPoint::Type<5, 3>>;
using Q5s2 = FixedPoint::Data<FixedPoint::Type<5, 2>>;
using Q5s1 = FixedPoint::Data<FixedPoint::Type<5, 1>>;

using Q4s3 = FixedPoint::Data<FixedPoint::Type<4, 3>>;
using Q4s2 = FixedPoint::Data<FixedPoint::Type<4, 2>>;
using Q4s1 = FixedPoint::Data<FixedPoint::Type<4, 1>>;

using Q3s2 = FixedPoint::Data<FixedPoint::Type<3, 2>>;
using Q3s1 = FixedPoint::Data<FixedPoint::Type<3, 1>>;

using Q2s1 = FixedPoint::Data<FixedPoint::Type<2, 1>>;

#endif //FIXEDPOINT_H
