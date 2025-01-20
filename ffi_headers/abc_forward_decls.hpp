#pragma once
// Forward declarations

namespace abc_helper
{
    struct FfiTerm;
    struct FfiAbcType;
    struct MaybeTerm;
    struct FfiTerm;
    struct FfiAbcType;
    struct Context;
    enum class ErrorCode;
    struct MaybeContext;
    struct FfiSummary;
    enum class BinaryOp;
    enum class CmpOp;
    struct Literal;
    struct MaybeAbcType;
    struct AbcScalar;

    /// Represents a scalar type. These are builtin types that have assumed
    /// bounds on them. E.g., an i32 is assumed to have bounds -2^31 to 2^31 - 1.
    ///
    /// Note that for Sint, Uint, and Float, the width is in **bytes**.
    struct AbcScalar
    {
        enum class Tag
        {
            /// Signed integer type. The width is in bytes.
            Sint,
            /// Unsigned integer type. The width is in bytes.
            Uint,
            /// IEEE-754 Floating point type.
            Float,
            /// Boolean type.
            Bool,
            /// Abstract integer type. That is, an integer type with unknown bounds.
            AbstractInt,
            /// Abstract floating point type. That is, a floating point type with unknown bounds.
            AbstractFloat,
        };

        struct Sint_Body
        {
            uint8_t _0;
        };

        struct Uint_Body
        {
            uint8_t _0;
        };

        struct Float_Body
        {
            uint8_t _0;
        };

        Tag tag;
        union
        {
            Sint_Body sint;
            Uint_Body uint;
            Float_Body float_;
        };

        static AbcScalar Sint(const uint8_t &_0)
        {
            AbcScalar result;
            ::new (&result.sint._0)(uint8_t)(_0);
            result.tag = Tag::Sint;
            return result;
        }

        bool IsSint() const
        {
            return tag == Tag::Sint;
        }

        const uint8_t &AsSint() const
        {
            assert(IsSint());
            return sint._0;
        }

        static AbcScalar Uint(const uint8_t &_0)
        {
            AbcScalar result;
            ::new (&result.uint._0)(uint8_t)(_0);
            result.tag = Tag::Uint;
            return result;
        }

        bool IsUint() const
        {
            return tag == Tag::Uint;
        }

        const uint8_t &AsUint() const
        {
            assert(IsUint());
            return uint._0;
        }

        static AbcScalar Float(const uint8_t &_0)
        {
            AbcScalar result;
            ::new (&result.float_._0)(uint8_t)(_0);
            result.tag = Tag::Float;
            return result;
        }

        bool IsFloat() const
        {
            return tag == Tag::Float;
        }

        const uint8_t &AsFloat() const
        {
            assert(IsFloat());
            return float_._0;
        }

        static AbcScalar Bool()
        {
            AbcScalar result;
            result.tag = Tag::Bool;
            return result;
        }

        bool IsBool() const
        {
            return tag == Tag::Bool;
        }

        static AbcScalar AbstractInt()
        {
            AbcScalar result;
            result.tag = Tag::AbstractInt;
            return result;
        }

        bool IsAbstractInt() const
        {
            return tag == Tag::AbstractInt;
        }

        static AbcScalar AbstractFloat()
        {
            AbcScalar result;
            result.tag = Tag::AbstractFloat;
            return result;
        }

        bool IsAbstractFloat() const
        {
            return tag == Tag::AbstractFloat;
        }
    };
}