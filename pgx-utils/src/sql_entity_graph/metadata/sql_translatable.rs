use core::any::TypeId;
use std::error::Error;

use super::{return_variant::ReturnVariantError, ReturnVariant};

#[derive(Clone, Copy, Debug, Hash, Ord, PartialOrd, PartialEq, Eq)]
pub enum ArgumentError {
    SetOf,
    Table,
    BareU8,
    SkipInArray,
}

impl std::fmt::Display for ArgumentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ArgumentError::SetOf => {
                write!(f, "Cannot use SetOfIterator as an argument")
            }
            ArgumentError::Table => {
                write!(f, "Canot use TableIterator as an argument")
            }
            ArgumentError::BareU8 => {
                write!(f, "Canot use bare u8")
            }
            ArgumentError::SkipInArray => {
                write!(f, "A SqlVariant::Skip inside Array is not valid")
            }
        }
    }
}

#[derive(Clone, Debug, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub enum SqlVariant {
    Mapped(String),
    Composite { requires_array_brackets: bool },
    Skip,
}

impl Error for ArgumentError {}

pub trait SqlTranslatable: 'static {
    fn type_id() -> TypeId {
        TypeId::of::<Self>()
    }
    fn type_name() -> &'static str {
        core::any::type_name::<Self>()
    }
    fn argument_sql() -> Result<SqlVariant, ArgumentError>;
    fn return_sql() -> Result<ReturnVariant, ReturnVariantError>;
    fn variadic() -> bool {
        false
    }
    fn optional() -> bool {
        false
    }
}

impl<T> SqlTranslatable for Option<T>
where
    T: SqlTranslatable,
{
    fn argument_sql() -> Result<SqlVariant, ArgumentError> {
        T::argument_sql()
    }
    fn return_sql() -> Result<ReturnVariant, ReturnVariantError> {
        T::return_sql()
    }
    fn optional() -> bool {
        true
    }
}

impl<T, E> SqlTranslatable for Result<T, E>
where
    T: SqlTranslatable,
    E: std::error::Error + 'static,
{
    fn argument_sql() -> Result<SqlVariant, ArgumentError> {
        T::argument_sql()
    }
    fn return_sql() -> Result<ReturnVariant, ReturnVariantError> {
        T::return_sql()
    }
}

impl<T> SqlTranslatable for Vec<T>
where
    T: SqlTranslatable,
{
    fn argument_sql() -> Result<SqlVariant, ArgumentError> {
        match T::type_id() {
            id if id == TypeId::of::<u8>() => Ok(SqlVariant::Mapped(format!("bytea"))),
            _ => match T::argument_sql() {
                Ok(SqlVariant::Mapped(val)) => Ok(SqlVariant::Mapped(format!("{val}[]"))),
                Ok(SqlVariant::Composite {
                    requires_array_brackets: _,
                }) => Ok(SqlVariant::Composite {
                    requires_array_brackets: true,
                }),
                Ok(SqlVariant::Skip) => Ok(SqlVariant::Skip),
                err @ Err(_) => err,
            },
        }
    }
    fn return_sql() -> Result<ReturnVariant, ReturnVariantError> {
        match T::type_id() {
            id if id == TypeId::of::<u8>() => {
                Ok(ReturnVariant::Plain(SqlVariant::Mapped(format!("bytea"))))
            }
            _ => match T::return_sql() {
                Ok(ReturnVariant::Plain(SqlVariant::Mapped(val))) => {
                    Ok(ReturnVariant::Plain(SqlVariant::Mapped(format!("{val}[]"))))
                }
                Ok(ReturnVariant::Plain(SqlVariant::Composite {
                    requires_array_brackets: _,
                })) => Ok(ReturnVariant::Plain(SqlVariant::Composite {
                    requires_array_brackets: true,
                })),
                Ok(ReturnVariant::Plain(SqlVariant::Skip)) => {
                    Ok(ReturnVariant::Plain(SqlVariant::Skip))
                }
                Ok(ReturnVariant::SetOf(_)) => Err(ReturnVariantError::SetOfInArray),
                Ok(ReturnVariant::Table(_)) => Err(ReturnVariantError::TableInArray),
                err @ Err(_) => err,
            },
        }
    }
}

impl SqlTranslatable for u8 {
    fn argument_sql() -> Result<SqlVariant, ArgumentError> {
        Err(ArgumentError::BareU8)
    }
    fn return_sql() -> Result<ReturnVariant, ReturnVariantError> {
        Err(ReturnVariantError::BareU8)
    }
}

impl SqlTranslatable for i32 {
    fn argument_sql() -> Result<SqlVariant, ArgumentError> {
        Ok(SqlVariant::Mapped(String::from("INT")))
    }
    fn return_sql() -> Result<ReturnVariant, ReturnVariantError> {
        Ok(ReturnVariant::Plain(SqlVariant::Mapped(String::from(
            "INT",
        ))))
    }
}

impl SqlTranslatable for String {
    fn argument_sql() -> Result<SqlVariant, ArgumentError> {
        Ok(SqlVariant::Mapped(String::from("TEXT")))
    }
    fn return_sql() -> Result<ReturnVariant, ReturnVariantError> {
        Ok(ReturnVariant::Plain(SqlVariant::Mapped(String::from(
            "TEXT",
        ))))
    }
}

impl SqlTranslatable for &'static str {
    fn argument_sql() -> Result<SqlVariant, ArgumentError> {
        Ok(SqlVariant::Mapped(String::from("TEXT")))
    }
    fn return_sql() -> Result<ReturnVariant, ReturnVariantError> {
        Ok(ReturnVariant::Plain(SqlVariant::Mapped(String::from(
            "TEXT",
        ))))
    }
}

impl SqlTranslatable for &'static [u8] {
    fn argument_sql() -> Result<SqlVariant, ArgumentError> {
        Ok(SqlVariant::Mapped(String::from("bytea")))
    }
    fn return_sql() -> Result<ReturnVariant, ReturnVariantError> {
        Ok(ReturnVariant::Plain(SqlVariant::Mapped(String::from(
            "bytea",
        ))))
    }
}

impl SqlTranslatable for i8 {
    fn argument_sql() -> Result<SqlVariant, ArgumentError> {
        Ok(SqlVariant::Mapped(String::from("char")))
    }
    fn return_sql() -> Result<ReturnVariant, ReturnVariantError> {
        Ok(ReturnVariant::Plain(SqlVariant::Mapped(String::from(
            "char",
        ))))
    }
}

impl SqlTranslatable for i16 {
    fn argument_sql() -> Result<SqlVariant, ArgumentError> {
        Ok(SqlVariant::Mapped(String::from("smallint")))
    }
    fn return_sql() -> Result<ReturnVariant, ReturnVariantError> {
        Ok(ReturnVariant::Plain(SqlVariant::Mapped(String::from(
            "smallint",
        ))))
    }
}

impl SqlTranslatable for i64 {
    fn argument_sql() -> Result<SqlVariant, ArgumentError> {
        Ok(SqlVariant::Mapped(String::from("bigint")))
    }
    fn return_sql() -> Result<ReturnVariant, ReturnVariantError> {
        Ok(ReturnVariant::Plain(SqlVariant::Mapped(String::from(
            "bigint",
        ))))
    }
}

impl SqlTranslatable for bool {
    fn argument_sql() -> Result<SqlVariant, ArgumentError> {
        Ok(SqlVariant::Mapped(String::from("bool")))
    }
    fn return_sql() -> Result<ReturnVariant, ReturnVariantError> {
        Ok(ReturnVariant::Plain(SqlVariant::Mapped(String::from(
            "bool",
        ))))
    }
}

impl SqlTranslatable for char {
    fn argument_sql() -> Result<SqlVariant, ArgumentError> {
        Ok(SqlVariant::Mapped(String::from("bool")))
    }
    fn return_sql() -> Result<ReturnVariant, ReturnVariantError> {
        Ok(ReturnVariant::Plain(SqlVariant::Mapped(String::from(
            "bool",
        ))))
    }
}

impl SqlTranslatable for f32 {
    fn argument_sql() -> Result<SqlVariant, ArgumentError> {
        Ok(SqlVariant::Mapped(String::from("real")))
    }
    fn return_sql() -> Result<ReturnVariant, ReturnVariantError> {
        Ok(ReturnVariant::Plain(SqlVariant::Mapped(String::from(
            "real",
        ))))
    }
}

impl SqlTranslatable for f64 {
    fn argument_sql() -> Result<SqlVariant, ArgumentError> {
        Ok(SqlVariant::Mapped(String::from("double precision")))
    }
    fn return_sql() -> Result<ReturnVariant, ReturnVariantError> {
        Ok(ReturnVariant::Plain(SqlVariant::Mapped(String::from(
            "double precision",
        ))))
    }
}

impl SqlTranslatable for std::ffi::CStr {
    fn argument_sql() -> Result<SqlVariant, ArgumentError> {
        Ok(SqlVariant::Mapped(String::from("cstring")))
    }
    fn return_sql() -> Result<ReturnVariant, ReturnVariantError> {
        Ok(ReturnVariant::Plain(SqlVariant::Mapped(String::from(
            "cstring",
        ))))
    }
}

impl SqlTranslatable for &'static cstr_core::CStr {
    fn argument_sql() -> Result<SqlVariant, ArgumentError> {
        Ok(SqlVariant::Mapped(String::from("cstring")))
    }
    fn return_sql() -> Result<ReturnVariant, ReturnVariantError> {
        Ok(ReturnVariant::Plain(SqlVariant::Mapped(String::from(
            "cstring",
        ))))
    }
}

impl SqlTranslatable for cstr_core::CStr {
    fn argument_sql() -> Result<SqlVariant, ArgumentError> {
        Ok(SqlVariant::Mapped(String::from("cstring")))
    }
    fn return_sql() -> Result<ReturnVariant, ReturnVariantError> {
        Ok(ReturnVariant::Plain(SqlVariant::Mapped(String::from(
            "cstring",
        ))))
    }
}