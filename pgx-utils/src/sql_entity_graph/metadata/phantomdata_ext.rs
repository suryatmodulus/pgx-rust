use core::{any::TypeId, marker::PhantomData};

use super::{
    return_variant::ReturnVariantError, ArgumentError, FunctionMetadataTypeEntity, ReturnVariant,
    SqlTranslatable, SqlVariant,
};

pub trait PhantomDataExt {
    fn type_id(&self) -> TypeId;
    fn type_name(&self) -> &'static str;
    fn argument_sql(&self) -> Result<SqlVariant, ArgumentError>;
    fn return_sql(&self) -> Result<ReturnVariant, ReturnVariantError>;
    fn variadic(&self) -> bool;
    fn optional(&self) -> bool;
    fn entity(&self) -> FunctionMetadataTypeEntity;
}

impl<T> PhantomDataExt for PhantomData<T>
where
    T: SqlTranslatable + 'static,
{
    fn type_id(&self) -> TypeId {
        T::type_id()
    }
    fn type_name(&self) -> &'static str {
        T::type_name()
    }
    fn argument_sql(&self) -> Result<SqlVariant, ArgumentError> {
        T::argument_sql()
    }
    fn return_sql(&self) -> Result<ReturnVariant, ReturnVariantError> {
        T::return_sql()
    }
    fn variadic(&self) -> bool {
        T::variadic()
    }
    fn optional(&self) -> bool {
        T::optional()
    }
    fn entity(&self) -> FunctionMetadataTypeEntity {
        FunctionMetadataTypeEntity {
            type_id: self.type_id(),
            type_name: self.type_name(),
            argument_sql: self.argument_sql(),
            return_sql: self.return_sql(),
            variadic: self.variadic(),
            optional: self.optional(),
        }
    }
}