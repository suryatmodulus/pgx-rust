use pgx_utils::sql_entity_graph::metadata::{
    ArgumentError, ReturnVariant, ReturnVariantError, SqlTranslatable, SqlVariant,
};

impl SqlTranslatable for crate::FunctionCallInfo {
    fn argument_sql() -> Result<SqlVariant, ArgumentError> {
        Ok(SqlVariant::Mapped(String::from("uuid")))
    }
    fn return_sql() -> Result<ReturnVariant, ReturnVariantError> {
        Ok(ReturnVariant::Plain(SqlVariant::Mapped(String::from(
            "uuid",
        ))))
    }
}