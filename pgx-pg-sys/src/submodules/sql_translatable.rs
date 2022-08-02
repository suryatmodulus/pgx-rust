use pgx_utils::sql_entity_graph::metadata::{
    ArgumentError, ReturnVariant, ReturnVariantError, SqlTranslatable, SqlVariant,
};

impl SqlTranslatable for crate::FunctionCallInfoBaseData {
    fn argument_sql() -> Result<SqlVariant, ArgumentError> {
        Ok(SqlVariant::Skip)
    }
    fn return_sql() -> Result<ReturnVariant, ReturnVariantError> {
        Ok(ReturnVariant::Plain(SqlVariant::Skip))
    }
}

impl SqlTranslatable for crate::PlannerInfo {
    fn argument_sql() -> Result<SqlVariant, ArgumentError> {
        Ok(SqlVariant::Mapped(String::from("internal")))
    }
    fn return_sql() -> Result<ReturnVariant, ReturnVariantError> {
        Ok(ReturnVariant::Plain(SqlVariant::Mapped(String::from(
            "internal",
        ))))
    }
}

impl SqlTranslatable for crate::IndexAmRoutine {
    fn argument_sql() -> Result<SqlVariant, ArgumentError> {
        Ok(SqlVariant::Mapped(String::from("box")))
    }
    fn return_sql() -> Result<ReturnVariant, ReturnVariantError> {
        Ok(ReturnVariant::Plain(SqlVariant::Mapped(String::from(
            "box",
        ))))
    }
}
impl SqlTranslatable for crate::FdwRoutine {
    fn argument_sql() -> Result<SqlVariant, ArgumentError> {
        Ok(SqlVariant::Mapped(String::from("box")))
    }
    fn return_sql() -> Result<ReturnVariant, ReturnVariantError> {
        Ok(ReturnVariant::Plain(SqlVariant::Mapped(String::from(
            "box",
        ))))
    }
}