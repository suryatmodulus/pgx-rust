use crate::errcodes::PgSqlErrorCode;
use crate::panic::CaughtError;
use std::collections::HashMap;
use std::panic::{catch_unwind, UnwindSafe};

pub struct PgTryBuilder<'a, R, F: FnOnce() -> R + UnwindSafe> {
    func: F,
    when: HashMap<PgSqlErrorCode, Box<dyn FnMut(CaughtError) -> R + 'a>>,
    catch_others: Option<Box<dyn FnMut(CaughtError) -> R + 'a>>,
    finally: Option<Box<dyn FnMut() + 'a>>,
}

impl<'a, R, F: FnOnce() -> R + UnwindSafe> PgTryBuilder<'a, R, F> {
    #[must_use = "must call `PgTryBuilder::execute(self)` in order for it to run"]
    pub fn new(func: F) -> Self {
        Self { func, when: Default::default(), catch_others: None, finally: None }
    }

    #[must_use = "must call `PgTryBuilder::execute(self)` in order for it to run"]
    pub fn catch_when(
        mut self,
        error: PgSqlErrorCode,
        f: impl FnMut(CaughtError) -> R + 'a,
    ) -> Self {
        self.when.insert(error, Box::new(f));
        self
    }

    #[must_use = "must call `PgTryBuilder::execute(self)` in order for it to run"]
    pub fn catch_others(mut self, f: impl FnMut(CaughtError) -> R + 'a) -> Self {
        self.catch_others = Some(Box::new(f));
        self
    }

    #[must_use = "must call `PgTryBuilder::execute(self)` in order for it to run"]
    pub fn finally(mut self, f: impl FnMut() + 'a) -> Self {
        self.finally = Some(Box::new(f));
        self
    }

    #[track_caller]
    pub fn execute(mut self) -> R {
        let result = catch_unwind(self.func);

        let finally = || {
            if let Some(mut finally) = self.finally {
                // finally block runs
                finally();
            }
        };

        let result = match result {
            Ok(result) => result,
            Err(error) => {
                let (sqlerrcode, panic) = match crate::panic::downcast_err(error) {
                    panic @ CaughtError::RustPanic { .. } => (PgSqlErrorCode::RustPanic, panic),
                    CaughtError::ErrorReport(ereport) => {
                        let sqlerrcode = ereport.ereport.errcode;
                        let panic = CaughtError::ErrorReport(ereport);
                        (sqlerrcode, panic)
                    }
                    CaughtError::PostgresError(errdata) => {
                        let sqlerrcode = errdata.sqlerrcode;
                        let panic = CaughtError::PostgresError(errdata);
                        (sqlerrcode, panic)
                    }
                };

                // Postgres source docs says that a PG_TRY/PG_CATCH/PG_FINALLY block can't have
                // both a CATCH and a FINALLY.
                //
                // We'll allow it, but if a catch handler (re)throws then the finally block will
                // not run.  Normal rust drop semantics during stack unwinding will, however.
                // So in practice it seems reasonable that users likely won't have finally blocks
                // anyways
                if let Some(handler) = self.when.get_mut(&sqlerrcode) {
                    // we have a registered catch handler for the error code we caught
                    handler(panic)
                } else if let Some(mut handler) = self.catch_others {
                    // we have a registered "catch others" handler
                    handler(panic)
                } else {
                    // we have no catch handler so we need call our finally handler and rethrow
                    finally();
                    panic.rethrow_ignore_location()
                }
            }
        };

        // make sure to run the finally block and return whatever the catch handler may have returned
        finally();
        result
    }
}