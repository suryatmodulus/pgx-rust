/*
Portions Copyright 2019-2021 ZomboDB, LLC.
Portions Copyright 2021-2022 Technology Concepts & Design, Inc. <support@tcdi.com>

All rights reserved.

Use of this source code is governed by the MIT license that can be found in the LICENSE file.
*/

//
// we allow improper_ctypes just to eliminate these warnings:
//      = note: `#[warn(improper_ctypes)]` on by default
//      = note: 128-bit integers don't currently have a known stable ABI
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]
#![allow(non_upper_case_globals)]
#![allow(improper_ctypes)]
#![allow(clippy::unneeded_field_pattern)]
#![cfg_attr(nightly, feature(strict_provenance))]

#[cfg(
    any(
        // no features at all will cause problems
        not(any(feature = "pg11", feature = "pg12", feature = "pg13", feature = "pg14", feature = "pg15")),
  ))]
std::compile_error!("exactly one one feature must be provided (pg11, pg12, pg13, pg14, pg15)");

pub mod submodules;

use core::ptr::NonNull;
use std::ffi::CStr;
use std::os::raw::c_char;

// for convenience we pull up everything submodules exposes
pub use submodules::*;

//
// our actual bindings modules -- these are generated by build.rs
//

// feature gate each pg version module
#[cfg(all(feature = "pg11", not(docsrs)))]
mod pg11 {
    include!(concat!(env!("OUT_DIR"), "/pg11.rs"));
}
#[cfg(all(feature = "pg11", docsrs))]
mod pg11;

#[cfg(all(feature = "pg12", not(docsrs)))]
mod pg12 {
    include!(concat!(env!("OUT_DIR"), "/pg12.rs"));
}
#[cfg(all(feature = "pg12", docsrs))]
mod pg12;

#[cfg(all(feature = "pg13", not(docsrs)))]
mod pg13 {
    include!(concat!(env!("OUT_DIR"), "/pg13.rs"));
}
#[cfg(all(feature = "pg13", docsrs))]
mod pg13;

#[cfg(all(feature = "pg14", not(docsrs)))]
mod pg14 {
    include!(concat!(env!("OUT_DIR"), "/pg14.rs"));
}
#[cfg(all(feature = "pg14", docsrs))]
mod pg14;

#[cfg(all(feature = "pg15", not(docsrs)))]
mod pg15 {
    include!(concat!(env!("OUT_DIR"), "/pg15.rs"));
}
#[cfg(all(feature = "pg15", docsrs))]
mod pg15;

// export each module publicly
#[cfg(feature = "pg11")]
pub use pg11::*;
#[cfg(feature = "pg12")]
pub use pg12::*;
#[cfg(feature = "pg13")]
pub use pg13::*;
#[cfg(feature = "pg14")]
pub use pg14::*;
#[cfg(feature = "pg15")]
pub use pg15::*;

// feature gate each pg-specific oid module
#[cfg(all(feature = "pg11", not(docsrs)))]
mod pg11_oids {
    include!(concat!(env!("OUT_DIR"), "/pg11_oids.rs"));
}
#[cfg(all(feature = "pg11", docsrs))]
mod pg11;

#[cfg(all(feature = "pg12", not(docsrs)))]
mod pg12_oids {
    include!(concat!(env!("OUT_DIR"), "/pg12_oids.rs"));
}
#[cfg(all(feature = "pg12", docsrs))]
mod pg12_oids;

#[cfg(all(feature = "pg13", not(docsrs)))]
mod pg13_oids {
    include!(concat!(env!("OUT_DIR"), "/pg13_oids.rs"));
}
#[cfg(all(feature = "pg13", docsrs))]
mod pg13_oids;

#[cfg(all(feature = "pg14", not(docsrs)))]
mod pg14_oids {
    include!(concat!(env!("OUT_DIR"), "/pg14_oids.rs"));
}
#[cfg(all(feature = "pg14", docsrs))]
mod pg14_oids;

#[cfg(all(feature = "pg15", not(docsrs)))]
mod pg15_oids {
    include!(concat!(env!("OUT_DIR"), "/pg15_oids.rs"));
}
#[cfg(all(feature = "pg15", docsrs))]
mod pg15_oids;

// export that module publicly
#[cfg(feature = "pg11")]
pub use pg11_oids::*;
#[cfg(feature = "pg12")]
pub use pg12_oids::*;
#[cfg(feature = "pg13")]
pub use pg13_oids::*;
#[cfg(feature = "pg14")]
pub use pg14_oids::*;
#[cfg(feature = "pg15")]
pub use pg15_oids::*;

// expose things we want available for all versions
pub use all_versions::*;

// and things that are version-specific
#[cfg(feature = "pg11")]
pub use internal::pg11::IndexBuildHeapScan;
#[cfg(feature = "pg11")]
pub use internal::pg11::*;

#[cfg(feature = "pg12")]
pub use internal::pg12::*;

#[cfg(feature = "pg13")]
pub use internal::pg13::*;

#[cfg(feature = "pg14")]
pub use internal::pg14::*;

#[cfg(feature = "pg15")]
pub use internal::pg15::*;

/// A trait applied to all Postgres `pg_sys::Node` types and subtypes
pub trait PgNode: seal::Sealed {
    /// Format this node
    #[inline]
    fn display_node(&self) -> std::string::String {
        // SAFETY: The trait is pub but this impl is private, and
        // this is only implemented for things known to be "Nodes"
        unsafe { display_node_impl(NonNull::from(self).cast()) }
    }
}

mod seal {
    pub trait Sealed {}
}

/// implementation function for `impl Display for $NodeType`
///
/// # Safety
/// Don't use this on anything that doesn't impl PgNode, or the type may be off
#[warn(unsafe_op_in_unsafe_fn)]
pub(crate) unsafe fn display_node_impl(node: NonNull<crate::Node>) -> std::string::String {
    // SAFETY: It's fine to call nodeToString with non-null well-typed pointers,
    // and pg_sys::nodeToString() returns data via palloc, which is never null
    // as Postgres will ERROR rather than giving us a null pointer,
    // and Postgres starts and finishes constructing StringInfos by writing '\0'
    unsafe {
        let node_cstr = crate::nodeToString(node.as_ptr().cast());

        let result = match CStr::from_ptr(node_cstr).to_str() {
            Ok(cstr) => cstr.to_string(),
            Err(e) => format!("<ffi error: {:?}>", e),
        };

        crate::pfree(node_cstr.cast());

        result
    }
}

/// A trait for converting a thing into a `char *` that is allocated by Postgres' palloc
pub trait AsPgCStr {
    /// Consumes `self` and converts it into a Postgres-allocated `char *`
    fn as_pg_cstr(self) -> *mut std::os::raw::c_char;
}

impl<'a> AsPgCStr for &'a str {
    fn as_pg_cstr(self) -> *mut std::os::raw::c_char {
        let self_bytes = self.as_bytes();
        let pg_cstr = unsafe { crate::palloc0(self_bytes.len() + 1) as *mut std::os::raw::c_uchar };
        let slice = unsafe { std::slice::from_raw_parts_mut(pg_cstr, self_bytes.len()) };
        slice.copy_from_slice(self_bytes);
        pg_cstr as *mut std::os::raw::c_char
    }
}

impl<'a> AsPgCStr for Option<&'a str> {
    fn as_pg_cstr(self) -> *mut c_char {
        match self {
            Some(s) => s.as_pg_cstr(),
            None => std::ptr::null_mut(),
        }
    }
}

impl AsPgCStr for std::string::String {
    fn as_pg_cstr(self) -> *mut std::os::raw::c_char {
        self.as_str().as_pg_cstr()
    }
}

impl AsPgCStr for &std::string::String {
    fn as_pg_cstr(self) -> *mut std::os::raw::c_char {
        self.as_str().as_pg_cstr()
    }
}

impl AsPgCStr for Option<std::string::String> {
    fn as_pg_cstr(self) -> *mut c_char {
        match self {
            Some(s) => s.as_pg_cstr(),
            None => std::ptr::null_mut(),
        }
    }
}

impl AsPgCStr for Option<&std::string::String> {
    fn as_pg_cstr(self) -> *mut c_char {
        match self {
            Some(s) => s.as_pg_cstr(),
            None => std::ptr::null_mut(),
        }
    }
}

impl AsPgCStr for &Option<std::string::String> {
    fn as_pg_cstr(self) -> *mut c_char {
        match self {
            Some(s) => s.as_pg_cstr(),
            None => std::ptr::null_mut(),
        }
    }
}

/// item declarations we want to add to all versions
mod all_versions {
    use crate as pg_sys;
    use pgx_macros::*;

    use memoffset::*;
    use std::str::FromStr;

    /// this comes from `postgres_ext.h`
    pub const InvalidOid: super::Oid = 0;
    pub const InvalidOffsetNumber: super::OffsetNumber = 0;
    pub const FirstOffsetNumber: super::OffsetNumber = 1;
    pub const MaxOffsetNumber: super::OffsetNumber =
        (super::BLCKSZ as usize / std::mem::size_of::<super::ItemIdData>()) as super::OffsetNumber;
    pub const InvalidBlockNumber: u32 = 0xFFFF_FFFF as crate::BlockNumber;
    pub const VARHDRSZ: usize = std::mem::size_of::<super::int32>();
    pub const InvalidTransactionId: super::TransactionId = 0 as super::TransactionId;
    pub const InvalidCommandId: super::CommandId = (!(0 as super::CommandId)) as super::CommandId;
    pub const FirstCommandId: super::CommandId = 0 as super::CommandId;
    pub const BootstrapTransactionId: super::TransactionId = 1 as super::TransactionId;
    pub const FrozenTransactionId: super::TransactionId = 2 as super::TransactionId;
    pub const FirstNormalTransactionId: super::TransactionId = 3 as super::TransactionId;
    pub const MaxTransactionId: super::TransactionId = 0xFFFF_FFFF as super::TransactionId;

    #[pgx_macros::pg_guard]
    extern "C" {
        pub fn pgx_list_nth(list: *mut super::List, nth: i32) -> *mut std::os::raw::c_void;
        pub fn pgx_list_nth_int(list: *mut super::List, nth: i32) -> i32;
        pub fn pgx_list_nth_oid(list: *mut super::List, nth: i32) -> super::Oid;
        pub fn pgx_list_nth_cell(list: *mut super::List, nth: i32) -> *mut super::ListCell;
        pub fn pgx_GETSTRUCT(tuple: pg_sys::HeapTuple) -> *mut std::os::raw::c_char;
    }

    //
    // TODO: [`TYPEALIGN`] and [`MAXALIGN`] are also part of PR #948 and when that's all merged,
    //       their uses should be switched to these
    //

    #[allow(non_snake_case)]
    #[inline(always)]
    pub const unsafe fn TYPEALIGN(alignval: usize, len: usize) -> usize {
        // #define TYPEALIGN(ALIGNVAL,LEN)  \
        // (((uintptr_t) (LEN) + ((ALIGNVAL) - 1)) & ~((uintptr_t) ((ALIGNVAL) - 1)))
        ((len) + ((alignval) - 1)) & !((alignval) - 1)
    }

    #[allow(non_snake_case)]
    #[inline(always)]
    pub const unsafe fn MAXALIGN(len: usize) -> usize {
        // #define MAXALIGN(LEN) TYPEALIGN(MAXIMUM_ALIGNOF, (LEN))
        TYPEALIGN(pg_sys::MAXIMUM_ALIGNOF as _, len)
    }

    ///  Given a currently-allocated chunk of Postgres allocated memory, determine the context
    ///  it belongs to.
    ///
    /// All chunks allocated by any memory context manager are required to be
    /// preceded by the corresponding MemoryContext stored, without padding, in the
    /// preceding sizeof(void*) bytes.  A currently-allocated chunk must contain a
    /// backpointer to its owning context.  The backpointer is used by pfree() and
    /// repalloc() to find the context to call.
    ///
    /// # Safety
    ///
    /// The specified `pointer` **must** be one allocated by Postgres (via [`palloc`] and friends).
    ///
    ///
    /// # Panics
    ///
    /// This function will panic if `pointer` is null, if it's not properly aligned, or if the memory
    /// it points do doesn't have the a header that looks like a memory context pointer
    #[allow(non_snake_case)]
    pub unsafe fn GetMemoryContextChunk(
        pointer: *mut std::os::raw::c_void,
    ) -> pg_sys::MemoryContext {
        /*
         * Try to detect bogus pointers handed to us, poorly though we can.
         * Presumably, a pointer that isn't MAXALIGNED isn't pointing at an
         * allocated chunk.
         */
        assert!(!pointer.is_null());
        assert_eq!(pointer, MAXALIGN(pointer as usize) as *mut ::std::os::raw::c_void);

        /*
         * OK, it's probably safe to look at the context.
         */
        // 	context = *(MemoryContext *) (((char *) pointer) - sizeof(void *));
        let context = unsafe {
            // SAFETY: the caller has assured us that `pointer` points to palloc'd memory, which
            // means it'll have this header before it
            *(pointer
                .cast::<::std::os::raw::c_char>()
                .sub(std::mem::size_of::<*mut ::std::os::raw::c_void>())
                .cast())
        };

        assert!(MemoryContextIsValid(context));

        context
    }

    /// Returns true if memory context is valid, as Postgres determines such a thing.
    ///
    /// # Safety
    ///
    /// Caller must determine that the specified `context` pointer, if it's probably a [`MemoryContextData`]
    /// pointer, really is.  This function is a best effort, not a guarantee.
    ///
    /// # Implementation Note
    ///
    /// If Postgres adds more memory context types in the future, we need to do that here too.
    #[allow(non_snake_case)]
    #[inline(always)]
    pub unsafe fn MemoryContextIsValid(context: *mut crate::MemoryContextData) -> bool {
        // #define MemoryContextIsValid(context) \
        // 	((context) != NULL && \
        // 	 (IsA((context), AllocSetContext) || \
        // 	  IsA((context), SlabContext) || \
        // 	  IsA((context), GenerationContext)))

        !context.is_null()
            && unsafe {
                // SAFETY:  we just determined that context isn't null, so it's safe to `.as_ref()`
                // and `.unwrap_unchecked()`
                let context = context.as_ref().unwrap_unchecked();

                context.type_ == crate::NodeTag_T_AllocSetContext
                    || context.type_ == crate::NodeTag_T_SlabContext
                    || context.type_ == crate::NodeTag_T_GenerationContext
            }
    }

    #[inline]
    pub fn VARHDRSZ_EXTERNAL() -> usize {
        offset_of!(super::varattrib_1b_e, va_data)
    }

    #[inline]
    pub fn VARHDRSZ_SHORT() -> usize {
        offset_of!(super::varattrib_1b, va_data)
    }

    #[inline]
    pub fn get_pg_major_version_string() -> &'static str {
        let mver = std::ffi::CStr::from_bytes_with_nul(super::PG_MAJORVERSION).unwrap();
        mver.to_str().unwrap()
    }

    #[inline]
    pub fn get_pg_major_version_num() -> u16 {
        u16::from_str(super::get_pg_major_version_string()).unwrap()
    }

    #[inline]
    pub fn get_pg_version_string() -> &'static str {
        let ver = std::ffi::CStr::from_bytes_with_nul(super::PG_VERSION_STR).unwrap();
        ver.to_str().unwrap()
    }

    #[inline]
    pub fn get_pg_major_minor_version_string() -> &'static str {
        let mver = std::ffi::CStr::from_bytes_with_nul(super::PG_VERSION).unwrap();
        mver.to_str().unwrap()
    }

    #[inline]
    pub fn TransactionIdIsNormal(xid: super::TransactionId) -> bool {
        xid >= FirstNormalTransactionId
    }

    /// ```c
    ///     #define type_is_array(typid)  (get_element_type(typid) != InvalidOid)
    /// ```
    #[inline]
    pub unsafe fn type_is_array(typoid: super::Oid) -> bool {
        super::get_element_type(typoid) != InvalidOid
    }

    #[inline]
    pub unsafe fn planner_rt_fetch(
        index: super::Index,
        root: *mut super::PlannerInfo,
    ) -> *mut super::RangeTblEntry {
        extern "C" {
            pub fn pgx_planner_rt_fetch(
                index: super::Index,
                root: *mut super::PlannerInfo,
            ) -> *mut super::RangeTblEntry;
        }

        pgx_planner_rt_fetch(index, root)
    }

    /// ```c
    /// #define rt_fetch(rangetable_index, rangetable) \
    ///     ((RangeTblEntry *) list_nth(rangetable, (rangetable_index)-1))
    /// ```
    #[inline]
    pub unsafe fn rt_fetch(
        index: super::Index,
        range_table: *mut super::List,
    ) -> *mut super::RangeTblEntry {
        pgx_list_nth(range_table, index as i32 - 1) as *mut super::RangeTblEntry
    }

    #[inline]
    pub fn HeapTupleHeaderGetXmin(
        htup_header: super::HeapTupleHeader,
    ) -> Option<super::TransactionId> {
        extern "C" {
            pub fn pgx_HeapTupleHeaderGetXmin(
                htup_header: super::HeapTupleHeader,
            ) -> super::TransactionId;
        }

        if htup_header.is_null() {
            None
        } else {
            Some(unsafe { pgx_HeapTupleHeaderGetXmin(htup_header) })
        }
    }

    #[inline]
    pub fn HeapTupleHeaderGetRawCommandId(
        htup_header: super::HeapTupleHeader,
    ) -> Option<super::CommandId> {
        extern "C" {
            pub fn pgx_HeapTupleHeaderGetRawCommandId(
                htup_header: super::HeapTupleHeader,
            ) -> super::CommandId;
        }

        if htup_header.is_null() {
            None
        } else {
            Some(unsafe { pgx_HeapTupleHeaderGetRawCommandId(htup_header) })
        }
    }

    /// #define HeapTupleHeaderIsHeapOnly(tup) \
    ///    ( \
    ///       ((tup)->t_infomask2 & HEAP_ONLY_TUPLE) != 0 \
    ///    )
    #[inline]
    pub unsafe fn HeapTupleHeaderIsHeapOnly(htup_header: super::HeapTupleHeader) -> bool {
        ((*htup_header).t_infomask2 & crate::HEAP_ONLY_TUPLE as u16) != 0
    }

    /// #define HeapTupleHeaderIsHotUpdated(tup) \
    /// ( \
    ///      ((tup)->t_infomask2 & HEAP_HOT_UPDATED) != 0 && \
    ///      ((tup)->t_infomask & HEAP_XMAX_INVALID) == 0 && \
    ///      !HeapTupleHeaderXminInvalid(tup) \
    /// )
    #[inline]
    pub unsafe fn HeapTupleHeaderIsHotUpdated(htup_header: super::HeapTupleHeader) -> bool {
        (*htup_header).t_infomask2 & crate::HEAP_HOT_UPDATED as u16 != 0
            && (*htup_header).t_infomask & crate::HEAP_XMAX_INVALID as u16 == 0
            && !HeapTupleHeaderXminInvalid(htup_header)
    }

    /// #define HeapTupleHeaderXminInvalid(tup) \
    /// ( \
    ///   ((tup)->t_infomask & (HEAP_XMIN_COMMITTED|HEAP_XMIN_INVALID)) == \
    ///      HEAP_XMIN_INVALID \
    /// )
    #[inline]
    pub unsafe fn HeapTupleHeaderXminInvalid(htup_header: super::HeapTupleHeader) -> bool {
        (*htup_header).t_infomask
            & (crate::HEAP_XMIN_COMMITTED as u16 | crate::HEAP_XMIN_INVALID as u16)
            == crate::HEAP_XMIN_INVALID as u16
    }

    /// #define BufferGetPage(buffer) ((Page)BufferGetBlock(buffer))
    #[inline]
    pub unsafe fn BufferGetPage(buffer: crate::Buffer) -> crate::Page {
        BufferGetBlock(buffer) as crate::Page
    }

    /// #define BufferGetBlock(buffer) \
    /// ( \
    ///      AssertMacro(BufferIsValid(buffer)), \
    ///      BufferIsLocal(buffer) ? \
    ///            LocalBufferBlockPointers[-(buffer) - 1] \
    ///      : \
    ///            (Block) (BufferBlocks + ((Size) ((buffer) - 1)) * BLCKSZ) \
    /// )
    #[inline]
    pub unsafe fn BufferGetBlock(buffer: crate::Buffer) -> crate::Block {
        if BufferIsLocal(buffer) {
            *crate::LocalBufferBlockPointers.offset(((-buffer) - 1) as isize)
        } else {
            crate::BufferBlocks
                .offset((((buffer as crate::Size) - 1) * crate::BLCKSZ as usize) as isize)
                as crate::Block
        }
    }

    /// #define BufferIsLocal(buffer)      ((buffer) < 0)
    #[inline]
    pub unsafe fn BufferIsLocal(buffer: crate::Buffer) -> bool {
        buffer < 0
    }

    #[inline]
    pub fn heap_tuple_get_struct<T>(htup: super::HeapTuple) -> *mut T {
        if htup.is_null() {
            0 as *mut T
        } else {
            unsafe { pgx_GETSTRUCT(htup) as *mut T }
        }
    }

    #[pg_guard]
    extern "C" {
        pub fn query_tree_walker(
            query: *mut super::Query,
            walker: ::std::option::Option<
                unsafe extern "C" fn(*mut super::Node, *mut ::std::os::raw::c_void) -> bool,
            >,
            context: *mut ::std::os::raw::c_void,
            flags: ::std::os::raw::c_int,
        ) -> bool;
    }

    #[pg_guard]
    extern "C" {
        pub fn expression_tree_walker(
            node: *mut super::Node,
            walker: ::std::option::Option<
                unsafe extern "C" fn(*mut super::Node, *mut ::std::os::raw::c_void) -> bool,
            >,
            context: *mut ::std::os::raw::c_void,
        ) -> bool;
    }

    #[pgx_macros::pg_guard]
    extern "C" {
        #[link_name = "pgx_SpinLockInit"]
        pub fn SpinLockInit(lock: *mut pg_sys::slock_t);
        #[link_name = "pgx_SpinLockAcquire"]
        pub fn SpinLockAcquire(lock: *mut pg_sys::slock_t);
        #[link_name = "pgx_SpinLockRelease"]
        pub fn SpinLockRelease(lock: *mut pg_sys::slock_t);
        #[link_name = "pgx_SpinLockFree"]
        pub fn SpinLockFree(lock: *mut pg_sys::slock_t) -> bool;
    }

    #[inline(always)]
    pub unsafe fn MemoryContextSwitchTo(context: crate::MemoryContext) -> crate::MemoryContext {
        let old = crate::CurrentMemoryContext;

        crate::CurrentMemoryContext = context;
        old
    }
}

mod internal {
    //
    // for specific versions
    //
    #[cfg(feature = "pg11")]
    pub(crate) mod pg11 {
        pub use crate::pg11::tupleDesc as TupleDescData;
        pub type QueryCompletion = std::os::raw::c_char;

        /// # Safety
        ///
        /// This function wraps Postgres' internal `IndexBuildHeapScan` method, and therefore, is
        /// inherently unsafe
        pub unsafe fn IndexBuildHeapScan<T>(
            heap_relation: crate::Relation,
            index_relation: crate::Relation,
            index_info: *mut crate::pg11::IndexInfo,
            build_callback: crate::IndexBuildCallback,
            build_callback_state: *mut T,
        ) {
            crate::pg11::IndexBuildHeapScan(
                heap_relation,
                index_relation,
                index_info,
                true,
                build_callback,
                build_callback_state as *mut std::os::raw::c_void,
                std::ptr::null_mut(),
            );
        }
    }

    #[cfg(feature = "pg12")]
    pub(crate) mod pg12 {
        pub use crate::pg12::AllocSetContextCreateInternal as AllocSetContextCreateExtended;
        pub type QueryCompletion = std::os::raw::c_char;

        pub const QTW_EXAMINE_RTES: u32 = crate::pg12::QTW_EXAMINE_RTES_BEFORE;

        /// # Safety
        ///
        /// This function wraps Postgres' internal `IndexBuildHeapScan` method, and therefore, is
        /// inherently unsafe
        pub unsafe fn IndexBuildHeapScan<T>(
            heap_relation: crate::Relation,
            index_relation: crate::Relation,
            index_info: *mut crate::pg12::IndexInfo,
            build_callback: crate::IndexBuildCallback,
            build_callback_state: *mut T,
        ) {
            let heap_relation_ref = heap_relation.as_ref().unwrap();
            let table_am = heap_relation_ref.rd_tableam.as_ref().unwrap();

            table_am.index_build_range_scan.unwrap()(
                heap_relation,
                index_relation,
                index_info,
                true,
                false,
                true,
                0,
                crate::InvalidBlockNumber,
                build_callback,
                build_callback_state as *mut std::os::raw::c_void,
                std::ptr::null_mut(),
            );
        }
    }

    #[cfg(feature = "pg13")]
    pub(crate) mod pg13 {
        pub use crate::pg13::AllocSetContextCreateInternal as AllocSetContextCreateExtended;

        pub const QTW_EXAMINE_RTES: u32 = crate::pg13::QTW_EXAMINE_RTES_BEFORE;

        /// # Safety
        ///
        /// This function wraps Postgres' internal `IndexBuildHeapScan` method, and therefore, is
        /// inherently unsafe
        pub unsafe fn IndexBuildHeapScan<T>(
            heap_relation: crate::Relation,
            index_relation: crate::Relation,
            index_info: *mut crate::IndexInfo,
            build_callback: crate::IndexBuildCallback,
            build_callback_state: *mut T,
        ) {
            let heap_relation_ref = heap_relation.as_ref().unwrap();
            let table_am = heap_relation_ref.rd_tableam.as_ref().unwrap();

            table_am.index_build_range_scan.unwrap()(
                heap_relation,
                index_relation,
                index_info,
                true,
                false,
                true,
                0,
                crate::InvalidBlockNumber,
                build_callback,
                build_callback_state as *mut std::os::raw::c_void,
                std::ptr::null_mut(),
            );
        }
    }

    #[cfg(feature = "pg14")]
    pub(crate) mod pg14 {
        pub use crate::pg14::AllocSetContextCreateInternal as AllocSetContextCreateExtended;

        pub const QTW_EXAMINE_RTES: u32 = crate::pg14::QTW_EXAMINE_RTES_BEFORE;

        /// # Safety
        ///
        /// This function wraps Postgres' internal `IndexBuildHeapScan` method, and therefore, is
        /// inherently unsafe
        pub unsafe fn IndexBuildHeapScan<T>(
            heap_relation: crate::Relation,
            index_relation: crate::Relation,
            index_info: *mut crate::IndexInfo,
            build_callback: crate::IndexBuildCallback,
            build_callback_state: *mut T,
        ) {
            let heap_relation_ref = heap_relation.as_ref().unwrap();
            let table_am = heap_relation_ref.rd_tableam.as_ref().unwrap();

            table_am.index_build_range_scan.unwrap()(
                heap_relation,
                index_relation,
                index_info,
                true,
                false,
                true,
                0,
                crate::InvalidBlockNumber,
                build_callback,
                build_callback_state as *mut std::os::raw::c_void,
                std::ptr::null_mut(),
            );
        }
    }

    #[cfg(feature = "pg15")]
    pub(crate) mod pg15 {
        pub use crate::pg15::AllocSetContextCreateInternal as AllocSetContextCreateExtended;

        pub const QTW_EXAMINE_RTES: u32 = crate::pg15::QTW_EXAMINE_RTES_BEFORE;

        /// # Safety
        ///
        /// This function wraps Postgres' internal `IndexBuildHeapScan` method, and therefore, is
        /// inherently unsafe
        pub unsafe fn IndexBuildHeapScan<T>(
            heap_relation: crate::Relation,
            index_relation: crate::Relation,
            index_info: *mut crate::IndexInfo,
            build_callback: crate::IndexBuildCallback,
            build_callback_state: *mut T,
        ) {
            let heap_relation_ref = heap_relation.as_ref().unwrap();
            let table_am = heap_relation_ref.rd_tableam.as_ref().unwrap();

            table_am.index_build_range_scan.unwrap()(
                heap_relation,
                index_relation,
                index_info,
                true,
                false,
                true,
                0,
                crate::InvalidBlockNumber,
                build_callback,
                build_callback_state as *mut std::os::raw::c_void,
                std::ptr::null_mut(),
            );
        }
    }
}

// Hack to fix linker errors that we get under amazonlinux2 on some PG versions
// due to our wrappers for various system library functions. Should be fairly
// harmless, but ideally we would not wrap these functions
// (https://github.com/tcdi/pgx/issues/730).
#[cfg(target_os = "linux")]
#[link(name = "resolv")]
extern "C" {}
