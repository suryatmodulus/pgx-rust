//! Enforces thread-safety in `pgx`.
//!
//! It's UB to call into Postgres functions from multiple threads. We handle
//! this by remembering the first thread to call into postgres functions, and
//! panicking if we're called from a different thread.
//!
//! This is called from the current crate from inside the setjmp shim, as that
//! code is *definitely* unsound to call in the presense of multiple threads.
//!
//! This is somewhat heavyhanded, and should be called from fewer places in the
//! future...

use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicUsize, Ordering};

static ACTIVE_THREAD: AtomicUsize = AtomicUsize::new(0);
#[track_caller]
pub(crate) fn check_active_thread() {
    let current_thread = thread_unique_addr();
    // Relaxed is sufficient as we're only interested in the effects on a single
    // atomic variable, and don't need synchronization beyond that.
    return match ACTIVE_THREAD.load(Ordering::Relaxed) {
        0 => init_active_thread(current_thread),
        thread_id => {
            if current_thread.get() != thread_id {
                thread_id_check_failed();
            }
        }
    };
}

#[track_caller]
fn init_active_thread(tid: NonZeroUsize) {
    match ACTIVE_THREAD.compare_exchange(0, tid.get(), Ordering::Relaxed, Ordering::Relaxed) {
        Ok(_) => unsafe {
            // We won the race. Register an atfork handler to clear the atomic
            // in any child processes we spawn.
            extern "C" fn clear_in_child() {
                ACTIVE_THREAD.store(0, Ordering::Relaxed);
            }
            libc::pthread_atfork(None, None, Some(clear_in_child));
        },
        Err(_) => {
            thread_id_check_failed();
        }
    }
}

#[cold]
#[inline(never)]
#[track_caller]
fn thread_id_check_failed() -> ! {
    panic!("`pgx` may not not be used from multiple threads.");
}

/// Get a unique pointer for a thread.
///
/// Conceptually, you can imagine that this returns a pointer to a thread local.
/// Unfortunately, under the TLS model relevant for pgx code (the global-dynamic
/// TLS model, which must be used by code that in modules that expect to be
/// dynamically loaded), actually accessing thread locals requires some hoops
/// that the C runtime must jump through -- calling `_tlv_bootstrap` (on Mach-O
/// targets), or `__tls_get_addr` (on ELF targets).
///
/// This is pretty fast, but we can make the check essentially free by using the
/// [similar] [tricks] to those used by thread-pooling allocators to detect
/// cross-thread frees.
///
/// Specifically, we don't need the pointer to any *particular* thread-local,
/// just a pointer that is guaranteed to be unique to the thread. This means we
/// can return any thread-specific pointer accessable through userspace such as
/// thread control/environment blocks, the TLS base pointer, etc. On many
/// targets, these can be accessed via inline assembly.
///
/// The details of doing this is... Very platform specific, to say the least. We
/// implement this for four targets
/// (`{x86_64,aarch64}`-`{apple-darwin,unknown-linux-gnu}`), rather than try to
/// implement it everywhere. It would be easy to add a few more (32-bit x86,
/// windows), but at the moment these are either unimportant to us, or
/// unsupported.
///
/// We explicitly exclude non-glibc toolchains on linux (although musl seems to
/// use the same conventions), and avoid builds under the x32 ABI (32-bit
/// pointers under x86_64). This mostly to reduce the cost of maintaining some
/// especially target-specific inline assembly. The approaches we take are as
/// follows:
///
/// - On x86_64 linux, we use `fs:0`, as `fs` is the thread register (which we
///   can't directly read from userspace), and (at least under glibc) the TCB is
///   in the first slot.
///
/// - On x86_64 macOS, we use `gs:0` -- same as above, but it uses `gs` for the
///   thread register.
///
/// - On aarch64 linux we read from the `tpidr_el0` system register, which holds
///   the thread base (no need to load the first address).
///
/// - On aarch64 macOS, we read from the `tpidrro_el0` system register (e.g. the
///   readonly version of the one aarch64 linux uses).
///
/// - As a fallback, we use the address of a thread-local.
///
/// If we care to support other linuxes, we can check the
/// [`llvm.thread.pointer`], but it should be both tested and manually verified
/// that it matches what e.g. glibc stores at that location -- at least
/// currently (in 2022) it seems to usually silently return the wrong result.
///
/// [similar]:
///     https://github.com/microsoft/mimalloc/blob/f2712f4a8f038a7fb4df2790f4c3b7e3ed9e219b/include/mimalloc-internal.h#L871
/// [tricks]:
///     https://github.com/mjansson/rpmalloc/blob/f56e2f6794eab5c280b089c90750c681679fde92/rpmalloc/rpmalloc.c#L774
/// [`llvm.thread.pointer`]:
///     https://llvm.org/docs/LangRef.html#llvm-thread-pointer-intrinsic
#[inline]
fn thread_unique_addr() -> NonZeroUsize {
    let thread_ptr: *const core::ffi::c_void;
    cfg_if::cfg_if! {
        if #[cfg(all(
            target_os = "linux",
            target_arch = "x86_64",
            // Avoid non-glibc, which may use different convention or not
            // consider this part of the ABI.
            target_env = "gnu",
            // Avoid x32 ABI (32 bit pointers on 64 bit arch), which requires
            // additional shenanigans.
            target_pointer_width = "64",
        ))] {
            // x86_64 linux uses fs for the thread segment, and (under glibc)
            // the first word is a pointer to the thread control block.
            unsafe {
                std::arch::asm!(
                    "mov {}, fs:0",
                    out(reg) thread_ptr,
                    options(nostack, readonly, preserves_flags),
                );
            }
        } else if #[cfg(all(target_os = "macos", target_arch = "x86_64"))] {
            // x86_64 macOS uses gs for the thread segment, and the first word
            // is a pointer to the thread control block.
            unsafe {
                std::arch::asm!(
                    "mov {}, gs:0",
                    out(reg) thread_ptr,
                    options(nostack, readonly, preserves_flags),
                );
            }
        } else if #[cfg(all(
            target_os = "linux",
            target_arch = "aarch64",
            // `tpidr_el0` is writable...
            target_env = "gnu",
            // I don't think there's a x32 equivalent for aarch64-linux, so this
            // is probably always true.
            target_pointer_width = "64",
        ))] {
            // aarch64 linux stores the TLS base pointer in tpidr_el0.
            unsafe {
                std::arch::asm!(
                    "mrs {}, tpidr_el0",
                    out(reg) thread_ptr,
                    options(nostack, readonly, preserves_flags),
                );
            }
        } else if #[cfg(all(target_os = "macos", target_arch = "aarch64"))] {
            // aarch64 macOS stores the TLS base pointer in tpidrro_el0
            unsafe {
                std::arch::asm!(
                    "mrs {}, tpidrro_el0",
                    out(reg) thread_ptr,
                    options(nostack, readonly, preserves_flags),
                );
            }
        } else {
            // For a fallback we use the address of some thread local.
            std::thread_local!(static BYTE: u8 = const { 0 });
            thread_ptr = BYTE.with(|p: &u8| {
                // Guaranteed not to be null.
                p as *const u8 as *const core::ffi::c_void
            });
        }
    }
    // Quickly smoke check that we didn't get something weird. If this gets hit
    // we should probably just force a fallback on that platform.
    debug_assert!(
        !thread_ptr.is_null(),
        "Thread pointer was null. Please file an issue against `pgx` \
        which includes the output of `rustc -Vv`, `rustc --print cfg`, \
        and `uname -a`",
    );
    // Avoid triggering the `unstable_name_collisions` lint.
    let thread_addr = sptr::Strict::addr(thread_ptr);
    // Valid addresses are never NULL, so this should always be fine.
    unsafe { NonZeroUsize::new_unchecked(thread_addr) }
}
