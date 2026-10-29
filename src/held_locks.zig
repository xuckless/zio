// SPDX-FileCopyrightText: 2026 xuckless
// SPDX-License-Identifier: MIT

//! A count of the locks the calling thread holds, for `Options.debug_held_locks`:
//! a task that suspends while it holds a lock panics at the suspension point.
//!
//! The application counts its own locks (`acquired` after taking one,
//! `released` before letting it go); zio checks the count where a task
//! suspends. Since a task never suspends with the count above zero, the
//! thread's count is always the running task's, and needs no per-task slot.
//! With the option off, every function here compiles to nothing.

const std = @import("std");
const zio_options = @import("options.zig").options;

pub const enabled = zio_options.debug_held_locks;

threadlocal var count: u32 = 0;

/// The calling thread took a lock.
pub inline fn acquired() void {
    if (enabled) count += 1;
}

/// The calling thread is letting a lock go.
pub inline fn released() void {
    if (enabled) {
        std.debug.assert(count > 0);
        count -= 1;
    }
}

/// The locks the calling thread holds (always 0 with the option off).
pub inline fn held() u32 {
    return if (enabled) count else 0;
}

/// The application's report for a task that suspends holding locks, set at
/// run time (null: a plain panic). It receives the count, can name where
/// each held lock was taken (the application keeps those sites, zio only
/// the count), and must not return; the panic's own stack trace is the
/// suspension site. Set it once, before any task can suspend holding a lock.
pub var report: ?*const fn (count: u32) noreturn = null;

/// Called where a task suspends.
pub inline fn checkSuspend() void {
    if (enabled and count != 0) suspendedHolding();
}

fn suspendedHolding() noreturn {
    @branchHint(.cold);
    if (report) |r| r(count);
    std.debug.panic("a task suspended while holding {d} lock(s): never hold a lock across a suspension point", .{count});
}

test "held_locks: counts up and down" {
    if (!enabled) return error.SkipZigTest;
    try std.testing.expectEqual(@as(u32, 0), held());
    acquired();
    acquired();
    try std.testing.expectEqual(@as(u32, 2), held());
    released();
    released();
    try std.testing.expectEqual(@as(u32, 0), held());
    checkSuspend();
}
