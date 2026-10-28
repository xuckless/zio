// SPDX-FileCopyrightText: 2026 xuckless (Anole's zio fork)
// SPDX-License-Identifier: MIT

//! Debug tools behind `Options.diagnostics`, compiled in Debug and test
//! builds only (`enabled`): a list of the live tasks of each executor with a
//! task dump that walks every suspended task's stack, and an executor stall
//! watchdog that warns when an executor stops passing its scheduler while
//! work waits. Neither ever fails or stops anything; both only print.
//!
//! The task dump (`dumpAll`, `Runtime.dumpTasks`) is for a hang: something
//! that already knows the process is stuck (a test runner's timeout) asks
//! for it from its own thread. Every live task is on an intrusive list of
//! the executor it was spawned on (linked in `AnyTask.create`, unlinked
//! before its stack is released, O(1) both ways, no allocation). The dump
//! walks the frame-pointer chain from each suspended task's saved context
//! (`coro.Context`), reading the other thread's stack racily as Go's
//! `tracebackothers` reads other goroutines' (runtime/traceback.go,
//! `forEachGRace`): a task may resume while it is read, so every frame
//! pointer is checked to stay inside the part of the task's stack that is
//! always mapped (the saved stack pointer up to the stack's base) and to
//! grow monotonically, and a frame that fails the check ends the walk. The
//! list's spin lock keeps a task's stack from being released while the dump
//! reads it; spawns and finishes take it too, uncontended unless a dump
//! runs. Processes that hold several runtimes (tests) reach them through a
//! process-wide registry (`registry`), joined in `Runtime.initStatic` and
//! left at the start of `Runtime.deinit`.
//!
//! The stall watchdog (`Watchdog`) is the reactor stall detector of Seastar
//! (src/core/reactor.cc, `cpu_stall_detector`), the retake check of Go's
//! sysmon (runtime/proc.go: a P whose `schedtick` has not moved for
//! `forcePreemptNS`) and tokio-supervisor's blocked-worker check
//! (`worker_poll_count` stops moving): each executor bumps a heartbeat
//! every time it passes its scheduler (`Executor.getNextTask`), a thread per
//! runtime checks the heartbeats every `StallWatchdog.interval`, and an
//! executor that has not passed its scheduler for `threshold` while it is
//! not parked and tasks wait in its run queue is reported once per stall,
//! at most `max_reports_per_minute` times a minute per runtime (Seastar's
//! `blocked_reactor_reports_per_minute`). The report names the executor and
//! what it is running and, where it can, that code's stack: the watchdog
//! sends the executor's thread SIGURG, whose handler (async-signal-safe: no
//! locks, no allocation, raw syscalls only) copies the interrupted pc and
//! walks the frame pointers into raw return addresses, as Seastar's
//! `block_notifier` does with `backtrace()`; the watchdog thread then
//! symbolizes them. SIGURG is the signal Go's asynchronous preemption uses
//! (runtime/signal_unix.go, `sigPreempt`, proposal 24543): debuggers pass it
//! through, libc does not use it, and a stray one is ignored by default.
//! zio already uses it to cancel blocking syscalls on pool workers
//! (os/syscall_cancel.zig), so there is one handler, installed there without
//! SA_RESTART and shared by reference count; it gives each signal to `onSignal`
//! first. A system call that the stalled executor's thread is blocked in may
//! therefore return EINTR, as Go programs see since its preemption signal
//! (go.dev/doc/go1.14#runtime). A signal that lands mid-context-switch, with
//! the stack pointer outside the stack of what the executor says it runs,
//! gives no stack (Seastar skips these too; Go's `isAsyncSafePoint` makes
//! the same stack-bounds check).

const std = @import("std");
const builtin = @import("builtin");
const zio_options = @import("options.zig").options;
const os = @import("os/root.zig");
const log = @import("common.zig").log;
const time = @import("time.zig");
const Duration = time.Duration;
const Timestamp = time.Timestamp;

const runtime_mod = @import("runtime.zig");
const Runtime = runtime_mod.Runtime;
const Executor = runtime_mod.Executor;
const AnyTask = @import("task.zig").AnyTask;

/// The task list, the dump and the watchdog are compiled in.
pub const enabled = zio_options.diagnostics and (builtin.mode == .debug or builtin.is_test);

/// Stacks can be walked by frame pointer: the targets whose saved context
/// and frame layout this file knows (fp -> {saved fp, return address}).
pub const can_walk = enabled and
    (builtin.cpu.arch == .x86_64 or builtin.cpu.arch == .aarch64) and
    (builtin.os.tag == .linux or builtin.os.tag.isDarwin());

/// The watchdog needs a thread and a futex to sleep on.
pub const watchdog_enabled = enabled and !builtin.single_threaded and os.Futex != void;

/// The watchdog can take a stalled executor's stack from a signal handler:
/// it needs libc for the thread's stack bounds (pthread) and the shared
/// SIGURG handler (os/syscall_cancel.zig) on top of `can_walk`.
pub const signal_capture = watchdog_enabled and can_walk and builtin.link_libc and os.syscall_cancel.enabled;

/// Return addresses kept per stack.
pub const max_frames = 32;

/// Stall watchdog settings (`RuntimeOptions.stall_watchdog`). Only used when
/// `Options.diagnostics` is on in a Debug or test build.
pub const StallWatchdog = struct {
    /// An executor that has not passed its scheduler for this long while
    /// tasks wait in its run queue is reported. `.zero` turns the watchdog
    /// off.
    threshold: Duration = .fromMilliseconds(200),
    /// How often the watchdog looks.
    interval: Duration = .fromMilliseconds(50),
    /// Reports beyond this many a minute are counted and mentioned in the
    /// next report instead.
    max_reports_per_minute: u32 = 5,
    /// How long the watchdog waits for the stalled thread to run the signal
    /// handler that takes its stack; the report says so when it did not.
    capture_wait: Duration = .fromMilliseconds(500),
    /// Where a report goes; `log.warn` when null. Called on the watchdog's
    /// thread, never on an executor.
    report: ?*const fn (text: []const u8) void = null,
};

/// A test-and-set lock that spins briefly, then yields the thread. Guards
/// only bounded sections: linking or unlinking one task, or copying a
/// bounded snapshot of a list (SAFETY.md L5). It never guards a wait, an
/// allocation or a print.
const SpinLock = struct {
    held: std.atomic.Value(bool) = .init(false),

    fn tryLock(self: *SpinLock) bool {
        return self.held.cmpxchgWeak(false, true, .acquire, .monotonic) == null;
    }

    fn lock(self: *SpinLock) void {
        var spins: u32 = 0;
        while (!self.tryLock()) {
            spins +%= 1;
            if (spins < 100) std.atomic.spinLoopHint() else os.thread.yield();
        }
    }

    /// `lock`, giving up after `limit` (the dump must not hang on a lock).
    fn lockFor(self: *SpinLock, limit: Duration) bool {
        const deadline = Timestamp.now(.monotonic).addDuration(limit);
        var spins: u32 = 0;
        while (!self.tryLock()) {
            spins +%= 1;
            if (spins < 100) {
                std.atomic.spinLoopHint();
            } else {
                if (Timestamp.now(.monotonic).value >= deadline.value) return false;
                os.thread.yield();
            }
        }
        return true;
    }

    fn unlock(self: *SpinLock) void {
        self.held.store(false, .release);
    }
};

/// A task's place on its executor's list (`AnyTask.debug_link`).
pub const TaskLink = struct {
    node: std.DoublyLinkedList.Node = .{},
    /// The list it is on; null once unlinked. Written by the spawner before
    /// anyone else can reach the task and read by the thread that finishes
    /// or destroys it, which the task's scheduling orders after the spawn.
    list: ?*TaskList = null,
};

/// The live tasks spawned on one executor. Under work stealing a task can
/// finish on another executor, so the lock is taken from any thread; it
/// is uncontended except while a dump copies the list.
pub const TaskList = struct {
    lock: SpinLock = .{},
    tasks: std.DoublyLinkedList = .{},
    len: usize = 0,

    pub fn link(self: *TaskList, l: *TaskLink) void {
        self.lock.lock();
        defer self.lock.unlock();
        self.tasks.append(&l.node);
        l.list = self;
        self.len += 1;
    }

    /// Takes `l` off its list, if it is on one. Must run before the task's
    /// stack is released: the dump reads the stacks of listed tasks.
    pub fn unlink(l: *TaskLink) void {
        const list = l.list orelse return;
        list.lock.lock();
        defer list.lock.unlock();
        list.tasks.remove(&l.node);
        l.list = null;
        list.len -= 1;
    }
};

/// What the watchdog and the dump read about an executor. Written only by
/// the executor's thread; read by the watchdog, the dump and the executor's
/// own signal handler.
pub const Watch = struct {
    /// Bumped at every scheduler pass.
    beats: std.atomic.Value(u32) = .init(0),
    /// Parked in its event loop waiting for work: not a stall.
    idle: std.atomic.Value(bool) = .init(false),
    /// Address of the task the executor runs or is about to switch to; 0
    /// while it runs scheduler code; `main_task` while the thread runs its
    /// own code outside the scheduler.
    running: std.atomic.Value(usize) = .init(0),
    main_task: usize = 0,
    /// The thread's identity and stack, recorded on the thread itself.
    tid: if (builtin.os.tag == .linux) std.os.linux.pid_t else void = if (builtin.os.tag == .linux) 0 else {},
    pthread: if (signal_capture and builtin.os.tag.isDarwin()) std.c.pthread_t else void = undefined,
    stack_lo: usize = 0,
    stack_hi: usize = 0,

    pub inline fn beat(self: *Watch, running: usize) void {
        self.beats.store(self.beats.load(.monotonic) +% 1, .monotonic);
        self.running.store(running, .monotonic);
    }
};

/// Per-executor state (`Executor.debug_diag`).
pub const ExecutorDiag = if (enabled) struct {
    tasks: TaskList = .{},
    watch: Watch = .{},
} else void;
pub const executor_diag_init: ExecutorDiag = if (enabled) .{} else {};

/// Per-runtime state (`Runtime.debug_diag`).
pub const RuntimeDiag = if (enabled) struct {
    node: std.DoublyLinkedList.Node = .{},
    watchdog: Watchdog = .{},
} else void;
pub const runtime_diag_init: RuntimeDiag = if (enabled) .{} else {};

/// On the executor's own thread, from `Executor.init`: who it is, where its
/// thread's stack is (for walking the stack of code that runs on it), and
/// that it runs its own code for now.
pub fn recordThread(ex: *Executor) void {
    const w = &ex.debug_diag.watch;
    w.main_task = @intFromPtr(&ex.main_task);
    w.running.store(w.main_task, .monotonic);
    if (builtin.os.tag == .linux) w.tid = std.os.linux.gettid();
    if (comptime !signal_capture) return;
    const bounds = threadStack() orelse return;
    w.stack_lo = bounds[0];
    w.stack_hi = bounds[1];
    if (comptime builtin.os.tag.isDarwin()) w.pthread = std.c.pthread_self();
}

const pthread_np = struct {
    extern "c" fn pthread_getattr_np(thread: std.c.pthread_t, attr: *std.c.pthread_attr_t) c_int;
    extern "c" fn pthread_attr_getstack(attr: *const std.c.pthread_attr_t, addr: *?*anyopaque, size: *usize) c_int;
    extern "c" fn pthread_get_stackaddr_np(thread: std.c.pthread_t) ?*anyopaque;
    extern "c" fn pthread_get_stacksize_np(thread: std.c.pthread_t) usize;
};

/// The calling thread's stack as [lo, hi).
fn threadStack() ?[2]usize {
    if (comptime builtin.os.tag.isDarwin()) {
        const self = std.c.pthread_self();
        const hi = @intFromPtr(pthread_np.pthread_get_stackaddr_np(self) orelse return null);
        const size = pthread_np.pthread_get_stacksize_np(self);
        return .{ hi - size, hi };
    } else {
        var attr: std.c.pthread_attr_t = undefined;
        if (pthread_np.pthread_getattr_np(std.c.pthread_self(), &attr) != 0) return null;
        defer _ = std.c.pthread_attr_destroy(&attr);
        var addr: ?*anyopaque = null;
        var size: usize = 0;
        if (pthread_np.pthread_attr_getstack(&attr, &addr, &size) != 0) return null;
        const lo = @intFromPtr(addr orelse return null);
        return .{ lo, lo + size };
    }
}

// ---------------------------------------------------------------------------
// Stack walking

/// One word of a stack that may belong to another thread, read racily: the
/// value may be stale, so callers check what they do with it.
inline fn readWord(addr: usize) usize {
    // SAFETY: callers pass an aligned address inside a stack range that
    // stays mapped and committed while they read (see `walk`); volatile, as
    // the owner may write it at the same time.
    const p: *const volatile usize = @ptrFromInt(addr);
    return p.*;
}

/// The return addresses of a stack whose innermost frame is at `pc` with
/// frame pointer `fp`, into `out`; every frame pointer read must lie in
/// [lo, hi) and grow towards `hi`. `pc` is stored plus one, so that it
/// symbolizes like the return addresses after it (`std.debug.writeStackTrace`
/// subtracts one from each). Async-signal-safe.
fn walk(pc: usize, fp0: usize, lo: usize, hi: usize, out: []usize) usize {
    if (out.len == 0 or pc == 0) return 0;
    out[0] = pc +| 1;
    var n: usize = 1;
    var fp = fp0;
    const word = @sizeOf(usize);
    while (n < out.len) {
        if (fp < lo or fp >= hi or hi - fp < 2 * word or !std.mem.isAligned(fp, word)) break;
        const next = readWord(fp);
        const ra = std.debug.stripInstructionPtrAuthCode(readWord(fp + word));
        if (ra <= 1) break;
        out[n] = ra;
        n += 1;
        if (next <= fp) break;
        fp = next;
    }
    return n;
}

/// The saved pc, frame pointer and stack pointer of a suspended task, read
/// racily (the task may resume meanwhile).
fn savedRegisters(task: *AnyTask) struct { pc: usize, fp: usize, sp: usize } {
    const ctx: *const volatile @TypeOf(task.coro.context) = &task.coro.context;
    return switch (builtin.cpu.arch) {
        .x86_64 => .{ .pc = ctx.rip, .fp = ctx.rbp, .sp = ctx.rsp },
        .aarch64 => .{ .pc = ctx.pc, .fp = ctx.fp, .sp = ctx.sp },
        else => .{ .pc = 0, .fp = 0, .sp = 0 },
    };
}

/// The committed part of a task's stack: [limit, base).
fn taskStack(task: *AnyTask) [2]usize {
    const info: *const volatile @TypeOf(task.coro.context.stack_info) = &task.coro.context.stack_info;
    return .{ info.limit, info.base };
}

// ---------------------------------------------------------------------------
// The registry of runtimes, the task dump

const registry = struct {
    var lock: os.Mutex = .init();
    var runtimes: std.DoublyLinkedList = .{};
};

/// From the end of `Runtime.initStatic`: the runtime joins the registry and
/// its watchdog starts.
pub fn attach(rt: *Runtime) !void {
    try rt.debug_diag.watchdog.start(rt);
    registry.lock.lock();
    defer registry.lock.unlock();
    registry.runtimes.append(&rt.debug_diag.node);
}

/// From the start of `Runtime.deinit`, before anything is torn down.
pub fn detach(rt: *Runtime) void {
    {
        registry.lock.lock();
        defer registry.lock.unlock();
        registry.runtimes.remove(&rt.debug_diag.node);
    }
    rt.debug_diag.watchdog.stop();
}

/// How long the dump waits for a lock before it skips what the lock guards.
const dump_lock_limit: Duration = .fromSeconds(2);
/// Tasks shown per executor.
const dump_max_tasks = 1024;

/// Every live task of every runtime in the process, with the stack of each
/// suspended one. For a hang dump: call it from a thread of its own, not
/// from an executor. Never waits longer than `dump_lock_limit` for a lock;
/// what it cannot lock it says it skipped.
pub fn dumpAll(w: *std.Io.Writer) void {
    if (comptime !enabled) {
        w.writeAll("zio task dump: not compiled in (needs Options.diagnostics in a Debug or test build)\n") catch {};
        return;
    }
    const deadline = Timestamp.now(.monotonic).addDuration(dump_lock_limit);
    while (!registry.lock.tryLock()) {
        if (Timestamp.now(.monotonic).value >= deadline.value) {
            w.writeAll("zio task dump: the runtime registry stayed locked; skipped\n") catch {};
            return;
        }
        os.time.sleep(.fromMilliseconds(1));
    }
    defer registry.lock.unlock();
    var count: usize = 0;
    var it = registry.runtimes.first;
    while (it) |node| : (it = node.next) count += 1;
    w.print("zio task dump: {d} runtime(s)\n", .{count}) catch return;
    it = registry.runtimes.first;
    while (it) |node| : (it = node.next) {
        const d: *RuntimeDiag = @fieldParentPtr("node", node);
        // SAFETY: `node` is the `debug_diag.node` of a registered Runtime
        // (`attach`), which stays listed until `detach`.
        const rt: *Runtime = @alignCast(@fieldParentPtr("debug_diag", d));
        dumpRuntime(rt, w);
    }
}

/// One task as copied under its list's lock.
const Snapshot = struct {
    addr: usize,
    state: AnyTask.State.Tag,
    running: bool,
    n: u8,
    frames: [max_frames]usize,
};

/// Every live task of `rt` (see `dumpAll`).
pub fn dumpRuntime(rt: *Runtime, w: *std.Io.Writer) void {
    if (comptime !enabled) return;
    const snaps = std.heap.page_allocator.alloc(Snapshot, dump_max_tasks) catch {
        w.writeAll("zio task dump: out of memory\n") catch {};
        return;
    };
    defer std.heap.page_allocator.free(snaps);
    w.print("runtime 0x{x}: {d} executor(s)\n", .{ @intFromPtr(rt), rt.executors.items.len }) catch return;
    if (builtin.omit_frame_pointer) w.writeAll("(zio is built without frame pointers: task stacks may stop early)\n") catch {};
    // What each executor runs now: a task running on one executor is not
    // walked from its saved context, which is stale.
    var running: [Executor.max_executors]usize = @splat(0);
    for (rt.executors.items, 0..) |ex, i| running[i] = ex.debug_diag.watch.running.load(.monotonic);
    const all_running = running[0..rt.executors.items.len];

    for (rt.executors.items, 0..) |ex, i| {
        const watch = &ex.debug_diag.watch;
        const list = &ex.debug_diag.tasks;
        const now_running = running[i];
        const what: []const u8 = if (watch.idle.load(.monotonic)) "parked waiting for work" else if (now_running == 0) "in scheduler code" else if (now_running == watch.main_task) "running the thread's own code (its main task)" else "running a task";
        // Copy under the lock, print after it: the lock is only held for
        // bounded reads, never across a print or symbolization.
        var n: usize = 0;
        var total: usize = 0;
        if (!list.lock.lockFor(dump_lock_limit)) {
            w.print("executor {d}: its task list stayed locked; skipped\n", .{i}) catch return;
            continue;
        }
        total = list.len;
        var it = list.tasks.first;
        while (it) |node| : (it = node.next) {
            if (n == snaps.len) break;
            const link: *TaskLink = @fieldParentPtr("node", node);
            const task: *AnyTask = @fieldParentPtr("debug_link", link);
            snaps[n] = snapshot(task, all_running);
            n += 1;
        }
        list.lock.unlock();

        w.print("executor {d}{f}: {s}, {d} live task(s), {d} queued\n", .{ i, ThreadName{ .watch = watch }, what, total, ex.run_queue.len() }) catch return;
        for (snaps[0..n]) |*s| printSnapshot(s, w) catch return;
        if (total > n) w.print("  ... and {d} more task(s) not shown\n", .{total - n}) catch return;
    }
}

const ThreadName = struct {
    watch: *const Watch,
    pub fn format(self: ThreadName, w: *std.Io.Writer) std.Io.Writer.Error!void {
        if (builtin.os.tag == .linux) try w.print(" (thread {d})", .{self.watch.tid});
    }
};

fn snapshot(task: *AnyTask, running: []const usize) Snapshot {
    var s: Snapshot = .{
        .addr = @intFromPtr(task),
        .state = task.state.load(.monotonic).tag,
        .running = std.mem.indexOfScalar(usize, running, @intFromPtr(task)) != null,
        .n = 0,
        .frames = undefined,
    };
    if (comptime !can_walk) return s;
    // A task that never ran has no stack yet, and a running one's saved
    // context is stale (the thread backtraces show where it is).
    if (s.running or s.state == .new or s.state == .finished) return s;
    const regs = savedRegisters(task);
    const stack = taskStack(task);
    if (regs.sp < stack[0] or regs.sp >= stack[1]) return s;
    s.n = @intCast(walk(regs.pc, regs.fp, regs.sp, stack[1], &s.frames));
    return s;
}

fn printSnapshot(s: *const Snapshot, w: *std.Io.Writer) std.Io.Writer.Error!void {
    const state: []const u8 = if (s.running) "running" else switch (s.state) {
        .new => "not started yet",
        .ready => "ready, queued to run",
        .waiting => "suspended",
        .finished => "finished",
    };
    try w.print("  task 0x{x}: {s}", .{ s.addr, state });
    if (s.n == 0) {
        try w.writeAll(if (s.running) " (its stack is in the thread backtraces)\n" else "\n");
        return;
    }
    // The raw addresses first, on one line: they stay usable (addr2line)
    // should symbolizing fail.
    try w.writeAll(" [");
    for (s.frames[0..s.n], 0..) |a, k| try w.print("{s}0x{x}", .{ if (k == 0) "" else " ", a });
    try w.writeAll("]\n");
    try writeTrace(s.frames[0..s.n], w);
}

/// `std.debug.writeStackTrace` over raw return addresses.
fn writeTrace(frames: []const usize, w: *std.Io.Writer) std.Io.Writer.Error!void {
    var copy: [max_frames]usize = undefined;
    const n = @min(frames.len, copy.len);
    @memcpy(copy[0..n], frames[0..n]);
    // A walk that filled the buffer may have stopped short of the end.
    const st: std.debug.StackTrace = .{ .return_addresses = copy[0..n], .skipped = if (n == max_frames) .unknown else .none };
    try std.debug.writeStackTrace(&st, .{ .writer = w, .mode = .no_color });
}

// ---------------------------------------------------------------------------
// The stall watchdog

pub const Watchdog = struct {
    thread: ?std.Thread = null,
    stop_flag: std.atomic.Value(u32) = .init(0),

    fn start(self: *Watchdog, rt: *Runtime) !void {
        if (comptime !watchdog_enabled) return;
        if (rt.options.stall_watchdog.threshold.value == 0) return;
        if (comptime signal_capture) os.syscall_cancel.installHandler();
        errdefer if (comptime signal_capture) os.syscall_cancel.uninstallHandler();
        self.thread = try std.Thread.spawn(.{}, run, .{ self, rt });
    }

    fn stop(self: *Watchdog) void {
        if (comptime !watchdog_enabled) return;
        const t = self.thread orelse return;
        self.stop_flag.store(1, .release);
        os.Futex.wake(&self.stop_flag, .one);
        t.join();
        self.thread = null;
        if (comptime signal_capture) os.syscall_cancel.uninstallHandler();
    }

    const Seen = struct {
        beats: u32 = 0,
        since: Timestamp = .zero,
        reported: bool = false,
    };

    fn run(self: *Watchdog, rt: *Runtime) void {
        const opts = rt.options.stall_watchdog;
        var seen: [Executor.max_executors]Seen = @splat(.{});
        var limiter: RateLimiter = .{ .max = opts.max_reports_per_minute };
        while (self.stop_flag.load(.acquire) == 0) {
            os.Futex.timedWait(&self.stop_flag, 0, opts.interval) catch {};
            if (self.stop_flag.load(.acquire) != 0) break;
            const now = Timestamp.now(.monotonic);
            for (rt.executors.items, 0..) |ex, i| {
                const s = &seen[i];
                const watch = &ex.debug_diag.watch;
                const beats = watch.beats.load(.monotonic);
                const queued = ex.run_queue.len();
                const waiting = queued > 0 or !ex.run_queue.overflow.isEmpty();
                if (beats != s.beats or watch.idle.load(.monotonic) or !waiting or s.since.value == 0) {
                    s.* = .{ .beats = beats, .since = now };
                    continue;
                }
                if (s.reported) continue;
                const stalled = s.since.durationTo(now);
                if (stalled.value < opts.threshold.value) continue;
                s.reported = true;
                if (limiter.allow(now)) |suppressed| report(rt, ex, i, stalled, queued, suppressed);
            }
        }
    }
};

const RateLimiter = struct {
    max: u32,
    window_start: Timestamp = .zero,
    count: u32 = 0,
    suppressed: u32 = 0,

    /// Null when the report is over the limit; else how many were
    /// suppressed since the last one that went out.
    fn allow(self: *RateLimiter, now: Timestamp) ?u32 {
        if (self.window_start.value == 0 or self.window_start.durationTo(now).value >= Duration.fromMinutes(1).value) {
            self.window_start = now;
            self.count = 0;
        }
        if (self.count >= self.max) {
            self.suppressed += 1;
            return null;
        }
        self.count += 1;
        const s = self.suppressed;
        self.suppressed = 0;
        return s;
    }
};

fn report(rt: *Runtime, ex: *Executor, index: usize, stalled: Duration, queued: u32, suppressed: u32) void {
    var buf: [16 * 1024]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    const watch = &ex.debug_diag.watch;
    const running = watch.running.load(.monotonic);
    var frames: [max_frames]usize = undefined;
    const captured = captureStack(watch, &frames, rt.options.stall_watchdog.capture_wait);
    w.print("executor {d}{f} stalled: no scheduler pass for {d} ms with {d} task(s) queued; it is ", .{
        index, ThreadName{ .watch = watch }, stalled.toMilliseconds(), queued,
    }) catch {};
    if (running == 0) {
        w.writeAll("in scheduler code") catch {};
    } else if (running == watch.main_task) {
        w.writeAll("running the thread's own code (its main task)") catch {};
    } else {
        w.print("running task 0x{x}", .{running}) catch {};
    }
    if (suppressed > 0) w.print(" ({d} earlier report(s) suppressed by the limit of {d} a minute)", .{ suppressed, rt.options.stall_watchdog.max_reports_per_minute }) catch {};
    w.writeAll("\n") catch {};
    switch (captured) {
        .frames => |n| {
            w.writeAll("its stack when the watchdog looked:\n") catch {};
            writeTrace(frames[0..n], &w) catch {};
        },
        .none => |why| w.print("no stack: {s}\n", .{why}) catch {},
    }
    const text = w.buffered();
    if (rt.options.stall_watchdog.report) |sink| sink(text) else log.warn("{s}", .{text});
}

// ---------------------------------------------------------------------------
// Taking a stalled executor's stack with SIGURG

const Captured = union(enum) {
    frames: usize,
    none: []const u8,
};

const capture = struct {
    /// One capture at a time, across every watchdog in the process.
    var lock: os.Mutex = .init();
    /// The `Watch` of the executor whose thread is being asked; the handler
    /// on that thread swaps it to 0 to claim the capture.
    var claim: std.atomic.Value(usize) = .init(0);
    /// Set by the handler when `frames`/`n`/`why` are written.
    var done: std.atomic.Value(u32) = .init(0);
    var frames: [max_frames]usize = undefined;
    var n: usize = 0;
    var why: []const u8 = "";
};

fn captureStack(watch: *Watch, out: *[max_frames]usize, capture_limit: Duration) Captured {
    if (comptime !signal_capture) return .{ .none = "not supported on this target" };
    if (watch.stack_hi == 0) return .{ .none = "the thread's stack bounds are unknown" };
    capture.lock.lock();
    defer capture.lock.unlock();
    capture.done.store(0, .monotonic);
    capture.n = 0;
    capture.why = "";
    capture.claim.store(@intFromPtr(watch), .release);
    if (!signalThread(watch)) {
        capture.claim.store(0, .release);
        return .{ .none = "the signal could not be sent" };
    }
    const deadline = Timestamp.now(.monotonic).addDuration(capture_limit);
    while (capture.done.load(.acquire) == 0) {
        if (Timestamp.now(.monotonic).value >= deadline.value) {
            // Unclaimed: take the claim back, and the handler, should it run
            // later, leaves everything alone. Claimed: the handler is
            // writing now and finishes in bounded time.
            if (capture.claim.cmpxchgStrong(@intFromPtr(watch), 0, .acq_rel, .monotonic) == null)
                return .{ .none = "the thread did not take the signal in time (signals blocked?)" };
            while (capture.done.load(.acquire) == 0) std.atomic.spinLoopHint();
            break;
        }
        os.time.sleep(.fromMilliseconds(1));
    }
    if (capture.n == 0) return .{ .none = capture.why };
    @memcpy(out[0..capture.n], capture.frames[0..capture.n]);
    return .{ .frames = capture.n };
}

fn signalThread(watch: *Watch) bool {
    if (comptime builtin.os.tag == .linux) {
        const rc = std.os.linux.tgkill(std.os.linux.getpid(), watch.tid, .URG);
        return std.os.linux.errno(rc) == .SUCCESS;
    } else if (comptime builtin.os.tag.isDarwin()) {
        return std.c.pthread_kill(watch.pthread, .URG) == 0;
    } else return false;
}

/// From the shared SIGURG handler (os/syscall_cancel.zig), on whatever
/// thread got the signal: true when it was the stack capture's, which this
/// thread then did. Async-signal-safe: atomics, a raw gettid (Linux) or
/// pthread_self (macOS), and plain reads and writes; no locks, no
/// allocation, no errno.
pub fn onSignal(uc: ?*anyopaque) bool {
    @disableInstrumentation();
    const claimed = capture.claim.load(.acquire);
    if (claimed == 0) return false;
    // SAFETY: `claim` holds the address of a live executor's Watch while
    // the watchdog waits for this handler (`captureStack`).
    const watch: *Watch = @ptrFromInt(claimed);
    if (!onThread(watch)) return false;
    if (capture.claim.cmpxchgStrong(claimed, 0, .acq_rel, .monotonic) != null) return false;
    takeStack(watch, uc);
    capture.done.store(1, .release);
    return true;
}

fn onThread(watch: *const Watch) bool {
    if (comptime builtin.os.tag == .linux) return std.os.linux.gettid() == watch.tid;
    if (comptime builtin.os.tag.isDarwin()) return std.c.pthread_self() == watch.pthread;
    return false;
}

/// In the handler: the interrupted code's stack, if the stack pointer is on
/// the stack of what the executor says it runs; otherwise the signal came
/// mid-context-switch and there is no stack to trust.
fn takeStack(watch: *const Watch, uc: ?*anyopaque) void {
    const ctx = std.debug.cpu_context.fromPosixSignalContext(uc) orelse {
        capture.why = "no register context in the signal";
        return;
    };
    const pc: usize, const fp: usize, const sp: usize = switch (builtin.cpu.arch) {
        .x86_64 => .{ ctx.gprs.get(.rip), ctx.gprs.get(.rbp), ctx.gprs.get(.rsp) },
        .aarch64 => .{ ctx.pc, ctx.x[29], ctx.sp },
        else => {
            capture.why = "not supported on this target";
            return;
        },
    };
    const running = watch.running.load(.monotonic);
    const lo: usize, const hi: usize = if (running == 0 or running == watch.main_task)
        .{ watch.stack_lo, watch.stack_hi }
    else blk: {
        // SAFETY: `running` is a task this thread is running; its stack
        // stays until the task finishes on this thread, after this handler.
        const task: *AnyTask = @ptrFromInt(running);
        const s = taskStack(task);
        break :blk .{ s[0], s[1] };
    };
    if (sp < lo or sp >= hi) {
        capture.why = "the signal came in the middle of a context switch";
        return;
    }
    capture.n = walk(pc, fp, sp, hi, &capture.frames);
    if (capture.n == 0) capture.why = "no frames";
}

// ---------------------------------------------------------------------------
// Tests

const Group = @import("group.zig").Group;

test "diag: the task dump shows a suspended task's function in its stack" {
    if (!enabled or !can_walk or builtin.omit_frame_pointer) return error.SkipZigTest;

    const rt = try Runtime.init(std.testing.allocator, .{ .executors = .exact(1) });
    defer rt.deinit();

    const S = struct {
        var event: @import("sync/Event.zig") = .init;
        noinline fn diagParkedForever() !void {
            try event.wait();
        }
    };
    S.event = .init;
    var handle = try rt.spawn(S.diagParkedForever, .{});
    defer handle.cancel();
    // Run it until it suspends in the wait.
    const task = AnyTask.fromAwaitable(handle.awaitable.?);
    while (task.state.load(.acquire).tag != .waiting) try rt.yield();

    var buf: [64 * 1024]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    dumpAll(&w);
    const text = w.buffered();
    S.event.set();
    try handle.join();

    errdefer std.debug.print("dump:\n{s}\n", .{text});
    try std.testing.expect(std.mem.indexOf(u8, text, "suspended") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "diagParkedForever") != null);
}

test "diag: a blocked executor with queued work is reported once, with the blocking function" {
    if (!watchdog_enabled or !signal_capture or builtin.omit_frame_pointer) return error.SkipZigTest;

    const S = struct {
        /// The blocker's task, once it runs.
        var blocker: std.atomic.Value(usize) = .init(0);
        /// Reports that name the blocker's task.
        var ours: std.atomic.Value(u32) = .init(0);
        var text: [16 * 1024]u8 = undefined;
        var text_len: usize = 0;

        /// Counts the reports of this stall: those that name the blocker's
        /// task as running. A report of another stall (the machine pausing
        /// the thread before the blocker runs) is not ours.
        fn sink(t: []const u8) void {
            const addr = blocker.load(.acquire);
            if (addr == 0) return;
            var needle_buf: [64]u8 = undefined;
            const needle = std.fmt.bufPrint(&needle_buf, "running task 0x{x}", .{addr}) catch unreachable;
            if (std.mem.indexOf(u8, t, needle) == null) return;
            if (ours.load(.acquire) == 0) {
                const n = @min(t.len, text.len);
                @memcpy(text[0..n], t[0..n]);
                text_len = n;
            }
            _ = ours.fetchAdd(1, .acq_rel);
            os.Futex.wake(&ours, .all);
        }

        /// Blocks its executor's thread (not the task: no zio wait) until
        /// the watchdog has reported this stall, or the test's deadline.
        noinline fn diagBlockExecutor() void {
            blocker.store(@intFromPtr(runtime_mod.getCurrentTask()), .release);
            const deadline = Timestamp.now(.monotonic).addDuration(test_deadline);
            while (ours.load(.acquire) == 0) {
                if (Timestamp.now(.monotonic).value >= deadline.value) return;
                os.Futex.timedWait(&ours, 0, .fromMilliseconds(100)) catch {};
            }
        }

        fn queued() void {}
    };
    S.blocker.store(0, .monotonic);
    S.ours.store(0, .monotonic);

    const rt = try Runtime.init(std.testing.allocator, .{
        .executors = .exact(1),
        // The only time limit is the test's deadline: a loaded machine may
        // take long to run the stalled thread's signal handler.
        .stall_watchdog = .{ .report = S.sink, .capture_wait = test_deadline },
    });
    defer rt.deinit();

    var group: Group = .init;
    defer group.cancel();
    try group.spawn(S.diagBlockExecutor, .{});
    // Queued behind the blocker on the same (only) executor.
    try group.spawn(S.queued, .{});
    try group.wait();

    const first = S.text[0..S.text_len];
    errdefer std.debug.print("{d} report(s) of the stall; first:\n{s}\n", .{ S.ours.load(.acquire), first });
    try std.testing.expectEqual(@as(u32, 1), S.ours.load(.acquire));
    try std.testing.expect(std.mem.indexOf(u8, first, "diagBlockExecutor") != null);
}

/// What a test waits for at most before it fails (TESTING.md D2): far past
/// any threshold here, so only a hang reaches it.
const test_deadline: Duration = .fromSeconds(60);
