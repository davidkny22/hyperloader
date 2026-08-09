//! Windows process identity from boot GUID and process creation time.

use super::{ProcessIdentity, ProcessObservation};
use std::io;
use windows_sys::Wdk::System::SystemInformation::NtQuerySystemInformation;
use windows_sys::Win32::Foundation::{
    CloseHandle, ERROR_INVALID_PARAMETER, FILETIME, WAIT_OBJECT_0, WAIT_TIMEOUT,
};
use windows_sys::Win32::System::Threading::{
    GetProcessTimes, OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION, PROCESS_SYNCHRONIZE,
    WaitForSingleObject,
};
use windows_sys::core::GUID;

const SYSTEM_BOOT_ENVIRONMENT_INFORMATION: i32 = 90;

pub(super) fn current_identity() -> Result<ProcessIdentity, String> {
    match observe(std::process::id()) {
        ProcessObservation::Live(identity) => Ok(identity),
        ProcessObservation::Missing => Err("calling process is absent".to_owned()),
        ProcessObservation::Ambiguous(detail) => Err(detail),
    }
}

pub(super) fn observe(pid: u32) -> ProcessObservation {
    // SAFETY: access flags and PID are value parameters and no handle inheritance is requested.
    let handle = unsafe {
        OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_SYNCHRONIZE,
            0,
            pid,
        )
    };
    if handle.is_null() {
        let error = io::Error::last_os_error();
        return if error.raw_os_error() == Some(ERROR_INVALID_PARAMETER as i32) {
            ProcessObservation::Missing
        } else {
            ProcessObservation::Ambiguous(format!("Windows process cannot be opened: {error}"))
        };
    }

    // SAFETY: `handle` is live and a zero timeout performs a nonblocking state query.
    let wait = unsafe { WaitForSingleObject(handle, 0) };
    if wait == WAIT_OBJECT_0 {
        // SAFETY: `handle` is live and is closed exactly once on this return path.
        unsafe { CloseHandle(handle) };
        return ProcessObservation::Missing;
    }
    if wait != WAIT_TIMEOUT {
        let error = io::Error::last_os_error();
        // SAFETY: `handle` is live and is closed exactly once on this return path.
        unsafe { CloseHandle(handle) };
        return ProcessObservation::Ambiguous(format!(
            "Windows process state cannot be queried: {error}"
        ));
    }

    let mut creation = FILETIME::default();
    let mut exit = FILETIME::default();
    let mut kernel = FILETIME::default();
    let mut user = FILETIME::default();
    // SAFETY: `handle` is live and all FILETIME pointers reference writable storage.
    let timing =
        unsafe { GetProcessTimes(handle, &mut creation, &mut exit, &mut kernel, &mut user) };
    // SAFETY: `handle` is live and is closed exactly once after the timing query.
    unsafe { CloseHandle(handle) };
    if timing == 0 {
        return ProcessObservation::Ambiguous(format!(
            "Windows process timing cannot be queried: {}",
            io::Error::last_os_error()
        ));
    }
    let proc_start = (u64::from(creation.dwHighDateTime) << 32) | u64::from(creation.dwLowDateTime);
    ProcessObservation::Live(ProcessIdentity {
        boot_id: boot_identifier(),
        proc_start,
    })
}

fn boot_identifier() -> String {
    #[repr(C)]
    #[derive(Default)]
    struct BootEnvironment {
        identifier: GUID,
        firmware_type: u32,
        boot_flags: u64,
    }

    let mut information = BootEnvironment::default();
    let mut returned = 0_u32;
    // SAFETY: `information` is writable storage of the declared length and `returned` is output.
    let status = unsafe {
        NtQuerySystemInformation(
            SYSTEM_BOOT_ENVIRONMENT_INFORMATION,
            (&mut information as *mut BootEnvironment).cast(),
            size_of::<BootEnvironment>() as u32,
            &mut returned,
        )
    };
    if status < 0 {
        return "windows-boot-id-unavailable".to_owned();
    }
    let guid = information.identifier;
    format!(
        "{:08x}-{:04x}-{:04x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        guid.data1,
        guid.data2,
        guid.data3,
        guid.data4[0],
        guid.data4[1],
        guid.data4[2],
        guid.data4[3],
        guid.data4[4],
        guid.data4[5],
        guid.data4[6],
        guid.data4[7]
    )
}
