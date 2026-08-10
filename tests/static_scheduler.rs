//! Fixed-frontier scheduler behavior at the engine boundary.

use _hyperloader::sched::{Dispatch, StaticSchedule};

#[test]
fn full_frontier_applies_backpressure_until_strict_commit() {
    let mut schedule = StaticSchedule::new(10, 15, 2, 2).expect("valid schedule");
    let first = schedule.next_dispatch().expect("first dispatch");
    assert_eq!(
        first,
        Dispatch {
            position: 10,
            worker: 0
        }
    );
    schedule.mark_dispatched(first).expect("first accepted");
    let second = schedule.next_dispatch().expect("second dispatch");
    assert_eq!(
        second,
        Dispatch {
            position: 11,
            worker: 1
        }
    );
    schedule.mark_dispatched(second).expect("second accepted");

    assert_eq!(schedule.next_dispatch(), None);
    schedule
        .mark_completed(second)
        .expect("second completes first");
    assert_eq!(schedule.try_commit(), None);
    assert_eq!(schedule.next_dispatch(), None);

    schedule.mark_completed(first).expect("head completes");
    assert_eq!(schedule.try_commit(), Some(10));
    assert_eq!(schedule.next_dispatch().map(|item| item.position), Some(12));
}

#[test]
fn out_of_order_completion_commits_every_position_once() {
    let mut schedule = StaticSchedule::new(0, 4, 4, 2).expect("valid schedule");
    let mut dispatches = Vec::new();
    while let Some(dispatch) = schedule.next_dispatch() {
        schedule
            .mark_dispatched(dispatch)
            .expect("dispatch accepted");
        dispatches.push(dispatch);
    }
    for index in [2, 0, 3, 1] {
        schedule
            .mark_completed(dispatches[index])
            .expect("completion accepted");
    }

    let mut committed = Vec::new();
    while let Some(position) = schedule.try_commit() {
        committed.push(position);
    }
    assert_eq!(committed, vec![0, 1, 2, 3]);
    assert!(schedule.is_complete());
    assert_eq!(schedule.occupied(), 0);
}

#[test]
fn invalid_transitions_are_rejected() {
    assert!(StaticSchedule::new(1, 0, 1, 1).is_err());
    assert!(StaticSchedule::new(0, 1, 0, 1).is_err());
    assert!(StaticSchedule::new(0, 1, 1, 0).is_err());

    let mut schedule = StaticSchedule::new(0, 2, 2, 1).expect("valid schedule");
    let dispatch = schedule.next_dispatch().expect("dispatch");
    assert!(
        schedule
            .mark_dispatched(Dispatch {
                position: 2,
                worker: 0
            })
            .is_err()
    );
    schedule
        .mark_dispatched(dispatch)
        .expect("dispatch accepted");
    assert!(
        schedule
            .mark_completed(Dispatch {
                position: 0,
                worker: 1
            })
            .is_err()
    );
    schedule
        .mark_completed(dispatch)
        .expect("completion accepted");
    assert!(schedule.mark_completed(dispatch).is_err());
}

#[test]
fn depth_growth_admits_more_work_without_reordering_commit() {
    let mut schedule = StaticSchedule::new(0, 5, 2, 2).expect("valid schedule");
    let first = schedule.next_dispatch().expect("first dispatch");
    schedule.mark_dispatched(first).expect("first accepted");
    let second = schedule.next_dispatch().expect("second dispatch");
    schedule.mark_dispatched(second).expect("second accepted");
    assert_eq!(schedule.next_dispatch(), None);
    assert!(schedule.set_depth(1).is_err());

    schedule.set_depth(4).expect("growth accepted");
    assert_eq!(schedule.next_dispatch().map(|item| item.position), Some(2));
    schedule.mark_completed(second).expect("second completed");
    assert_eq!(schedule.try_commit(), None);
    schedule.mark_completed(first).expect("first completed");
    assert_eq!(schedule.try_commit(), Some(0));
    assert_eq!(schedule.try_commit(), Some(1));
}

#[test]
fn selected_dispatch_reorders_execution_without_reordering_commit() {
    let mut schedule = StaticSchedule::new(0, 4, 4, 2).expect("valid schedule");
    assert_eq!(
        schedule.dispatch_candidates().collect::<Vec<_>>(),
        vec![0, 1, 2, 3]
    );

    let costly = schedule.dispatch_at(3).expect("costly dispatch");
    schedule.mark_dispatched(costly).expect("costly accepted");
    let next = schedule.dispatch_at(1).expect("next dispatch");
    schedule.mark_dispatched(next).expect("next accepted");
    schedule.mark_completed(costly).expect("costly completed");
    schedule.mark_completed(next).expect("next completed");

    assert_eq!(schedule.try_commit(), None);
    assert_eq!(schedule.next_dispatch().map(|item| item.position), Some(0));
}
