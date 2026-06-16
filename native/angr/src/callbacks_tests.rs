use super::*;

#[test]
fn test_callbacks_creation() {
    Python::initialize();
    let callbacks = PythonCallbacks::new();
    assert!(!callbacks.is_ready());
}

#[test]
fn test_loop_execution_event() {
    let event = LoopExecutionEvent::from_run_result(
        RunResult::BlockEnd {
            next_addr: 0x1000,
            jumpkind: "Ijk_Boring".to_string(),
        },
        5,
    );
    assert_eq!(event.event_type, "block_end");
    assert_eq!(event.pc, Some(0x1000));
    assert_eq!(event.blocks_executed, 5);
}
