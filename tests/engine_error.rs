use _hyperloader::error::{EngineError, EngineErrorKind};

#[test]
fn error_preserves_category_and_message() {
    let error = EngineError::new(
        EngineErrorKind::InvalidConfiguration,
        "batch size must be positive",
    );

    assert_eq!(error.kind(), EngineErrorKind::InvalidConfiguration);
    assert_eq!(error.message(), "batch size must be positive");
    assert_eq!(
        error.to_string(),
        "invalid configuration: batch size must be positive"
    );
}
