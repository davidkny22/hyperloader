"""Machine-readable live-training evaluation protocol."""

from .ambient import AmbientDecision, AmbientProbe, compare_ambient
from .anchors import NamedTrainingAnchor, default_named_anchors
from .decision import TrainingDecision, decide
from .dial import TransformerDialPoint, default_dial, validate_dial
from .feeders import (
    IteratorTokenFeeder,
    ResidentTokenFeeder,
    TokenBatch,
    collate_token_batch,
)
from .image_batches import ImageBatch
from .lease import FileLease, LeaseRecord, LeaseUnavailable
from .live_cell import run_training_observation, warm_training_process
from .models import (
    DecisionRule,
    TrainingCellConfig,
    TrainingEnvironment,
    TrainingHalf,
    TrainingObservation,
)
from .output import write_result
from .public_feeders import PublicLoaderFeeder, TrainingBatch, build_public_feeder
from .resume import replay_loader_hash_chain, run_resume_leg
from .resume_records import (
    ARITHMETIC_CONTRACT,
    CheckpointRecord,
    ResumeBundle,
    ResumeLeg,
    validate_resume_bundle,
    write_resume_bundle,
)
from .training_checkpoint import (
    ResumeCursor,
    load_training_checkpoint,
    save_training_checkpoint,
)
from .validation import TrainingProtocolError, validate_observations

__all__ = [
    "AmbientDecision",
    "AmbientProbe",
    "ARITHMETIC_CONTRACT",
    "CheckpointRecord",
    "DecisionRule",
    "FileLease",
    "ImageBatch",
    "IteratorTokenFeeder",
    "LeaseRecord",
    "LeaseUnavailable",
    "NamedTrainingAnchor",
    "PublicLoaderFeeder",
    "ResumeBundle",
    "ResumeCursor",
    "ResumeLeg",
    "ResidentTokenFeeder",
    "TokenBatch",
    "TrainingCellConfig",
    "TrainingBatch",
    "TrainingDecision",
    "TrainingEnvironment",
    "TrainingHalf",
    "TrainingObservation",
    "TrainingProtocolError",
    "TransformerDialPoint",
    "compare_ambient",
    "collate_token_batch",
    "decide",
    "default_dial",
    "default_named_anchors",
    "build_public_feeder",
    "load_training_checkpoint",
    "replay_loader_hash_chain",
    "run_training_observation",
    "run_resume_leg",
    "save_training_checkpoint",
    "validate_dial",
    "validate_observations",
    "validate_resume_bundle",
    "warm_training_process",
    "write_result",
    "write_resume_bundle",
]
