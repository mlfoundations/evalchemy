import uuid

from sqlalchemy import Column, Text, Boolean, ForeignKey, TIMESTAMP, Float, CHAR, Integer
from sqlalchemy.dialects.postgresql.json import JSONB
from sqlalchemy.dialects.postgresql.base import UUID
from sqlalchemy.orm import declarative_base
from sqlalchemy import func


Base = declarative_base()


class Dataset(Base):
    """
    SQLAlchemy model for datasets table.

    Attributes:
        id: Unique identifier (UUID)
        name: Human-readable name for the dataset
        size: Size/length of the dataset
        created_by: User who created the dataset
        creation_time: UTC timestamp of creation
        creation_location: Environment where dataset was created
        data_location: Storage location (S3/GCS/HuggingFace)
        generation_yaml: YAML configuration used for generation
        dataset_type: Type of dataset (SFT/RLHF)
        content_hash: SHA256 hash of dataset content
        generated_externally: Flag for external generation
        external_link: Original dataset source URL
    """

    __tablename__ = "datasets"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(Text, nullable=False, comment="Non-unique pretty name, defaults to YAML name field")
    created_by = Column(Text, nullable=False, comment="Creator ($USER, $SLURM_USER)")
    creation_time = Column(TIMESTAMP(timezone=True), nullable=False, comment="UTC timestamp of creation")
    creation_location = Column(Text, nullable=False, comment="Environment (bespoke_ray, local, TACC, etc)")
    data_location = Column(Text, nullable=False, comment="S3/GCS directory or HuggingFace link")
    generation_parameters = Column(JSONB, nullable=False, comment="YAML pipeline configuration")
    dataset_type = Column(Text, nullable=False, comment="Dataset type (SFT/RLHF)")
    external_link = Column(Text, nullable=True, comment="Original dataset source URL")
    data_generation_hash = Column(Text, nullable=True, comment="Fingerprint of dataset")
    hf_fingerprint = Column(Text, nullable=True, comment="Fingerprint in HF")

    def __repr__(self):
        return (
            f"Dataset(id={self.id}, name={self.name}, created_by={self.created_by}, "
            f"creation_location={self.creation_location}, creation_time={self.creation_time}, "
            f"data_location={self.data_location}, dataset_type={self.dataset_type}), "
            f"hf_fingerprint={self.hf_fingerprint}"
        )

    def to_dict(self):
        return {
            "id": str(self.id),
            "name": self.name,
            "created_by": self.created_by,
            "creation_time": self.creation_time,
            "creation_location": self.creation_location,
            "data_location": self.data_location,
            "generation_parameters": self.generation_parameters,
            "dataset_type": self.dataset_type,
            "external_link": self.external_link,
            "hf_fingerprint": self.hf_fingerprint,
        }


class Model(Base):
    __tablename__ = "models"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(Text, nullable=False)
    base_model_id = Column(UUID(as_uuid=True), ForeignKey("models.id"), nullable=True)
    created_by = Column(Text, nullable=False)
    creation_location = Column(Text, nullable=False)
    creation_time = Column(TIMESTAMP(timezone=True), nullable=True)
    training_start = Column(TIMESTAMP(timezone=True), nullable=False)
    training_end = Column(TIMESTAMP(timezone=True), nullable=True)
    training_parameters = Column(JSONB, nullable=False)
    training_status = Column(Text, nullable=True)
    dataset_id = Column(UUID(as_uuid=True), ForeignKey("datasets.id"))
    is_external = Column(Boolean, nullable=False)
    weights_location = Column(Text, nullable=False)
    wandb_link = Column(Text, nullable=True)
    git_commit_hash = Column(Text, nullable=True, comment="Commit in HF")
    last_modified = Column(TIMESTAMP(timezone=True), nullable=True)

    def __repr__(self):
        return (
            f"Model(id={self.id}, name={self.name}, base_model_id={self.base_model_id}, "
            f"created_by={self.created_by}, creation_location={self.creation_location}, creation_time={self.creation_time}"
            f"training_start={self.training_start}, training_end={self.training_end}, "
            f"training_status={self.training_status}, dataset_id={self.dataset_id}, "
            f"is_external={self.is_external}, weights_location={self.weights_location}), "
            f"git_commit_hash={self.git_commit_hash}, last_modified={self.last_modified}"
        )

    def to_dict(self):
        return {
            "id": str(self.id),
            "name": self.name,
            "base_model_id": str(self.base_model_id) if self.base_model_id else None,
            "created_by": self.created_by,
            "creation_location": self.creation_location,
            "training_start": self.training_start,
            "training_end": self.training_end,
            "training_parameters": self.training_parameters,
            "training_status": self.training_status,
            "dataset_id": str(self.dataset_id) if self.dataset_id else None,
            "is_external": self.is_external,
            "weights_location": self.weights_location,
            "wandb_link": self.wandb_link,
            "git_commit_hash": self.git_commit_hash,
            "last_modified": self.last_modified,
        }


class EvalResult(Base):
    __tablename__ = "evalresults"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_id = Column(UUID(as_uuid=True), ForeignKey("models.id"))
    eval_setting_id = Column(UUID(as_uuid=True), ForeignKey("evalsettings.id"))
    score = Column(Float, nullable=True)
    dataset_id = Column(UUID(as_uuid=True), ForeignKey("datasets.id"))
    created_by = Column(Text, nullable=False)
    creation_time = Column(TIMESTAMP(timezone=True), nullable=False)
    creation_location = Column(Text, nullable=False)
    completions_location = Column(Text, nullable=False)

    def __repr__(self):
        return (
            f"EvalResult(id={self.id}, model_id={self.model_id}, eval_setting_id={self.eval_setting_id}, "
            f"score={self.score}, dataset_id={self.dataset_id}, created_by={self.created_by}, "
            f"creation_time={self.creation_time}, completions_location={self.completions_location})"
        )

    def to_dict(self):
        return {
            "id": str(self.id),
            "model_id": str(self.model_id),
            "eval_setting_id": str(self.eval_setting_id),
            "score": self.score,
            "dataset_id": str(self.dataset_id),
            "created_by": self.created_by,
            "creation_time": self.creation_time,
            "creation_location": self.creation_location,
            "completions_location": self.completions_location,
        }


class EvalSetting(Base):
    __tablename__ = "evalsettings"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(Text, nullable=False)
    parameters = Column(JSONB, nullable=False)
    eval_version_hash = Column(CHAR(64))
    display_order = Column(Integer, nullable=True)

    @classmethod
    def determine_display_order(cls, session, name: str) -> int:
        """
        Determine the display order for a new eval setting based on its prefix.

        Args:
            session: Database session
            name: Name of the eval setting

        Returns:
            Integer representing the display order
        """
        prefix = name.split("_")[0]

        # Check if there's an existing eval setting with the same prefix
        existing_setting = session.query(cls).filter(cls.name.like(f"{prefix}%")).first()

        if existing_setting and existing_setting.display_order:
            return existing_setting.display_order

        # For other prefixes, find the next available order number
        max_order = session.query(func.max(cls.display_order)).scalar() or 1000

        # Check if there are other settings with the same prefix
        other_prefix_order = session.query(cls.display_order).filter(cls.name.like(f"{prefix}%")).first()

        if other_prefix_order:
            return other_prefix_order[0]

        # Get all existing prefixes and their orders
        existing_prefixes = (
            session.query(func.split_part(cls.name, "_", 1).label("prefix"), cls.display_order)
            .group_by(func.split_part(cls.name, "_", 1), cls.display_order)
            .all()
        )

        # Find the next available order number for new prefixes
        used_orders = {row[1] for row in existing_prefixes}
        next_order = max_order + 1000 if not used_orders else max(used_orders) + 1000

        return next_order

    def __repr__(self):
        return (
            f"EvalSetting(id={self.id}, name={self.name}, parameters={self.parameters}, "
            f"eval_version_hash={self.eval_version_hash})"
        )

    def to_dict(self):
        return {
            "id": str(self.id),
            "name": self.name,
            "parameters": self.parameters,
            "eval_version_hash": self.eval_version_hash,
        }
