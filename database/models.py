import uuid
from sqlalchemy import create_engine, Column, Text, Boolean, ForeignKey, TIMESTAMP, Float, CHAR
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime


Base = declarative_base()

class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(Text, nullable=False)
    created_by = Column(Text, nullable=False)
    creation_location = Column(Text, nullable=False)
    creation_time = Column(TIMESTAMP(timezone=True), nullable=False)
    data_location = Column(Text, nullable=False)
    generation_parameters = Column(JSONB, nullable=False)
    dataset_type = Column(Text, nullable=False)

class Model(Base):
    __tablename__ = "models"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(Text, nullable=False)
    base_model_id = Column(UUID(as_uuid=True), ForeignKey("models.id"), nullable=True)
    created_by = Column(Text, nullable=False)
    creation_location = Column(Text, nullable=False)
    training_start = Column(TIMESTAMP(timezone=True), nullable=False)
    training_end = Column(TIMESTAMP(timezone=True), nullable=True)
    training_parameters = Column(JSONB, nullable=False)
    training_status = Column(Text, nullable=True)
    dataset_id = Column(UUID(as_uuid=True), ForeignKey("datasets.id"))
    is_external = Column(Boolean, nullable=False)
    weights_location = Column(Text, nullable=False)
    wandb_link = Column(Text, nullable=True)

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

class EvalSetting(Base):
    __tablename__ = "evalsettings"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(Text, nullable=False)
    parameters = Column(JSONB, nullable=False)
    eval_version_hash = Column(CHAR(64))
