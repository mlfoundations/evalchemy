from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from eval.database.config import DATABASE_URL
from eval.database.models import Base, Dataset, Model, EvalResult, EvalSetting

def create_db_engine():
    engine = create_engine(DATABASE_URL)
    create_tables(engine)
    return engine, sessionmaker(bind=engine)

def create_tables(engine):
    Base.metadata.create_all(engine)