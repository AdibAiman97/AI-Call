import os
from sqlalchemy import create_engine, Column, Integer, String, Double # type: ignore
from sqlalchemy.orm import sessionmaker, declarative_base # type: ignore
from dotenv import load_dotenv # type: ignore

load_dotenv()

DB_USER = "postgres"
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = "localhost"
DB_NAME = "voxis_ai"
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"

engine = create_engine(DATABASE_URL)
Base = declarative_base()

#Create a session factory
Session = sessionmaker(bind=engine)
session = Session()

class CallSession(Base):
    __tablename__ = "call_sessions"
    id = Column(Integer, primary_key=True, index=True)
    cust_id = Column(String, nullable=False)
    start_time = Column(String, nullable=False)
    end_time = Column(String, nullable=False)
    duration = Column(Double, nullable=False)
    sentiment = Column(String, nullable=False)
    summarized_content = Column(String, nullable=False)

Base.metadata.create_all(bind=engine)