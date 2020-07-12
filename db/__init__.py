from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker
from config.settings import DB_HOST_PATH

engine = create_engine(DB_HOST_PATH, echo=False)
Session = scoped_session(sessionmaker(bind=engine))