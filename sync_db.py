# sync_db.py
import os
from sqlalchemy import create_engine, inspect, text, Column, Integer, String, Boolean, LargeBinary, DateTime
from sqlalchemy.orm import declarative_base
from datetime import datetime

# -----------------------------
# Configuration
# -----------------------------
basedir = os.path.abspath(os.path.dirname(__file__))
db_path = os.path.join(basedir, "instance", "cleaningapp.db")
os.makedirs(os.path.dirname(db_path), exist_ok=True)

DATABASE_URI = f"sqlite:///{db_path}"
engine = create_engine(DATABASE_URI, echo=True)
Base = declarative_base()

print("Using database:", db_path)

# -----------------------------
# Models
# -----------------------------
class CleanupReport(Base):
    __tablename__ = "cleanup_report"
    id = Column(Integer, primary_key=True)
    location = Column(String(200))
    reported_by = Column(String(150))  # ensure this exists
    notes = Column(String)
    photo = Column(LargeBinary)
    photo_filename = Column(String(255))
    submitted_photo = Column(LargeBinary)
    submitted_photo_filename = Column(String(255))
    status = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)
    is_complete = Column(Boolean, default=False)
    points_awarded = Column(Integer, default=0)
    submitted_by_cleaner_id = Column(Integer)
    rating = Column(Integer)

# -----------------------------
# Sync function
# -----------------------------
def sync_db():
    """Ensure all tables exist and missing columns are added."""
    insp = inspect(engine)

    for model in [CleanupReport]:
        table_name = model.__tablename__
        if table_name not in insp.get_table_names():
            print(f"Table '{table_name}' does not exist. Creating...")
            Base.metadata.create_all(engine)
            continue

        # Table exists: check for missing columns
        existing_cols = [col['name'] for col in insp.get_columns(table_name)]
        for column in model.__table__.columns:
            if column.name not in existing_cols:
                # Determine SQL type
                if isinstance(column.type, Integer):
                    sql_type = "INTEGER"
                elif isinstance(column.type, Boolean):
                    sql_type = "BOOLEAN"
                elif isinstance(column.type, DateTime):
                    sql_type = "DATETIME"
                else:
                    sql_type = "TEXT"
                print(f"Adding missing column: {table_name}.{column.name} ({sql_type})")
                # Use a connection context
                with engine.begin() as conn:
                    conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {column.name} {sql_type}"))

    print("Database schema is now synchronized with the model.")

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    sync_db()
