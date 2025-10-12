"""Add rating fields

Revision ID: f183e4ea24a9
Revises: 1e5eab7d28ed
Create Date: 2025-09-30 09:37:43.874683

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'f183e4ea24a9'
down_revision = '1e5eab7d28ed'
branch_labels = None
depends_on = None


def upgrade():
    conn = op.get_bind()
    inspector = sa.inspect(conn)

    # Check if column exists in cleanup_report
    cleanup_columns = [col['name'] for col in inspector.get_columns('cleanup_report')]
    if 'rating' not in cleanup_columns:
        with op.batch_alter_table('cleanup_report', schema=None) as batch_op:
            batch_op.add_column(sa.Column('rating', sa.Integer(), nullable=True))

    # Check if column exists in user
    user_columns = [col['name'] for col in inspector.get_columns('user')]
    if 'avg_rating' not in user_columns:
        with op.batch_alter_table('user', schema=None) as batch_op:
            batch_op.add_column(sa.Column('avg_rating', sa.Float(), nullable=True))



def downgrade():
    with op.batch_alter_table('cleanup_report', schema=None) as batch_op:
        batch_op.drop_column('rating')
    
    with op.batch_alter_table('user', schema=None) as batch_op:
        batch_op.drop_column('avg_rating')


    # ### end Alembic commands ###
