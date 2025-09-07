"""Add last_crawled to documents"""
from alembic import op
import sqlalchemy as sa

revision = "003"
down_revision = "002"
branch_labels = None
depends_on = None

def upgrade():
    op.add_column("documents", sa.Column("last_crawled", sa.DateTime(timezone=True), nullable=True))

def downgrade():
    op.drop_column("documents", "last_crawled")