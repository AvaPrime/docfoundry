"""Job processing system for DocFoundry.

Provides background job processing with Redis and APScheduler for periodic crawls,
ingestion tasks, and other background operations.
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict

import redis
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.jobstores.redis import RedisJobStore
from apscheduler.executors.asyncio import AsyncIOExecutor
from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    """Job status enumeration."""
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass
class JobRecord:
    """Job record for tracking job state."""
    id: str
    type: str
    status: JobStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    parameters: Dict[str, Any] = None
    logs: List[str] = None
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}
        if self.logs is None:
            self.logs = []
    
    def add_log(self, message: str):
        """Add a log message with timestamp."""
        timestamp = datetime.now().isoformat()
        self.logs.append(f"[{timestamp}] {message}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        # Convert datetime objects to ISO strings
        for field in ['created_at', 'started_at', 'completed_at']:
            if data[field]:
                data[field] = data[field].isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'JobRecord':
        """Create JobRecord from dictionary."""
        # Convert ISO strings back to datetime objects
        for field in ['created_at', 'started_at', 'completed_at']:
            if data.get(field):
                data[field] = datetime.fromisoformat(data[field])
        return cls(**data)


class JobManager:
    """Manages background jobs with Redis and APScheduler."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.redis_url = redis_url
        self.redis_client = None
        self.scheduler = None
        self.job_handlers: Dict[str, Callable] = {}
        self._running = False
        self._memory_jobs = {}
    
    async def initialize(self):
        """Initialize Redis connection and scheduler."""
        redis_available = False
        
        try:
            # Initialize Redis client
            self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
            await asyncio.get_event_loop().run_in_executor(None, self.redis_client.ping)
            logger.info("Connected to Redis successfully")
            redis_available = True
            
        except Exception as e:
            logger.warning(f"Redis not available: {e}. Job manager will run in memory-only mode.")
            self.redis_client = None
        
        try:
            # Configure APScheduler
            if redis_available:
                jobstores = {
                    'default': RedisJobStore(host='localhost', port=6379, db=1)
                }
            else:
                jobstores = {}
            
            executors = {
                'default': AsyncIOExecutor()
            }
            job_defaults = {
                'coalesce': False,
                'max_instances': 3
            }
            
            self.scheduler = AsyncIOScheduler(
                jobstores=jobstores,
                executors=executors,
                job_defaults=job_defaults
            )
            
            # Add event listeners
            self.scheduler.add_listener(self._job_executed, EVENT_JOB_EXECUTED)
            self.scheduler.add_listener(self._job_error, EVENT_JOB_ERROR)
            
            self.scheduler.start()
            self._running = True
            logger.info("Job scheduler initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize job manager: {e}")
            # Don't raise the exception to allow the server to start
            self._running = False
    
    async def shutdown(self):
        """Shutdown the job manager."""
        self._running = False
        if self.scheduler:
            self.scheduler.shutdown()
        if self.redis_client:
            await asyncio.get_event_loop().run_in_executor(None, self.redis_client.close)
        logger.info("Job manager shutdown complete")
    
    def register_handler(self, job_type: str, handler: Callable):
        """Register a job handler function."""
        self.job_handlers[job_type] = handler
        logger.info(f"Registered handler for job type: {job_type}")
    
    async def enqueue_job(self, job_type: str, parameters: Dict[str, Any] = None) -> str:
        """Enqueue a new job."""
        if not self._running:
            raise RuntimeError("Job manager not initialized")
        
        if job_type not in self.job_handlers:
            raise ValueError(f"No handler registered for job type: {job_type}")
        
        job_id = str(uuid.uuid4())
        job_record = JobRecord(
            id=job_id,
            type=job_type,
            status=JobStatus.QUEUED,
            created_at=datetime.now(),
            parameters=parameters or {}
        )
        
        # Store job record
        if self.redis_client:
            await self._store_job_record(job_record)
        else:
            if not hasattr(self, '_memory_jobs'):
                self._memory_jobs = {}
            self._memory_jobs[job_id] = job_record
        
        # Schedule job execution
        self.scheduler.add_job(
            self._execute_job,
            'date',
            run_date=datetime.now() + timedelta(seconds=1),
            args=[job_id],
            id=job_id
        )
        
        logger.info(f"Enqueued job {job_id} of type {job_type}")
        return job_id
    
    async def schedule_periodic_job(
        self, 
        job_type: str, 
        cron_expression: str, 
        parameters: Dict[str, Any] = None,
        job_id: Optional[str] = None
    ) -> str:
        """Schedule a periodic job using cron expression."""
        if not self._running:
            raise RuntimeError("Job manager not initialized")
        
        if job_id is None:
            job_id = f"periodic_{job_type}_{uuid.uuid4().hex[:8]}"
        
        # Parse cron expression (simplified - supports minute, hour, day, month, day_of_week)
        cron_parts = cron_expression.split()
        if len(cron_parts) != 5:
            raise ValueError("Cron expression must have 5 parts: minute hour day month day_of_week")
        
        minute, hour, day, month, day_of_week = cron_parts
        
        self.scheduler.add_job(
            self._execute_periodic_job,
            'cron',
            minute=minute,
            hour=hour,
            day=day,
            month=month,
            day_of_week=day_of_week,
            args=[job_type, parameters or {}],
            id=job_id
        )
        
        logger.info(f"Scheduled periodic job {job_id} with cron: {cron_expression}")
        return job_id
    
    async def get_job_status(self, job_id: str) -> Optional[JobRecord]:
        """Get job status and details."""
        if not self._running:
            raise RuntimeError("Job manager not initialized")
        
        if self.redis_client:
            job_data = await asyncio.get_event_loop().run_in_executor(
                None, self.redis_client.get, f"job:{job_id}"
            )
            
            if job_data:
                return JobRecord.from_dict(json.loads(job_data))
        else:
            # Get from memory when Redis is not available
            if hasattr(self, '_memory_jobs'):
                return self._memory_jobs.get(job_id)
        
        return None
    
    async def list_jobs(self, status: Optional[JobStatus] = None, limit: int = 100) -> List[JobRecord]:
        """List jobs, optionally filtered by status."""
        if not self._running:
            raise RuntimeError("Job manager not initialized")
        
        jobs = []
        
        if self.redis_client:
            # Get all job keys from Redis
            job_keys = await asyncio.get_event_loop().run_in_executor(
                None, self.redis_client.keys, "job:*"
            )
            
            for key in job_keys[:limit]:
                job_data = await asyncio.get_event_loop().run_in_executor(
                    None, self.redis_client.get, key
                )
                if job_data:
                    job_record = JobRecord.from_dict(json.loads(job_data))
                    if status is None or job_record.status == status:
                        jobs.append(job_record)
        else:
            # Get from memory when Redis is not available
            if hasattr(self, '_memory_jobs'):
                for job_record in self._memory_jobs.values():
                    if status is None or job_record.status == status:
                        jobs.append(job_record)
        
        # Sort by creation time (newest first)
        jobs.sort(key=lambda x: x.created_at, reverse=True)
        return jobs[:limit]
    
    async def _execute_job(self, job_id: str):
        """Execute a single job."""
        job_record = await self.get_job_status(job_id)
        if not job_record:
            logger.error(f"Job {job_id} not found")
            return
        
        handler = self.job_handlers.get(job_record.type)
        if not handler:
            job_record.status = JobStatus.FAILED
            job_record.error = f"No handler registered for job type: {job_record.type}"
            job_record.add_log(f"Failed: {job_record.error}")
            if self.redis_client:
                await self._store_job_record(job_record)
            else:
                self._memory_jobs[job_id] = job_record
            return
        
        # Update job status to running
        job_record.status = JobStatus.RUNNING
        job_record.started_at = datetime.now()
        job_record.add_log("Job started")
        if self.redis_client:
            await self._store_job_record(job_record)
        else:
            self._memory_jobs[job_id] = job_record
        
        try:
            # Execute the job handler
            result = await handler(job_record.id, job_record.parameters)
            
            # Update job status to done
            job_record.status = JobStatus.DONE
            job_record.completed_at = datetime.now()
            job_record.result = result
            job_record.add_log("Job completed successfully")
            
        except Exception as e:
            # Update job status to failed
            job_record.status = JobStatus.FAILED
            job_record.completed_at = datetime.now()
            job_record.error = str(e)
            job_record.add_log(f"Job failed: {e}")
            logger.error(f"Job {job_id} failed: {e}")
        
        if self.redis_client:
            await self._store_job_record(job_record)
        else:
            self._memory_jobs[job_id] = job_record
    
    async def _execute_periodic_job(self, job_type: str, parameters: Dict[str, Any]):
        """Execute a periodic job by creating a new job instance."""
        job_id = await self.enqueue_job(job_type, parameters)
        logger.info(f"Created periodic job instance {job_id} for type {job_type}")
    
    async def _store_job_record(self, job_record: JobRecord):
        """Store job record in Redis or memory."""
        if self.redis_client:
            job_data = json.dumps(job_record.to_dict())
            await asyncio.get_event_loop().run_in_executor(
                None, 
                self.redis_client.setex, 
                f"job:{job_record.id}", 
                86400 * 7,  # 7 days TTL
                job_data
            )
        else:
            # Store in memory when Redis is not available
            if not hasattr(self, '_memory_jobs'):
                self._memory_jobs = {}
            self._memory_jobs[job_record.id] = job_record
    
    def _job_executed(self, event):
        """Handle job execution event."""
        logger.info(f"Job {event.job_id} executed successfully")
    
    def _job_error(self, event):
        """Handle job error event."""
        logger.error(f"Job {event.job_id} failed: {event.exception}")


# Global job manager instance
job_manager = JobManager()


# Register default job handlers
def register_default_handlers():
    """
    Register default job handlers with the job manager.
    """
    from .job_handlers import crawl_source_job, reindex_job, periodic_crawl_job
    
    job_manager.register_handler("crawl_source", crawl_source_job)
    job_manager.register_handler("reindex", reindex_job)
    job_manager.register_handler("periodic_crawl", periodic_crawl_job)
    
    logger.info("Default job handlers registered successfully")