"""
Laboratory Information Management System (LIMS) Integration Module

Provides interfaces for:
- Sample tracking and management
- Experiment data upload
- Equipment integration
- Audit trails
- Compliance management
"""

import asyncio
import json
import logging
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path


logger = logging.getLogger(__name__)


class SampleStatus(Enum):
    """Sample lifecycle status"""
    REGISTERED = auto()
    QUEUED = auto()
    IN_PROGRESS = auto()
    COMPLETED = auto()
    FAILED = auto()
    ARCHIVED = auto()


class ExperimentStatus(Enum):
    """Experiment execution status"""
    PLANNED = auto()
    RUNNING = auto()
    PAUSED = auto()
    COMPLETED = auto()
    CANCELLED = auto()
    ERROR = auto()


@dataclass
class SampleMetadata:
    """Sample metadata structure"""
    sample_id: str
    name: str
    material_type: str
    created_by: str
    created_at: datetime = field(default_factory=datetime.now)
    project: str = ""
    batch_id: str = ""
    priority: str = "normal"  # low, normal, high, urgent
    notes: str = ""
    custom_fields: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'sample_id': self.sample_id,
            'name': self.name,
            'material_type': self.material_type,
            'created_by': self.created_by,
            'created_at': self.created_at.isoformat(),
            'project': self.project,
            'batch_id': self.batch_id,
            'priority': self.priority,
            'notes': self.notes,
            'custom_fields': self.custom_fields
        }


@dataclass
class ExperimentRecord:
    """Experiment execution record"""
    experiment_id: str
    sample_id: str
    experiment_type: str
    status: ExperimentStatus
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    operator: str = ""
    equipment_used: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    results: Dict[str, Any] = field(default_factory=dict)
    raw_data_paths: List[str] = field(default_factory=list)
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'experiment_id': self.experiment_id,
            'sample_id': self.sample_id,
            'experiment_type': self.experiment_type,
            'status': self.status.name,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'operator': self.operator,
            'equipment_used': self.equipment_used,
            'parameters': self.parameters,
            'results': self.results,
            'raw_data_paths': self.raw_data_paths,
            'notes': self.notes
        }


@dataclass
class AuditEntry:
    """Audit trail entry"""
    timestamp: datetime
    action: str
    user: str
    entity_type: str
    entity_id: str
    old_value: Optional[Any] = None
    new_value: Optional[Any] = None
    reason: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'action': self.action,
            'user': self.user,
            'entity_type': self.entity_type,
            'entity_id': self.entity_id,
            'old_value': self.old_value,
            'new_value': self.new_value,
            'reason': self.reason
        }


class LIMSClient(ABC):
    """Abstract base class for LIMS clients"""
    
    def __init__(self, base_url: str, api_key: Optional[str] = None):
        self.base_url = base_url
        self.api_key = api_key
        self._connected = False
    
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to LIMS system"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """Disconnect from LIMS"""
        pass
    
    @abstractmethod
    async def create_sample(self, metadata: SampleMetadata) -> str:
        """Create new sample record"""
        pass
    
    @abstractmethod
    async def get_sample(self, sample_id: str) -> Optional[SampleMetadata]:
        """Retrieve sample information"""
        pass
    
    @abstractmethod
    async def update_sample_status(self, 
                                  sample_id: str,
                                  status: SampleStatus) -> bool:
        """Update sample status"""
        pass
    
    @abstractmethod
    async def log_experiment(self, record: ExperimentRecord) -> str:
        """Log experiment execution"""
        pass
    
    @abstractmethod
    async def upload_data(self,
                         sample_id: str,
                         experiment_id: str,
                         data: Dict[str, Any],
                         file_paths: Optional[List[str]] = None) -> bool:
        """Upload experimental data"""
        pass
    
    @abstractmethod
    async def query_experiments(self,
                               filters: Dict[str, Any]) -> List[ExperimentRecord]:
        """Query experiment records"""
        pass


class MockLIMSClient(LIMSClient):
    """
    Mock LIMS client for testing and development
    Stores all data in memory
    """
    
    def __init__(self, base_url: str = "mock://lims"):
        super().__init__(base_url)
        self._samples: Dict[str, SampleMetadata] = {}
        self._experiments: Dict[str, ExperimentRecord] = {}
        self._audit_log: List[AuditEntry] = []
        self._data_store: Dict[str, Dict[str, Any]] = {}
    
    async def connect(self) -> bool:
        """Mock connection"""
        self._connected = True
        logger.info("Connected to Mock LIMS")
        return True
    
    async def disconnect(self) -> bool:
        """Mock disconnection"""
        self._connected = False
        logger.info("Disconnected from Mock LIMS")
        return True
    
    async def create_sample(self, metadata: SampleMetadata) -> str:
        """Create sample record"""
        self._samples[metadata.sample_id] = metadata
        
        self._audit_log.append(AuditEntry(
            timestamp=datetime.now(),
            action="CREATE",
            user=metadata.created_by,
            entity_type="SAMPLE",
            entity_id=metadata.sample_id,
            new_value=metadata.to_dict()
        ))
        
        logger.info(f"Created sample: {metadata.sample_id}")
        return metadata.sample_id
    
    async def get_sample(self, sample_id: str) -> Optional[SampleMetadata]:
        """Get sample record"""
        return self._samples.get(sample_id)
    
    async def update_sample_status(self,
                                  sample_id: str,
                                  status: SampleStatus) -> bool:
        """Update sample status"""
        if sample_id not in self._samples:
            return False
        
        sample = self._samples[sample_id]
        old_status = sample.custom_fields.get('status', 'UNKNOWN')
        sample.custom_fields['status'] = status.name
        
        self._audit_log.append(AuditEntry(
            timestamp=datetime.now(),
            action="STATUS_CHANGE",
            user="system",
            entity_type="SAMPLE",
            entity_id=sample_id,
            old_value=old_status,
            new_value=status.name
        ))
        
        return True
    
    async def log_experiment(self, record: ExperimentRecord) -> str:
        """Log experiment"""
        self._experiments[record.experiment_id] = record
        
        self._audit_log.append(AuditEntry(
            timestamp=datetime.now(),
            action="EXPERIMENT_LOG",
            user=record.operator,
            entity_type="EXPERIMENT",
            entity_id=record.experiment_id,
            new_value=record.to_dict()
        ))
        
        logger.info(f"Logged experiment: {record.experiment_id}")
        return record.experiment_id
    
    async def upload_data(self,
                         sample_id: str,
                         experiment_id: str,
                         data: Dict[str, Any],
                         file_paths: Optional[List[str]] = None) -> bool:
        """Upload data"""
        key = f"{sample_id}/{experiment_id}"
        self._data_store[key] = {
            'data': data,
            'file_paths': file_paths or [],
            'uploaded_at': datetime.now().isoformat()
        }
        
        logger.info(f"Uploaded data for {key}")
        return True
    
    async def query_experiments(self,
                               filters: Dict[str, Any]) -> List[ExperimentRecord]:
        """Query experiments"""
        results = []
        
        for exp in self._experiments.values():
            match = True
            
            if 'sample_id' in filters and exp.sample_id != filters['sample_id']:
                match = False
            if 'experiment_type' in filters and exp.experiment_type != filters['experiment_type']:
                match = False
            if 'status' in filters and exp.status.name != filters['status']:
                match = False
            
            if match:
                results.append(exp)
        
        return results
    
    def get_audit_log(self, 
                     entity_type: Optional[str] = None,
                     entity_id: Optional[str] = None) -> List[AuditEntry]:
        """Get audit log entries"""
        entries = self._audit_log
        
        if entity_type:
            entries = [e for e in entries if e.entity_type == entity_type]
        if entity_id:
            entries = [e for e in entries if e.entity_id == entity_id]
        
        return entries


class SampleTracker:
    """
    High-level sample tracking interface
    Manages sample lifecycle and workflow
    """
    
    def __init__(self, lims_client: LIMSClient):
        self.lims = lims_client
        self._active_samples: Dict[str, SampleStatus] = {}
    
    async def register_sample(self,
                            sample_id: str,
                            name: str,
                            material_type: str,
                            created_by: str,
                            **kwargs) -> str:
        """Register new sample"""
        metadata = SampleMetadata(
            sample_id=sample_id,
            name=name,
            material_type=material_type,
            created_by=created_by,
            **kwargs
        )
        
        await self.lims.create_sample(metadata)
        await self.lims.update_sample_status(sample_id, SampleStatus.REGISTERED)
        
        self._active_samples[sample_id] = SampleStatus.REGISTERED
        
        return sample_id
    
    async def start_processing(self, sample_id: str) -> bool:
        """Mark sample as in progress"""
        success = await self.lims.update_sample_status(
            sample_id, SampleStatus.IN_PROGRESS
        )
        if success:
            self._active_samples[sample_id] = SampleStatus.IN_PROGRESS
        return success
    
    async def complete_processing(self, sample_id: str) -> bool:
        """Mark sample as completed"""
        success = await self.lims.update_sample_status(
            sample_id, SampleStatus.COMPLETED
        )
        if success:
            self._active_samples[sample_id] = SampleStatus.COMPLETED
        return success
    
    async def fail_processing(self, sample_id: str, reason: str = "") -> bool:
        """Mark sample as failed"""
        success = await self.lims.update_sample_status(
            sample_id, SampleStatus.FAILED
        )
        if success:
            self._active_samples[sample_id] = SampleStatus.FAILED
        return success
    
    def get_sample_status(self, sample_id: str) -> Optional[SampleStatus]:
        """Get current sample status"""
        return self._active_samples.get(sample_id)
    
    def list_active_samples(self) -> List[str]:
        """List all active sample IDs"""
        return [
            sid for sid, status in self._active_samples.items()
            if status not in [SampleStatus.COMPLETED, SampleStatus.ARCHIVED]
        ]


class DataUploader:
    """
    Handles data upload to LIMS
    Includes validation, checksums, and batching
    """
    
    def __init__(self, lims_client: LIMSClient):
        self.lims = lims_client
        self._pending_uploads: List[Dict[str, Any]] = []
    
    def calculate_checksum(self, data: Dict[str, Any]) -> str:
        """Calculate data checksum"""
        json_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]
    
    async def upload_experiment_data(self,
                                    sample_id: str,
                                    experiment_id: str,
                                    data: Dict[str, Any],
                                    file_paths: Optional[List[str]] = None,
                                    validate: bool = True) -> bool:
        """Upload experiment data"""
        
        if validate:
            if not self._validate_data(data):
                logger.error("Data validation failed")
                return False
        
        # Add metadata
        data['_metadata'] = {
            'checksum': self.calculate_checksum(data),
            'uploaded_at': datetime.now().isoformat(),
            'version': '1.0'
        }
        
        success = await self.lims.upload_data(
            sample_id, experiment_id, data, file_paths
        )
        
        if success:
            logger.info(f"Uploaded data for {sample_id}/{experiment_id}")
        
        return success
    
    def _validate_data(self, data: Dict[str, Any]) -> bool:
        """Validate data structure"""
        # Basic validation
        if not isinstance(data, dict):
            return False
        
        # Check for required fields if present
        if 'sample_id' in data and not isinstance(data['sample_id'], str):
            return False
        
        return True
    
    def queue_upload(self,
                    sample_id: str,
                    experiment_id: str,
                    data: Dict[str, Any],
                    file_paths: Optional[List[str]] = None):
        """Queue data for batch upload"""
        self._pending_uploads.append({
            'sample_id': sample_id,
            'experiment_id': experiment_id,
            'data': data,
            'file_paths': file_paths
        })
    
    async def flush_uploads(self) -> Dict[str, bool]:
        """Upload all queued data"""
        results = {}
        
        for upload in self._pending_uploads:
            key = f"{upload['sample_id']}/{upload['experiment_id']}"
            results[key] = await self.upload_experiment_data(
                upload['sample_id'],
                upload['experiment_id'],
                upload['data'],
                upload['file_paths']
            )
        
        self._pending_uploads.clear()
        return results


class ExperimentLogger:
    """
    Comprehensive experiment logging
    Tracks all aspects of experiment execution
    """
    
    def __init__(self, lims_client: LIMSClient):
        self.lims = lims_client
        self._current_experiments: Dict[str, ExperimentRecord] = {}
    
    async def start_experiment(self,
                              experiment_id: str,
                              sample_id: str,
                              experiment_type: str,
                              operator: str = "",
                              parameters: Optional[Dict[str, Any]] = None) -> str:
        """Start logging new experiment"""
        record = ExperimentRecord(
            experiment_id=experiment_id,
            sample_id=sample_id,
            experiment_type=experiment_type,
            status=ExperimentStatus.RUNNING,
            started_at=datetime.now(),
            operator=operator,
            parameters=parameters or {}
        )
        
        await self.lims.log_experiment(record)
        self._current_experiments[experiment_id] = record
        
        return experiment_id
    
    async def log_step(self,
                      experiment_id: str,
                      step_name: str,
                      parameters: Optional[Dict[str, Any]] = None,
                      results: Optional[Dict[str, Any]] = None):
        """Log experiment step"""
        if experiment_id not in self._current_experiments:
            logger.warning(f"Experiment {experiment_id} not found")
            return
        
        record = self._current_experiments[experiment_id]
        
        if 'steps' not in record.results:
            record.results['steps'] = []
        
        record.results['steps'].append({
            'step_name': step_name,
            'timestamp': datetime.now().isoformat(),
            'parameters': parameters,
            'results': results
        })
    
    async def add_equipment(self,
                           experiment_id: str,
                           equipment_id: str):
        """Record equipment used"""
        if experiment_id in self._current_experiments:
            record = self._current_experiments[experiment_id]
            if equipment_id not in record.equipment_used:
                record.equipment_used.append(equipment_id)
    
    async def add_data_file(self,
                           experiment_id: str,
                           file_path: str):
        """Record data file path"""
        if experiment_id in self._current_experiments:
            record = self._current_experiments[experiment_id]
            record.raw_data_paths.append(file_path)
    
    async def complete_experiment(self,
                                 experiment_id: str,
                                 final_results: Optional[Dict[str, Any]] = None,
                                 notes: str = "") -> bool:
        """Mark experiment as completed"""
        if experiment_id not in self._current_experiments:
            return False
        
        record = self._current_experiments[experiment_id]
        record.status = ExperimentStatus.COMPLETED
        record.completed_at = datetime.now()
        record.notes = notes
        
        if final_results:
            record.results.update(final_results)
        
        await self.lims.log_experiment(record)
        
        return True
    
    async def fail_experiment(self,
                             experiment_id: str,
                             error_message: str) -> bool:
        """Mark experiment as failed"""
        if experiment_id not in self._current_experiments:
            return False
        
        record = self._current_experiments[experiment_id]
        record.status = ExperimentStatus.ERROR
        record.completed_at = datetime.now()
        record.notes = error_message
        
        await self.lims.log_experiment(record)
        
        return True
    
    def get_experiment_status(self, experiment_id: str) -> Optional[ExperimentStatus]:
        """Get experiment status"""
        if experiment_id in self._current_experiments:
            return self._current_experiments[experiment_id].status
        return None
    
    def generate_report(self, experiment_id: str) -> str:
        """Generate experiment report"""
        if experiment_id not in self._current_experiments:
            return f"Experiment {experiment_id} not found"
        
        record = self._current_experiments[experiment_id]
        
        lines = [
            f"Experiment Report: {experiment_id}",
            "=" * 50,
            f"Sample ID: {record.sample_id}",
            f"Type: {record.experiment_type}",
            f"Status: {record.status.name}",
            f"Operator: {record.operator}",
            f"Started: {record.started_at}",
            f"Completed: {record.completed_at}",
            "",
            "Equipment Used:",
        ]
        
        for eq in record.equipment_used:
            lines.append(f"  - {eq}")
        
        lines.extend([
            "",
            "Parameters:",
            json.dumps(record.parameters, indent=2),
            "",
            "Results:",
            json.dumps(record.results, indent=2),
        ])
        
        if record.notes:
            lines.extend(["", "Notes:", record.notes])
        
        return "\n".join(lines)


class RESTLIMSClient(LIMSClient):
    """
    REST API client for LIMS integration
    Supports standard REST/JSON APIs
    """
    
    def __init__(self, 
                 base_url: str,
                 api_key: str,
                 timeout: float = 30.0):
        super().__init__(base_url, api_key)
        self.timeout = timeout
        self._session = None
    
    async def connect(self) -> bool:
        """Connect to REST API"""
        try:
            import aiohttp
            
            self._session = aiohttp.ClientSession(
                headers={'Authorization': f'Bearer {self.api_key}'}
            )
            
            # Test connection
            async with self._session.get(
                f"{self.base_url}/api/health",
                timeout=self.timeout
            ) as response:
                self._connected = response.status == 200
                return self._connected
                
        except Exception as e:
            logger.error(f"Failed to connect to LIMS: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from API"""
        if self._session:
            await self._session.close()
            self._session = None
        self._connected = False
        return True
    
    async def create_sample(self, metadata: SampleMetadata) -> str:
        """Create sample via REST API"""
        if not self._session:
            raise RuntimeError("Not connected")
        
        async with self._session.post(
            f"{self.base_url}/api/samples",
            json=metadata.to_dict(),
            timeout=self.timeout
        ) as response:
            if response.status == 201:
                result = await response.json()
                return result['sample_id']
            else:
                raise RuntimeError(f"Failed to create sample: {response.status}")
    
    async def get_sample(self, sample_id: str) -> Optional[SampleMetadata]:
        """Get sample via REST API"""
        if not self._session:
            raise RuntimeError("Not connected")
        
        async with self._session.get(
            f"{self.base_url}/api/samples/{sample_id}",
            timeout=self.timeout
        ) as response:
            if response.status == 200:
                data = await response.json()
                return SampleMetadata(**data)
            return None
    
    async def update_sample_status(self,
                                  sample_id: str,
                                  status: SampleStatus) -> bool:
        """Update sample status via REST API"""
        if not self._session:
            raise RuntimeError("Not connected")
        
        async with self._session.patch(
            f"{self.base_url}/api/samples/{sample_id}/status",
            json={'status': status.name},
            timeout=self.timeout
        ) as response:
            return response.status == 200
    
    async def log_experiment(self, record: ExperimentRecord) -> str:
        """Log experiment via REST API"""
        if not self._session:
            raise RuntimeError("Not connected")
        
        async with self._session.post(
            f"{self.base_url}/api/experiments",
            json=record.to_dict(),
            timeout=self.timeout
        ) as response:
            if response.status == 201:
                result = await response.json()
                return result['experiment_id']
            else:
                raise RuntimeError(f"Failed to log experiment: {response.status}")
    
    async def upload_data(self,
                         sample_id: str,
                         experiment_id: str,
                         data: Dict[str, Any],
                         file_paths: Optional[List[str]] = None) -> bool:
        """Upload data via REST API"""
        if not self._session:
            raise RuntimeError("Not connected")
        
        # Upload metadata
        async with self._session.post(
            f"{self.base_url}/api/data",
            json={
                'sample_id': sample_id,
                'experiment_id': experiment_id,
                'data': data
            },
            timeout=self.timeout
        ) as response:
            if response.status != 201:
                return False
        
        # Upload files if provided
        if file_paths:
            for file_path in file_paths:
                with open(file_path, 'rb') as f:
                    async with self._session.post(
                        f"{self.base_url}/api/files",
                        data={'file': f},
                        timeout=self.timeout
                    ) as response:
                        if response.status != 201:
                            logger.warning(f"Failed to upload file: {file_path}")
        
        return True
    
    async def query_experiments(self,
                               filters: Dict[str, Any]) -> List[ExperimentRecord]:
        """Query experiments via REST API"""
        if not self._session:
            raise RuntimeError("Not connected")
        
        async with self._session.get(
            f"{self.base_url}/api/experiments",
            params=filters,
            timeout=self.timeout
        ) as response:
            if response.status == 200:
                data = await response.json()
                return [ExperimentRecord(**exp) for exp in data['experiments']]
            return []
