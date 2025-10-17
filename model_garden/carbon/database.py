"""Persistent storage for carbon emissions data."""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime


class EmissionsDatabase:
    """
    Persistent storage for carbon emissions records.
    
    Stores emissions data in a central database file for easy querying
    and persistence across restarts.
    """
    
    def __init__(self, db_path: Path = Path("storage/emissions_db.json")):
        """
        Initialize the emissions database.
        
        Args:
            db_path: Path to the JSON database file
        """
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_db_exists()
    
    def _ensure_db_exists(self) -> None:
        """Create the database file if it doesn't exist."""
        if not self.db_path.exists():
            self._write_db({
                "version": "1.0",
                "created_at": datetime.utcnow().isoformat(),
                "emissions": []
            })
    
    def _read_db(self) -> Dict[str, Any]:
        """Read the database file."""
        try:
            with open(self.db_path, 'r') as f:
                content = f.read()
                # Handle empty file
                if not content.strip():
                    return {
                        "version": "1.0",
                        "created_at": datetime.utcnow().isoformat(),
                        "emissions": []
                    }
                return json.loads(content)
        except (json.JSONDecodeError, FileNotFoundError):
            # Corrupt or missing, recreate
            self._ensure_db_exists()
            with open(self.db_path, 'r') as f:
                return json.load(f)
    
    def _write_db(self, data: Dict[str, Any]) -> None:
        """Write to the database file."""
        with open(self.db_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def add_emission(self, emission_data: Dict[str, Any]) -> None:
        """
        Add an emission record to the database.
        
        Args:
            emission_data: Emission data dictionary
        """
        db = self._read_db()
        
        # Add timestamp if not present
        if 'timestamp' not in emission_data:
            emission_data['timestamp'] = datetime.utcnow().isoformat()
        
        # Add or update the record
        job_id = emission_data.get('job_id')
        if job_id:
            # Remove existing record for this job if it exists
            db['emissions'] = [e for e in db['emissions'] if e.get('job_id') != job_id]
        
        db['emissions'].append(emission_data)
        
        # Sort by timestamp (newest first)
        db['emissions'].sort(
            key=lambda x: x.get('timestamp', ''),
            reverse=True
        )
        
        self._write_db(db)
    
    def get_emission(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get emission record by job ID.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Emission data or None if not found
        """
        db = self._read_db()
        for emission in db['emissions']:
            if emission.get('job_id') == job_id:
                return emission
        return None
    
    def get_all_emissions(
        self,
        job_type: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get all emission records.
        
        Args:
            job_type: Filter by job type ('training', 'inference', None for all)
            limit: Maximum number of records to return
            
        Returns:
            List of emission data dictionaries
        """
        db = self._read_db()
        emissions = db['emissions']
        
        # Filter by job type if specified
        if job_type:
            emissions = [e for e in emissions if e.get('job_type') == job_type]
        
        # Limit results if specified
        if limit:
            emissions = emissions[:limit]
        
        return emissions
    
    def get_total_emissions(self) -> Dict[str, Any]:
        """
        Get aggregate statistics for all emissions.
        
        Returns:
            Dictionary with total emissions, energy, and count
        """
        db = self._read_db()
        emissions = db['emissions']
        
        total_co2 = sum(e.get('emissions_kg_co2', 0) for e in emissions)
        total_energy = sum(e.get('energy_consumed_kwh', 0) for e in emissions)
        total_duration = sum(e.get('duration_seconds', 0) for e in emissions)
        
        # Group by job type
        by_type = {}
        for emission in emissions:
            job_type = emission.get('job_type', 'unknown')
            if job_type not in by_type:
                by_type[job_type] = {
                    'count': 0,
                    'total_co2': 0.0,
                    'total_energy': 0.0,
                    'total_duration': 0.0
                }
            by_type[job_type]['count'] += 1
            by_type[job_type]['total_co2'] += emission.get('emissions_kg_co2', 0)
            by_type[job_type]['total_energy'] += emission.get('energy_consumed_kwh', 0)
            by_type[job_type]['total_duration'] += emission.get('duration_seconds', 0)
        
        return {
            'total_emissions_kg_co2': total_co2,
            'total_energy_kwh': total_energy,
            'total_duration_seconds': total_duration,
            'total_count': len(emissions),
            'by_type': by_type,
            'equivalents': {
                'km_driven': total_co2 * 4.6,
                'smartphones_charged': int(total_co2 * 121),
                'tree_months': total_co2 / 0.006,
            }
        }
    
    def delete_emission(self, job_id: str) -> bool:
        """
        Delete an emission record.
        
        Args:
            job_id: Job identifier
            
        Returns:
            True if deleted, False if not found
        """
        db = self._read_db()
        original_len = len(db['emissions'])
        db['emissions'] = [e for e in db['emissions'] if e.get('job_id') != job_id]
        
        if len(db['emissions']) < original_len:
            self._write_db(db)
            return True
        return False


# Global database instance
_emissions_db: Optional[EmissionsDatabase] = None


def get_emissions_db() -> EmissionsDatabase:
    """Get the global emissions database instance."""
    global _emissions_db
    if _emissions_db is None:
        _emissions_db = EmissionsDatabase()
    return _emissions_db
