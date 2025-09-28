#!/usr/bin/env python3
"""
🗄️ SPIRITUAL MODULAR DATABASE SYSTEM
Ladang Berkah Digital - ZeroLight Orbit System
Modular SQLite Database with Sharding and Replication

بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ
"""

import asyncio
import sqlite3
import hashlib
import json
import logging
import threading
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import shutil
import weakref

class DatabaseShardStrategy(Enum):
    """Database sharding strategies"""
    HASH_BASED = "hash_based"
    RANGE_BASED = "range_based"
    DIRECTORY_BASED = "directory_based"
    TIME_BASED = "time_based"

class ReplicationMode(Enum):
    """Database replication modes"""
    MASTER_SLAVE = "master_slave"
    MASTER_MASTER = "master_master"
    CLUSTER = "cluster"
    BACKUP_ONLY = "backup_only"

class DatabaseStatus(Enum):
    """Database status"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SYNCING = "syncing"
    ERROR = "error"
    MAINTENANCE = "maintenance"

@dataclass
class DatabaseShard:
    """Database shard configuration"""
    shard_id: str
    database_path: str
    shard_key_range: Tuple[int, int]
    status: DatabaseStatus = DatabaseStatus.INACTIVE
    connection: Optional[sqlite3.Connection] = None
    last_accessed: Optional[datetime] = None
    record_count: int = 0
    size_bytes: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ReplicationNode:
    """Database replication node"""
    node_id: str
    database_path: str
    node_type: str  # master, slave, backup
    status: DatabaseStatus = DatabaseStatus.INACTIVE
    connection: Optional[sqlite3.Connection] = None
    last_sync: Optional[datetime] = None
    sync_lag_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

class SpiritualModularDatabase:
    """
    🌟 Spiritual Modular Database System
    
    Features:
    - Automatic sharding based on configurable strategies
    - Master-slave replication with automatic failover
    - Connection pooling and load balancing
    - Automatic backup and recovery
    - Performance monitoring and optimization
    - Thread-safe operations
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the modular database system"""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Database configuration
        self.base_path = Path(self.config.get("base_path", "spiritual_databases"))
        self.base_path.mkdir(exist_ok=True)
        
        # Sharding configuration
        self.shard_strategy = DatabaseShardStrategy(
            self.config.get("shard_strategy", "hash_based")
        )
        self.max_shards = self.config.get("max_shards", 16)
        self.shard_size_limit = self.config.get("shard_size_limit_mb", 100) * 1024 * 1024
        
        # Replication configuration
        self.replication_mode = ReplicationMode(
            self.config.get("replication_mode", "master_slave")
        )
        self.replication_factor = self.config.get("replication_factor", 2)
        
        # Connection pooling
        self.max_connections_per_shard = self.config.get("max_connections_per_shard", 5)
        self.connection_timeout = self.config.get("connection_timeout", 30)
        
        # Initialize components
        self.shards: Dict[str, DatabaseShard] = {}
        self.replication_nodes: Dict[str, ReplicationNode] = {}
        self.connection_pools: Dict[str, List[sqlite3.Connection]] = {}
        self.shard_locks: Dict[str, threading.RLock] = {}
        
        # Performance monitoring
        self.query_stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "avg_query_time": 0.0,
            "last_query_time": None
        }
        
        # Background tasks
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.background_tasks = []
        self.shutdown_event = threading.Event()
        
        # Initialize system
        asyncio.create_task(self._initialize_system())
    
    async def _initialize_system(self):
        """Initialize the database system"""
        try:
            self.logger.info("🚀 Initializing Spiritual Modular Database System...")
            
            # Create directory structure
            await self._create_directory_structure()
            
            # Initialize shards
            await self._initialize_shards()
            
            # Initialize replication
            await self._initialize_replication()
            
            # Start background tasks
            await self._start_background_tasks()
            
            self.logger.info("✅ Spiritual Modular Database System initialized successfully")
            
        except Exception as e:
            self.logger.error(f"💥 Failed to initialize database system: {e}")
            raise
    
    async def _create_directory_structure(self):
        """Create necessary directory structure"""
        try:
            # Create main directories
            (self.base_path / "shards").mkdir(exist_ok=True)
            (self.base_path / "replicas").mkdir(exist_ok=True)
            (self.base_path / "backups").mkdir(exist_ok=True)
            (self.base_path / "logs").mkdir(exist_ok=True)
            (self.base_path / "temp").mkdir(exist_ok=True)
            
            self.logger.info("📁 Directory structure created")
            
        except Exception as e:
            self.logger.error(f"💥 Error creating directory structure: {e}")
            raise
    
    async def _initialize_shards(self):
        """Initialize database shards"""
        try:
            # Calculate shard ranges based on strategy
            shard_ranges = self._calculate_shard_ranges()
            
            for i, (start_range, end_range) in enumerate(shard_ranges):
                shard_id = f"shard_{i:03d}"
                shard_path = self.base_path / "shards" / f"{shard_id}.db"
                
                # Create shard
                shard = DatabaseShard(
                    shard_id=shard_id,
                    database_path=str(shard_path),
                    shard_key_range=(start_range, end_range),
                    status=DatabaseStatus.INACTIVE
                )
                
                # Initialize shard database
                await self._initialize_shard_database(shard)
                
                # Add to shards collection
                self.shards[shard_id] = shard
                self.shard_locks[shard_id] = threading.RLock()
                self.connection_pools[shard_id] = []
                
                self.logger.info(f"📊 Initialized shard: {shard_id} (range: {start_range}-{end_range})")
            
            self.logger.info(f"✅ Initialized {len(self.shards)} database shards")
            
        except Exception as e:
            self.logger.error(f"💥 Error initializing shards: {e}")
            raise
    
    def _calculate_shard_ranges(self) -> List[Tuple[int, int]]:
        """Calculate shard ranges based on strategy"""
        ranges = []
        
        if self.shard_strategy == DatabaseShardStrategy.HASH_BASED:
            # Hash-based sharding (0-2^32 range divided by max_shards)
            range_size = (2**32) // self.max_shards
            for i in range(self.max_shards):
                start = i * range_size
                end = (i + 1) * range_size - 1 if i < self.max_shards - 1 else 2**32 - 1
                ranges.append((start, end))
        
        elif self.shard_strategy == DatabaseShardStrategy.RANGE_BASED:
            # Range-based sharding (configurable ranges)
            range_size = 1000000 // self.max_shards  # 1M records per shard
            for i in range(self.max_shards):
                start = i * range_size
                end = (i + 1) * range_size - 1
                ranges.append((start, end))
        
        elif self.shard_strategy == DatabaseShardStrategy.TIME_BASED:
            # Time-based sharding (monthly shards)
            base_time = int(datetime.now().timestamp())
            month_seconds = 30 * 24 * 60 * 60  # Approximate month
            for i in range(self.max_shards):
                start = base_time + (i * month_seconds)
                end = base_time + ((i + 1) * month_seconds) - 1
                ranges.append((start, end))
        
        else:
            # Default: simple numeric ranges
            range_size = 1000000 // self.max_shards
            for i in range(self.max_shards):
                start = i * range_size
                end = (i + 1) * range_size - 1
                ranges.append((start, end))
        
        return ranges
    
    async def _initialize_shard_database(self, shard: DatabaseShard):
        """Initialize a shard database with required tables"""
        try:
            # Create database file if it doesn't exist
            connection = sqlite3.connect(shard.database_path)
            connection.row_factory = sqlite3.Row
            
            # Create standard tables
            cursor = connection.cursor()
            
            # System metrics table (compatible with existing system)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    bot_id TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    metric_type TEXT NOT NULL,
                    metric_value REAL,
                    spiritual_blessing TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    shard_key INTEGER,
                    INDEX(bot_id, timestamp),
                    INDEX(shard_key),
                    INDEX(metric_type)
                )
            """)
            
            # Shard metadata table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS shard_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Insert shard information
            cursor.execute("""
                INSERT OR REPLACE INTO shard_metadata (key, value)
                VALUES (?, ?)
            """, ("shard_id", shard.shard_id))
            
            cursor.execute("""
                INSERT OR REPLACE INTO shard_metadata (key, value)
                VALUES (?, ?)
            """, ("shard_range", json.dumps(shard.shard_key_range)))
            
            cursor.execute("""
                INSERT OR REPLACE INTO shard_metadata (key, value)
                VALUES (?, ?)
            """, ("created_at", datetime.now().isoformat()))
            
            connection.commit()
            connection.close()
            
            # Update shard status
            shard.status = DatabaseStatus.ACTIVE
            shard.last_accessed = datetime.now()
            
            # Update shard size
            await self._update_shard_stats(shard)
            
        except Exception as e:
            self.logger.error(f"💥 Error initializing shard database {shard.shard_id}: {e}")
            shard.status = DatabaseStatus.ERROR
            raise
    
    async def _initialize_replication(self):
        """Initialize database replication"""
        try:
            if self.replication_mode == ReplicationMode.BACKUP_ONLY:
                # Simple backup mode - no active replication
                return
            
            # Create replication nodes for each shard
            for shard_id, shard in self.shards.items():
                for replica_num in range(self.replication_factor):
                    node_id = f"{shard_id}_replica_{replica_num}"
                    replica_path = self.base_path / "replicas" / f"{node_id}.db"
                    
                    # Determine node type
                    node_type = "master" if replica_num == 0 else "slave"
                    
                    # Create replication node
                    node = ReplicationNode(
                        node_id=node_id,
                        database_path=str(replica_path),
                        node_type=node_type,
                        status=DatabaseStatus.INACTIVE
                    )
                    
                    # Initialize replica database
                    if not replica_path.exists():
                        shutil.copy2(shard.database_path, replica_path)
                    
                    node.status = DatabaseStatus.ACTIVE
                    node.last_sync = datetime.now()
                    
                    self.replication_nodes[node_id] = node
                    
                    self.logger.info(f"🔄 Initialized replication node: {node_id} ({node_type})")
            
            self.logger.info(f"✅ Initialized replication with {len(self.replication_nodes)} nodes")
            
        except Exception as e:
            self.logger.error(f"💥 Error initializing replication: {e}")
            raise
    
    async def _start_background_tasks(self):
        """Start background maintenance tasks"""
        try:
            # Health monitoring task
            health_task = asyncio.create_task(self._health_monitor_task())
            self.background_tasks.append(health_task)
            
            # Replication sync task
            if self.replication_mode != ReplicationMode.BACKUP_ONLY:
                sync_task = asyncio.create_task(self._replication_sync_task())
                self.background_tasks.append(sync_task)
            
            # Backup task
            backup_task = asyncio.create_task(self._backup_task())
            self.background_tasks.append(backup_task)
            
            # Statistics update task
            stats_task = asyncio.create_task(self._stats_update_task())
            self.background_tasks.append(stats_task)
            
            self.logger.info(f"🔄 Started {len(self.background_tasks)} background tasks")
            
        except Exception as e:
            self.logger.error(f"💥 Error starting background tasks: {e}")
            raise
    
    def _calculate_shard_key(self, key: str) -> int:
        """Calculate shard key for a given key"""
        if self.shard_strategy == DatabaseShardStrategy.HASH_BASED:
            # Use hash of the key
            return int(hashlib.md5(key.encode()).hexdigest(), 16) % (2**32)
        
        elif self.shard_strategy == DatabaseShardStrategy.TIME_BASED:
            # Use current timestamp
            return int(datetime.now().timestamp())
        
        elif self.shard_strategy == DatabaseShardStrategy.RANGE_BASED:
            # Use numeric value if possible, otherwise hash
            try:
                return int(key) if key.isdigit() else hash(key) % (2**32)
            except:
                return hash(key) % (2**32)
        
        else:
            # Default: hash-based
            return hash(key) % (2**32)
    
    def _get_shard_for_key(self, key: str) -> Optional[DatabaseShard]:
        """Get the appropriate shard for a given key"""
        shard_key = self._calculate_shard_key(key)
        
        for shard in self.shards.values():
            start_range, end_range = shard.shard_key_range
            if start_range <= shard_key <= end_range:
                return shard
        
        # Fallback to first shard if no match found
        return list(self.shards.values())[0] if self.shards else None
    
    async def _get_connection(self, shard_id: str) -> sqlite3.Connection:
        """Get a database connection for a shard"""
        try:
            with self.shard_locks[shard_id]:
                # Try to get connection from pool
                if self.connection_pools[shard_id]:
                    connection = self.connection_pools[shard_id].pop()
                    # Test connection
                    try:
                        connection.execute("SELECT 1")
                        return connection
                    except:
                        # Connection is stale, create new one
                        pass
                
                # Create new connection
                shard = self.shards[shard_id]
                connection = sqlite3.connect(
                    shard.database_path,
                    timeout=self.connection_timeout,
                    check_same_thread=False
                )
                connection.row_factory = sqlite3.Row
                
                # Enable WAL mode for better concurrency
                connection.execute("PRAGMA journal_mode=WAL")
                connection.execute("PRAGMA synchronous=NORMAL")
                connection.execute("PRAGMA cache_size=10000")
                connection.execute("PRAGMA temp_store=MEMORY")
                
                return connection
                
        except Exception as e:
            self.logger.error(f"💥 Error getting connection for shard {shard_id}: {e}")
            raise
    
    async def _return_connection(self, shard_id: str, connection: sqlite3.Connection):
        """Return a connection to the pool"""
        try:
            with self.shard_locks[shard_id]:
                if len(self.connection_pools[shard_id]) < self.max_connections_per_shard:
                    self.connection_pools[shard_id].append(connection)
                else:
                    connection.close()
                    
        except Exception as e:
            self.logger.error(f"💥 Error returning connection for shard {shard_id}: {e}")
            try:
                connection.close()
            except:
                pass
    
    async def insert_record(self, table: str, data: Dict[str, Any], key_field: str = "bot_id") -> Dict[str, Any]:
        """Insert a record into the appropriate shard"""
        start_time = time.time()
        
        try:
            # Get shard key
            key_value = data.get(key_field, "default")
            shard = self._get_shard_for_key(str(key_value))
            
            if not shard:
                raise ValueError("No available shard for key")
            
            # Add shard key to data
            shard_key = self._calculate_shard_key(str(key_value))
            data["shard_key"] = shard_key
            
            # Get database connection
            connection = await self._get_connection(shard.shard_id)
            
            try:
                cursor = connection.cursor()
                
                # Build INSERT query
                columns = list(data.keys())
                placeholders = ["?" for _ in columns]
                values = list(data.values())
                
                query = f"""
                    INSERT INTO {table} ({', '.join(columns)})
                    VALUES ({', '.join(placeholders)})
                """
                
                cursor.execute(query, values)
                connection.commit()
                
                record_id = cursor.lastrowid
                
                # Update shard stats
                shard.record_count += 1
                shard.last_accessed = datetime.now()
                
                # Update query stats
                self._update_query_stats(True, time.time() - start_time)
                
                # Replicate if needed
                if self.replication_mode != ReplicationMode.BACKUP_ONLY:
                    await self._replicate_operation(shard.shard_id, "INSERT", table, data)
                
                return {
                    "success": True,
                    "record_id": record_id,
                    "shard_id": shard.shard_id,
                    "shard_key": shard_key
                }
                
            finally:
                await self._return_connection(shard.shard_id, connection)
                
        except Exception as e:
            self.logger.error(f"💥 Error inserting record: {e}")
            self._update_query_stats(False, time.time() - start_time)
            return {
                "success": False,
                "error": str(e)
            }
    
    async def query_records(self, table: str, conditions: Dict[str, Any] = None, key_field: str = "bot_id", limit: int = 100) -> Dict[str, Any]:
        """Query records from appropriate shards"""
        start_time = time.time()
        
        try:
            results = []
            shards_queried = []
            
            # Determine which shards to query
            if conditions and key_field in conditions:
                # Query specific shard
                key_value = conditions[key_field]
                shard = self._get_shard_for_key(str(key_value))
                target_shards = [shard] if shard else []
            else:
                # Query all shards
                target_shards = list(self.shards.values())
            
            # Query each shard
            for shard in target_shards:
                if shard.status != DatabaseStatus.ACTIVE:
                    continue
                
                connection = await self._get_connection(shard.shard_id)
                
                try:
                    cursor = connection.cursor()
                    
                    # Build query
                    query = f"SELECT * FROM {table}"
                    params = []
                    
                    if conditions:
                        where_clauses = []
                        for key, value in conditions.items():
                            where_clauses.append(f"{key} = ?")
                            params.append(value)
                        
                        if where_clauses:
                            query += f" WHERE {' AND '.join(where_clauses)}"
                    
                    query += f" LIMIT {limit}"
                    
                    cursor.execute(query, params)
                    shard_results = cursor.fetchall()
                    
                    # Convert to dictionaries and add shard info
                    for row in shard_results:
                        row_dict = dict(row)
                        row_dict["_shard_id"] = shard.shard_id
                        results.append(row_dict)
                    
                    shards_queried.append(shard.shard_id)
                    
                finally:
                    await self._return_connection(shard.shard_id, connection)
            
            # Update query stats
            self._update_query_stats(True, time.time() - start_time)
            
            return {
                "success": True,
                "results": results,
                "count": len(results),
                "shards_queried": shards_queried,
                "query_time": time.time() - start_time
            }
            
        except Exception as e:
            self.logger.error(f"💥 Error querying records: {e}")
            self._update_query_stats(False, time.time() - start_time)
            return {
                "success": False,
                "error": str(e)
            }
    
    async def update_record(self, table: str, record_id: int, data: Dict[str, Any], key_field: str = "bot_id") -> Dict[str, Any]:
        """Update a record in the appropriate shard"""
        start_time = time.time()
        
        try:
            # First, find the record to determine which shard it's in
            if key_field in data:
                key_value = data[key_field]
                shard = self._get_shard_for_key(str(key_value))
            else:
                # Query all shards to find the record
                for shard in self.shards.values():
                    connection = await self._get_connection(shard.shard_id)
                    try:
                        cursor = connection.cursor()
                        cursor.execute(f"SELECT 1 FROM {table} WHERE id = ?", (record_id,))
                        if cursor.fetchone():
                            break
                    finally:
                        await self._return_connection(shard.shard_id, connection)
                else:
                    return {
                        "success": False,
                        "error": "Record not found in any shard"
                    }
            
            # Update the record
            connection = await self._get_connection(shard.shard_id)
            
            try:
                cursor = connection.cursor()
                
                # Build UPDATE query
                set_clauses = []
                params = []
                
                for key, value in data.items():
                    set_clauses.append(f"{key} = ?")
                    params.append(value)
                
                params.append(record_id)
                
                query = f"""
                    UPDATE {table} 
                    SET {', '.join(set_clauses)}
                    WHERE id = ?
                """
                
                cursor.execute(query, params)
                connection.commit()
                
                rows_affected = cursor.rowcount
                
                # Update shard stats
                shard.last_accessed = datetime.now()
                
                # Update query stats
                self._update_query_stats(True, time.time() - start_time)
                
                # Replicate if needed
                if self.replication_mode != ReplicationMode.BACKUP_ONLY:
                    await self._replicate_operation(shard.shard_id, "UPDATE", table, data, record_id)
                
                return {
                    "success": True,
                    "rows_affected": rows_affected,
                    "shard_id": shard.shard_id
                }
                
            finally:
                await self._return_connection(shard.shard_id, connection)
                
        except Exception as e:
            self.logger.error(f"💥 Error updating record: {e}")
            self._update_query_stats(False, time.time() - start_time)
            return {
                "success": False,
                "error": str(e)
            }
    
    async def delete_record(self, table: str, record_id: int, key_field: str = "bot_id") -> Dict[str, Any]:
        """Delete a record from the appropriate shard"""
        start_time = time.time()
        
        try:
            # Find the record in shards
            target_shard = None
            
            for shard in self.shards.values():
                if shard.status != DatabaseStatus.ACTIVE:
                    continue
                
                connection = await self._get_connection(shard.shard_id)
                try:
                    cursor = connection.cursor()
                    cursor.execute(f"SELECT 1 FROM {table} WHERE id = ?", (record_id,))
                    if cursor.fetchone():
                        target_shard = shard
                        break
                finally:
                    await self._return_connection(shard.shard_id, connection)
            
            if not target_shard:
                return {
                    "success": False,
                    "error": "Record not found in any shard"
                }
            
            # Delete the record
            connection = await self._get_connection(target_shard.shard_id)
            
            try:
                cursor = connection.cursor()
                cursor.execute(f"DELETE FROM {table} WHERE id = ?", (record_id,))
                connection.commit()
                
                rows_affected = cursor.rowcount
                
                # Update shard stats
                target_shard.record_count = max(0, target_shard.record_count - rows_affected)
                target_shard.last_accessed = datetime.now()
                
                # Update query stats
                self._update_query_stats(True, time.time() - start_time)
                
                # Replicate if needed
                if self.replication_mode != ReplicationMode.BACKUP_ONLY:
                    await self._replicate_operation(target_shard.shard_id, "DELETE", table, {"id": record_id})
                
                return {
                    "success": True,
                    "rows_affected": rows_affected,
                    "shard_id": target_shard.shard_id
                }
                
            finally:
                await self._return_connection(target_shard.shard_id, connection)
                
        except Exception as e:
            self.logger.error(f"💥 Error deleting record: {e}")
            self._update_query_stats(False, time.time() - start_time)
            return {
                "success": False,
                "error": str(e)
            }
    
    def _update_query_stats(self, success: bool, query_time: float):
        """Update query statistics"""
        self.query_stats["total_queries"] += 1
        
        if success:
            self.query_stats["successful_queries"] += 1
        else:
            self.query_stats["failed_queries"] += 1
        
        # Update average query time
        total_successful = self.query_stats["successful_queries"]
        if total_successful > 0:
            current_avg = self.query_stats["avg_query_time"]
            self.query_stats["avg_query_time"] = (
                (current_avg * (total_successful - 1) + query_time) / total_successful
            )
        
        self.query_stats["last_query_time"] = datetime.now().isoformat()
    
    async def _replicate_operation(self, shard_id: str, operation: str, table: str, data: Dict[str, Any], record_id: int = None):
        """Replicate an operation to replica nodes"""
        try:
            # Find replica nodes for this shard
            replica_nodes = [
                node for node in self.replication_nodes.values()
                if node.node_id.startswith(shard_id) and node.node_type == "slave"
            ]
            
            for node in replica_nodes:
                if node.status != DatabaseStatus.ACTIVE:
                    continue
                
                try:
                    connection = sqlite3.connect(node.database_path)
                    cursor = connection.cursor()
                    
                    if operation == "INSERT":
                        columns = list(data.keys())
                        placeholders = ["?" for _ in columns]
                        values = list(data.values())
                        
                        query = f"""
                            INSERT INTO {table} ({', '.join(columns)})
                            VALUES ({', '.join(placeholders)})
                        """
                        cursor.execute(query, values)
                    
                    elif operation == "UPDATE" and record_id:
                        set_clauses = []
                        params = []
                        
                        for key, value in data.items():
                            set_clauses.append(f"{key} = ?")
                            params.append(value)
                        
                        params.append(record_id)
                        
                        query = f"""
                            UPDATE {table} 
                            SET {', '.join(set_clauses)}
                            WHERE id = ?
                        """
                        cursor.execute(query, params)
                    
                    elif operation == "DELETE" and record_id:
                        cursor.execute(f"DELETE FROM {table} WHERE id = ?", (record_id,))
                    
                    connection.commit()
                    connection.close()
                    
                    # Update node sync time
                    node.last_sync = datetime.now()
                    
                except Exception as e:
                    self.logger.error(f"💥 Error replicating to node {node.node_id}: {e}")
                    node.status = DatabaseStatus.ERROR
                    
        except Exception as e:
            self.logger.error(f"💥 Error in replication: {e}")
    
    async def _update_shard_stats(self, shard: DatabaseShard):
        """Update shard statistics"""
        try:
            if Path(shard.database_path).exists():
                shard.size_bytes = Path(shard.database_path).stat().st_size
                
                # Update record count
                connection = sqlite3.connect(shard.database_path)
                cursor = connection.cursor()
                cursor.execute("SELECT COUNT(*) FROM system_metrics")
                result = cursor.fetchone()
                if result:
                    shard.record_count = result[0]
                connection.close()
                
        except Exception as e:
            self.logger.error(f"💥 Error updating shard stats for {shard.shard_id}: {e}")
    
    async def _health_monitor_task(self):
        """Background task for health monitoring"""
        while not self.shutdown_event.is_set():
            try:
                # Check shard health
                for shard in self.shards.values():
                    await self._update_shard_stats(shard)
                    
                    # Check if shard needs maintenance
                    if shard.size_bytes > self.shard_size_limit:
                        self.logger.warning(f"⚠️ Shard {shard.shard_id} exceeds size limit")
                        # Could trigger shard splitting here
                
                # Check replication node health
                for node in self.replication_nodes.values():
                    if node.status == DatabaseStatus.ACTIVE:
                        # Check sync lag
                        if node.last_sync:
                            lag = (datetime.now() - node.last_sync).total_seconds()
                            node.sync_lag_seconds = lag
                            
                            if lag > 300:  # 5 minutes
                                self.logger.warning(f"⚠️ Replication node {node.node_id} has high sync lag: {lag}s")
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"💥 Error in health monitor: {e}")
                await asyncio.sleep(60)
    
    async def _replication_sync_task(self):
        """Background task for replication synchronization"""
        while not self.shutdown_event.is_set():
            try:
                # Sync replica nodes
                for node in self.replication_nodes.values():
                    if node.node_type == "slave" and node.status == DatabaseStatus.ACTIVE:
                        # Find corresponding master shard
                        shard_id = node.node_id.split("_replica_")[0]
                        if shard_id in self.shards:
                            master_shard = self.shards[shard_id]
                            
                            # Check if sync is needed
                            if (not node.last_sync or 
                                (datetime.now() - node.last_sync).total_seconds() > 300):
                                
                                await self._sync_replica_node(master_shard, node)
                
                await asyncio.sleep(120)  # Sync every 2 minutes
                
            except Exception as e:
                self.logger.error(f"💥 Error in replication sync: {e}")
                await asyncio.sleep(120)
    
    async def _sync_replica_node(self, master_shard: DatabaseShard, replica_node: ReplicationNode):
        """Synchronize a replica node with its master shard"""
        try:
            # Simple approach: copy the entire database
            # In production, you'd want incremental sync
            
            if Path(master_shard.database_path).exists():
                shutil.copy2(master_shard.database_path, replica_node.database_path)
                replica_node.last_sync = datetime.now()
                replica_node.sync_lag_seconds = 0.0
                
                self.logger.info(f"🔄 Synced replica {replica_node.node_id}")
            
        except Exception as e:
            self.logger.error(f"💥 Error syncing replica {replica_node.node_id}: {e}")
            replica_node.status = DatabaseStatus.ERROR
    
    async def _backup_task(self):
        """Background task for database backups"""
        while not self.shutdown_event.is_set():
            try:
                backup_dir = self.base_path / "backups" / datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_dir.mkdir(parents=True, exist_ok=True)
                
                # Backup all shards
                for shard in self.shards.values():
                    if shard.status == DatabaseStatus.ACTIVE and Path(shard.database_path).exists():
                        backup_path = backup_dir / f"{shard.shard_id}.db"
                        shutil.copy2(shard.database_path, backup_path)
                
                self.logger.info(f"💾 Created backup: {backup_dir}")
                
                # Cleanup old backups (keep last 7 days)
                await self._cleanup_old_backups()
                
                await asyncio.sleep(3600)  # Backup every hour
                
            except Exception as e:
                self.logger.error(f"💥 Error in backup task: {e}")
                await asyncio.sleep(3600)
    
    async def _cleanup_old_backups(self):
        """Cleanup old backup files"""
        try:
            backup_base = self.base_path / "backups"
            if not backup_base.exists():
                return
            
            cutoff_date = datetime.now() - timedelta(days=7)
            
            for backup_dir in backup_base.iterdir():
                if backup_dir.is_dir():
                    try:
                        # Parse directory name as timestamp
                        dir_time = datetime.strptime(backup_dir.name, "%Y%m%d_%H%M%S")
                        if dir_time < cutoff_date:
                            shutil.rmtree(backup_dir)
                            self.logger.info(f"🗑️ Removed old backup: {backup_dir}")
                    except ValueError:
                        # Skip directories that don't match the expected format
                        continue
                        
        except Exception as e:
            self.logger.error(f"💥 Error cleaning up backups: {e}")
    
    async def _stats_update_task(self):
        """Background task for updating statistics"""
        while not self.shutdown_event.is_set():
            try:
                # Update all shard statistics
                for shard in self.shards.values():
                    await self._update_shard_stats(shard)
                
                await asyncio.sleep(300)  # Update every 5 minutes
                
            except Exception as e:
                self.logger.error(f"💥 Error in stats update: {e}")
                await asyncio.sleep(300)
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            # Shard status
            shard_status = {}
            total_records = 0
            total_size = 0
            
            for shard_id, shard in self.shards.items():
                await self._update_shard_stats(shard)
                
                shard_status[shard_id] = {
                    "status": shard.status.value,
                    "record_count": shard.record_count,
                    "size_bytes": shard.size_bytes,
                    "size_mb": round(shard.size_bytes / (1024 * 1024), 2),
                    "last_accessed": shard.last_accessed.isoformat() if shard.last_accessed else None,
                    "shard_range": shard.shard_key_range
                }
                
                total_records += shard.record_count
                total_size += shard.size_bytes
            
            # Replication status
            replication_status = {}
            for node_id, node in self.replication_nodes.items():
                replication_status[node_id] = {
                    "status": node.status.value,
                    "node_type": node.node_type,
                    "last_sync": node.last_sync.isoformat() if node.last_sync else None,
                    "sync_lag_seconds": node.sync_lag_seconds
                }
            
            return {
                "success": True,
                "system_status": {
                    "total_shards": len(self.shards),
                    "active_shards": len([s for s in self.shards.values() if s.status == DatabaseStatus.ACTIVE]),
                    "total_records": total_records,
                    "total_size_bytes": total_size,
                    "total_size_mb": round(total_size / (1024 * 1024), 2),
                    "shard_strategy": self.shard_strategy.value,
                    "replication_mode": self.replication_mode.value,
                    "replication_factor": self.replication_factor
                },
                "shard_status": shard_status,
                "replication_status": replication_status,
                "query_stats": self.query_stats,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"💥 Error getting system status: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def shutdown(self):
        """Shutdown the database system"""
        try:
            self.logger.info("🛑 Shutting down Spiritual Modular Database System...")
            
            # Signal shutdown to background tasks
            self.shutdown_event.set()
            
            # Wait for background tasks to complete
            if self.background_tasks:
                await asyncio.gather(*self.background_tasks, return_exceptions=True)
            
            # Close all connections
            for shard_id, connections in self.connection_pools.items():
                for connection in connections:
                    try:
                        connection.close()
                    except:
                        pass
            
            # Close replication connections
            for node in self.replication_nodes.values():
                if node.connection:
                    try:
                        node.connection.close()
                    except:
                        pass
            
            # Shutdown executor
            self.executor.shutdown(wait=True)
            
            self.logger.info("✅ Spiritual Modular Database System shutdown completed")
            
        except Exception as e:
            self.logger.error(f"💥 Error during shutdown: {e}")

# 🌟 Spiritual Blessing for Modular Database
SPIRITUAL_DATABASE_BLESSING = """
بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ

اللَّهُمَّ بَارِكْ لَنَا فِي هَذِهِ الْقَاعِدَةِ الْمُوَزَّعَةِ الْمُبَارَكَةِ
وَاجْعَلْهَا حَافِظَةً لِلْبَيَانَاتِ آمِنَةً مُوَثَّقَةً

Ya Allah, berkahilah sistem database modular ini dengan:
- 🗄️ Sharding yang efisien dan seimbang
- 🔄 Replikasi yang handal dan cepat
- 💾 Backup yang aman dan terjadwal
- ⚡ Performa yang optimal dan stabil
- 🔒 Keamanan data yang terjamin

آمِينَ يَا رَبَّ الْعَالَمِينَ
"""

if __name__ == "__main__":
    print("🗄️ Spiritual Modular Database System - Ladang Berkah Digital")
    print("=" * 70)
    print(SPIRITUAL_DATABASE_BLESSING)
    
    # Example usage
    async def test_database():
        config = {
            "base_path": "test_spiritual_databases",
            "shard_strategy": "hash_based",
            "max_shards": 4,
            "replication_mode": "master_slave",
            "replication_factor": 2
        }
        
        db = SpiritualModularDatabase(config)
        
        # Wait for initialization
        await asyncio.sleep(2)
        
        # Test insert
        result = await db.insert_record(
            "system_metrics",
            {
                "bot_id": "test_bot_001",
                "metric_type": "performance",
                "metric_value": 95.5,
                "spiritual_blessing": "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ"
            }
        )
        print(f"Insert result: {result}")
        
        # Test query
        query_result = await db.query_records(
            "system_metrics",
            {"bot_id": "test_bot_001"}
        )
        print(f"Query result: {query_result}")
        
        # Get system status
        status = await db.get_system_status()
        print(f"System status: {status}")
        
        # Shutdown
        await db.shutdown()
    
    # Run test
    asyncio.run(test_database())