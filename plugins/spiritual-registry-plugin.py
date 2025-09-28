#!/usr/bin/env python3
"""
📋 SPIRITUAL REGISTRY PLUGIN
Ladang Berkah Digital - ZeroLight Orbit System
Plugin Implementation for Spiritual Bot Registry Management

بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ
"""

import asyncio
import sys
import os
import sqlite3
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import importlib.util
import json

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from spiritual_plugin_interface import (
    SpiritualPlugin, PluginMetadata, PluginPriority, PluginStatus
)

class SpiritualRegistryPlugin(SpiritualPlugin):
    """
    🌟 Plugin for Spiritual Bot Registry Management
    
    Manages the bot registry functionality including:
    - Bot registration and tracking
    - Performance metrics collection
    - Registry database operations
    - Bot lifecycle monitoring
    """
    
    def _create_metadata(self) -> PluginMetadata:
        """Create plugin metadata"""
        return PluginMetadata(
            name="spiritual-registry",
            version="1.0.0",
            description="Spiritual Bot Registry Management Plugin",
            author="ZeroLight Orbit Team",
            dependencies=["spiritual-framework"],  # Depends on framework
            priority=PluginPriority.HIGH,
            category="core",
            spiritual_blessing="بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ"
        )
    
    async def initialize(self) -> bool:
        """Initialize the registry plugin"""
        try:
            self.logger.info("🚀 Initializing Spiritual Registry Plugin...")
            
            # Import the registry module dynamically
            registry_path = Path(__file__).parent.parent / "spiritual-static-bot-registry.py"
            
            if not registry_path.exists():
                self.logger.error(f"❌ Registry file not found: {registry_path}")
                return False
            
            # Load registry module
            spec = importlib.util.spec_from_file_location(
                "spiritual_static_bot_registry", 
                registry_path
            )
            self.registry_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.registry_module)
            
            # Initialize registry components
            self.registry = None
            self.registry_config = self.config.get("registry", {})
            self.database_path = self.registry_config.get(
                "database_path", 
                str(Path(__file__).parent.parent / "spiritual_bot_registry.db")
            )
            
            # Setup registry
            await self._setup_registry()
            
            self.logger.info("✅ Spiritual Registry Plugin initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"💥 Failed to initialize registry plugin: {e}")
            return False
    
    async def _setup_registry(self):
        """Setup registry components"""
        try:
            # Initialize registry if available in module
            if hasattr(self.registry_module, 'SpiritualAdvancedBotRegistry'):
                self.registry = self.registry_module.SpiritualAdvancedBotRegistry(
                    db_path=self.database_path,
                    config=self.registry_config
                )
                self.logger.info("📋 Bot Registry initialized")
            
            # Initialize database connection
            await self._init_database()
            
            # Initialize registry stats
            self.registry_stats = {
                "bots_registered": 0,
                "bots_active": 0,
                "total_queries": 0,
                "last_query": None,
                "database_size": 0
            }
            
            # Update initial stats
            await self._update_stats()
            
        except Exception as e:
            self.logger.error(f"💥 Error setting up registry: {e}")
            raise
    
    async def _init_database(self):
        """Initialize database connection and tables"""
        try:
            self.db_connection = sqlite3.connect(self.database_path)
            self.db_connection.row_factory = sqlite3.Row
            
            # Ensure required tables exist
            await self._ensure_tables()
            
            self.logger.info(f"🗄️ Database initialized: {self.database_path}")
            
        except Exception as e:
            self.logger.error(f"💥 Error initializing database: {e}")
            raise
    
    async def _ensure_tables(self):
        """Ensure required database tables exist"""
        try:
            cursor = self.db_connection.cursor()
            
            # Check if system_metrics table exists
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='system_metrics'
            """)
            
            if not cursor.fetchone():
                # Create basic system_metrics table
                cursor.execute("""
                    CREATE TABLE system_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        bot_id TEXT NOT NULL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        metric_type TEXT NOT NULL,
                        metric_value REAL,
                        spiritual_blessing TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                self.logger.info("📊 Created system_metrics table")
            
            # Create index for better performance
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_bot_id_timestamp 
                ON system_metrics(bot_id, timestamp)
            """)
            
            self.db_connection.commit()
            
        except Exception as e:
            self.logger.error(f"💥 Error ensuring tables: {e}")
            raise
    
    async def execute(self, action: str, *args, **kwargs) -> Dict[str, Any]:
        """Execute registry actions"""
        try:
            self.registry_stats["total_queries"] += 1
            self.registry_stats["last_query"] = datetime.now().isoformat()
            
            if action == "register_bot":
                return await self._register_bot(**kwargs)
            
            elif action == "unregister_bot":
                return await self._unregister_bot(**kwargs)
            
            elif action == "get_bot_info":
                return await self._get_bot_info(**kwargs)
            
            elif action == "list_bots":
                return await self._list_bots(**kwargs)
            
            elif action == "update_bot_metrics":
                return await self._update_bot_metrics(**kwargs)
            
            elif action == "get_performance_report":
                return await self._get_performance_report(**kwargs)
            
            elif action == "get_registry_stats":
                return await self._get_registry_stats()
            
            elif action == "search_bots":
                return await self._search_bots(**kwargs)
            
            elif action == "cleanup_old_data":
                return await self._cleanup_old_data(**kwargs)
            
            elif action == "health_check":
                return await self._health_check()
            
            else:
                return {
                    "success": False,
                    "error": f"Unknown action: {action}",
                    "available_actions": [
                        "register_bot", "unregister_bot", "get_bot_info", 
                        "list_bots", "update_bot_metrics", "get_performance_report",
                        "get_registry_stats", "search_bots", "cleanup_old_data", 
                        "health_check"
                    ]
                }
                
        except Exception as e:
            self.logger.error(f"💥 Error executing action {action}: {e}")
            return {
                "success": False,
                "error": str(e),
                "action": action
            }
    
    async def _register_bot(self, **kwargs) -> Dict[str, Any]:
        """Register a new bot in the registry"""
        try:
            bot_id = kwargs.get("bot_id")
            bot_info = kwargs.get("bot_info", {})
            
            if not bot_id:
                return {
                    "success": False,
                    "error": "bot_id is required"
                }
            
            if self.registry and hasattr(self.registry, 'register_bot'):
                result = await self.registry.register_bot(bot_id, bot_info)
                
                if result.get("success"):
                    self.registry_stats["bots_registered"] += 1
                    self.registry_stats["bots_active"] += 1
                
                return result
            else:
                # Fallback implementation using direct database
                cursor = self.db_connection.cursor()
                
                # Insert bot registration record
                cursor.execute("""
                    INSERT INTO system_metrics 
                    (bot_id, metric_type, metric_value, spiritual_blessing)
                    VALUES (?, ?, ?, ?)
                """, (
                    bot_id, 
                    "registration", 
                    1.0, 
                    self.metadata.spiritual_blessing
                ))
                
                self.db_connection.commit()
                
                self.registry_stats["bots_registered"] += 1
                self.registry_stats["bots_active"] += 1
                
                return {
                    "success": True,
                    "bot_id": bot_id,
                    "registered_at": datetime.now().isoformat(),
                    "message": "Bot registered successfully"
                }
                
        except Exception as e:
            self.logger.error(f"💥 Error registering bot: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _unregister_bot(self, **kwargs) -> Dict[str, Any]:
        """Unregister a bot from the registry"""
        try:
            bot_id = kwargs.get("bot_id")
            
            if not bot_id:
                return {
                    "success": False,
                    "error": "bot_id is required"
                }
            
            if self.registry and hasattr(self.registry, 'unregister_bot'):
                result = await self.registry.unregister_bot(bot_id)
                
                if result.get("success"):
                    self.registry_stats["bots_active"] = max(0, self.registry_stats["bots_active"] - 1)
                
                return result
            else:
                # Fallback implementation
                cursor = self.db_connection.cursor()
                
                # Insert unregistration record
                cursor.execute("""
                    INSERT INTO system_metrics 
                    (bot_id, metric_type, metric_value, spiritual_blessing)
                    VALUES (?, ?, ?, ?)
                """, (
                    bot_id, 
                    "unregistration", 
                    0.0, 
                    self.metadata.spiritual_blessing
                ))
                
                self.db_connection.commit()
                
                self.registry_stats["bots_active"] = max(0, self.registry_stats["bots_active"] - 1)
                
                return {
                    "success": True,
                    "bot_id": bot_id,
                    "unregistered_at": datetime.now().isoformat(),
                    "message": "Bot unregistered successfully"
                }
                
        except Exception as e:
            self.logger.error(f"💥 Error unregistering bot: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _get_bot_info(self, **kwargs) -> Dict[str, Any]:
        """Get information about a specific bot"""
        try:
            bot_id = kwargs.get("bot_id")
            
            if not bot_id:
                return {
                    "success": False,
                    "error": "bot_id is required"
                }
            
            if self.registry and hasattr(self.registry, 'get_bot_info'):
                return await self.registry.get_bot_info(bot_id)
            else:
                # Fallback implementation
                cursor = self.db_connection.cursor()
                
                cursor.execute("""
                    SELECT * FROM system_metrics 
                    WHERE bot_id = ? 
                    ORDER BY timestamp DESC 
                    LIMIT 10
                """, (bot_id,))
                
                records = cursor.fetchall()
                
                if records:
                    return {
                        "success": True,
                        "bot_id": bot_id,
                        "metrics": [dict(record) for record in records],
                        "last_seen": records[0]["timestamp"] if records else None
                    }
                else:
                    return {
                        "success": False,
                        "error": f"Bot {bot_id} not found in registry"
                    }
                
        except Exception as e:
            self.logger.error(f"💥 Error getting bot info: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _list_bots(self, **kwargs) -> Dict[str, Any]:
        """List all registered bots"""
        try:
            limit = kwargs.get("limit", 100)
            offset = kwargs.get("offset", 0)
            
            if self.registry and hasattr(self.registry, 'list_bots'):
                return await self.registry.list_bots(limit=limit, offset=offset)
            else:
                # Fallback implementation
                cursor = self.db_connection.cursor()
                
                cursor.execute("""
                    SELECT DISTINCT bot_id, 
                           MAX(timestamp) as last_seen,
                           COUNT(*) as metric_count
                    FROM system_metrics 
                    GROUP BY bot_id
                    ORDER BY last_seen DESC
                    LIMIT ? OFFSET ?
                """, (limit, offset))
                
                bots = cursor.fetchall()
                
                return {
                    "success": True,
                    "bots": [dict(bot) for bot in bots],
                    "total_count": len(bots),
                    "limit": limit,
                    "offset": offset
                }
                
        except Exception as e:
            self.logger.error(f"💥 Error listing bots: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _update_bot_metrics(self, **kwargs) -> Dict[str, Any]:
        """Update metrics for a specific bot"""
        try:
            bot_id = kwargs.get("bot_id")
            metrics = kwargs.get("metrics", {})
            
            if not bot_id:
                return {
                    "success": False,
                    "error": "bot_id is required"
                }
            
            if self.registry and hasattr(self.registry, 'update_metrics'):
                return await self.registry.update_metrics(bot_id, metrics)
            else:
                # Fallback implementation
                cursor = self.db_connection.cursor()
                
                # Insert metrics
                for metric_type, metric_value in metrics.items():
                    cursor.execute("""
                        INSERT INTO system_metrics 
                        (bot_id, metric_type, metric_value, spiritual_blessing)
                        VALUES (?, ?, ?, ?)
                    """, (
                        bot_id, 
                        metric_type, 
                        float(metric_value) if isinstance(metric_value, (int, float)) else 0.0,
                        self.metadata.spiritual_blessing
                    ))
                
                self.db_connection.commit()
                
                return {
                    "success": True,
                    "bot_id": bot_id,
                    "metrics_updated": len(metrics),
                    "updated_at": datetime.now().isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"💥 Error updating bot metrics: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _get_performance_report(self, **kwargs) -> Dict[str, Any]:
        """Get performance report for the registry"""
        try:
            if self.registry and hasattr(self.registry, 'get_performance_report'):
                return await self.registry.get_performance_report()
            else:
                # Fallback implementation
                cursor = self.db_connection.cursor()
                
                # Get basic statistics
                cursor.execute("""
                    SELECT 
                        COUNT(DISTINCT bot_id) as unique_bots,
                        COUNT(*) as total_metrics,
                        AVG(metric_value) as avg_metric_value,
                        MAX(timestamp) as last_update
                    FROM system_metrics
                """)
                
                stats = cursor.fetchone()
                
                return {
                    "success": True,
                    "performance_report": dict(stats) if stats else {},
                    "registry_stats": self.registry_stats,
                    "generated_at": datetime.now().isoformat(),
                    "spiritual_blessing": self.metadata.spiritual_blessing
                }
                
        except Exception as e:
            self.logger.error(f"💥 Error getting performance report: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics"""
        try:
            await self._update_stats()
            
            return {
                "success": True,
                "stats": self.registry_stats,
                "plugin_info": {
                    "name": self.metadata.name,
                    "version": self.metadata.version,
                    "status": self.status.value
                },
                "database_info": {
                    "path": self.database_path,
                    "exists": Path(self.database_path).exists(),
                    "size_bytes": Path(self.database_path).stat().st_size if Path(self.database_path).exists() else 0
                },
                "spiritual_blessing": self.metadata.spiritual_blessing
            }
            
        except Exception as e:
            self.logger.error(f"💥 Error getting registry stats: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _search_bots(self, **kwargs) -> Dict[str, Any]:
        """Search for bots based on criteria"""
        try:
            query = kwargs.get("query", "")
            metric_type = kwargs.get("metric_type")
            limit = kwargs.get("limit", 50)
            
            cursor = self.db_connection.cursor()
            
            sql = "SELECT DISTINCT bot_id FROM system_metrics WHERE 1=1"
            params = []
            
            if query:
                sql += " AND bot_id LIKE ?"
                params.append(f"%{query}%")
            
            if metric_type:
                sql += " AND metric_type = ?"
                params.append(metric_type)
            
            sql += " LIMIT ?"
            params.append(limit)
            
            cursor.execute(sql, params)
            results = cursor.fetchall()
            
            return {
                "success": True,
                "results": [dict(result) for result in results],
                "query": query,
                "metric_type": metric_type,
                "count": len(results)
            }
            
        except Exception as e:
            self.logger.error(f"💥 Error searching bots: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _cleanup_old_data(self, **kwargs) -> Dict[str, Any]:
        """Cleanup old data from registry"""
        try:
            days_old = kwargs.get("days_old", 30)
            cutoff_date = datetime.now() - timedelta(days=days_old)
            
            cursor = self.db_connection.cursor()
            
            # Delete old metrics
            cursor.execute("""
                DELETE FROM system_metrics 
                WHERE timestamp < ?
            """, (cutoff_date.isoformat(),))
            
            deleted_count = cursor.rowcount
            self.db_connection.commit()
            
            return {
                "success": True,
                "deleted_records": deleted_count,
                "cutoff_date": cutoff_date.isoformat(),
                "cleaned_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"💥 Error cleaning up old data: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _health_check(self) -> Dict[str, Any]:
        """Perform registry health check"""
        try:
            health_status = {
                "registry_module": self.registry_module is not None,
                "registry_instance": self.registry is not None,
                "database_connection": self.db_connection is not None,
                "database_file_exists": Path(self.database_path).exists(),
                "stats_available": bool(self.registry_stats)
            }
            
            # Test database connection
            try:
                cursor = self.db_connection.cursor()
                cursor.execute("SELECT 1")
                health_status["database_accessible"] = True
            except:
                health_status["database_accessible"] = False
            
            # Check registry health if available
            if self.registry and hasattr(self.registry, 'health_check'):
                registry_health = await self.registry.health_check()
                health_status["registry_health"] = registry_health
            
            all_healthy = all(
                status for key, status in health_status.items() 
                if isinstance(status, bool)
            )
            
            return {
                "success": True,
                "healthy": all_healthy,
                "health_status": health_status,
                "timestamp": datetime.now().isoformat(),
                "spiritual_blessing": self.metadata.spiritual_blessing
            }
            
        except Exception as e:
            self.logger.error(f"💥 Error in health check: {e}")
            return {
                "success": False,
                "healthy": False,
                "error": str(e)
            }
    
    async def _update_stats(self):
        """Update internal statistics"""
        try:
            if Path(self.database_path).exists():
                self.registry_stats["database_size"] = Path(self.database_path).stat().st_size
            
            # Update other stats from database if available
            if self.db_connection:
                cursor = self.db_connection.cursor()
                
                cursor.execute("SELECT COUNT(DISTINCT bot_id) FROM system_metrics")
                result = cursor.fetchone()
                if result:
                    self.registry_stats["bots_registered"] = result[0]
                
        except Exception as e:
            self.logger.error(f"💥 Error updating stats: {e}")
    
    async def cleanup(self) -> bool:
        """Cleanup registry plugin resources"""
        try:
            self.logger.info("🧹 Cleaning up Spiritual Registry Plugin...")
            
            # Cleanup registry if available
            if self.registry and hasattr(self.registry, 'cleanup'):
                await self.registry.cleanup()
            
            # Close database connection
            if self.db_connection:
                self.db_connection.close()
            
            # Clear references
            self.registry = None
            self.registry_module = None
            self.db_connection = None
            self.registry_config = {}
            self.registry_stats = {}
            
            self.logger.info("✅ Spiritual Registry Plugin cleanup completed")
            return True
            
        except Exception as e:
            self.logger.error(f"💥 Error during cleanup: {e}")
            return False

# 🌟 Plugin Factory Function
def create_plugin(config: Dict[str, Any] = None) -> SpiritualRegistryPlugin:
    """
    Factory function to create Spiritual Registry Plugin instance
    
    Args:
        config: Plugin configuration
        
    Returns:
        SpiritualRegistryPlugin instance
    """
    return SpiritualRegistryPlugin(config)

# 🌟 Plugin Registration Helper
def register_plugin(registry):
    """
    Helper function to register this plugin with a registry
    
    Args:
        registry: SpiritualPluginRegistry instance
    """
    registry.register_plugin(
        plugin_name="spiritual-registry",
        plugin_class=SpiritualRegistryPlugin,
        config={
            "registry": {
                "database_path": "spiritual_bot_registry.db",
                "auto_cleanup": True,
                "cleanup_interval_days": 30,
                "health_check_interval": 120
            }
        },
        auto_start=True,
        singleton=True
    )

# 🌟 Spiritual Blessing for Registry Plugin
SPIRITUAL_REGISTRY_BLESSING = """
بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ

اللَّهُمَّ بَارِكْ لَنَا فِي هَذَا السِّجِلِّ الْمُبَارَكِ
وَاجْعَلْهُ حَافِظًا لِبَيَانَاتِ الْبُوتَاتِ الرُّوحِيَّةِ

Ya Allah, berkahilah registry plugin ini dengan:
- 📋 Pencatatan yang akurat dan terpercaya
- 🗄️ Database yang stabil dan aman
- 📊 Monitoring performa yang efektif
- 🔍 Kemampuan pencarian yang cepat

آمِينَ يَا رَبَّ الْعَالَمِينَ
"""

if __name__ == "__main__":
    print("📋 Spiritual Registry Plugin - Ladang Berkah Digital")
    print("=" * 60)
    print(SPIRITUAL_REGISTRY_BLESSING)
    
    # Test plugin creation
    plugin = create_plugin({
        "registry": {
            "database_path": "test_registry.db",
            "auto_cleanup": True
        }
    })
    
    print(f"\n✅ Plugin created: {plugin}")
    print(f"📋 Metadata: {plugin.get_metadata()}")