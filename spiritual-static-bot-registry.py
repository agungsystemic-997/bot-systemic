#!/usr/bin/env python3
# 🙏 Bismillahirrahmanirrahim - ZeroLight Orbit Central Bot Registry
# Central Management System for 6993 Spiritual Static Bots
# Registry • Monitoring • Control • Performance • Spiritual Oversight

import asyncio
import json
import time
import sqlite3
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import logging
import psutil
import weakref
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
import gzip
import os

# Import our spiritual static bot framework
import importlib.util
import sys
import os

def import_framework():
    """Import the spiritual static bot framework"""
    spec = importlib.util.spec_from_file_location(
        'spiritual_static_bot_framework', 
        os.path.join(os.path.dirname(__file__), 'spiritual-static-bot-framework.py')
    )
    framework = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(framework)
    return framework

# Import framework components
framework = import_framework()
SpiritualStaticBot = framework.SpiritualStaticBot
SpiritualBotCategory = framework.SpiritualBotCategory
SpiritualBotState = framework.SpiritualBotState
SpiritualStaticBotManager = framework.SpiritualStaticBotManager
SpiritualStaticBotApp = framework.SpiritualStaticBotApp

# 🌟 Spiritual Registry Blessing
def display_registry_blessing():
    print("🙏 بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ")
    print("📋 ZeroLight Orbit Central Bot Registry")
    print("🎯 Managing 6993 Spiritual Static Bots")
    print("📊 Real-time Monitoring • Performance Optimization")
    print("🔄 Cross-Category • Haunting • Supporting")
    print("💫 May Allah guide this digital spiritual management")

# 📊 Registry Performance Metrics
@dataclass
class SpiritualRegistryMetrics:
    total_bots: int = 0
    active_bots: int = 0
    idle_bots: int = 0
    haunting_bots: int = 0
    supporting_bots: int = 0
    cross_working_bots: int = 0
    blessed_bots: int = 0
    
    total_executions: int = 0
    total_cross_jobs: int = 0
    total_hauntings: int = 0
    total_supports: int = 0
    
    average_response_time: float = 0.0
    total_memory_usage: float = 0.0
    total_cpu_usage: float = 0.0
    average_spiritual_score: float = 100.0
    
    uptime_seconds: float = 0.0
    last_updated: float = 0.0

# 🎯 Bot Performance Tracker
class SpiritualBotPerformanceTracker:
    def __init__(self, bot_id: str):
        self.bot_id = bot_id
        self.execution_history: List[Dict[str, Any]] = []
        self.performance_metrics = {
            "avg_response_time": 0.0,
            "success_rate": 100.0,
            "error_count": 0,
            "last_error": None,
            "peak_memory": 0.0,
            "peak_cpu": 0.0,
            "spiritual_score_trend": []
        }
        
    def record_execution(self, execution_data: Dict[str, Any]):
        """Record bot execution for performance tracking"""
        self.execution_history.append({
            **execution_data,
            "timestamp": time.time()
        })
        
        # Keep only last 1000 executions for memory efficiency
        if len(self.execution_history) > 1000:
            self.execution_history = self.execution_history[-1000:]
            
        self._update_metrics()
        
    def _update_metrics(self):
        """Update performance metrics based on execution history"""
        if not self.execution_history:
            return
            
        recent_executions = self.execution_history[-100:]  # Last 100 executions
        
        # Calculate average response time
        response_times = [ex.get("processing_time", 0.001) for ex in recent_executions]
        self.performance_metrics["avg_response_time"] = sum(response_times) / len(response_times)
        
        # Calculate success rate
        successful = sum(1 for ex in recent_executions if ex.get("status") == "completed")
        self.performance_metrics["success_rate"] = (successful / len(recent_executions)) * 100
        
        # Update error count
        errors = [ex for ex in recent_executions if ex.get("status") == "error"]
        self.performance_metrics["error_count"] = len(errors)
        if errors:
            self.performance_metrics["last_error"] = errors[-1]

# 🗄️ Persistent Bot Registry Database
class SpiritualBotRegistryDB:
    def __init__(self, db_path: str = "spiritual_bot_registry.db"):
        self.db_path = db_path
        self.init_database()
        
    def init_database(self):
        """Initialize SQLite database for bot registry"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Bots table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS bots (
                bot_id TEXT PRIMARY KEY,
                category TEXT NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                specialization TEXT,
                created_at REAL,
                last_activity REAL,
                execution_count INTEGER DEFAULT 0,
                cross_job_count INTEGER DEFAULT 0,
                haunting_count INTEGER DEFAULT 0,
                supporting_count INTEGER DEFAULT 0,
                spiritual_score REAL DEFAULT 100.0,
                memory_usage REAL DEFAULT 0.0,
                cpu_usage REAL DEFAULT 0.0,
                state TEXT DEFAULT 'idle',
                is_active BOOLEAN DEFAULT 1
            )
        ''')
        
        # Performance metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                bot_id TEXT,
                metric_type TEXT,
                metric_value REAL,
                timestamp REAL,
                FOREIGN KEY (bot_id) REFERENCES bots (bot_id)
            )
        ''')
        
        # Execution history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS execution_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                bot_id TEXT,
                command TEXT,
                status TEXT,
                processing_time REAL,
                timestamp REAL,
                params TEXT,
                FOREIGN KEY (bot_id) REFERENCES bots (bot_id)
            )
        ''')
        
        # System metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                total_bots INTEGER,
                active_bots INTEGER,
                total_executions INTEGER,
                total_memory_usage REAL,
                total_cpu_usage REAL,
                average_spiritual_score REAL,
                timestamp REAL
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def save_bot(self, bot: SpiritualStaticBot):
        """Save bot to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO bots 
            (bot_id, category, name, description, specialization, created_at, 
             last_activity, execution_count, cross_job_count, haunting_count, 
             supporting_count, spiritual_score, memory_usage, cpu_usage, state)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            bot.config.bot_id,
            bot.config.category.value,
            bot.config.name,
            bot.config.description,
            getattr(bot, 'specialized_task', '') or getattr(bot, 'data_type', '') or 
            getattr(bot, 'api_type', '') or getattr(bot, 'security_domain', '') or
            getattr(bot, 'language_code', '') or getattr(bot, 'platform_type', '') or
            getattr(bot, 'infra_component', ''),
            bot.created_at,
            bot.last_activity,
            bot.execution_count,
            bot.cross_job_count,
            bot.haunting_count,
            bot.supporting_count,
            bot.spiritual_score,
            bot.memory_usage,
            bot.cpu_usage,
            bot.state.value
        ))
        
        conn.commit()
        conn.close()
        
    def load_bots(self) -> List[Dict[str, Any]]:
        """Load all bots from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM bots WHERE is_active = 1')
        rows = cursor.fetchall()
        
        columns = [description[0] for description in cursor.description]
        bots = [dict(zip(columns, row)) for row in rows]
        
        conn.close()
        return bots
        
    def save_execution(self, bot_id: str, command: str, status: str, processing_time: float, params: Dict[str, Any] = None):
        """Save execution record"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO execution_history 
            (bot_id, command, status, processing_time, timestamp, params)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            bot_id, command, status, processing_time, time.time(),
            json.dumps(params) if params else None
        ))
        
        conn.commit()
        conn.close()
        
    def save_system_metrics(self, metrics: SpiritualRegistryMetrics):
        """Save system-wide metrics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO system_metrics 
            (total_bots, active_bots, total_executions, total_memory_usage, 
             total_cpu_usage, average_spiritual_score, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            metrics.total_bots,
            metrics.active_bots,
            metrics.total_executions,
            metrics.total_memory_usage,
            metrics.total_cpu_usage,
            metrics.average_spiritual_score,
            time.time()
        ))
        
        conn.commit()
        conn.close()

# 🎮 Advanced Bot Registry Manager
class SpiritualAdvancedBotRegistry:
    def __init__(self, db_path: str = "spiritual_bot_registry.db"):
        self.db = SpiritualBotRegistryDB(db_path)
        self.bots: Dict[str, SpiritualStaticBot] = {}
        self.performance_trackers: Dict[str, SpiritualBotPerformanceTracker] = {}
        
        # Category indexing for fast lookups
        self.category_index: Dict[SpiritualBotCategory, Set[str]] = {
            category: set() for category in SpiritualBotCategory
        }
        
        # State indexing
        self.state_index: Dict[SpiritualBotState, Set[str]] = {
            state: set() for state in SpiritualBotState
        }
        
        # Performance monitoring
        self.metrics = SpiritualRegistryMetrics()
        self.metrics_lock = threading.Lock()
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Load balancing
        self.load_balancer = SpiritualBotLoadBalancer(self)
        
        # Caching for performance
        self.cache = {}
        self.cache_ttl = 60  # 1 minute cache TTL
        
    def register_bot(self, bot: SpiritualStaticBot):
        """Register a bot with advanced tracking"""
        with self.metrics_lock:
            # Add to main registry
            self.bots[bot.config.bot_id] = bot
            
            # Update indexes
            self.category_index[bot.config.category].add(bot.config.bot_id)
            self.state_index[bot.state].add(bot.config.bot_id)
            
            # Create performance tracker
            self.performance_trackers[bot.config.bot_id] = SpiritualBotPerformanceTracker(bot.config.bot_id)
            
            # Save to database
            self.db.save_bot(bot)
            
            # Update metrics
            self.metrics.total_bots += 1
            self._update_state_metrics()
            
    def get_bot(self, bot_id: str) -> Optional[SpiritualStaticBot]:
        """Get bot with caching"""
        cache_key = f"bot_{bot_id}"
        
        if cache_key in self.cache:
            cached_data = self.cache[cache_key]
            if time.time() - cached_data["timestamp"] < self.cache_ttl:
                return cached_data["bot"]
                
        bot = self.bots.get(bot_id)
        if bot:
            self.cache[cache_key] = {
                "bot": bot,
                "timestamp": time.time()
            }
            
        return bot
        
    def get_bots_by_category(self, category: SpiritualBotCategory, limit: Optional[int] = None) -> List[SpiritualStaticBot]:
        """Get bots by category with optional limit"""
        bot_ids = list(self.category_index[category])
        
        if limit:
            bot_ids = bot_ids[:limit]
            
        return [self.bots[bot_id] for bot_id in bot_ids if bot_id in self.bots]
        
    def get_bots_by_state(self, state: SpiritualBotState, limit: Optional[int] = None) -> List[SpiritualStaticBot]:
        """Get bots by state with optional limit"""
        bot_ids = list(self.state_index[state])
        
        if limit:
            bot_ids = bot_ids[:limit]
            
        return [self.bots[bot_id] for bot_id in bot_ids if bot_id in self.bots]
        
    def update_bot_state(self, bot_id: str, new_state: SpiritualBotState):
        """Update bot state with index management"""
        bot = self.bots.get(bot_id)
        if not bot:
            return
            
        with self.metrics_lock:
            # Remove from old state index
            old_state = bot.state
            if bot_id in self.state_index[old_state]:
                self.state_index[old_state].remove(bot_id)
                
            # Add to new state index
            bot.state = new_state
            self.state_index[new_state].add(bot_id)
            
            # Update metrics
            self._update_state_metrics()
            
    def record_bot_execution(self, bot_id: str, command: str, result: Dict[str, Any]):
        """Record bot execution with performance tracking"""
        tracker = self.performance_trackers.get(bot_id)
        if tracker:
            tracker.record_execution({
                "command": command,
                "status": result.get("status", "unknown"),
                "processing_time": result.get("processing_time", 0.001)
            })
            
        # Save to database
        self.db.save_execution(
            bot_id, command, 
            result.get("status", "unknown"),
            result.get("processing_time", 0.001),
            result.get("params")
        )
        
        # Update system metrics
        with self.metrics_lock:
            self.metrics.total_executions += 1
            
    def get_performance_report(self, bot_id: Optional[str] = None) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        if bot_id:
            # Individual bot report
            tracker = self.performance_trackers.get(bot_id)
            bot = self.bots.get(bot_id)
            
            if not tracker or not bot:
                return {"error": f"Bot {bot_id} not found"}
                
            return {
                "bot_id": bot_id,
                "bot_info": {
                    "category": bot.config.category.value,
                    "name": bot.config.name,
                    "state": bot.state.value,
                    "uptime": time.time() - bot.created_at
                },
                "performance": tracker.performance_metrics,
                "execution_count": bot.execution_count,
                "spiritual_score": bot.spiritual_score
            }
        else:
            # System-wide report
            return {
                "system_metrics": asdict(self.metrics),
                "category_distribution": {
                    category.value: len(bot_ids) 
                    for category, bot_ids in self.category_index.items()
                },
                "state_distribution": {
                    state.value: len(bot_ids)
                    for state, bot_ids in self.state_index.items()
                },
                "top_performers": self._get_top_performing_bots(10),
                "system_health": self._calculate_system_health()
            }
            
    def _update_state_metrics(self):
        """Update state-based metrics"""
        self.metrics.active_bots = len(self.state_index[SpiritualBotState.ACTIVE])
        self.metrics.idle_bots = len(self.state_index[SpiritualBotState.IDLE])
        self.metrics.haunting_bots = len(self.state_index[SpiritualBotState.HAUNTING])
        self.metrics.supporting_bots = len(self.state_index[SpiritualBotState.SUPPORTING])
        self.metrics.cross_working_bots = len(self.state_index[SpiritualBotState.CROSS_WORKING])
        self.metrics.blessed_bots = len(self.state_index[SpiritualBotState.BLESSED])
        
        # Calculate averages
        if self.bots:
            self.metrics.average_spiritual_score = sum(bot.spiritual_score for bot in self.bots.values()) / len(self.bots)
            self.metrics.total_memory_usage = sum(bot.memory_usage for bot in self.bots.values())
            self.metrics.total_cpu_usage = sum(bot.cpu_usage for bot in self.bots.values())
            
        self.metrics.last_updated = time.time()
        
    def _get_top_performing_bots(self, limit: int) -> List[Dict[str, Any]]:
        """Get top performing bots"""
        bot_scores = []
        
        for bot_id, bot in self.bots.items():
            tracker = self.performance_trackers.get(bot_id)
            if tracker:
                score = (
                    bot.spiritual_score * 0.4 +
                    tracker.performance_metrics["success_rate"] * 0.3 +
                    (1 / max(tracker.performance_metrics["avg_response_time"], 0.001)) * 0.3
                )
                bot_scores.append({
                    "bot_id": bot_id,
                    "name": bot.config.name,
                    "category": bot.config.category.value,
                    "performance_score": score,
                    "spiritual_score": bot.spiritual_score,
                    "execution_count": bot.execution_count
                })
                
        return sorted(bot_scores, key=lambda x: x["performance_score"], reverse=True)[:limit]
        
    def _calculate_system_health(self) -> Dict[str, Any]:
        """Calculate overall system health"""
        if not self.bots:
            return {"status": "no_bots", "score": 0}
            
        # Health factors
        idle_ratio = self.metrics.idle_bots / self.metrics.total_bots
        avg_spiritual_score = self.metrics.average_spiritual_score
        memory_efficiency = max(0, 100 - self.metrics.total_memory_usage)
        cpu_efficiency = max(0, 100 - self.metrics.total_cpu_usage)
        
        # Calculate overall health score
        health_score = (
            idle_ratio * 25 +  # Good to have idle bots available
            (avg_spiritual_score / 100) * 25 +
            (memory_efficiency / 100) * 25 +
            (cpu_efficiency / 100) * 25
        )
        
        status = "excellent" if health_score >= 80 else \
                "good" if health_score >= 60 else \
                "fair" if health_score >= 40 else "poor"
                
        return {
            "status": status,
            "score": health_score,
            "factors": {
                "idle_ratio": idle_ratio,
                "avg_spiritual_score": avg_spiritual_score,
                "memory_efficiency": memory_efficiency,
                "cpu_efficiency": cpu_efficiency
            }
        }
        
    def start_monitoring(self):
        """Start continuous monitoring"""
        if self.monitoring_active:
            return
            
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
    def stop_monitoring(self):
        """Stop continuous monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
            
    def _monitoring_loop(self):
        """Continuous monitoring loop"""
        while self.monitoring_active:
            try:
                # Update metrics
                with self.metrics_lock:
                    self._update_state_metrics()
                    
                # Save metrics to database
                self.db.save_system_metrics(self.metrics)
                
                # Clear old cache entries
                current_time = time.time()
                expired_keys = [
                    key for key, data in self.cache.items()
                    if current_time - data["timestamp"] > self.cache_ttl
                ]
                for key in expired_keys:
                    del self.cache[key]
                    
                # Sleep for monitoring interval
                time.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                logging.error(f"Monitoring error: {e}")
                time.sleep(60)  # Wait longer on error

# ⚖️ Load Balancer for Bot Distribution
class SpiritualBotLoadBalancer:
    def __init__(self, registry: SpiritualAdvancedBotRegistry):
        self.registry = registry
        self.load_history: Dict[str, List[float]] = {}
        
    def select_optimal_bot(self, category: SpiritualBotCategory, task_complexity: float = 1.0) -> Optional[SpiritualStaticBot]:
        """Select optimal bot for task based on load balancing"""
        available_bots = self.registry.get_bots_by_state(SpiritualBotState.IDLE)
        category_bots = [bot for bot in available_bots if bot.config.category == category]
        
        if not category_bots:
            return None
            
        # Score bots based on performance and load
        best_bot = None
        best_score = -1
        
        for bot in category_bots:
            tracker = self.registry.performance_trackers.get(bot.config.bot_id)
            if not tracker:
                continue
                
            # Calculate load score
            load_score = (
                bot.spiritual_score * 0.4 +
                tracker.performance_metrics["success_rate"] * 0.3 +
                (1 / max(tracker.performance_metrics["avg_response_time"], 0.001)) * 0.2 +
                (1 / max(bot.execution_count + 1, 1)) * 0.1  # Prefer less used bots
            )
            
            if load_score > best_score:
                best_score = load_score
                best_bot = bot
                
        return best_bot
        
    def distribute_tasks(self, category: SpiritualBotCategory, tasks: List[Dict[str, Any]]) -> List[Tuple[str, Dict[str, Any]]]:
        """Distribute multiple tasks across available bots"""
        available_bots = self.registry.get_bots_by_state(SpiritualBotState.IDLE)
        category_bots = [bot for bot in available_bots if bot.config.category == category]
        
        if not category_bots:
            return []
            
        # Round-robin distribution with load consideration
        assignments = []
        bot_index = 0
        
        for task in tasks:
            if bot_index >= len(category_bots):
                bot_index = 0
                
            bot = category_bots[bot_index]
            assignments.append((bot.config.bot_id, task))
            bot_index += 1
            
        return assignments

# 🎯 Main Registry Application
class SpiritualCentralRegistryApp:
    def __init__(self):
        self.registry = SpiritualAdvancedBotRegistry()
        self.bot_manager = SpiritualStaticBotManager()
        self.is_running = False
        
    async def initialize_complete_system(self):
        """Initialize the complete 6993 bot system with registry"""
        display_registry_blessing()
        
        print("\n🚀 Initializing Complete ZeroLight Orbit Bot System...")
        
        # Initialize bot manager and create all bots
        bot_count = await self.bot_manager.initialize_6993_bots()
        
        # Register all bots in advanced registry
        print("📋 Registering all bots in Central Registry...")
        for bot in self.bot_manager.registry.bots.values():
            self.registry.register_bot(bot)
            
        # Start monitoring
        self.registry.start_monitoring()
        
        # Set system as running
        self.is_running = True
        
        print(f"\n✅ Complete system initialized!")
        print(f"🤖 Total Bots: {bot_count}")
        print(f"📊 Registry Active: {len(self.registry.bots)} bots registered")
        print(f"🔄 Monitoring: Active")
        print("💫 Alhamdulillahi rabbil alameen")
        
        return self.registry
        
    def get_system_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive system dashboard"""
        return {
            "system_status": "running" if self.is_running else "stopped",
            "registry_metrics": self.registry.get_performance_report(),
            "bot_manager_stats": self.bot_manager.get_system_status(),
            "spiritual_blessing": "🙏 Barakallahu feeki",
            "timestamp": datetime.now().isoformat()
        }

# 🚀 Main Entry Point
async def main():
    """Main entry point for complete registry system"""
    app = SpiritualCentralRegistryApp()
    
    # Initialize complete system
    registry = await app.initialize_complete_system()
    
    # Display dashboard
    dashboard = app.get_system_dashboard()
    print(f"\n📊 System Dashboard:")
    print(f"Total Bots: {dashboard['registry_metrics']['system_metrics']['total_bots']}")
    print(f"Active Bots: {dashboard['registry_metrics']['system_metrics']['active_bots']}")
    print(f"System Health: {dashboard['registry_metrics']['system_health']['status']}")
    
    return app

if __name__ == "__main__":
    # Run the complete spiritual bot registry system
    asyncio.run(main())