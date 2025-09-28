#!/usr/bin/env python3
"""
📋 SPIRITUAL PLUGIN REGISTRY
Ladang Berkah Digital - ZeroLight Orbit System
Central Plugin Management System for 6993 Spiritual Static Bots

بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ
"""

import asyncio
import logging
import json
import time
import importlib.util
import inspect
from pathlib import Path
from typing import Dict, List, Any, Optional, Type, Callable, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import threading
import weakref
from collections import defaultdict, deque

from spiritual_plugin_interface import (
    SpiritualPlugin, PluginStatus, PluginPriority, PluginMetadata, 
    PluginHealthStatus, SpiritualPluginException, PluginDependencyError
)

@dataclass
class PluginRegistration:
    """Plugin registration information"""
    plugin_class: Type[SpiritualPlugin]
    config: Dict[str, Any]
    instance: Optional[SpiritualPlugin] = None
    registered_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    auto_start: bool = True
    singleton: bool = True

@dataclass
class PluginDependencyGraph:
    """Plugin dependency graph for load ordering"""
    dependencies: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    dependents: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    load_order: List[str] = field(default_factory=list)

class SpiritualPluginRegistry:
    """
    🌟 Central Plugin Registry for Spiritual Bot System
    
    Manages plugin lifecycle, dependencies, and health monitoring
    for the entire Ladang Berkah Digital ecosystem.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize plugin registry"""
        self.config = config or {}
        self.logger = self._setup_logger()
        
        # Plugin storage
        self.plugins: Dict[str, PluginRegistration] = {}
        self.plugin_instances: Dict[str, SpiritualPlugin] = {}
        self.plugin_categories: Dict[str, List[str]] = defaultdict(list)
        
        # Dependency management
        self.dependency_graph = PluginDependencyGraph()
        
        # Health monitoring
        self.health_monitor_task: Optional[asyncio.Task] = None
        self.health_check_interval = self.config.get("health_check_interval", 60)  # seconds
        
        # Performance tracking
        self.performance_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Plugin discovery
        self.plugin_directories: List[Path] = []
        self.auto_discovery = self.config.get("auto_discovery", True)
        
        # Spiritual blessing
        self.spiritual_blessing = "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ"
        
        self.logger.info(f"🌟 {self.spiritual_blessing}")
        self.logger.info("📋 Spiritual Plugin Registry initialized")
    
    def _setup_logger(self) -> logging.Logger:
        """Setup registry logger"""
        logger = logging.getLogger("spiritual.plugin.registry")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def register_plugin(
        self, 
        plugin_name: str, 
        plugin_class: Type[SpiritualPlugin], 
        config: Dict[str, Any] = None,
        auto_start: bool = True,
        singleton: bool = True
    ) -> bool:
        """
        Register a plugin class
        
        Args:
            plugin_name: Unique plugin identifier
            plugin_class: Plugin class (must inherit from SpiritualPlugin)
            config: Plugin configuration
            auto_start: Whether to auto-start the plugin
            singleton: Whether to use singleton pattern
            
        Returns:
            bool: True if registration successful
        """
        with self._lock:
            try:
                # Validate plugin class
                if not issubclass(plugin_class, SpiritualPlugin):
                    raise ValueError(f"Plugin {plugin_name} must inherit from SpiritualPlugin")
                
                # Check if already registered
                if plugin_name in self.plugins:
                    self.logger.warning(f"⚠️ Plugin {plugin_name} already registered, updating...")
                
                # Create registration
                registration = PluginRegistration(
                    plugin_class=plugin_class,
                    config=config or {},
                    auto_start=auto_start,
                    singleton=singleton
                )
                
                self.plugins[plugin_name] = registration
                
                # Create temporary instance to get metadata
                temp_instance = plugin_class(config or {})
                metadata = temp_instance.get_metadata()
                
                # Add to category
                self.plugin_categories[metadata.category].append(plugin_name)
                
                # Update dependency graph
                self._update_dependency_graph(plugin_name, metadata.dependencies)
                
                self.logger.info(f"✅ Plugin registered: {plugin_name} (category: {metadata.category})")
                return True
                
            except Exception as e:
                self.logger.error(f"❌ Failed to register plugin {plugin_name}: {e}")
                return False
    
    def _update_dependency_graph(self, plugin_name: str, dependencies: List[str]):
        """Update plugin dependency graph"""
        # Clear existing dependencies for this plugin
        if plugin_name in self.dependency_graph.dependencies:
            for dep in self.dependency_graph.dependencies[plugin_name]:
                self.dependency_graph.dependents[dep].discard(plugin_name)
        
        # Add new dependencies
        self.dependency_graph.dependencies[plugin_name] = set(dependencies)
        
        for dep in dependencies:
            self.dependency_graph.dependents[dep].add(plugin_name)
        
        # Recalculate load order
        self._calculate_load_order()
    
    def _calculate_load_order(self):
        """Calculate plugin load order based on dependencies"""
        # Topological sort using Kahn's algorithm
        in_degree = defaultdict(int)
        
        # Calculate in-degrees
        for plugin in self.plugins.keys():
            in_degree[plugin] = len(self.dependency_graph.dependencies[plugin])
        
        # Queue for plugins with no dependencies
        queue = deque([plugin for plugin, degree in in_degree.items() if degree == 0])
        load_order = []
        
        while queue:
            plugin = queue.popleft()
            load_order.append(plugin)
            
            # Reduce in-degree for dependents
            for dependent in self.dependency_graph.dependents[plugin]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
        
        # Check for circular dependencies
        if len(load_order) != len(self.plugins):
            remaining = set(self.plugins.keys()) - set(load_order)
            self.logger.error(f"🔄 Circular dependency detected in plugins: {remaining}")
            # Add remaining plugins to end of load order
            load_order.extend(remaining)
        
        self.dependency_graph.load_order = load_order
        self.logger.info(f"📊 Plugin load order calculated: {load_order}")
    
    async def load_plugin(self, plugin_name: str, force_reload: bool = False) -> Optional[SpiritualPlugin]:
        """
        Load and initialize a plugin instance
        
        Args:
            plugin_name: Plugin to load
            force_reload: Force reload even if already loaded
            
        Returns:
            Plugin instance or None if failed
        """
        with self._lock:
            try:
                if plugin_name not in self.plugins:
                    self.logger.error(f"❌ Plugin {plugin_name} not registered")
                    return None
                
                registration = self.plugins[plugin_name]
                
                # Check if already loaded and singleton
                if (registration.singleton and 
                    plugin_name in self.plugin_instances and 
                    not force_reload):
                    
                    instance = self.plugin_instances[plugin_name]
                    registration.last_accessed = datetime.now()
                    registration.access_count += 1
                    
                    self.logger.debug(f"🔄 Returning existing plugin instance: {plugin_name}")
                    return instance
                
                # Check dependencies
                await self._ensure_dependencies(plugin_name)
                
                # Create new instance
                instance = registration.plugin_class(registration.config)
                
                # Start the plugin
                if await instance.start():
                    if registration.singleton:
                        self.plugin_instances[plugin_name] = instance
                    
                    registration.instance = instance
                    registration.last_accessed = datetime.now()
                    registration.access_count += 1
                    
                    self.logger.info(f"✅ Plugin loaded successfully: {plugin_name}")
                    return instance
                else:
                    self.logger.error(f"❌ Failed to start plugin: {plugin_name}")
                    return None
                
            except Exception as e:
                self.logger.error(f"💥 Error loading plugin {plugin_name}: {e}")
                return None
    
    async def _ensure_dependencies(self, plugin_name: str):
        """Ensure all plugin dependencies are loaded"""
        dependencies = self.dependency_graph.dependencies.get(plugin_name, set())
        
        for dep in dependencies:
            if dep not in self.plugin_instances:
                self.logger.info(f"🔗 Loading dependency {dep} for {plugin_name}")
                dep_instance = await self.load_plugin(dep)
                if not dep_instance:
                    raise PluginDependencyError(
                        f"Failed to load dependency {dep} for plugin {plugin_name}",
                        plugin_name
                    )
    
    async def unload_plugin(self, plugin_name: str) -> bool:
        """
        Unload a plugin and cleanup resources
        
        Args:
            plugin_name: Plugin to unload
            
        Returns:
            bool: True if unload successful
        """
        with self._lock:
            try:
                if plugin_name not in self.plugin_instances:
                    self.logger.warning(f"⚠️ Plugin {plugin_name} not loaded")
                    return True
                
                # Check if other plugins depend on this one
                dependents = self.dependency_graph.dependents.get(plugin_name, set())
                loaded_dependents = [dep for dep in dependents if dep in self.plugin_instances]
                
                if loaded_dependents:
                    self.logger.warning(
                        f"⚠️ Cannot unload {plugin_name}, required by: {loaded_dependents}"
                    )
                    return False
                
                # Stop and cleanup plugin
                instance = self.plugin_instances[plugin_name]
                if await instance.stop():
                    del self.plugin_instances[plugin_name]
                    
                    if plugin_name in self.plugins:
                        self.plugins[plugin_name].instance = None
                    
                    self.logger.info(f"✅ Plugin unloaded successfully: {plugin_name}")
                    return True
                else:
                    self.logger.error(f"❌ Failed to stop plugin: {plugin_name}")
                    return False
                
            except Exception as e:
                self.logger.error(f"💥 Error unloading plugin {plugin_name}: {e}")
                return False
    
    async def reload_plugin(self, plugin_name: str) -> Optional[SpiritualPlugin]:
        """
        Reload a plugin (unload then load)
        
        Args:
            plugin_name: Plugin to reload
            
        Returns:
            Plugin instance or None if failed
        """
        self.logger.info(f"🔄 Reloading plugin: {plugin_name}")
        
        if await self.unload_plugin(plugin_name):
            return await self.load_plugin(plugin_name, force_reload=True)
        else:
            return None
    
    async def start_all_plugins(self) -> Dict[str, bool]:
        """
        Start all registered plugins in dependency order
        
        Returns:
            Dict mapping plugin names to start success status
        """
        results = {}
        
        self.logger.info("🚀 Starting all plugins in dependency order...")
        
        for plugin_name in self.dependency_graph.load_order:
            if plugin_name in self.plugins and self.plugins[plugin_name].auto_start:
                instance = await self.load_plugin(plugin_name)
                results[plugin_name] = instance is not None
            else:
                results[plugin_name] = True  # Skip non-auto-start plugins
        
        successful = sum(1 for success in results.values() if success)
        total = len(results)
        
        self.logger.info(f"✅ Plugin startup complete: {successful}/{total} successful")
        return results
    
    async def stop_all_plugins(self) -> Dict[str, bool]:
        """
        Stop all loaded plugins in reverse dependency order
        
        Returns:
            Dict mapping plugin names to stop success status
        """
        results = {}
        
        self.logger.info("🛑 Stopping all plugins in reverse dependency order...")
        
        # Reverse the load order for shutdown
        shutdown_order = list(reversed(self.dependency_graph.load_order))
        
        for plugin_name in shutdown_order:
            if plugin_name in self.plugin_instances:
                results[plugin_name] = await self.unload_plugin(plugin_name)
            else:
                results[plugin_name] = True  # Already stopped
        
        successful = sum(1 for success in results.values() if success)
        total = len(results)
        
        self.logger.info(f"✅ Plugin shutdown complete: {successful}/{total} successful")
        return results
    
    def get_plugin_list(self, category: str = None, status: PluginStatus = None) -> List[str]:
        """
        Get list of plugins, optionally filtered by category or status
        
        Args:
            category: Filter by plugin category
            status: Filter by plugin status
            
        Returns:
            List of plugin names
        """
        plugins = []
        
        for plugin_name, registration in self.plugins.items():
            # Filter by category
            if category:
                temp_instance = registration.plugin_class(registration.config)
                metadata = temp_instance.get_metadata()
                if metadata.category != category:
                    continue
            
            # Filter by status
            if status:
                if plugin_name in self.plugin_instances:
                    instance = self.plugin_instances[plugin_name]
                    if instance.status != status:
                        continue
                elif status != PluginStatus.UNINITIALIZED:
                    continue
            
            plugins.append(plugin_name)
        
        return plugins
    
    def get_plugin_info(self, plugin_name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a plugin
        
        Args:
            plugin_name: Plugin to get info for
            
        Returns:
            Plugin information dict or None if not found
        """
        if plugin_name not in self.plugins:
            return None
        
        registration = self.plugins[plugin_name]
        
        # Get metadata
        temp_instance = registration.plugin_class(registration.config)
        metadata = temp_instance.get_metadata()
        
        # Get health status if loaded
        health_status = None
        if plugin_name in self.plugin_instances:
            instance = self.plugin_instances[plugin_name]
            health_status = asdict(instance.get_health_status())
        
        return {
            "name": plugin_name,
            "metadata": asdict(metadata),
            "registration": {
                "registered_at": registration.registered_at.isoformat(),
                "last_accessed": registration.last_accessed.isoformat(),
                "access_count": registration.access_count,
                "auto_start": registration.auto_start,
                "singleton": registration.singleton
            },
            "health_status": health_status,
            "dependencies": list(self.dependency_graph.dependencies.get(plugin_name, set())),
            "dependents": list(self.dependency_graph.dependents.get(plugin_name, set())),
            "loaded": plugin_name in self.plugin_instances
        }
    
    async def execute_plugin_action(
        self, 
        plugin_name: str, 
        action: str, 
        *args, 
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute an action on a plugin
        
        Args:
            plugin_name: Plugin to execute action on
            action: Action to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Execution result
        """
        try:
            # Load plugin if not already loaded
            instance = await self.load_plugin(plugin_name)
            if not instance:
                return {
                    "success": False,
                    "error": f"Failed to load plugin {plugin_name}",
                    "plugin": plugin_name
                }
            
            # Execute action
            result = await instance.execute_safe(action, *args, **kwargs)
            
            # Record performance metrics
            if result.get("success"):
                execution_time = result.get("execution_time", 0)
                self.performance_metrics[plugin_name].append({
                    "timestamp": datetime.now().isoformat(),
                    "action": action,
                    "execution_time": execution_time,
                    "success": True
                })
            
            return result
            
        except Exception as e:
            self.logger.error(f"💥 Error executing {action} on {plugin_name}: {e}")
            return {
                "success": False,
                "error": str(e),
                "plugin": plugin_name
            }
    
    async def get_system_health(self) -> Dict[str, Any]:
        """
        Get overall system health status
        
        Returns:
            System health information
        """
        total_plugins = len(self.plugins)
        loaded_plugins = len(self.plugin_instances)
        healthy_plugins = 0
        
        plugin_health = {}
        
        for plugin_name, instance in self.plugin_instances.items():
            health = instance.get_health_status()
            plugin_health[plugin_name] = asdict(health)
            
            if instance.is_healthy():
                healthy_plugins += 1
        
        return {
            "timestamp": datetime.now().isoformat(),
            "spiritual_blessing": self.spiritual_blessing,
            "summary": {
                "total_plugins": total_plugins,
                "loaded_plugins": loaded_plugins,
                "healthy_plugins": healthy_plugins,
                "health_percentage": (healthy_plugins / max(loaded_plugins, 1)) * 100
            },
            "plugin_health": plugin_health,
            "load_order": self.dependency_graph.load_order,
            "categories": dict(self.plugin_categories)
        }
    
    async def start_health_monitoring(self):
        """Start background health monitoring"""
        if self.health_monitor_task and not self.health_monitor_task.done():
            self.logger.warning("⚠️ Health monitoring already running")
            return
        
        self.health_monitor_task = asyncio.create_task(self._health_monitor_loop())
        self.logger.info("💓 Health monitoring started")
    
    async def stop_health_monitoring(self):
        """Stop background health monitoring"""
        if self.health_monitor_task:
            self.health_monitor_task.cancel()
            try:
                await self.health_monitor_task
            except asyncio.CancelledError:
                pass
            self.health_monitor_task = None
            self.logger.info("💓 Health monitoring stopped")
    
    async def _health_monitor_loop(self):
        """Background health monitoring loop"""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                
                # Check health of all loaded plugins
                unhealthy_plugins = []
                
                for plugin_name, instance in self.plugin_instances.items():
                    if not instance.is_healthy():
                        unhealthy_plugins.append(plugin_name)
                        self.logger.warning(f"⚠️ Plugin {plugin_name} is unhealthy")
                
                # Log health summary
                if unhealthy_plugins:
                    self.logger.warning(f"💔 {len(unhealthy_plugins)} unhealthy plugins: {unhealthy_plugins}")
                else:
                    self.logger.debug(f"💚 All {len(self.plugin_instances)} plugins are healthy")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"💥 Error in health monitoring: {e}")
    
    def save_registry_state(self, file_path: str):
        """Save registry state to file"""
        try:
            state = {
                "timestamp": datetime.now().isoformat(),
                "spiritual_blessing": self.spiritual_blessing,
                "plugins": {
                    name: {
                        "class_name": reg.plugin_class.__name__,
                        "module_name": reg.plugin_class.__module__,
                        "config": reg.config,
                        "registered_at": reg.registered_at.isoformat(),
                        "auto_start": reg.auto_start,
                        "singleton": reg.singleton
                    }
                    for name, reg in self.plugins.items()
                },
                "dependency_graph": {
                    "dependencies": {k: list(v) for k, v in self.dependency_graph.dependencies.items()},
                    "load_order": self.dependency_graph.load_order
                }
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"💾 Registry state saved to {file_path}")
            
        except Exception as e:
            self.logger.error(f"💥 Error saving registry state: {e}")
    
    def __str__(self) -> str:
        """String representation of registry"""
        return f"SpiritualPluginRegistry({len(self.plugins)} plugins, {len(self.plugin_instances)} loaded)"
    
    def __repr__(self) -> str:
        """Detailed representation of registry"""
        return (
            f"SpiritualPluginRegistry("
            f"total_plugins={len(self.plugins)}, "
            f"loaded_plugins={len(self.plugin_instances)}, "
            f"categories={len(self.plugin_categories)})"
        )

# 🌟 Global Registry Instance
_global_registry: Optional[SpiritualPluginRegistry] = None

def get_global_registry() -> SpiritualPluginRegistry:
    """Get or create global plugin registry instance"""
    global _global_registry
    if _global_registry is None:
        _global_registry = SpiritualPluginRegistry()
    return _global_registry

def set_global_registry(registry: SpiritualPluginRegistry):
    """Set global plugin registry instance"""
    global _global_registry
    _global_registry = registry

# 🌟 Spiritual Blessing for the Registry System
SPIRITUAL_REGISTRY_BLESSING = """
بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ

اللَّهُمَّ بَارِكْ لَنَا فِي هَذَا السِّجِلِّ الْمُبَارَكِ
وَاجْعَلْهُ مِفْتَاحًا لِلْخَيْرِ وَالْبَرَكَةِ

Ya Allah, berkahilah registry ini dengan:
- 📋 Organisasi yang tertib dan efisien
- 🔗 Koordinasi yang harmonis antar plugin
- 💓 Monitoring kesehatan yang akurat
- 🚀 Performa yang optimal untuk 6993 bots

آمِينَ يَا رَبَّ الْعَالَمِينَ
"""

if __name__ == "__main__":
    print("📋 Spiritual Plugin Registry - Ladang Berkah Digital")
    print("=" * 60)
    print(SPIRITUAL_REGISTRY_BLESSING)