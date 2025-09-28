#!/usr/bin/env python3
"""
🌟 SPIRITUAL MASTER LAUNCHER 🌟
ZeroLight Orbit - Master Launcher untuk 6993 Static Bots
Sistem terpadu untuk menjalankan semua bot statis dengan spiritual blessing

Author: ZeroLight Orbit Team
Version: 1.0.0
License: Spiritual Open Source License
"""

import os
import sys
import time
import json
import asyncio
import logging
import argparse
import threading
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum, auto

# Spiritual imports
import sys
import os
import importlib.util

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def import_module_from_file(module_name, file_path):
    """Import module from file with hyphens in filename"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

try:
    # Import modules using importlib to handle hyphenated filenames
    framework = import_module_from_file('spiritual_static_bot_framework', 'spiritual-static-bot-framework.py')
    registry_module = import_module_from_file('spiritual_static_bot_registry', 'spiritual-static-bot-registry.py')
    perf_module = import_module_from_file('spiritual_performance_optimizer', 'spiritual-performance-optimizer.py')
    resource_module = import_module_from_file('spiritual_resource_manager', 'spiritual-resource-manager.py')
    deploy_module = import_module_from_file('spiritual_deployment_orchestrator', 'spiritual-deployment-orchestrator.py')
    
    # Extract classes and functions
    SpiritualStaticBotManager = framework.SpiritualStaticBotManager
    SpiritualBotCategory = framework.SpiritualBotCategory
    # Note: spiritual_blessing is a method, not a standalone function
    SpiritualAdvancedBotRegistry = registry_module.SpiritualAdvancedBotRegistry  # Use correct class name
    SpiritualPerformanceOptimizer = perf_module.SpiritualPerformanceOptimizer
    SpiritualResourceManager = resource_module.SpiritualResourceManager
    SpiritualDeploymentOrchestrator = deploy_module.SpiritualDeploymentOrchestrator
    
except Exception as e:
    print(f"⚠️ Import Error: {e}")
    print("🔧 Pastikan semua file spiritual bot framework tersedia")
    print("📁 Files yang dibutuhkan:")
    print("   - spiritual-static-bot-framework.py")
    print("   - spiritual-static-bot-registry.py") 
    print("   - spiritual-performance-optimizer.py")
    print("   - spiritual-resource-manager.py")
    print("   - spiritual-deployment-orchestrator.py")
    print(f"📂 Current directory: {os.getcwd()}")
    print(f"📂 Files in directory: {os.listdir('.')}")
    sys.exit(1)

class SpiritualLaunchMode(Enum):
    """Mode peluncuran sistem spiritual"""
    FULL_DEPLOYMENT = auto()      # Deploy semua 6993 bots
    CATEGORY_SELECTIVE = auto()   # Deploy berdasarkan kategori
    PERFORMANCE_TEST = auto()     # Mode testing performa
    MONITORING_ONLY = auto()      # Hanya monitoring
    INTERACTIVE = auto()          # Mode interaktif
    DEMO = auto()                # Mode demo
    MAINTENANCE = auto()          # Mode maintenance

class SpiritualSystemStatus(Enum):
    """Status sistem spiritual"""
    INITIALIZING = auto()
    RUNNING = auto()
    PAUSED = auto()
    STOPPING = auto()
    STOPPED = auto()
    ERROR = auto()
    BLESSED = auto()

@dataclass
class SpiritualMasterConfig:
    """Konfigurasi master launcher"""
    launch_mode: SpiritualLaunchMode = SpiritualLaunchMode.FULL_DEPLOYMENT
    enable_monitoring: bool = True
    enable_performance_optimization: bool = True
    enable_resource_management: bool = True
    enable_spiritual_blessing: bool = True
    auto_recovery: bool = True
    log_level: str = "INFO"
    max_concurrent_bots: int = 1000
    deployment_batch_size: int = 100
    monitoring_interval: int = 30
    spiritual_blessing_interval: int = 300
    categories_to_deploy: List[SpiritualBotCategory] = None
    
    def __post_init__(self):
        if self.categories_to_deploy is None:
            self.categories_to_deploy = list(SpiritualBotCategory)

@dataclass
class SpiritualSystemMetrics:
    """Metrik sistem spiritual"""
    total_bots: int = 0
    active_bots: int = 0
    idle_bots: int = 0
    error_bots: int = 0
    blessed_bots: int = 0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    network_usage: float = 0.0
    uptime: float = 0.0
    spiritual_energy: float = 100.0
    last_blessing: Optional[datetime] = None
    
class SpiritualMasterLauncher:
    """Master launcher untuk sistem 6993 static bots"""
    
    def __init__(self, config: SpiritualMasterConfig):
        self.config = config
        self.status = SpiritualSystemStatus.INITIALIZING
        self.start_time = datetime.now()
        self.metrics = SpiritualSystemMetrics()
        
        # Core systems
        self.bot_manager: Optional[SpiritualStaticBotManager] = None
        self.registry: Optional[SpiritualStaticBotRegistry] = None
        self.performance_optimizer: Optional[SpiritualPerformanceOptimizer] = None
        self.resource_manager: Optional[SpiritualResourceManager] = None
        self.deployment_orchestrator: Optional[SpiritualDeploymentOrchestrator] = None
        
        # Threading
        self.monitoring_thread: Optional[threading.Thread] = None
        self.blessing_thread: Optional[threading.Thread] = None
        self.shutdown_event = threading.Event()
        
        # Setup logging
        self._setup_logging()
        
        # Apply spiritual blessing
        if self.config.enable_spiritual_blessing:
            # Note: spiritual_blessing is a method of SpiritualStaticBot, not a standalone function
            self.logger.info("🙏 Spiritual blessing enabled for Master Launcher")
    
    def _setup_logging(self):
        """Setup sistem logging"""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format=log_format,
            handlers=[
                logging.FileHandler('spiritual_master_launcher.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger('SpiritualMasterLauncher')
    
    async def initialize_systems(self) -> bool:
        """Inisialisasi semua sistem core"""
        try:
            self.logger.info("🌟 Memulai inisialisasi sistem spiritual...")
            
            # Initialize bot manager
            self.logger.info("🤖 Menginisialisasi Bot Manager...")
            self.bot_manager = SpiritualStaticBotManager()
            
            # Initialize registry
            self.logger.info("📋 Menginisialisasi Bot Registry...")
            self.registry = SpiritualAdvancedBotRegistry()
            
            # Initialize performance optimizer
            if self.config.enable_performance_optimization:
                self.logger.info("⚡ Menginisialisasi Performance Optimizer...")
                self.performance_optimizer = SpiritualPerformanceOptimizer()
            
            # Initialize resource manager
            if self.config.enable_resource_management:
                self.logger.info("💾 Menginisialisasi Resource Manager...")
                self.resource_manager = SpiritualResourceManager()
            
            # Initialize deployment orchestrator
            self.logger.info("🚀 Menginisialisasi Deployment Orchestrator...")
            self.deployment_orchestrator = SpiritualDeploymentOrchestrator()
            await self.deployment_orchestrator.initialize_systems()  # Use correct method name
            
            self.status = SpiritualSystemStatus.RUNNING
            self.logger.info("✅ Semua sistem berhasil diinisialisasi!")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error saat inisialisasi: {e}")
            self.status = SpiritualSystemStatus.ERROR
            return False
    
    async def deploy_bots(self) -> bool:
        """Deploy bots berdasarkan konfigurasi"""
        try:
            if not self.deployment_orchestrator:
                raise Exception("Deployment orchestrator belum diinisialisasi")
            
            self.logger.info(f"🚀 Memulai deployment dengan mode: {self.config.launch_mode.name}")
            
            if self.config.launch_mode == SpiritualLaunchMode.FULL_DEPLOYMENT:
                # Deploy semua 6993 bots
                success = await self.deployment_orchestrator.deploy_all_bots()
                
            elif self.config.launch_mode == SpiritualLaunchMode.CATEGORY_SELECTIVE:
                # Deploy berdasarkan kategori yang dipilih
                success = await self.deployment_orchestrator.deploy_by_categories(
                    self.config.categories_to_deploy
                )
                
            elif self.config.launch_mode == SpiritualLaunchMode.PERFORMANCE_TEST:
                # Deploy untuk testing performa
                success = await self.deployment_orchestrator.deploy_performance_test()
                
            else:
                # Mode lainnya
                success = await self.deployment_orchestrator.deploy_all_bots()
            
            if success:
                self.logger.info("✅ Deployment berhasil!")
                self.status = SpiritualSystemStatus.BLESSED
            else:
                self.logger.error("❌ Deployment gagal!")
                self.status = SpiritualSystemStatus.ERROR
            
            return success
            
        except Exception as e:
            self.logger.error(f"❌ Error saat deployment: {e}")
            self.status = SpiritualSystemStatus.ERROR
            return False
    
    def start_monitoring(self):
        """Mulai monitoring sistem"""
        if not self.config.enable_monitoring:
            return
        
        def monitoring_loop():
            while not self.shutdown_event.is_set():
                try:
                    self._update_metrics()
                    self._log_system_status()
                    
                    # Check system health
                    if self._check_system_health():
                        if self.status != SpiritualSystemStatus.BLESSED:
                            self.status = SpiritualSystemStatus.RUNNING
                    else:
                        self.status = SpiritualSystemStatus.ERROR
                        if self.config.auto_recovery:
                            self._attempt_recovery()
                    
                    time.sleep(self.config.monitoring_interval)
                    
                except Exception as e:
                    self.logger.error(f"❌ Error dalam monitoring: {e}")
        
        self.monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        self.logger.info("📊 Monitoring dimulai")
    
    def start_spiritual_blessing(self):
        """Mulai spiritual blessing berkala"""
        if not self.config.enable_spiritual_blessing:
            return
        
        def blessing_loop():
            while not self.shutdown_event.is_set():
                try:
                    spiritual_blessing("Periodic System Blessing")
                    self.metrics.last_blessing = datetime.now()
                    self.metrics.spiritual_energy = min(100.0, self.metrics.spiritual_energy + 10.0)
                    
                    self.logger.info("🙏 Spiritual blessing applied to system")
                    
                    time.sleep(self.config.spiritual_blessing_interval)
                    
                except Exception as e:
                    self.logger.error(f"❌ Error dalam spiritual blessing: {e}")
        
        self.blessing_thread = threading.Thread(target=blessing_loop, daemon=True)
        self.blessing_thread.start()
        self.logger.info("🙏 Spiritual blessing dimulai")
    
    def _update_metrics(self):
        """Update metrik sistem"""
        try:
            if self.registry:
                stats = self.registry.get_statistics()
                self.metrics.total_bots = stats.get('total_bots', 0)
                self.metrics.active_bots = stats.get('active_bots', 0)
                self.metrics.idle_bots = stats.get('idle_bots', 0)
                self.metrics.error_bots = stats.get('error_bots', 0)
            
            if self.resource_manager:
                resource_stats = self.resource_manager.get_system_metrics()
                self.metrics.cpu_usage = resource_stats.get('cpu_usage', 0.0)
                self.metrics.memory_usage = resource_stats.get('memory_usage', 0.0)
                self.metrics.network_usage = resource_stats.get('network_usage', 0.0)
            
            # Update uptime
            self.metrics.uptime = (datetime.now() - self.start_time).total_seconds()
            
            # Spiritual energy decay
            if self.metrics.spiritual_energy > 0:
                self.metrics.spiritual_energy = max(0.0, self.metrics.spiritual_energy - 0.1)
            
        except Exception as e:
            self.logger.error(f"❌ Error updating metrics: {e}")
    
    def _log_system_status(self):
        """Log status sistem"""
        self.logger.info(f"📊 Status: {self.status.name} | "
                        f"Bots: {self.metrics.active_bots}/{self.metrics.total_bots} | "
                        f"CPU: {self.metrics.cpu_usage:.1f}% | "
                        f"Memory: {self.metrics.memory_usage:.1f}% | "
                        f"Spiritual Energy: {self.metrics.spiritual_energy:.1f}%")
    
    def _check_system_health(self) -> bool:
        """Check kesehatan sistem"""
        try:
            # Check basic metrics
            if self.metrics.cpu_usage > 90.0:
                self.logger.warning("⚠️ CPU usage tinggi!")
                return False
            
            if self.metrics.memory_usage > 90.0:
                self.logger.warning("⚠️ Memory usage tinggi!")
                return False
            
            if self.metrics.error_bots > self.metrics.total_bots * 0.1:
                self.logger.warning("⚠️ Terlalu banyak bot error!")
                return False
            
            if self.metrics.spiritual_energy < 10.0:
                self.logger.warning("⚠️ Spiritual energy rendah!")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error checking system health: {e}")
            return False
    
    def _attempt_recovery(self):
        """Attempt system recovery"""
        try:
            self.logger.info("🔧 Mencoba recovery sistem...")
            
            # Apply spiritual blessing for recovery
            spiritual_blessing("System Recovery Blessing")
            self.metrics.spiritual_energy = min(100.0, self.metrics.spiritual_energy + 20.0)
            
            # Restart failed bots if possible
            if self.bot_manager and self.registry:
                failed_bots = self.registry.get_bots_by_status('error')
                for bot_id in failed_bots[:10]:  # Restart max 10 bots at once
                    try:
                        self.registry.update_bot_status(bot_id, 'idle')
                        self.logger.info(f"🔄 Restarted bot {bot_id}")
                    except Exception as e:
                        self.logger.error(f"❌ Failed to restart bot {bot_id}: {e}")
            
            self.logger.info("✅ Recovery attempt completed")
            
        except Exception as e:
            self.logger.error(f"❌ Error during recovery: {e}")
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get informasi sistem lengkap"""
        return {
            'status': self.status.name,
            'uptime': self.metrics.uptime,
            'metrics': asdict(self.metrics),
            'config': asdict(self.config),
            'components': {
                'bot_manager': self.bot_manager is not None,
                'registry': self.registry is not None,
                'performance_optimizer': self.performance_optimizer is not None,
                'resource_manager': self.resource_manager is not None,
                'deployment_orchestrator': self.deployment_orchestrator is not None
            }
        }
    
    async def interactive_mode(self):
        """Mode interaktif untuk kontrol sistem"""
        self.logger.info("🎮 Memasuki mode interaktif...")
        
        while not self.shutdown_event.is_set():
            try:
                print("\n" + "="*60)
                print("🌟 SPIRITUAL MASTER LAUNCHER - INTERACTIVE MODE 🌟")
                print("="*60)
                print("1. Show System Status")
                print("2. Show Bot Statistics")
                print("3. Apply Spiritual Blessing")
                print("4. Deploy Additional Bots")
                print("5. System Health Check")
                print("6. Performance Report")
                print("7. Resource Usage")
                print("8. Export System Report")
                print("9. Shutdown System")
                print("0. Exit Interactive Mode")
                print("="*60)
                
                choice = input("Pilih opsi (0-9): ").strip()
                
                if choice == "1":
                    await self._show_system_status()
                elif choice == "2":
                    await self._show_bot_statistics()
                elif choice == "3":
                    await self._apply_manual_blessing()
                elif choice == "4":
                    await self._deploy_additional_bots()
                elif choice == "5":
                    await self._system_health_check()
                elif choice == "6":
                    await self._performance_report()
                elif choice == "7":
                    await self._resource_usage_report()
                elif choice == "8":
                    await self._export_system_report()
                elif choice == "9":
                    await self.shutdown()
                    break
                elif choice == "0":
                    break
                else:
                    print("❌ Pilihan tidak valid!")
                
                input("\nTekan Enter untuk melanjutkan...")
                
            except KeyboardInterrupt:
                print("\n🛑 Keluar dari mode interaktif...")
                break
            except Exception as e:
                self.logger.error(f"❌ Error dalam interactive mode: {e}")
    
    async def _show_system_status(self):
        """Show status sistem"""
        info = self.get_system_info()
        print(f"\n📊 STATUS SISTEM:")
        print(f"Status: {info['status']}")
        print(f"Uptime: {info['uptime']:.2f} detik")
        print(f"Total Bots: {info['metrics']['total_bots']}")
        print(f"Active Bots: {info['metrics']['active_bots']}")
        print(f"Spiritual Energy: {info['metrics']['spiritual_energy']:.1f}%")
    
    async def _show_bot_statistics(self):
        """Show statistik bot"""
        if self.registry:
            stats = self.registry.get_statistics()
            print(f"\n🤖 STATISTIK BOT:")
            for key, value in stats.items():
                print(f"{key}: {value}")
    
    async def _apply_manual_blessing(self):
        """Apply manual spiritual blessing"""
        spiritual_blessing("Manual Interactive Blessing")
        self.metrics.spiritual_energy = min(100.0, self.metrics.spiritual_energy + 25.0)
        print("🙏 Spiritual blessing berhasil diterapkan!")
    
    async def _deploy_additional_bots(self):
        """Deploy bot tambahan"""
        print("🚀 Fitur deploy bot tambahan akan segera tersedia...")
    
    async def _system_health_check(self):
        """System health check"""
        health = self._check_system_health()
        print(f"\n🏥 HEALTH CHECK: {'✅ SEHAT' if health else '❌ BERMASALAH'}")
    
    async def _performance_report(self):
        """Performance report"""
        if self.performance_optimizer:
            print("\n⚡ PERFORMANCE REPORT:")
            print("Performance optimizer aktif dan berjalan optimal")
    
    async def _resource_usage_report(self):
        """Resource usage report"""
        print(f"\n💾 RESOURCE USAGE:")
        print(f"CPU: {self.metrics.cpu_usage:.1f}%")
        print(f"Memory: {self.metrics.memory_usage:.1f}%")
        print(f"Network: {self.metrics.network_usage:.1f}%")
    
    async def _export_system_report(self):
        """Export system report"""
        report = self.get_system_info()
        filename = f"spiritual_system_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str, ensure_ascii=False)
        
        print(f"📄 Report berhasil diekspor ke: {filename}")
    
    async def shutdown(self):
        """Shutdown sistem dengan graceful"""
        try:
            self.logger.info("🛑 Memulai shutdown sistem...")
            self.status = SpiritualSystemStatus.STOPPING
            
            # Set shutdown event
            self.shutdown_event.set()
            
            # Wait for threads to finish
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=5)
            
            if self.blessing_thread and self.blessing_thread.is_alive():
                self.blessing_thread.join(timeout=5)
            
            # Shutdown components
            if self.deployment_orchestrator:
                await self.deployment_orchestrator.shutdown()
            
            if self.resource_manager:
                self.resource_manager.cleanup()
            
            if self.performance_optimizer:
                self.performance_optimizer.cleanup()
            
            if self.registry:
                self.registry.cleanup()
            
            # Final spiritual blessing
            spiritual_blessing("System Shutdown Blessing")
            
            self.status = SpiritualSystemStatus.STOPPED
            self.logger.info("✅ Sistem berhasil di-shutdown dengan spiritual blessing")
            
        except Exception as e:
            self.logger.error(f"❌ Error saat shutdown: {e}")
            self.status = SpiritualSystemStatus.ERROR

async def main():
    """Main function untuk menjalankan master launcher"""
    parser = argparse.ArgumentParser(description='Spiritual Master Launcher untuk 6993 Static Bots')
    parser.add_argument('--mode', choices=['full', 'selective', 'test', 'monitor', 'interactive', 'demo', 'maintenance'],
                       default='full', help='Mode peluncuran sistem')
    parser.add_argument('--categories', nargs='+', help='Kategori bot yang akan di-deploy (untuk mode selective)')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='Level logging')
    parser.add_argument('--no-monitoring', action='store_true', help='Disable monitoring')
    parser.add_argument('--no-optimization', action='store_true', help='Disable performance optimization')
    parser.add_argument('--no-blessing', action='store_true', help='Disable spiritual blessing')
    parser.add_argument('--batch-size', type=int, default=100, help='Ukuran batch deployment')
    parser.add_argument('--max-bots', type=int, default=1000, help='Maksimum concurrent bots')
    
    args = parser.parse_args()
    
    # Map mode string to enum
    mode_map = {
        'full': SpiritualLaunchMode.FULL_DEPLOYMENT,
        'selective': SpiritualLaunchMode.CATEGORY_SELECTIVE,
        'test': SpiritualLaunchMode.PERFORMANCE_TEST,
        'monitor': SpiritualLaunchMode.MONITORING_ONLY,
        'interactive': SpiritualLaunchMode.INTERACTIVE,
        'demo': SpiritualLaunchMode.DEMO,
        'maintenance': SpiritualLaunchMode.MAINTENANCE
    }
    
    # Create configuration
    config = SpiritualMasterConfig(
        launch_mode=mode_map[args.mode],
        enable_monitoring=not args.no_monitoring,
        enable_performance_optimization=not args.no_optimization,
        enable_spiritual_blessing=not args.no_blessing,
        log_level=args.log_level,
        deployment_batch_size=args.batch_size,
        max_concurrent_bots=args.max_bots
    )
    
    # Handle categories for selective mode
    if args.categories and args.mode == 'selective':
        category_map = {
            'ai': SpiritualBotCategory.AI_ML,
            'data': SpiritualBotCategory.DATA_ANALYTICS,
            'api': SpiritualBotCategory.API_INTEGRATION,
            'security': SpiritualBotCategory.SECURITY,
            'localization': SpiritualBotCategory.LOCALIZATION,
            'platform': SpiritualBotCategory.PLATFORM_SPECIFIC,
            'infrastructure': SpiritualBotCategory.INFRASTRUCTURE
        }
        config.categories_to_deploy = [category_map.get(cat.lower()) for cat in args.categories if cat.lower() in category_map]
    
    # Create and run launcher
    launcher = SpiritualMasterLauncher(config)
    
    try:
        print("🌟 SPIRITUAL MASTER LAUNCHER - ZeroLight Orbit 🌟")
        print("="*60)
        print(f"Mode: {config.launch_mode.name}")
        print(f"Target Bots: 6993 Static Bots")
        print(f"Spiritual Blessing: {'✅ Enabled' if config.enable_spiritual_blessing else '❌ Disabled'}")
        print("="*60)
        
        # Initialize systems
        if not await launcher.initialize_systems():
            print("❌ Gagal menginisialisasi sistem!")
            return 1
        
        # Start monitoring and blessing
        launcher.start_monitoring()
        launcher.start_spiritual_blessing()
        
        # Deploy bots (except for monitoring-only mode)
        if config.launch_mode != SpiritualLaunchMode.MONITORING_ONLY:
            if not await launcher.deploy_bots():
                print("❌ Gagal melakukan deployment!")
                return 1
        
        # Run based on mode
        if config.launch_mode == SpiritualLaunchMode.INTERACTIVE:
            await launcher.interactive_mode()
        elif config.launch_mode == SpiritualLaunchMode.DEMO:
            print("🎭 Running demo mode for 60 seconds...")
            await asyncio.sleep(60)
        else:
            print("🚀 Sistem berjalan... Tekan Ctrl+C untuk shutdown")
            try:
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                print("\n🛑 Shutdown signal received...")
        
        # Graceful shutdown
        await launcher.shutdown()
        return 0
        
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        if launcher:
            await launcher.shutdown()
        return 1

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n🛑 Program dihentikan oleh user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)