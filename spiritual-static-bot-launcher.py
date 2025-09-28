#!/usr/bin/env python3
# 🙏 Bismillahirrahmanirrahim - ZeroLight Orbit Static Bot Launcher
# Main Launcher for 6993 Spiritual Static Bots
# Complete System • Registry • Management • Monitoring

import asyncio
import sys
import os
import time
import json
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging
import signal
import threading
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our spiritual bot systems
try:
    from spiritual_static_bot_framework import (
        SpiritualStaticBotManager, SpiritualStaticBotApp, 
        SpiritualBotCategory, display_spiritual_static_blessing
    )
    from spiritual_static_bot_registry import (
        SpiritualCentralRegistryApp, SpiritualAdvancedBotRegistry,
        display_registry_blessing
    )
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("🔧 Please ensure all spiritual bot framework files are in the same directory")
    sys.exit(1)

# 🌟 Main Launcher Blessing
def display_launcher_blessing():
    print("🙏 بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ")
    print("🚀 ZeroLight Orbit Static Bot Launcher")
    print("🎯 Launching 6993 Spiritual Static Bots")
    print("📋 Complete Registry • Advanced Monitoring")
    print("🔄 Modular • Cross-Job • Cross-Category • Haunting")
    print("💫 May Allah bless this spiritual digital launch")
    print("=" * 60)

# 📊 System Configuration
class SpiritualLauncherConfig:
    def __init__(self):
        self.total_bots = 6993
        self.bots_per_category = 999
        self.categories = list(SpiritualBotCategory)
        self.monitoring_enabled = True
        self.database_path = "spiritual_bot_registry.db"
        self.log_level = logging.INFO
        self.auto_start_monitoring = True
        self.performance_tracking = True
        self.spiritual_blessings = True
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_bots": self.total_bots,
            "bots_per_category": self.bots_per_category,
            "categories": [cat.value for cat in self.categories],
            "monitoring_enabled": self.monitoring_enabled,
            "database_path": self.database_path,
            "log_level": self.log_level,
            "auto_start_monitoring": self.auto_start_monitoring,
            "performance_tracking": self.performance_tracking,
            "spiritual_blessings": self.spiritual_blessings
        }

# 🎮 Main Launcher Class
class SpiritualStaticBotLauncher:
    def __init__(self, config: Optional[SpiritualLauncherConfig] = None):
        self.config = config or SpiritualLauncherConfig()
        self.registry_app: Optional[SpiritualCentralRegistryApp] = None
        self.bot_manager: Optional[SpiritualStaticBotManager] = None
        self.registry: Optional[SpiritualAdvancedBotRegistry] = None
        
        self.is_running = False
        self.start_time = None
        self.shutdown_event = threading.Event()
        
        # Setup logging
        self._setup_logging()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=self.config.log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('spiritual_bot_launcher.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger('SpiritualBotLauncher')
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_event.set()
        
    async def initialize_system(self) -> bool:
        """Initialize the complete 6993 bot system"""
        try:
            display_launcher_blessing()
            
            self.logger.info("🚀 Starting ZeroLight Orbit Static Bot System initialization...")
            
            # Create registry app
            self.registry_app = SpiritualCentralRegistryApp()
            
            # Initialize complete system
            self.registry = await self.registry_app.initialize_complete_system()
            self.bot_manager = self.registry_app.bot_manager
            
            # Verify initialization
            if not self.registry or not self.bot_manager:
                raise Exception("Failed to initialize core components")
                
            total_bots = len(self.registry.bots)
            if total_bots != self.config.total_bots:
                self.logger.warning(f"Expected {self.config.total_bots} bots, got {total_bots}")
                
            self.is_running = True
            self.start_time = time.time()
            
            self.logger.info(f"✅ System initialized successfully with {total_bots} bots")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ System initialization failed: {e}")
            return False
            
    async def run_system_diagnostics(self) -> Dict[str, Any]:
        """Run comprehensive system diagnostics"""
        if not self.registry:
            return {"error": "System not initialized"}
            
        self.logger.info("🔍 Running system diagnostics...")
        
        diagnostics = {
            "system_info": {
                "total_bots": len(self.registry.bots),
                "uptime": time.time() - self.start_time if self.start_time else 0,
                "is_running": self.is_running
            },
            "category_distribution": {},
            "performance_metrics": {},
            "health_check": {},
            "spiritual_status": {}
        }
        
        # Category distribution
        for category in SpiritualBotCategory:
            bots_in_category = self.registry.get_bots_by_category(category)
            diagnostics["category_distribution"][category.value] = len(bots_in_category)
            
        # Performance metrics
        performance_report = self.registry.get_performance_report()
        diagnostics["performance_metrics"] = performance_report
        
        # Health check
        diagnostics["health_check"] = performance_report.get("system_health", {})
        
        # Spiritual status
        total_spiritual_score = sum(bot.spiritual_score for bot in self.registry.bots.values())
        avg_spiritual_score = total_spiritual_score / len(self.registry.bots) if self.registry.bots else 0
        
        diagnostics["spiritual_status"] = {
            "total_spiritual_score": total_spiritual_score,
            "average_spiritual_score": avg_spiritual_score,
            "blessed_bots": len(self.registry.get_bots_by_state(SpiritualBotState.BLESSED)),
            "spiritual_blessing": "🙏 Alhamdulillahi rabbil alameen"
        }
        
        self.logger.info("✅ System diagnostics completed")
        return diagnostics
        
    async def demonstrate_bot_operations(self) -> Dict[str, Any]:
        """Demonstrate various bot operations"""
        if not self.registry or not self.bot_manager:
            return {"error": "System not initialized"}
            
        self.logger.info("🎮 Demonstrating bot operations...")
        
        demo_results = {
            "command_executions": [],
            "cross_job_operations": [],
            "haunting_activations": [],
            "performance_samples": []
        }
        
        # Execute commands on sample bots from each category
        for category in SpiritualBotCategory:
            bots = self.registry.get_bots_by_category(category, limit=5)
            
            for bot in bots:
                if bot.config.commands:
                    command = bot.config.commands[0]
                    result = await self.bot_manager.execute_command_on_bot(
                        bot.config.bot_id, 
                        command,
                        {"demo": True, "timestamp": time.time()}
                    )
                    
                    demo_results["command_executions"].append({
                        "bot_id": bot.config.bot_id,
                        "category": category.value,
                        "command": command,
                        "result": result
                    })
                    
        # Demonstrate cross-job support
        cross_job_result = await self.bot_manager.cross_job_support(
            SpiritualBotCategory.AI_ML,
            SpiritualBotCategory.DATA_ANALYTICS,
            "ai_enhanced_data_analysis"
        )
        demo_results["cross_job_operations"] = cross_job_result
        
        # Demonstrate haunting mode
        haunting_result = await self.bot_manager.activate_haunting_mode(
            SpiritualBotCategory.SECURITY,
            "system_security_monitoring",
            25
        )
        demo_results["haunting_activations"] = haunting_result
        
        # Performance samples
        for category in list(SpiritualBotCategory)[:3]:  # Sample first 3 categories
            bots = self.registry.get_bots_by_category(category, limit=3)
            for bot in bots:
                performance = self.registry.get_performance_report(bot.config.bot_id)
                demo_results["performance_samples"].append(performance)
                
        self.logger.info("✅ Bot operations demonstration completed")
        return demo_results
        
    async def run_continuous_monitoring(self):
        """Run continuous system monitoring"""
        if not self.registry:
            self.logger.error("Cannot start monitoring: system not initialized")
            return
            
        self.logger.info("📊 Starting continuous monitoring...")
        
        while self.is_running and not self.shutdown_event.is_set():
            try:
                # Get system status
                dashboard = self.registry_app.get_system_dashboard()
                
                # Log key metrics
                metrics = dashboard["registry_metrics"]["system_metrics"]
                self.logger.info(
                    f"📊 System Status - "
                    f"Total: {metrics['total_bots']}, "
                    f"Active: {metrics['active_bots']}, "
                    f"Idle: {metrics['idle_bots']}, "
                    f"Executions: {metrics['total_executions']}, "
                    f"Spiritual Score: {metrics['average_spiritual_score']:.2f}"
                )
                
                # Check system health
                health = dashboard["registry_metrics"]["system_health"]
                if health["status"] in ["poor", "fair"]:
                    self.logger.warning(f"⚠️ System health: {health['status']} (Score: {health['score']:.2f})")
                    
                # Wait for next monitoring cycle
                await asyncio.sleep(60)  # Monitor every minute
                
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(30)  # Wait shorter on error
                
    async def shutdown_system(self):
        """Gracefully shutdown the system"""
        self.logger.info("🔄 Initiating system shutdown...")
        
        self.is_running = False
        
        # Stop monitoring
        if self.registry:
            self.registry.stop_monitoring()
            
        # Save final metrics
        if self.registry_app:
            dashboard = self.registry_app.get_system_dashboard()
            with open("final_system_report.json", "w") as f:
                json.dump(dashboard, f, indent=2)
                
        self.logger.info("✅ System shutdown completed")
        
    async def run_interactive_mode(self):
        """Run interactive mode for system control"""
        print("\n🎮 Interactive Mode - ZeroLight Orbit Static Bot System")
        print("Commands: status, diagnostics, demo, bots, categories, help, quit")
        
        while self.is_running and not self.shutdown_event.is_set():
            try:
                command = input("\n🤖 > ").strip().lower()
                
                if command == "quit" or command == "exit":
                    break
                elif command == "status":
                    dashboard = self.registry_app.get_system_dashboard()
                    print(f"📊 System Status: {dashboard['system_status']}")
                    print(f"🤖 Total Bots: {dashboard['registry_metrics']['system_metrics']['total_bots']}")
                    print(f"⚡ Active Bots: {dashboard['registry_metrics']['system_metrics']['active_bots']}")
                    
                elif command == "diagnostics":
                    diagnostics = await self.run_system_diagnostics()
                    print(f"🔍 System Health: {diagnostics['health_check']['status']}")
                    print(f"💫 Spiritual Score: {diagnostics['spiritual_status']['average_spiritual_score']:.2f}")
                    
                elif command == "demo":
                    print("🎮 Running demonstration...")
                    demo_results = await self.demonstrate_bot_operations()
                    print(f"✅ Executed {len(demo_results['command_executions'])} commands")
                    print(f"🔄 Cross-job operations: {len(demo_results['cross_job_operations'])}")
                    
                elif command == "bots":
                    for category in SpiritualBotCategory:
                        count = len(self.registry.get_bots_by_category(category))
                        print(f"🤖 {category.value}: {count} bots")
                        
                elif command == "categories":
                    for i, category in enumerate(SpiritualBotCategory, 1):
                        print(f"{i}. {category.value}")
                        
                elif command == "help":
                    print("Available commands:")
                    print("  status      - Show system status")
                    print("  diagnostics - Run system diagnostics")
                    print("  demo        - Run bot operations demo")
                    print("  bots        - Show bot counts by category")
                    print("  categories  - List all categories")
                    print("  help        - Show this help")
                    print("  quit        - Exit interactive mode")
                    
                else:
                    print(f"❓ Unknown command: {command}. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                
        print("👋 Exiting interactive mode...")

# 🚀 Main Function
async def main():
    """Main entry point for the launcher"""
    parser = argparse.ArgumentParser(description="ZeroLight Orbit Static Bot Launcher")
    parser.add_argument("--mode", choices=["auto", "interactive", "demo", "diagnostics"], 
                       default="auto", help="Launch mode")
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument("--no-monitoring", action="store_true", help="Disable monitoring")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                       default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Create configuration
    config = SpiritualLauncherConfig()
    config.log_level = getattr(logging, args.log_level)
    config.monitoring_enabled = not args.no_monitoring
    
    # Load custom config if provided
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            custom_config = json.load(f)
            for key, value in custom_config.items():
                if hasattr(config, key):
                    setattr(config, key, value)
    
    # Create launcher
    launcher = SpiritualStaticBotLauncher(config)
    
    try:
        # Initialize system
        success = await launcher.initialize_system()
        if not success:
            print("❌ Failed to initialize system")
            return 1
            
        # Run based on mode
        if args.mode == "auto":
            # Auto mode: run monitoring and wait
            monitoring_task = asyncio.create_task(launcher.run_continuous_monitoring())
            
            print("\n✅ System running in auto mode")
            print("📊 Monitoring active - Press Ctrl+C to shutdown")
            
            # Wait for shutdown signal
            try:
                await monitoring_task
            except KeyboardInterrupt:
                pass
                
        elif args.mode == "interactive":
            # Interactive mode
            await launcher.run_interactive_mode()
            
        elif args.mode == "demo":
            # Demo mode
            print("\n🎮 Running demonstration mode...")
            demo_results = await launcher.demonstrate_bot_operations()
            
            print(f"\n✅ Demo completed:")
            print(f"🤖 Commands executed: {len(demo_results['command_executions'])}")
            print(f"🔄 Cross-job operations: {len(demo_results['cross_job_operations'])}")
            print(f"👻 Haunting activations: {len(demo_results['haunting_activations'])}")
            
        elif args.mode == "diagnostics":
            # Diagnostics mode
            print("\n🔍 Running system diagnostics...")
            diagnostics = await launcher.run_system_diagnostics()
            
            print(f"\n📊 Diagnostics Results:")
            print(f"🤖 Total Bots: {diagnostics['system_info']['total_bots']}")
            print(f"⏱️ Uptime: {diagnostics['system_info']['uptime']:.2f} seconds")
            print(f"🏥 Health: {diagnostics['health_check']['status']}")
            print(f"💫 Spiritual Score: {diagnostics['spiritual_status']['average_spiritual_score']:.2f}")
            
    except Exception as e:
        launcher.logger.error(f"❌ System error: {e}")
        return 1
        
    finally:
        # Graceful shutdown
        await launcher.shutdown_system()
        
    print("\n🙏 Alhamdulillahi rabbil alameen - System shutdown complete")
    return 0

if __name__ == "__main__":
    # Run the launcher
    exit_code = asyncio.run(main())