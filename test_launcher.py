#!/usr/bin/env python3
"""
Test script untuk menjalankan spiritual master launcher
"""
import asyncio
import sys
import os

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import the launcher using importlib
import importlib.util

def import_module_from_file(module_name, file_path):
    """Import module from file with hyphens in filename"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Import the launcher module
launcher_module = import_module_from_file("spiritual_master_launcher", 
                                        os.path.join(current_dir, "spiritual-master-launcher.py"))

SpiritualMasterLauncher = launcher_module.SpiritualMasterLauncher
SpiritualMasterConfig = launcher_module.SpiritualMasterConfig

async def test_launcher():
    """Test the spiritual master launcher"""
    print("🌾 Membuka ladang berkah digital...")
    
    # Create config for demo mode
    config = SpiritualMasterConfig(
        launch_mode=launcher_module.SpiritualLaunchMode.DEMO,
        log_level="INFO",
        enable_spiritual_blessing=True,
        max_concurrent_bots=10
    )
    
    # Create and run launcher
    launcher = SpiritualMasterLauncher(config)
    
    try:
        print("🚀 Menginisialisasi sistem...")
        success = await launcher.initialize_systems()
        
        if success:
            print("✅ Sistem berhasil diinisialisasi!")
            print("📡 Jalur aktif untuk spiritual bot system")
            print("🙏 Setiap bot adalah titipan. Terima kasih, Tuhan.")
        else:
            print("❌ Gagal menginisialisasi sistem")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        await launcher.shutdown()

if __name__ == "__main__":
    asyncio.run(test_launcher())