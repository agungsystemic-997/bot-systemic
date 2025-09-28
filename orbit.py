#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌟 ORBIT - ZeroLight Command Launcher
Ladang Berkah Digital - Spiritual Web Discovery
"""

import sys
import subprocess
import os

def main():
    """Launcher untuk sistem orbit dengan penamaan spiritual"""
    
    if len(sys.argv) < 2:
        print("🌟 ORBIT - ZeroLight Command Launcher")
        print("=" * 40)
        print("Penggunaan:")
        print("  orbit --aktifkan planet earth [domain1] [domain2] ...")
        print("  orbit --wilayah sea --makhluk water --target 150")
        print()
        print("Contoh lengkap:")
        print("  orbit --aktifkan planet earth \\")
        print("        --wilayah sea \\")
        print("        --makhluk water \\")
        print("        --fungsi listener \\")
        print("        --log ./log/sea_water.csv \\")
        print("        --refleksi ./log/sea_refleksi.txt \\")
        print("        --target 150 \\")
        print("        --penutupan otomatis \\")
        print("        --tampilan syukur \\")
        print("        --mode lembut")
        print()
        print("✨ Ladang Berkah Digital - ZeroLight Orbit System ✨")
        return
    
    # Path ke script utama
    script_path = os.path.join(os.path.dirname(__file__), "orbit-earth-sea-water-listener.py")
    
    # Jalankan script utama dengan semua argumen
    try:
        cmd = [sys.executable, script_path] + sys.argv[1:]
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Terjadi kesalahan saat menjalankan orbit: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"⚠️ File orbit-earth-sea-water-listener.py tidak ditemukan di: {script_path}")
        sys.exit(1)

if __name__ == "__main__":
    main()