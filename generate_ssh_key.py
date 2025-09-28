#!/usr/bin/env python3
"""
🔑 SPIRITUAL SSH KEY GENERATOR FOR GITHUB 🔑
Bismillahirrahmanirrahim - Generator SSH Key yang diberkahi
"""

import os
import subprocess
import sys
from pathlib import Path

def create_ssh_key():
    """Generate SSH key untuk GitHub dengan berkah"""
    
    print("🌟 SPIRITUAL SSH KEY GENERATOR 🌟")
    print("=" * 50)
    print("Bismillahirrahmanirrahim")
    print()
    
    # Buat direktori .ssh jika belum ada
    ssh_dir = Path.home() / ".ssh"
    ssh_dir.mkdir(exist_ok=True)
    
    # Path untuk key files
    private_key_path = ssh_dir / "id_rsa_github_spiritual"
    public_key_path = ssh_dir / "id_rsa_github_spiritual.pub"
    
    # Email untuk SSH key
    email = input("📧 Masukkan email GitHub Anda (atau tekan Enter untuk default): ").strip()
    if not email:
        email = "spiritual.bot@github.com"
    
    print(f"\n🔐 Membuat SSH Key dengan email: {email}")
    print("🕐 Mohon tunggu sebentar...")
    
    try:
        # Generate SSH key menggunakan subprocess
        cmd = [
            "ssh-keygen",
            "-t", "rsa",
            "-b", "4096",
            "-C", email,
            "-f", str(private_key_path),
            "-N", ""  # No passphrase
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        print("✅ SSH Key berhasil dibuat!")
        print(f"📁 Private key: {private_key_path}")
        print(f"📁 Public key: {public_key_path}")
        print()
        
        # Baca dan tampilkan public key
        if public_key_path.exists():
            with open(public_key_path, 'r') as f:
                public_key_content = f.read().strip()
            
            print("🔑 PUBLIC KEY UNTUK GITHUB:")
            print("=" * 60)
            print(public_key_content)
            print("=" * 60)
            print()
            
            print("📋 LANGKAH SELANJUTNYA:")
            print("1. Copy public key di atas")
            print("2. Buka GitHub.com → Settings → SSH and GPG keys")
            print("3. Klik 'New SSH key'")
            print("4. Paste public key dan beri nama 'Spiritual Bot Key'")
            print("5. Klik 'Add SSH key'")
            print()
            
            # Simpan ke file untuk mudah copy
            output_file = Path("github_ssh_public_key.txt")
            with open(output_file, 'w') as f:
                f.write(public_key_content)
            
            print(f"💾 Public key juga disimpan di: {output_file}")
            print()
            
            # Test SSH connection
            print("🧪 Testing SSH connection ke GitHub...")
            test_cmd = ["ssh", "-T", "git@github.com", "-i", str(private_key_path)]
            test_result = subprocess.run(test_cmd, capture_output=True, text=True)
            
            if "successfully authenticated" in test_result.stderr:
                print("✅ SSH connection berhasil!")
            else:
                print("⚠️  SSH belum terkonfigurasi. Silakan tambahkan key ke GitHub dulu.")
            
        else:
            print("❌ Error: Public key file tidak ditemukan")
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Error generating SSH key: {e}")
        print("💡 Pastikan OpenSSH terinstall di sistem Anda")
        
        # Alternative method menggunakan Git Bash jika ada
        git_bash_paths = [
            r"C:\Program Files\Git\bin\ssh-keygen.exe",
            r"C:\Program Files (x86)\Git\bin\ssh-keygen.exe"
        ]
        
        for git_path in git_bash_paths:
            if os.path.exists(git_path):
                print(f"🔄 Mencoba dengan Git Bash: {git_path}")
                try:
                    cmd_git = [
                        git_path,
                        "-t", "rsa",
                        "-b", "4096", 
                        "-C", email,
                        "-f", str(private_key_path),
                        "-N", ""
                    ]
                    subprocess.run(cmd_git, check=True)
                    print("✅ SSH Key berhasil dibuat dengan Git Bash!")
                    break
                except subprocess.CalledProcessError:
                    continue
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    print("\n🌙 Alhamdulillahirabbil'alamiin")
    print("SSH Key telah dibuat dengan berkah Allah SWT")

if __name__ == "__main__":
    create_ssh_key()