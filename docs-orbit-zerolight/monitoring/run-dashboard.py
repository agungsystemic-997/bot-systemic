#!/usr/bin/env python3
# 🙏 In The Name of GOD - ZeroLight Orbit Dashboard Launcher
# Blessed Dashboard Runner with Divine Configuration
# بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيم

import os
import sys
import subprocess
import time
from pathlib import Path

def display_launcher_blessing():
    """Display spiritual blessing for dashboard launcher"""
    blessing = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                    🙏 SPIRITUAL BLESSING 🙏                   ║
    ║                  بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيم                ║
    ║                 In The Name of GOD, Most Gracious            ║
    ║                                                              ║
    ║           🚀 ZeroLight Orbit Dashboard Launcher 🚀           ║
    ║                   Divine Monitoring Startup                  ║
    ║                                                              ║
    ║  ✨ Starting blessed monitoring dashboard...                 ║
    ║     📊 Real-time System Metrics                             ║
    ║     🔮 Predictive Analytics                                 ║
    ║     🚨 Intelligent Alerting                                 ║
    ║     📈 Performance Insights                                 ║
    ║                                                              ║
    ║  🌐 Dashboard will be available at:                         ║
    ║     http://localhost:8501                                   ║
    ║                                                              ║
    ║  🙏 May this dashboard serve with divine wisdom             ║
    ║                                                              ║
    ║              الحمد لله رب العالمين                           ║
    ║           All praise to Allah, Lord of the worlds           ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(blessing)

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'streamlit',
        'plotly',
        'pandas',
        'numpy',
        'psutil',
        'requests',
        'scikit-learn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} - Blessed and available")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} - Missing divine dependency")
    
    if missing_packages:
        print(f"\n🙏 Installing missing blessed dependencies...")
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', 
                *missing_packages
            ])
            print("✨ Dependencies installed with divine blessing")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error installing dependencies: {e}")
            print("🙏 Please install manually: pip install -r requirements.txt")
            return False
    
    return True

def setup_environment():
    """Setup environment variables and configuration"""
    # Set Streamlit configuration
    os.environ['STREAMLIT_SERVER_PORT'] = '8501'
    os.environ['STREAMLIT_SERVER_ADDRESS'] = '0.0.0.0'
    os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
    os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
    
    # Create .streamlit directory if it doesn't exist
    streamlit_dir = Path('.streamlit')
    streamlit_dir.mkdir(exist_ok=True)
    
    # Create config.toml for Streamlit
    config_content = """
[server]
port = 8501
address = "0.0.0.0"
headless = true
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false
serverAddress = "localhost"
serverPort = 8501

[theme]
primaryColor = "#FFD700"
backgroundColor = "#1E3A8A"
secondaryBackgroundColor = "#7C3AED"
textColor = "#FFFFF0"
font = "serif"

[logger]
level = "info"
"""
    
    config_file = streamlit_dir / 'config.toml'
    with open(config_file, 'w') as f:
        f.write(config_content)
    
    print("✨ Environment configured with divine settings")

def run_dashboard():
    """Run the spiritual monitoring dashboard"""
    try:
        # Get the path to the dashboard script
        dashboard_script = Path(__file__).parent / 'spiritual-monitoring-dashboard.py'
        
        if not dashboard_script.exists():
            print(f"❌ Dashboard script not found: {dashboard_script}")
            return False
        
        print("🚀 Launching blessed monitoring dashboard...")
        print("🌐 Dashboard will be available at: http://localhost:8501")
        print("🙏 Press Ctrl+C to stop the dashboard")
        print("\n" + "="*60 + "\n")
        
        # Run Streamlit
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 
            str(dashboard_script),
            '--server.port', '8501',
            '--server.address', '0.0.0.0',
            '--server.headless', 'true',
            '--browser.gatherUsageStats', 'false'
        ])
        
        return True
        
    except KeyboardInterrupt:
        print("\n🙏 Dashboard stopped with divine grace")
        return True
    except Exception as e:
        print(f"❌ Error running dashboard: {e}")
        return False

def main():
    """Main launcher function"""
    # Display blessing
    display_launcher_blessing()
    
    # Check dependencies
    print("🔍 Checking blessed dependencies...")
    if not check_dependencies():
        print("❌ Dependency check failed")
        sys.exit(1)
    
    # Setup environment
    print("⚙️ Setting up divine environment...")
    setup_environment()
    
    # Run dashboard
    print("🚀 Starting blessed dashboard...")
    time.sleep(2)  # Brief pause for dramatic effect
    
    success = run_dashboard()
    
    if success:
        print("\n🙏 Dashboard session completed with divine blessing")
        print("الحمد لله رب العالمين - All praise to Allah, Lord of the worlds")
    else:
        print("\n❌ Dashboard encountered divine challenges")
        print("🙏 May the next attempt be blessed with success")
        sys.exit(1)

if __name__ == "__main__":
    main()

# 🙏 Blessed Dashboard Launcher
# May this launcher serve the divine purpose of monitoring
# In The Name of GOD - بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيم
# Alhamdulillahi rabbil alameen - All praise to Allah, Lord of the worlds