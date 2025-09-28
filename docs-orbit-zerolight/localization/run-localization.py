#!/usr/bin/env python3
# 🙏 In The Name of GOD - ZeroLight Orbit Localization Launcher
# بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيم
# Blessed Launcher for Divine Multi-Language Experience

"""
ZeroLight Orbit Localization System Launcher
===========================================

This script launches the comprehensive internationalization system
for the ZeroLight Orbit platform, providing blessed multi-language
support with cultural adaptation.

Features:
- Multi-language support (20+ languages)
- Cultural themes and adaptations
- RTL/LTR text direction support
- Islamic calendar and prayer times
- Regional number and date formatting
- Dynamic translation capabilities

Usage:
    python run-localization.py [options]

Options:
    --language, -l    Set default language (en, ar, tr, fa, etc.)
    --theme, -t       Set cultural theme (islamic, modern, classic)
    --demo, -d        Run demonstration mode
    --export, -e      Export all translations
    --server, -s      Start localization server
    --help, -h        Show this help message

Examples:
    python run-localization.py --language ar --theme islamic
    python run-localization.py --demo
    python run-localization.py --server --language en
"""

import os
import sys
import argparse
import subprocess
import json
import time
from pathlib import Path
from datetime import datetime
import logging

# Add the current directory to Python path
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('localization.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def display_spiritual_launcher_blessing():
    """Display spiritual blessing for the launcher"""
    blessing = """
    🙏 ═══════════════════════════════════════════════════════════════
    
    بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيم
    In The Name of GOD, Most Gracious, Most Merciful
    
    🌟 ZeroLight Orbit Localization System 🌟
    Divine Multi-Language Platform Launcher
    
    ═══════════════════════════════════════════════════════════════
    
    🌍 Supported Languages:
    • Arabic (العربية) - RTL Support
    • English - Base Language  
    • Turkish (Türkçe) - Cultural Adaptation
    • Persian (فارسی) - RTL Support
    • Urdu (اردو) - RTL Support
    • Indonesian - Islamic Culture
    • Malay - Islamic Culture
    • French (Français)
    • German (Deutsch)
    • Spanish (Español)
    • Russian (Русский)
    • Chinese (中文)
    • Japanese (日本語)
    • Korean (한국어)
    • Hindi (हिन्दी)
    • Bengali (বাংলা)
    • Swahili
    • Hausa
    • Portuguese (Português)
    • Italian (Italiano)
    
    🕌 Cultural Features:
    • Islamic Calendar & Prayer Times
    • Cultural Color Schemes
    • RTL/LTR Text Direction
    • Regional Formatting
    • Spiritual Themes
    
    الحمد لله رب العالمين
    All Praise to Allah, Lord of the Worlds
    
    ═══════════════════════════════════════════════════════════════ 🙏
    """
    print(blessing)

def check_dependencies():
    """Check and install required dependencies"""
    logger.info("🔍 Checking dependencies...")
    
    required_packages = [
        'babel',
        'gettext',
        'googletrans==4.0.0rc1',
        'langdetect',
        'pytz',
        'hijri-converter',
        'lunardate',
        'flask',
        'streamlit',
        'fastapi',
        'uvicorn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if '==' in package:
                pkg_name = package.split('==')[0]
            else:
                pkg_name = package
            __import__(pkg_name.replace('-', '_'))
            logger.info(f"✅ {package} - Available")
        except ImportError:
            missing_packages.append(package)
            logger.warning(f"❌ {package} - Missing")
    
    if missing_packages:
        logger.info("📦 Installing missing packages...")
        for package in missing_packages:
            try:
                subprocess.check_call([
                    sys.executable, '-m', 'pip', 'install', package
                ])
                logger.info(f"✅ Installed {package}")
            except subprocess.CalledProcessError as e:
                logger.error(f"❌ Failed to install {package}: {e}")
                return False
    
    logger.info("🎉 All dependencies are ready!")
    return True

def setup_environment():
    """Setup environment variables and configuration"""
    logger.info("⚙️ Setting up environment...")
    
    # Set environment variables
    os.environ['PYTHONPATH'] = str(current_dir)
    os.environ['FLASK_APP'] = 'spiritual-i18n-manager.py'
    os.environ['FLASK_ENV'] = 'development'
    
    # Create necessary directories
    directories = [
        'translations',
        'translations/en',
        'translations/ar', 
        'translations/tr',
        'translations/fa',
        'translations/ur',
        'translations/id',
        'translations/ms',
        'translations/fr',
        'translations/de',
        'translations/es',
        'translations/ru',
        'translations/zh',
        'translations/ja',
        'translations/ko',
        'translations/hi',
        'translations/bn',
        'translations/sw',
        'translations/ha',
        'translations/pt',
        'translations/it',
        'cultural-themes',
        'fonts',
        'logs'
    ]
    
    for directory in directories:
        dir_path = current_dir / directory
        dir_path.mkdir(exist_ok=True)
        logger.info(f"📁 Created directory: {directory}")
    
    logger.info("✅ Environment setup complete!")

def run_demo_mode():
    """Run demonstration of localization features"""
    logger.info("🎭 Starting demonstration mode...")
    
    try:
        # Import the localization manager
        from spiritual_i18n_manager import SpiritualLocalizationApp
        
        # Create and run demo
        app = SpiritualLocalizationApp()
        
        print("\n🌟 Localization Demo Starting...")
        print("=" * 50)
        
        # Demo different languages
        languages = ['en', 'ar', 'tr', 'fa']
        
        for lang in languages:
            print(f"\n🌍 Language: {lang.upper()}")
            print("-" * 30)
            
            app.set_language(lang)
            
            # Demo basic translations
            print(f"App Name: {app.get_translation('app_name')}")
            print(f"Welcome: {app.get_translation('dashboard_welcome')}")
            print(f"Settings: {app.get_translation('nav_settings')}")
            print(f"Blessing: {app.get_translation('app_blessing')}")
            
            # Demo prayer times for Islamic languages
            if lang in ['ar', 'tr', 'fa', 'ur', 'id', 'ms']:
                print(f"Prayer Times:")
                prayers = ['prayer_fajr', 'prayer_dhuhr', 'prayer_asr', 'prayer_maghrib', 'prayer_isha']
                for prayer in prayers:
                    print(f"  {app.get_translation(prayer)}")
            
            time.sleep(1)  # Brief pause for readability
        
        print("\n🎉 Demo completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        return False
    
    return True

def export_translations():
    """Export all translations to various formats"""
    logger.info("📤 Exporting translations...")
    
    try:
        from spiritual_i18n_manager import SpiritualLocalizationApp
        
        app = SpiritualLocalizationApp()
        
        # Export to JSON
        export_dir = current_dir / 'exports'
        export_dir.mkdir(exist_ok=True)
        
        languages = ['en', 'ar', 'tr', 'fa', 'ur', 'id', 'ms']
        
        for lang in languages:
            app.set_language(lang)
            
            # Get all translations
            translations = {}
            
            # Sample translation keys
            keys = [
                'app_name', 'app_tagline', 'app_blessing',
                'nav_home', 'nav_dashboard', 'nav_settings',
                'auth_login', 'auth_register', 'auth_logout',
                'dashboard_welcome', 'settings_title',
                'prayer_fajr', 'prayer_dhuhr', 'prayer_asr',
                'blessing_bismillah', 'blessing_alhamdulillah'
            ]
            
            for key in keys:
                translations[key] = app.get_translation(key)
            
            # Export to JSON
            json_file = export_dir / f'{lang}_translations.json'
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(translations, f, ensure_ascii=False, indent=2)
            
            logger.info(f"✅ Exported {lang} translations to {json_file}")
        
        # Export cultural CSS
        css_export_dir = export_dir / 'css'
        css_export_dir.mkdir(exist_ok=True)
        
        # Copy cultural themes
        themes_dir = current_dir / 'cultural-themes'
        if themes_dir.exists():
            import shutil
            for css_file in themes_dir.glob('*.css'):
                shutil.copy2(css_file, css_export_dir / css_file.name)
                logger.info(f"✅ Exported CSS theme: {css_file.name}")
        
        logger.info("🎉 Export completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Export failed: {e}")
        return False
    
    return True

def start_localization_server(language='en', port=8080):
    """Start the localization web server"""
    logger.info(f"🚀 Starting localization server on port {port}...")
    
    try:
        # Create a simple Flask server
        server_code = f'''
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from flask import Flask, render_template_string, request, jsonify
from spiritual_i18n_manager import SpiritualLocalizationApp

app = Flask(__name__)
localization_app = SpiritualLocalizationApp()

# Set default language
localization_app.set_language('{language}')

@app.route('/')
def index():
    return render_template_string("""
    <!DOCTYPE html>
    <html dir="{{{{ direction }}}}">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{{{{ app_name }}}} - {{{{ tagline }}}}</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <link href="/static/css/islamic-theme.css" rel="stylesheet">
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }}
            .hero {{ background: linear-gradient(135deg, #006633, #003366); color: white; padding: 4rem 0; }}
            .feature-card {{ transition: transform 0.3s ease; }}
            .feature-card:hover {{ transform: translateY(-5px); }}
        </style>
    </head>
    <body class="islamic-theme">
        <div class="hero text-center">
            <div class="container">
                <h1 class="display-4">{{{{ app_name }}}}</h1>
                <p class="lead">{{{{ tagline }}}}</p>
                <p class="blessing">{{{{ blessing }}}}</p>
            </div>
        </div>
        
        <div class="container my-5">
            <div class="row">
                <div class="col-md-4 mb-4">
                    <div class="card feature-card islamic">
                        <div class="card-body text-center">
                            <h5 class="card-title">{{{{ nav_dashboard }}}}</h5>
                            <p class="card-text">{{{{ dashboard_welcome }}}}</p>
                        </div>
                    </div>
                </div>
                <div class="col-md-4 mb-4">
                    <div class="card feature-card islamic">
                        <div class="card-body text-center">
                            <h5 class="card-title">{{{{ nav_analytics }}}}</h5>
                            <p class="card-text">{{{{ analytics_title }}}}</p>
                        </div>
                    </div>
                </div>
                <div class="col-md-4 mb-4">
                    <div class="card feature-card islamic">
                        <div class="card-body text-center">
                            <h5 class="card-title">{{{{ nav_security }}}}</h5>
                            <p class="card-text">{{{{ security_title }}}}</p>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="row mt-5">
                <div class="col-12">
                    <h3 class="text-center mb-4">Language Selection</h3>
                    <div class="text-center">
                        <a href="/set_language/en" class="btn btn-outline-primary m-2">English</a>
                        <a href="/set_language/ar" class="btn btn-outline-primary m-2">العربية</a>
                        <a href="/set_language/tr" class="btn btn-outline-primary m-2">Türkçe</a>
                        <a href="/set_language/fa" class="btn btn-outline-primary m-2">فارسی</a>
                    </div>
                </div>
            </div>
        </div>
        
        <footer class="footer islamic mt-5">
            <div class="container text-center">
                <p class="blessing">{{{{ blessing_bismillah }}}}</p>
                <p>{{{{ culture_peace }}}} • {{{{ culture_wisdom }}}} • {{{{ culture_unity }}}}</p>
            </div>
        </footer>
    </body>
    </html>
    """,
        direction='rtl' if localization_app.current_language in ['ar', 'fa', 'ur'] else 'ltr',
        app_name=localization_app.get_translation('app_name'),
        tagline=localization_app.get_translation('app_tagline'),
        blessing=localization_app.get_translation('app_blessing'),
        nav_dashboard=localization_app.get_translation('nav_dashboard'),
        nav_analytics=localization_app.get_translation('nav_analytics'),
        nav_security=localization_app.get_translation('nav_security'),
        dashboard_welcome=localization_app.get_translation('dashboard_welcome'),
        analytics_title=localization_app.get_translation('analytics_title'),
        security_title=localization_app.get_translation('security_title'),
        blessing_bismillah=localization_app.get_translation('blessing_bismillah'),
        culture_peace=localization_app.get_translation('culture_peace'),
        culture_wisdom=localization_app.get_translation('culture_wisdom'),
        culture_unity=localization_app.get_translation('culture_unity')
    )

@app.route('/set_language/<lang>')
def set_language(lang):
    localization_app.set_language(lang)
    return redirect('/')

@app.route('/api/translate/<key>')
def translate_api(key):
    return jsonify({{
        'key': key,
        'translation': localization_app.get_translation(key),
        'language': localization_app.current_language
    }})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port={port}, debug=True)
'''
        
        # Write server file
        server_file = current_dir / 'localization_server.py'
        with open(server_file, 'w', encoding='utf-8') as f:
            f.write(server_code)
        
        # Start server
        logger.info(f"🌐 Server starting at http://localhost:{port}")
        subprocess.run([sys.executable, str(server_file)])
        
    except Exception as e:
        logger.error(f"❌ Server failed to start: {e}")
        return False
    
    return True

def main():
    """Main launcher function"""
    parser = argparse.ArgumentParser(
        description='ZeroLight Orbit Localization System Launcher',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run-localization.py --demo
  python run-localization.py --language ar --theme islamic
  python run-localization.py --server --port 8080
  python run-localization.py --export
        """
    )
    
    parser.add_argument(
        '--language', '-l',
        default='en',
        choices=['en', 'ar', 'tr', 'fa', 'ur', 'id', 'ms', 'fr', 'de', 'es', 'ru', 'zh', 'ja', 'ko', 'hi', 'bn', 'sw', 'ha', 'pt', 'it'],
        help='Set default language'
    )
    
    parser.add_argument(
        '--theme', '-t',
        default='islamic',
        choices=['islamic', 'modern', 'classic'],
        help='Set cultural theme'
    )
    
    parser.add_argument(
        '--demo', '-d',
        action='store_true',
        help='Run demonstration mode'
    )
    
    parser.add_argument(
        '--export', '-e',
        action='store_true',
        help='Export all translations'
    )
    
    parser.add_argument(
        '--server', '-s',
        action='store_true',
        help='Start localization server'
    )
    
    parser.add_argument(
        '--port', '-p',
        type=int,
        default=8080,
        help='Server port (default: 8080)'
    )
    
    args = parser.parse_args()
    
    # Display blessing
    display_spiritual_launcher_blessing()
    
    # Check dependencies
    if not check_dependencies():
        logger.error("❌ Dependency check failed!")
        return 1
    
    # Setup environment
    setup_environment()
    
    # Execute requested action
    success = True
    
    if args.demo:
        success = run_demo_mode()
    elif args.export:
        success = export_translations()
    elif args.server:
        success = start_localization_server(args.language, args.port)
    else:
        # Default: run demo
        success = run_demo_mode()
    
    if success:
        logger.info("🎉 Localization system completed successfully!")
        print("\n🙏 May this blessed system serve all users with peace and wisdom")
        print("الحمد لله رب العالمين - All Praise to Allah, Lord of the Worlds")
        return 0
    else:
        logger.error("❌ Localization system encountered errors!")
        return 1

if __name__ == '__main__':
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n🙏 Gracefully shutting down...")
        print("السلام عليكم - Peace be upon you")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        print(f"\n❌ Error: {e}")
        print("🙏 May Allah guide us through all difficulties")
        sys.exit(1)