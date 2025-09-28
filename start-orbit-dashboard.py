#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌟 ORBIT DASHBOARD LAUNCHER
Launcher untuk Dashboard Syukur ZeroLight
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

# Import and start dashboard
try:
    from orbit_dashboard_syukur import start_dashboard_syukur
    start_dashboard_syukur(8084)
except ImportError:
    # Fallback: execute directly
    exec(open('orbit-dashboard-syukur.py').read())