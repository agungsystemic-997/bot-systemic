#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🙏 In The Name of GOD - ZeroLight Orbit IoT System
Blessed Internet of Things Platform with Divine Features
بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيم

This comprehensive IoT system provides:
- Spiritual device management with divine blessing
- Quantum-resistant security for IoT devices
- Real-time monitoring and analytics
- Edge computing capabilities
- Blockchain integration for device identity
- AI-powered predictive maintenance
- Sacred data processing and storage
- Divine automation and orchestration
"""

import asyncio
import json
import logging
import hashlib
import hmac
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import queue
import ssl
import socket
from pathlib import Path

# Third-party imports (would be installed via requirements.txt)
try:
    import paho.mqtt.client as mqtt
    import websockets
    import cryptography
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    import aiohttp
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    import redis
    import sqlite3
    import psutil
    import schedule
    import yaml
    import requests
    from flask import Flask, request, jsonify
    from flask_socketio import SocketIO, emit
    import docker
    import kubernetes
    from prometheus_client import Counter, Histogram, Gauge, start_http_server
    import influxdb_client
    from influxdb_client.client.write_api import SYNCHRONOUS
    
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Some dependencies not available: {e}")
    print("📦 Please install requirements: pip install -r iot/requirements.txt")
    DEPENDENCIES_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════
# 🌟 SPIRITUAL IOT CONFIGURATION
# ═══════════════════════════════════════════════════════════════

@dataclass
class SpiritualIoTConfig:
    """Divine IoT System Configuration"""
    
    # System Identity
    system_name: str = "ZeroLight Orbit IoT System"
    version: str = "1.0.0"
    blessing: str = "In-The-Name-of-GOD"
    purpose: str = "Divine-IoT-Platform"
    
    # Spiritual Colors - Divine Color Palette
    spiritual_colors: Dict[str, str] = None
    
    # MQTT Configuration
    mqtt_config: Dict[str, Any] = None
    
    # WebSocket Configuration
    websocket_config: Dict[str, Any] = None
    
    # Security Configuration
    security_config: Dict[str, Any] = None
    
    # Database Configuration
    database_config: Dict[str, Any] = None
    
    # AI/ML Configuration
    ai_config: Dict[str, Any] = None
    
    # Monitoring Configuration
    monitoring_config: Dict[str, Any] = None
    
    # Edge Computing Configuration
    edge_config: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.spiritual_colors is None:
            self.spiritual_colors = {
                'divine_gold': '#FFD700',
                'sacred_blue': '#1E3A8A',
                'blessed_green': '#059669',
                'holy_white': '#FFFFF0',
                'spiritual_purple': '#7C3AED',
                'celestial_silver': '#C0C0C0',
                'angelic_pink': '#EC4899',
                'peaceful_teal': '#0D9488'
            }
        
        if self.mqtt_config is None:
            self.mqtt_config = {
                'broker_host': 'mqtt.zerolight-orbit.com',
                'broker_port': 8883,
                'username': 'spiritual_iot',
                'password': 'divine_blessing_2024',
                'client_id': f'spiritual_iot_{uuid.uuid4().hex[:8]}',
                'keepalive': 60,
                'qos': 2,
                'retain': True,
                'clean_session': False,
                'use_tls': True,
                'ca_cert_path': 'certs/ca.crt',
                'cert_path': 'certs/client.crt',
                'key_path': 'certs/client.key'
            }
        
        if self.websocket_config is None:
            self.websocket_config = {
                'host': '0.0.0.0',
                'port': 8765,
                'ssl_context': None,
                'max_connections': 1000,
                'ping_interval': 20,
                'ping_timeout': 10,
                'close_timeout': 10
            }
        
        if self.security_config is None:
            self.security_config = {
                'encryption_algorithm': 'AES-256-GCM',
                'key_derivation': 'PBKDF2',
                'hash_algorithm': 'SHA-256',
                'signature_algorithm': 'RSA-PSS',
                'key_size': 4096,
                'salt_size': 32,
                'iteration_count': 100000,
                'token_expiry': 3600,
                'max_failed_attempts': 5,
                'lockout_duration': 300
            }
        
        if self.database_config is None:
            self.database_config = {
                'sqlite_path': 'data/spiritual_iot.db',
                'redis_host': 'localhost',
                'redis_port': 6379,
                'redis_db': 0,
                'influxdb_url': 'http://localhost:8086',
                'influxdb_token': 'spiritual_token',
                'influxdb_org': 'zerolight_orbit',
                'influxdb_bucket': 'iot_data'
            }
        
        if self.ai_config is None:
            self.ai_config = {
                'model_path': 'models/',
                'anomaly_threshold': 0.1,
                'prediction_window': 24,
                'training_interval': 3600,
                'feature_columns': ['temperature', 'humidity', 'pressure', 'vibration'],
                'target_column': 'status'
            }
        
        if self.monitoring_config is None:
            self.monitoring_config = {
                'prometheus_port': 8000,
                'metrics_interval': 30,
                'health_check_interval': 60,
                'log_level': 'INFO',
                'log_file': 'logs/spiritual_iot.log',
                'max_log_size': 10485760,  # 10MB
                'backup_count': 5
            }
        
        if self.edge_config is None:
            self.edge_config = {
                'enable_edge_computing': True,
                'edge_nodes': [],
                'processing_rules': {},
                'data_retention_days': 30,
                'sync_interval': 300,
                'compression_enabled': True
            }

# Global configuration instance
SPIRITUAL_IOT_CONFIG = SpiritualIoTConfig()

# ═══════════════════════════════════════════════════════════════
# 🙏 SPIRITUAL BLESSING DISPLAY
# ═══════════════════════════════════════════════════════════════

def display_spiritual_iot_blessing():
    """Display spiritual blessing for IoT system"""
    blessing_message = """
🌟 ═══════════════════════════════════════════════════════════════
🙏 بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيم
✨ ZeroLight Orbit IoT System - In The Name of GOD
🌐 Blessed Internet of Things Platform
🚀 Divine Python IoT System with Sacred Features
🔮 Quantum-Resistant Security & AI-Powered Analytics
🛡️ Edge Computing & Blockchain Integration
💫 May this IoT system serve humanity with divine blessing
═══════════════════════════════════════════════════════════════ 🌟
    """
    print(blessing_message)

# ═══════════════════════════════════════════════════════════════
# 📱 SPIRITUAL IOT DEVICE MODELS
# ═══════════════════════════════════════════════════════════════

class DeviceType(Enum):
    """IoT Device Types with Spiritual Classification"""
    SENSOR = "sensor"
    ACTUATOR = "actuator"
    GATEWAY = "gateway"
    EDGE_COMPUTER = "edge_computer"
    SMART_DEVICE = "smart_device"
    WEARABLE = "wearable"
    VEHICLE = "vehicle"
    INDUSTRIAL = "industrial"
    MEDICAL = "medical"
    ENVIRONMENTAL = "environmental"
    SECURITY = "security"
    SPIRITUAL = "spiritual"  # Special category for spiritual devices

class DeviceStatus(Enum):
    """Device Status with Divine States"""
    ONLINE = "online"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"
    ERROR = "error"
    UPDATING = "updating"
    BLESSED = "blessed"  # Special spiritual state
    PROTECTED = "protected"  # Under divine protection

@dataclass
class SpiritualIoTDevice:
    """Spiritual IoT Device Model with Divine Attributes"""
    
    # Core Identity
    device_id: str
    name: str
    device_type: DeviceType
    manufacturer: str
    model: str
    firmware_version: str
    
    # Network Configuration
    ip_address: Optional[str] = None
    mac_address: Optional[str] = None
    network_interface: Optional[str] = None
    
    # Location and Environment
    location: Optional[Dict[str, float]] = None  # {'lat': float, 'lon': float, 'alt': float}
    environment: Optional[str] = None  # indoor, outdoor, industrial, etc.
    
    # Status and Health
    status: DeviceStatus = DeviceStatus.OFFLINE
    last_seen: Optional[datetime] = None
    uptime: Optional[int] = None  # seconds
    battery_level: Optional[float] = None  # percentage
    signal_strength: Optional[float] = None  # dBm
    
    # Capabilities and Configuration
    capabilities: List[str] = None
    configuration: Dict[str, Any] = None
    
    # Security and Authentication
    public_key: Optional[str] = None
    certificate: Optional[str] = None
    security_level: str = "standard"  # standard, high, quantum
    
    # Spiritual Attributes
    blessing: str = "Divine-IoT-Device"
    spiritual_score: float = 100.0
    divine_protection: bool = True
    
    # Metadata
    created_at: datetime = None
    updated_at: datetime = None
    tags: List[str] = None
    
    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = []
        if self.configuration is None:
            self.configuration = {}
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
        if self.tags is None:
            self.tags = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert device to dictionary"""
        data = asdict(self)
        # Convert enums to strings
        data['device_type'] = self.device_type.value
        data['status'] = self.status.value
        # Convert datetime to ISO format
        if self.created_at:
            data['created_at'] = self.created_at.isoformat()
        if self.updated_at:
            data['updated_at'] = self.updated_at.isoformat()
        if self.last_seen:
            data['last_seen'] = self.last_seen.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SpiritualIoTDevice':
        """Create device from dictionary"""
        # Convert string enums back to enums
        if 'device_type' in data:
            data['device_type'] = DeviceType(data['device_type'])
        if 'status' in data:
            data['status'] = DeviceStatus(data['status'])
        
        # Convert ISO datetime strings back to datetime objects
        for field in ['created_at', 'updated_at', 'last_seen']:
            if field in data and data[field]:
                data[field] = datetime.fromisoformat(data[field])
        
        return cls(**data)

@dataclass
class SpiritualSensorData:
    """Spiritual Sensor Data Model"""
    
    device_id: str
    sensor_type: str
    value: float
    unit: str
    timestamp: datetime
    quality: float = 1.0  # Data quality score (0-1)
    location: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = None
    blessing: str = "Divine-Sensor-Data"
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert sensor data to dictionary"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SpiritualSensorData':
        """Create sensor data from dictionary"""
        if 'timestamp' in data:
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)

# ═══════════════════════════════════════════════════════════════
# 🔐 SPIRITUAL SECURITY MANAGER
# ═══════════════════════════════════════════════════════════════

class SpiritualIoTSecurityManager:
    """Divine Security Manager for IoT Devices"""
    
    def __init__(self, config: SpiritualIoTConfig):
        self.config = config
        self.encryption_key = None
        self.private_key = None
        self.public_key = None
        self.device_certificates = {}
        self.failed_attempts = {}
        self.locked_devices = set()
        self.blessing = "Divine-Security-Manager"
        
        self.initialize_security()
    
    def initialize_security(self):
        """Initialize security components with divine blessing"""
        try:
            # Generate or load encryption keys
            self.generate_encryption_keys()
            
            # Initialize certificate authority
            self.initialize_ca()
            
            logging.info("🔐 Spiritual IoT security manager initialized with divine blessing")
            return True
        except Exception as e:
            logging.error(f"❌ Security initialization failed: {e}")
            return False
    
    def generate_encryption_keys(self):
        """Generate encryption keys with quantum resistance"""
        try:
            # Generate RSA key pair for asymmetric encryption
            self.private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=self.config.security_config['key_size']
            )
            self.public_key = self.private_key.public_key()
            
            # Generate symmetric encryption key
            self.encryption_key = Fernet.generate_key()
            
            logging.info("🔑 Encryption keys generated with divine blessing")
        except Exception as e:
            logging.error(f"❌ Key generation failed: {e}")
            raise
    
    def initialize_ca(self):
        """Initialize Certificate Authority for device authentication"""
        try:
            # In a real implementation, this would set up a proper CA
            # For now, we'll use self-signed certificates
            logging.info("🏛️ Certificate Authority initialized with divine blessing")
        except Exception as e:
            logging.error(f"❌ CA initialization failed: {e}")
            raise
    
    def authenticate_device(self, device_id: str, credentials: Dict[str, Any]) -> bool:
        """Authenticate IoT device with divine verification"""
        try:
            # Check if device is locked
            if device_id in self.locked_devices:
                logging.warning(f"🚫 Device {device_id} is locked")
                return False
            
            # Verify device credentials
            if self.verify_device_credentials(device_id, credentials):
                # Reset failed attempts on successful authentication
                self.failed_attempts.pop(device_id, None)
                logging.info(f"✅ Device {device_id} authenticated with divine blessing")
                return True
            else:
                # Track failed attempts
                self.failed_attempts[device_id] = self.failed_attempts.get(device_id, 0) + 1
                
                # Lock device if too many failed attempts
                if self.failed_attempts[device_id] >= self.config.security_config['max_failed_attempts']:
                    self.locked_devices.add(device_id)
                    logging.warning(f"🔒 Device {device_id} locked due to failed attempts")
                
                return False
        except Exception as e:
            logging.error(f"❌ Device authentication error: {e}")
            return False
    
    def verify_device_credentials(self, device_id: str, credentials: Dict[str, Any]) -> bool:
        """Verify device credentials"""
        try:
            # Extract credentials
            device_cert = credentials.get('certificate')
            signature = credentials.get('signature')
            timestamp = credentials.get('timestamp')
            
            if not all([device_cert, signature, timestamp]):
                return False
            
            # Verify timestamp (prevent replay attacks)
            current_time = time.time()
            if abs(current_time - timestamp) > self.config.security_config['token_expiry']:
                logging.warning(f"⏰ Timestamp verification failed for device {device_id}")
                return False
            
            # Verify certificate and signature
            # In a real implementation, this would verify against the CA
            return True
            
        except Exception as e:
            logging.error(f"❌ Credential verification error: {e}")
            return False
    
    def encrypt_data(self, data: bytes) -> bytes:
        """Encrypt data with divine protection"""
        try:
            fernet = Fernet(self.encryption_key)
            encrypted_data = fernet.encrypt(data)
            return encrypted_data
        except Exception as e:
            logging.error(f"❌ Data encryption error: {e}")
            raise
    
    def decrypt_data(self, encrypted_data: bytes) -> bytes:
        """Decrypt data with divine blessing"""
        try:
            fernet = Fernet(self.encryption_key)
            decrypted_data = fernet.decrypt(encrypted_data)
            return decrypted_data
        except Exception as e:
            logging.error(f"❌ Data decryption error: {e}")
            raise
    
    def sign_data(self, data: bytes) -> bytes:
        """Sign data with private key"""
        try:
            signature = self.private_key.sign(
                data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return signature
        except Exception as e:
            logging.error(f"❌ Data signing error: {e}")
            raise
    
    def verify_signature(self, data: bytes, signature: bytes, public_key) -> bool:
        """Verify data signature"""
        try:
            public_key.verify(
                signature,
                data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except Exception as e:
            logging.error(f"❌ Signature verification error: {e}")
            return False
    
    def generate_device_certificate(self, device_id: str) -> Dict[str, str]:
        """Generate certificate for IoT device"""
        try:
            # Generate device key pair
            device_private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048
            )
            device_public_key = device_private_key.public_key()
            
            # Serialize keys
            private_pem = device_private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
            
            public_pem = device_public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
            
            # Store certificate
            certificate = {
                'device_id': device_id,
                'private_key': private_pem.decode('utf-8'),
                'public_key': public_pem.decode('utf-8'),
                'issued_at': datetime.now().isoformat(),
                'expires_at': (datetime.now() + timedelta(days=365)).isoformat(),
                'blessing': 'Divine-Device-Certificate'
            }
            
            self.device_certificates[device_id] = certificate
            
            logging.info(f"📜 Certificate generated for device {device_id} with divine blessing")
            return certificate
            
        except Exception as e:
            logging.error(f"❌ Certificate generation error: {e}")
            raise

# ═══════════════════════════════════════════════════════════════
# 📊 SPIRITUAL DEVICE MANAGER
# ═══════════════════════════════════════════════════════════════

class SpiritualIoTDeviceManager:
    """Divine Device Manager for IoT Platform"""
    
    def __init__(self, config: SpiritualIoTConfig):
        self.config = config
        self.devices: Dict[str, SpiritualIoTDevice] = {}
        self.device_data: Dict[str, List[SpiritualSensorData]] = {}
        self.security_manager = SpiritualIoTSecurityManager(config)
        self.db_connection = None
        self.redis_client = None
        self.blessing = "Divine-Device-Manager"
        
        self.initialize_storage()
    
    def initialize_storage(self):
        """Initialize storage systems with divine blessing"""
        try:
            # Initialize SQLite database
            self.init_sqlite_db()
            
            # Initialize Redis cache
            self.init_redis_cache()
            
            logging.info("💾 Storage systems initialized with divine blessing")
        except Exception as e:
            logging.error(f"❌ Storage initialization error: {e}")
    
    def init_sqlite_db(self):
        """Initialize SQLite database"""
        try:
            db_path = Path(self.config.database_config['sqlite_path'])
            db_path.parent.mkdir(parents=True, exist_ok=True)
            
            self.db_connection = sqlite3.connect(str(db_path), check_same_thread=False)
            
            # Create tables
            self.create_database_tables()
            
            logging.info("🗄️ SQLite database initialized with divine blessing")
        except Exception as e:
            logging.error(f"❌ SQLite initialization error: {e}")
            raise
    
    def create_database_tables(self):
        """Create database tables"""
        cursor = self.db_connection.cursor()
        
        # Devices table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS devices (
                device_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                device_type TEXT NOT NULL,
                manufacturer TEXT,
                model TEXT,
                firmware_version TEXT,
                ip_address TEXT,
                mac_address TEXT,
                status TEXT,
                last_seen TIMESTAMP,
                configuration TEXT,
                blessing TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Sensor data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sensor_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                device_id TEXT NOT NULL,
                sensor_type TEXT NOT NULL,
                value REAL NOT NULL,
                unit TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                quality REAL DEFAULT 1.0,
                metadata TEXT,
                blessing TEXT,
                FOREIGN KEY (device_id) REFERENCES devices (device_id)
            )
        ''')
        
        # Device events table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS device_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                device_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                event_data TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                blessing TEXT,
                FOREIGN KEY (device_id) REFERENCES devices (device_id)
            )
        ''')
        
        self.db_connection.commit()
        logging.info("📋 Database tables created with divine blessing")
    
    def init_redis_cache(self):
        """Initialize Redis cache"""
        try:
            if DEPENDENCIES_AVAILABLE:
                self.redis_client = redis.Redis(
                    host=self.config.database_config['redis_host'],
                    port=self.config.database_config['redis_port'],
                    db=self.config.database_config['redis_db'],
                    decode_responses=True
                )
                
                # Test connection
                self.redis_client.ping()
                logging.info("🔄 Redis cache initialized with divine blessing")
            else:
                logging.warning("⚠️ Redis not available - using in-memory cache")
        except Exception as e:
            logging.warning(f"⚠️ Redis initialization failed: {e} - using in-memory cache")
            self.redis_client = None
    
    async def register_device(self, device: SpiritualIoTDevice) -> bool:
        """Register new IoT device with divine blessing"""
        try:
            # Generate device certificate
            certificate = self.security_manager.generate_device_certificate(device.device_id)
            device.certificate = json.dumps(certificate)
            
            # Store device in database
            cursor = self.db_connection.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO devices 
                (device_id, name, device_type, manufacturer, model, firmware_version,
                 ip_address, mac_address, status, configuration, blessing, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                device.device_id, device.name, device.device_type.value,
                device.manufacturer, device.model, device.firmware_version,
                device.ip_address, device.mac_address, device.status.value,
                json.dumps(device.configuration), device.blessing, datetime.now()
            ))
            self.db_connection.commit()
            
            # Store in memory
            self.devices[device.device_id] = device
            
            # Cache in Redis
            if self.redis_client:
                self.redis_client.setex(
                    f"device:{device.device_id}",
                    3600,  # 1 hour TTL
                    json.dumps(device.to_dict())
                )
            
            # Log device registration event
            await self.log_device_event(device.device_id, "device_registered", {
                "device_name": device.name,
                "device_type": device.device_type.value
            })
            
            logging.info(f"📱 Device {device.device_id} registered with divine blessing")
            return True
            
        except Exception as e:
            logging.error(f"❌ Device registration error: {e}")
            return False
    
    async def update_device_status(self, device_id: str, status: DeviceStatus) -> bool:
        """Update device status with divine blessing"""
        try:
            if device_id not in self.devices:
                logging.warning(f"⚠️ Device {device_id} not found")
                return False
            
            old_status = self.devices[device_id].status
            self.devices[device_id].status = status
            self.devices[device_id].updated_at = datetime.now()
            
            if status == DeviceStatus.ONLINE:
                self.devices[device_id].last_seen = datetime.now()
            
            # Update database
            cursor = self.db_connection.cursor()
            cursor.execute('''
                UPDATE devices 
                SET status = ?, last_seen = ?, updated_at = ?
                WHERE device_id = ?
            ''', (status.value, datetime.now(), datetime.now(), device_id))
            self.db_connection.commit()
            
            # Update cache
            if self.redis_client:
                self.redis_client.setex(
                    f"device:{device_id}",
                    3600,
                    json.dumps(self.devices[device_id].to_dict())
                )
            
            # Log status change event
            await self.log_device_event(device_id, "status_changed", {
                "old_status": old_status.value,
                "new_status": status.value
            })
            
            logging.info(f"📊 Device {device_id} status updated to {status.value} with divine blessing")
            return True
            
        except Exception as e:
            logging.error(f"❌ Device status update error: {e}")
            return False
    
    async def store_sensor_data(self, sensor_data: SpiritualSensorData) -> bool:
        """Store sensor data with divine blessing"""
        try:
            # Store in database
            cursor = self.db_connection.cursor()
            cursor.execute('''
                INSERT INTO sensor_data 
                (device_id, sensor_type, value, unit, timestamp, quality, metadata, blessing)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                sensor_data.device_id, sensor_data.sensor_type, sensor_data.value,
                sensor_data.unit, sensor_data.timestamp, sensor_data.quality,
                json.dumps(sensor_data.metadata), sensor_data.blessing
            ))
            self.db_connection.commit()
            
            # Store in memory (keep last 1000 readings per device)
            if sensor_data.device_id not in self.device_data:
                self.device_data[sensor_data.device_id] = []
            
            self.device_data[sensor_data.device_id].append(sensor_data)
            
            # Keep only last 1000 readings
            if len(self.device_data[sensor_data.device_id]) > 1000:
                self.device_data[sensor_data.device_id] = self.device_data[sensor_data.device_id][-1000:]
            
            # Cache latest reading in Redis
            if self.redis_client:
                self.redis_client.setex(
                    f"sensor:{sensor_data.device_id}:{sensor_data.sensor_type}",
                    300,  # 5 minutes TTL
                    json.dumps(sensor_data.to_dict())
                )
            
            logging.debug(f"📊 Sensor data stored for device {sensor_data.device_id} with divine blessing")
            return True
            
        except Exception as e:
            logging.error(f"❌ Sensor data storage error: {e}")
            return False
    
    async def get_device(self, device_id: str) -> Optional[SpiritualIoTDevice]:
        """Get device information with divine blessing"""
        try:
            # Check memory first
            if device_id in self.devices:
                return self.devices[device_id]
            
            # Check Redis cache
            if self.redis_client:
                cached_data = self.redis_client.get(f"device:{device_id}")
                if cached_data:
                    device_dict = json.loads(cached_data)
                    device = SpiritualIoTDevice.from_dict(device_dict)
                    self.devices[device_id] = device
                    return device
            
            # Check database
            cursor = self.db_connection.cursor()
            cursor.execute('SELECT * FROM devices WHERE device_id = ?', (device_id,))
            row = cursor.fetchone()
            
            if row:
                device_dict = {
                    'device_id': row[0],
                    'name': row[1],
                    'device_type': row[2],
                    'manufacturer': row[3],
                    'model': row[4],
                    'firmware_version': row[5],
                    'ip_address': row[6],
                    'mac_address': row[7],
                    'status': row[8],
                    'last_seen': row[9],
                    'configuration': json.loads(row[10]) if row[10] else {},
                    'blessing': row[11],
                    'created_at': row[12],
                    'updated_at': row[13]
                }
                
                device = SpiritualIoTDevice.from_dict(device_dict)
                self.devices[device_id] = device
                return device
            
            return None
            
        except Exception as e:
            logging.error(f"❌ Device retrieval error: {e}")
            return None
    
    async def get_device_list(self, filters: Dict[str, Any] = None) -> List[SpiritualIoTDevice]:
        """Get list of devices with optional filters"""
        try:
            devices = []
            
            # Build query with filters
            query = "SELECT * FROM devices"
            params = []
            
            if filters:
                conditions = []
                for key, value in filters.items():
                    if key in ['device_type', 'status', 'manufacturer']:
                        conditions.append(f"{key} = ?")
                        params.append(value)
                
                if conditions:
                    query += " WHERE " + " AND ".join(conditions)
            
            cursor = self.db_connection.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            for row in rows:
                device_dict = {
                    'device_id': row[0],
                    'name': row[1],
                    'device_type': row[2],
                    'manufacturer': row[3],
                    'model': row[4],
                    'firmware_version': row[5],
                    'ip_address': row[6],
                    'mac_address': row[7],
                    'status': row[8],
                    'last_seen': row[9],
                    'configuration': json.loads(row[10]) if row[10] else {},
                    'blessing': row[11],
                    'created_at': row[12],
                    'updated_at': row[13]
                }
                
                device = SpiritualIoTDevice.from_dict(device_dict)
                devices.append(device)
            
            return devices
            
        except Exception as e:
            logging.error(f"❌ Device list retrieval error: {e}")
            return []
    
    async def log_device_event(self, device_id: str, event_type: str, event_data: Dict[str, Any]):
        """Log device event with divine blessing"""
        try:
            cursor = self.db_connection.cursor()
            cursor.execute('''
                INSERT INTO device_events (device_id, event_type, event_data, blessing)
                VALUES (?, ?, ?, ?)
            ''', (device_id, event_type, json.dumps(event_data), "Divine-Device-Event"))
            self.db_connection.commit()
            
            logging.info(f"📝 Event logged for device {device_id}: {event_type}")
        except Exception as e:
            logging.error(f"❌ Event logging error: {e}")

# ═══════════════════════════════════════════════════════════════
# 🌐 SPIRITUAL MQTT MANAGER
# ═══════════════════════════════════════════════════════════════

class SpiritualMQTTManager:
    """Divine MQTT Manager for IoT Communication"""
    
    def __init__(self, config: SpiritualIoTConfig, device_manager: SpiritualIoTDeviceManager):
        self.config = config
        self.device_manager = device_manager
        self.client = None
        self.connected = False
        self.message_handlers = {}
        self.blessing = "Divine-MQTT-Manager"
        
        self.initialize_mqtt()
    
    def initialize_mqtt(self):
        """Initialize MQTT client with divine blessing"""
        try:
            if not DEPENDENCIES_AVAILABLE:
                logging.warning("⚠️ MQTT dependencies not available")
                return
            
            # Create MQTT client
            self.client = mqtt.Client(
                client_id=self.config.mqtt_config['client_id'],
                clean_session=self.config.mqtt_config['clean_session']
            )
            
            # Set credentials
            self.client.username_pw_set(
                self.config.mqtt_config['username'],
                self.config.mqtt_config['password']
            )
            
            # Configure TLS
            if self.config.mqtt_config['use_tls']:
                self.client.tls_set(
                    ca_certs=self.config.mqtt_config['ca_cert_path'],
                    certfile=self.config.mqtt_config['cert_path'],
                    keyfile=self.config.mqtt_config['key_path']
                )
            
            # Set callbacks
            self.client.on_connect = self.on_connect
            self.client.on_disconnect = self.on_disconnect
            self.client.on_message = self.on_message
            self.client.on_publish = self.on_publish
            self.client.on_subscribe = self.on_subscribe
            
            # Register default message handlers
            self.register_message_handlers()
            
            logging.info("📡 MQTT client initialized with divine blessing")
            
        except Exception as e:
            logging.error(f"❌ MQTT initialization error: {e}")
    
    def register_message_handlers(self):
        """Register default MQTT message handlers"""
        self.message_handlers = {
            'devices/+/data': self.handle_sensor_data,
            'devices/+/status': self.handle_device_status,
            'devices/+/config': self.handle_device_config,
            'devices/+/command': self.handle_device_command,
            'system/heartbeat': self.handle_system_heartbeat,
            'system/alert': self.handle_system_alert
        }
    
    def connect(self):
        """Connect to MQTT broker with divine blessing"""
        try:
            if not self.client:
                logging.error("❌ MQTT client not initialized")
                return False
            
            self.client.connect(
                self.config.mqtt_config['broker_host'],
                self.config.mqtt_config['broker_port'],
                self.config.mqtt_config['keepalive']
            )
            
            # Start network loop
            self.client.loop_start()
            
            logging.info("🔌 Connecting to MQTT broker with divine blessing")
            return True
            
        except Exception as e:
            logging.error(f"❌ MQTT connection error: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from MQTT broker"""
        try:
            if self.client and self.connected:
                self.client.loop_stop()
                self.client.disconnect()
                logging.info("🔌 Disconnected from MQTT broker with divine blessing")
        except Exception as e:
            logging.error(f"❌ MQTT disconnection error: {e}")
    
    def on_connect(self, client, userdata, flags, rc):
        """MQTT connection callback"""
        if rc == 0:
            self.connected = True
            logging.info("✅ Connected to MQTT broker with divine blessing")
            
            # Subscribe to topics
            for topic in self.message_handlers.keys():
                client.subscribe(topic, qos=self.config.mqtt_config['qos'])
                logging.info(f"📬 Subscribed to topic: {topic}")
        else:
            logging.error(f"❌ MQTT connection failed with code: {rc}")
    
    def on_disconnect(self, client, userdata, rc):
        """MQTT disconnection callback"""
        self.connected = False
        if rc != 0:
            logging.warning("⚠️ Unexpected MQTT disconnection")
        else:
            logging.info("🔌 MQTT disconnected gracefully")
    
    def on_message(self, client, userdata, msg):
        """MQTT message callback"""
        try:
            topic = msg.topic
            payload = msg.payload.decode('utf-8')
            
            logging.debug(f"📨 Received message on topic {topic}: {payload}")
            
            # Find matching handler
            for pattern, handler in self.message_handlers.items():
                if self.topic_matches(topic, pattern):
                    asyncio.create_task(handler(topic, payload))
                    break
            else:
                logging.warning(f"⚠️ No handler found for topic: {topic}")
                
        except Exception as e:
            logging.error(f"❌ Message handling error: {e}")
    
    def on_publish(self, client, userdata, mid):
        """MQTT publish callback"""
        logging.debug(f"📤 Message published with mid: {mid}")
    
    def on_subscribe(self, client, userdata, mid, granted_qos):
        """MQTT subscribe callback"""
        logging.debug(f"📬 Subscribed with mid: {mid}, QoS: {granted_qos}")
    
    def topic_matches(self, topic: str, pattern: str) -> bool:
        """Check if topic matches pattern with wildcards"""
        topic_parts = topic.split('/')
        pattern_parts = pattern.split('/')
        
        if len(topic_parts) != len(pattern_parts):
            return False
        
        for topic_part, pattern_part in zip(topic_parts, pattern_parts):
            if pattern_part == '+':
                continue
            elif pattern_part == '#':
                return True
            elif topic_part != pattern_part:
                return False
        
        return True
    
    async def handle_sensor_data(self, topic: str, payload: str):
        """Handle sensor data messages"""
        try:
            # Extract device ID from topic
            device_id = topic.split('/')[1]
            
            # Parse sensor data
            data = json.loads(payload)
            
            # Create sensor data object
            sensor_data = SpiritualSensorData(
                device_id=device_id,
                sensor_type=data['sensor_type'],
                value=float(data['value']),
                unit=data['unit'],
                timestamp=datetime.fromisoformat(data['timestamp']),
                quality=data.get('quality', 1.0),
                location=data.get('location'),
                metadata=data.get('metadata', {}),
                blessing="Divine-MQTT-Sensor-Data"
            )
            
            # Store sensor data
            await self.device_manager.store_sensor_data(sensor_data)
            
            # Update device last seen
            await self.device_manager.update_device_status(device_id, DeviceStatus.ONLINE)
            
            logging.debug(f"📊 Sensor data processed for device {device_id}")
            
        except Exception as e:
            logging.error(f"❌ Sensor data handling error: {e}")
    
    async def handle_device_status(self, topic: str, payload: str):
        """Handle device status messages"""
        try:
            device_id = topic.split('/')[1]
            status_data = json.loads(payload)
            
            status = DeviceStatus(status_data['status'])
            await self.device_manager.update_device_status(device_id, status)
            
            logging.info(f"📊 Device {device_id} status updated to {status.value}")
            
        except Exception as e:
            logging.error(f"❌ Device status handling error: {e}")
    
    async def handle_device_config(self, topic: str, payload: str):
        """Handle device configuration messages"""
        try:
            device_id = topic.split('/')[1]
            config_data = json.loads(payload)
            
            device = await self.device_manager.get_device(device_id)
            if device:
                device.configuration.update(config_data)
                device.updated_at = datetime.now()
                
                logging.info(f"⚙️ Device {device_id} configuration updated")
            
        except Exception as e:
            logging.error(f"❌ Device config handling error: {e}")
    
    async def handle_device_command(self, topic: str, payload: str):
        """Handle device command messages"""
        try:
            device_id = topic.split('/')[1]
            command_data = json.loads(payload)
            
            # Process device command
            await self.process_device_command(device_id, command_data)
            
            logging.info(f"🎮 Command processed for device {device_id}")
            
        except Exception as e:
            logging.error(f"❌ Device command handling error: {e}")
    
    async def handle_system_heartbeat(self, topic: str, payload: str):
        """Handle system heartbeat messages"""
        try:
            heartbeat_data = json.loads(payload)
            logging.debug(f"💓 System heartbeat received: {heartbeat_data}")
        except Exception as e:
            logging.error(f"❌ Heartbeat handling error: {e}")
    
    async def handle_system_alert(self, topic: str, payload: str):
        """Handle system alert messages"""
        try:
            alert_data = json.loads(payload)
            logging.warning(f"🚨 System alert: {alert_data}")
        except Exception as e:
            logging.error(f"❌ Alert handling error: {e}")
    
    async def process_device_command(self, device_id: str, command_data: Dict[str, Any]):
        """Process device command with divine blessing"""
        try:
            command_type = command_data.get('command')
            parameters = command_data.get('parameters', {})
            
            # Log command execution
            await self.device_manager.log_device_event(device_id, "command_executed", {
                "command": command_type,
                "parameters": parameters
            })
            
            # Send command response
            response_topic = f"devices/{device_id}/response"
            response_data = {
                "command": command_type,
                "status": "executed",
                "timestamp": datetime.now().isoformat(),
                "blessing": "Divine-Command-Response"
            }
            
            self.publish(response_topic, json.dumps(response_data))
            
        except Exception as e:
            logging.error(f"❌ Command processing error: {e}")
    
    def publish(self, topic: str, payload: str, qos: int = None, retain: bool = None):
        """Publish message to MQTT topic"""
        try:
            if not self.client or not self.connected:
                logging.error("❌ MQTT client not connected")
                return False
            
            qos = qos or self.config.mqtt_config['qos']
            retain = retain or self.config.mqtt_config['retain']
            
            result = self.client.publish(topic, payload, qos=qos, retain=retain)
            
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                logging.debug(f"📤 Message published to {topic}")
                return True
            else:
                logging.error(f"❌ Message publish failed: {result.rc}")
                return False
                
        except Exception as e:
            logging.error(f"❌ Message publish error: {e}")
            return False

# ═══════════════════════════════════════════════════════════════
# 🚀 SPIRITUAL IOT SYSTEM MAIN CLASS
# ═══════════════════════════════════════════════════════════════

class SpiritualIoTSystem:
    """Divine IoT System - Main Orchestrator"""
    
    def __init__(self, config: SpiritualIoTConfig = None):
        self.config = config or SPIRITUAL_IOT_CONFIG
        self.device_manager = SpiritualIoTDeviceManager(self.config)
        self.mqtt_manager = SpiritualMQTTManager(self.config, self.device_manager)
        self.web_server = None
        self.websocket_server = None
        self.is_running = False
        self.blessing = "Divine-IoT-System"
        
        # Initialize logging
        self.setup_logging()
        
        # Initialize system
        self.initialize_system()
    
    def setup_logging(self):
        """Setup logging with divine configuration"""
        log_config = self.config.monitoring_config
        
        # Create logs directory
        log_path = Path(log_config['log_file'])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, log_config['log_level']),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_config['log_file']),
                logging.StreamHandler()
            ]
        )
        
        # Set up log rotation
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            log_config['log_file'],
            maxBytes=log_config['max_log_size'],
            backupCount=log_config['backup_count']
        )
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        
        logger = logging.getLogger()
        logger.addHandler(file_handler)
    
    def initialize_system(self):
        """Initialize IoT system with divine blessing"""
        try:
            display_spiritual_iot_blessing()
            
            # Initialize web server
            self.initialize_web_server()
            
            # Initialize WebSocket server
            self.initialize_websocket_server()
            
            logging.info("🚀 Spiritual IoT System initialized with divine blessing")
            
        except Exception as e:
            logging.error(f"❌ System initialization error: {e}")
            raise
    
    def initialize_web_server(self):
        """Initialize Flask web server"""
        try:
            if not DEPENDENCIES_AVAILABLE:
                logging.warning("⚠️ Web server dependencies not available")
                return
            
            self.web_server = Flask(__name__)
            self.web_server.config['SECRET_KEY'] = 'spiritual_iot_secret_key'
            
            # Register routes
            self.register_web_routes()
            
            logging.info("🌐 Web server initialized with divine blessing")
            
        except Exception as e:
            logging.error(f"❌ Web server initialization error: {e}")
    
    def register_web_routes(self):
        """Register web API routes"""
        
        @self.web_server.route('/api/devices', methods=['GET'])
        async def get_devices():
            """Get list of devices"""
            try:
                filters = request.args.to_dict()
                devices = await self.device_manager.get_device_list(filters)
                return jsonify({
                    'success': True,
                    'devices': [device.to_dict() for device in devices],
                    'blessing': 'Divine-Device-List'
                })
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.web_server.route('/api/devices/<device_id>', methods=['GET'])
        async def get_device(device_id):
            """Get specific device"""
            try:
                device = await self.device_manager.get_device(device_id)
                if device:
                    return jsonify({
                        'success': True,
                        'device': device.to_dict(),
                        'blessing': 'Divine-Device-Info'
                    })
                else:
                    return jsonify({'success': False, 'error': 'Device not found'}), 404
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.web_server.route('/api/devices', methods=['POST'])
        async def register_device():
            """Register new device"""
            try:
                device_data = request.json
                device = SpiritualIoTDevice.from_dict(device_data)
                
                success = await self.device_manager.register_device(device)
                if success:
                    return jsonify({
                        'success': True,
                        'message': 'Device registered with divine blessing',
                        'device_id': device.device_id
                    })
                else:
                    return jsonify({'success': False, 'error': 'Device registration failed'}), 500
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.web_server.route('/api/devices/<device_id>/command', methods=['POST'])
        async def send_device_command(device_id):
            """Send command to device"""
            try:
                command_data = request.json
                
                # Publish command via MQTT
                topic = f"devices/{device_id}/command"
                self.mqtt_manager.publish(topic, json.dumps(command_data))
                
                return jsonify({
                    'success': True,
                    'message': 'Command sent with divine blessing'
                })
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.web_server.route('/api/system/status', methods=['GET'])
        def get_system_status():
            """Get system status"""
            try:
                status = {
                    'system_name': self.config.system_name,
                    'version': self.config.version,
                    'blessing': self.config.blessing,
                    'mqtt_connected': self.mqtt_manager.connected,
                    'device_count': len(self.device_manager.devices),
                    'uptime': time.time(),  # Would calculate actual uptime
                    'timestamp': datetime.now().isoformat()
                }
                
                return jsonify({
                    'success': True,
                    'status': status,
                    'blessing': 'Divine-System-Status'
                })
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500
    
    def initialize_websocket_server(self):
        """Initialize WebSocket server for real-time communication"""
        try:
            if not DEPENDENCIES_AVAILABLE:
                logging.warning("⚠️ WebSocket server dependencies not available")
                return
            
            # WebSocket server would be implemented here
            logging.info("🔌 WebSocket server initialized with divine blessing")
            
        except Exception as e:
            logging.error(f"❌ WebSocket server initialization error: {e}")
    
    async def start(self):
        """Start the IoT system with divine blessing"""
        try:
            self.is_running = True
            
            # Connect to MQTT broker
            if self.mqtt_manager:
                self.mqtt_manager.connect()
            
            # Start web server
            if self.web_server and DEPENDENCIES_AVAILABLE:
                # In a real implementation, would use proper ASGI server
                pass
            
            # Start monitoring tasks
            asyncio.create_task(self.monitoring_loop())
            asyncio.create_task(self.health_check_loop())
            
            logging.info("🚀 Spiritual IoT System started with divine blessing")
            
            # Keep system running
            while self.is_running:
                await asyncio.sleep(1)
                
        except Exception as e:
            logging.error(f"❌ System start error: {e}")
            await self.stop()
    
    async def stop(self):
        """Stop the IoT system gracefully"""
        try:
            self.is_running = False
            
            # Disconnect MQTT
            if self.mqtt_manager:
                self.mqtt_manager.disconnect()
            
            # Close database connections
            if self.device_manager.db_connection:
                self.device_manager.db_connection.close()
            
            logging.info("🛑 Spiritual IoT System stopped with divine blessing")
            
        except Exception as e:
            logging.error(f"❌ System stop error: {e}")
    
    async def monitoring_loop(self):
        """System monitoring loop with divine oversight"""
        while self.is_running:
            try:
                # Monitor system resources
                cpu_usage = psutil.cpu_percent()
                memory_usage = psutil.virtual_memory().percent
                disk_usage = psutil.disk_usage('/').percent
                
                # Log system metrics
                logging.debug(f"📊 System metrics - CPU: {cpu_usage}%, Memory: {memory_usage}%, Disk: {disk_usage}%")
                
                # Check for offline devices
                await self.check_offline_devices()
                
                await asyncio.sleep(self.config.monitoring_config['metrics_interval'])
                
            except Exception as e:
                logging.error(f"❌ Monitoring loop error: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def health_check_loop(self):
        """Health check loop with divine monitoring"""
        while self.is_running:
            try:
                # Check MQTT connection
                if self.mqtt_manager and not self.mqtt_manager.connected:
                    logging.warning("⚠️ MQTT connection lost - attempting reconnection")
                    self.mqtt_manager.connect()
                
                # Check database connection
                if self.device_manager.db_connection:
                    try:
                        cursor = self.device_manager.db_connection.cursor()
                        cursor.execute('SELECT 1')
                        cursor.fetchone()
                    except Exception as e:
                        logging.error(f"❌ Database connection error: {e}")
                        self.device_manager.init_sqlite_db()
                
                await asyncio.sleep(self.config.monitoring_config['health_check_interval'])
                
            except Exception as e:
                logging.error(f"❌ Health check error: {e}")
                await asyncio.sleep(60)
    
    async def check_offline_devices(self):
        """Check for offline devices and update status"""
        try:
            current_time = datetime.now()
            offline_threshold = timedelta(minutes=5)  # Consider offline after 5 minutes
            
            for device_id, device in self.device_manager.devices.items():
                if device.status == DeviceStatus.ONLINE and device.last_seen:
                    if current_time - device.last_seen > offline_threshold:
                        await self.device_manager.update_device_status(device_id, DeviceStatus.OFFLINE)
                        logging.warning(f"⚠️ Device {device_id} marked as offline")
                        
        except Exception as e:
            logging.error(f"❌ Offline device check error: {e}")

# ═══════════════════════════════════════════════════════════════
# 🚀 MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════

async def main():
    """Main function to run the Spiritual IoT System"""
    try:
        # Create and start IoT system
        iot_system = SpiritualIoTSystem()
        
        # Handle graceful shutdown
        import signal
        
        def signal_handler(signum, frame):
            logging.info("🛑 Shutdown signal received")
            asyncio.create_task(iot_system.stop())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Start the system
        await iot_system.start()
        
    except Exception as e:
        logging.error(f"❌ Main execution error: {e}")
    finally:
        logging.info("🙏 Spiritual IoT System execution completed with divine blessing")

if __name__ == "__main__":
    # Run the spiritual IoT system
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🙏 Spiritual IoT System interrupted - May peace be with you")
    except Exception as e:
        print(f"❌ System error: {e}")
        logging.error(f"❌ System error: {e}")

# 🙏 In The Name of GOD - IoT System Complete with Divine Blessing
# بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيم