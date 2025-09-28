// 🙏 In The Name of GOD - ZeroLight Orbit Mobile Application
// Blessed Cross-Platform Mobile Experience with Divine Flutter
// بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيم

import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:math';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter/foundation.dart';
import 'package:provider/provider.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:connectivity_plus/connectivity_plus.dart';
import 'package:device_info_plus/device_info_plus.dart';
import 'package:package_info_plus/package_info_plus.dart';
import 'package:local_auth/local_auth.dart';
import 'package:crypto/crypto.dart';
import 'package:http/http.dart' as http;
import 'package:web_socket_channel/web_socket_channel.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import 'package:flutter_local_notifications/flutter_local_notifications.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:geolocator/geolocator.dart';
import 'package:camera/camera.dart';
import 'package:image_picker/image_picker.dart';
import 'package:file_picker/file_picker.dart';
import 'package:path_provider/path_provider.dart';
import 'package:sqflite/sqflite.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:lottie/lottie.dart';
import 'package:rive/rive.dart';

// 🌟 Spiritual Mobile Configuration
class SpiritualMobileConfig {
  static const String appName = 'ZeroLight Orbit';
  static const String appVersion = '1.0.0';
  static const String blessing = 'In-The-Name-of-GOD';
  static const String purpose = 'Divine-Mobile-Experience';
  
  // Spiritual Colors - Divine Color Palette
  static const Map<String, Color> spiritualColors = {
    'divine_gold': Color(0xFFFFD700),
    'sacred_blue': Color(0xFF1E3A8A),
    'blessed_green': Color(0xFF059669),
    'holy_white': Color(0xFFFFFFF0),
    'spiritual_purple': Color(0xFF7C3AED),
    'celestial_silver': Color(0xFFC0C0C0),
    'angelic_pink': Color(0xFFEC4899),
    'peaceful_teal': Color(0xFF0D9488),
  };
  
  // Spiritual Themes
  static const Map<String, dynamic> lightTheme = {
    'primary': Color(0xFF1E3A8A),
    'secondary': Color(0xFFFFD700),
    'background': Color(0xFFFFFFF0),
    'surface': Color(0xFFFFFFFF),
    'blessing': 'Divine-Light-Theme'
  };
  
  static const Map<String, dynamic> darkTheme = {
    'primary': Color(0xFF3B82F6),
    'secondary': Color(0xFFFBBF24),
    'background': Color(0xFF0F172A),
    'surface': Color(0xFF1E293B),
    'blessing': 'Sacred-Dark-Theme'
  };
  
  // API Configuration
  static const String baseApiUrl = 'https://api.zerolight-orbit.com';
  static const String websocketUrl = 'wss://ws.zerolight-orbit.com';
  static const Duration requestTimeout = Duration(seconds: 30);
  
  // Security Configuration
  static const String encryptionKey = 'spiritual-mobile-encryption-key';
  static const int maxLoginAttempts = 5;
  static const Duration sessionTimeout = Duration(hours: 24);
  
  // Spiritual Features
  static const List<String> spiritualFeatures = [
    'Divine Authentication',
    'Sacred Data Sync',
    'Blessed Notifications',
    'Spiritual Analytics',
    'Holy Offline Mode',
    'Celestial UI/UX',
    'Angelic Performance',
    'Peaceful Meditation'
  ];
}

// 🙏 Spiritual Blessing Display
void displaySpiritualMobileBlessing() {
  if (kDebugMode) {
    print('\n🌟 ═══════════════════════════════════════════════════════════════');
    print('🙏 بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيم');
    print('✨ ZeroLight Orbit Mobile - In The Name of GOD');
    print('📱 Blessed Cross-Platform Mobile Experience');
    print('🚀 Divine Flutter Application with Sacred Features');
    print('═══════════════════════════════════════════════════════════════ 🌟\n');
  }
}

// 📱 Spiritual Mobile Data Models
class SpiritualUser {
  final String id;
  final String username;
  final String email;
  final String displayName;
  final String? profileImageUrl;
  final DateTime createdAt;
  final DateTime lastLoginAt;
  final Map<String, dynamic> preferences;
  final List<String> permissions;
  final double spiritualScore;
  final String blessing;
  
  SpiritualUser({
    required this.id,
    required this.username,
    required this.email,
    required this.displayName,
    this.profileImageUrl,
    required this.createdAt,
    required this.lastLoginAt,
    required this.preferences,
    required this.permissions,
    this.spiritualScore = 0.0,
    this.blessing = 'Divine-User-Blessed',
  });
  
  factory SpiritualUser.fromJson(Map<String, dynamic> json) {
    return SpiritualUser(
      id: json['id'],
      username: json['username'],
      email: json['email'],
      displayName: json['display_name'],
      profileImageUrl: json['profile_image_url'],
      createdAt: DateTime.parse(json['created_at']),
      lastLoginAt: DateTime.parse(json['last_login_at']),
      preferences: json['preferences'] ?? {},
      permissions: List<String>.from(json['permissions'] ?? []),
      spiritualScore: json['spiritual_score']?.toDouble() ?? 0.0,
      blessing: json['blessing'] ?? 'Divine-User-Blessed',
    );
  }
  
  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'username': username,
      'email': email,
      'display_name': displayName,
      'profile_image_url': profileImageUrl,
      'created_at': createdAt.toIso8601String(),
      'last_login_at': lastLoginAt.toIso8601String(),
      'preferences': preferences,
      'permissions': permissions,
      'spiritual_score': spiritualScore,
      'blessing': blessing,
    };
  }
}

class SpiritualNotification {
  final String id;
  final String title;
  final String body;
  final String type;
  final Map<String, dynamic> data;
  final DateTime createdAt;
  final bool isRead;
  final String blessing;
  
  SpiritualNotification({
    required this.id,
    required this.title,
    required this.body,
    required this.type,
    required this.data,
    required this.createdAt,
    this.isRead = false,
    this.blessing = 'Divine-Notification-Blessed',
  });
  
  factory SpiritualNotification.fromJson(Map<String, dynamic> json) {
    return SpiritualNotification(
      id: json['id'],
      title: json['title'],
      body: json['body'],
      type: json['type'],
      data: json['data'] ?? {},
      createdAt: DateTime.parse(json['created_at']),
      isRead: json['is_read'] ?? false,
      blessing: json['blessing'] ?? 'Divine-Notification-Blessed',
    );
  }
}

// 🔐 Spiritual Authentication Service
class SpiritualAuthService extends ChangeNotifier {
  SpiritualUser? _currentUser;
  bool _isAuthenticated = false;
  bool _isLoading = false;
  String? _errorMessage;
  
  final FlutterSecureStorage _secureStorage = const FlutterSecureStorage();
  final LocalAuthentication _localAuth = LocalAuthentication();
  
  SpiritualUser? get currentUser => _currentUser;
  bool get isAuthenticated => _isAuthenticated;
  bool get isLoading => _isLoading;
  String? get errorMessage => _errorMessage;
  
  // Initialize authentication service with divine blessing
  Future<void> initialize() async {
    _setLoading(true);
    
    try {
      // Check for stored authentication token
      final token = await _secureStorage.read(key: 'auth_token');
      if (token != null) {
        await _validateStoredToken(token);
      }
      
      // Check biometric availability
      await _checkBiometricAvailability();
      
    } catch (e) {
      _setError('Authentication initialization failed: $e');
    } finally {
      _setLoading(false);
    }
  }
  
  // Spiritual login with divine authentication
  Future<bool> login(String username, String password) async {
    _setLoading(true);
    _clearError();
    
    try {
      final response = await http.post(
        Uri.parse('${SpiritualMobileConfig.baseApiUrl}/auth/login'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'username': username,
          'password': password,
          'device_info': await _getDeviceInfo(),
          'blessing': 'Divine-Login-Request',
        }),
      ).timeout(SpiritualMobileConfig.requestTimeout);
      
      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        
        // Store authentication token securely
        await _secureStorage.write(key: 'auth_token', value: data['token']);
        
        // Create user object
        _currentUser = SpiritualUser.fromJson(data['user']);
        _isAuthenticated = true;
        
        // Store user data
        await _storeUserData(_currentUser!);
        
        notifyListeners();
        return true;
      } else {
        final error = jsonDecode(response.body);
        _setError(error['message'] ?? 'Login failed');
        return false;
      }
      
    } catch (e) {
      _setError('Network error: $e');
      return false;
    } finally {
      _setLoading(false);
    }
  }
  
  // Biometric authentication with spiritual blessing
  Future<bool> authenticateWithBiometrics() async {
    try {
      final isAvailable = await _localAuth.canCheckBiometrics;
      if (!isAvailable) {
        _setError('Biometric authentication not available');
        return false;
      }
      
      final isAuthenticated = await _localAuth.authenticate(
        localizedReason: 'Authenticate with divine biometric blessing',
        options: const AuthenticationOptions(
          biometricOnly: true,
          stickyAuth: true,
        ),
      );
      
      if (isAuthenticated) {
        // Load stored user data
        await _loadStoredUserData();
        return true;
      }
      
      return false;
      
    } catch (e) {
      _setError('Biometric authentication failed: $e');
      return false;
    }
  }
  
  // Spiritual logout with divine blessing
  Future<void> logout() async {
    _setLoading(true);
    
    try {
      // Notify server of logout
      final token = await _secureStorage.read(key: 'auth_token');
      if (token != null) {
        await http.post(
          Uri.parse('${SpiritualMobileConfig.baseApiUrl}/auth/logout'),
          headers: {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer $token',
          },
          body: jsonEncode({'blessing': 'Divine-Logout-Request'}),
        ).timeout(SpiritualMobileConfig.requestTimeout);
      }
      
      // Clear stored data
      await _secureStorage.deleteAll();
      await _clearUserData();
      
      _currentUser = null;
      _isAuthenticated = false;
      
      notifyListeners();
      
    } catch (e) {
      _setError('Logout error: $e');
    } finally {
      _setLoading(false);
    }
  }
  
  // Helper methods
  void _setLoading(bool loading) {
    _isLoading = loading;
    notifyListeners();
  }
  
  void _setError(String error) {
    _errorMessage = error;
    notifyListeners();
  }
  
  void _clearError() {
    _errorMessage = null;
    notifyListeners();
  }
  
  Future<Map<String, dynamic>> _getDeviceInfo() async {
    final deviceInfo = DeviceInfoPlugin();
    final packageInfo = await PackageInfo.fromPlatform();
    
    if (Platform.isAndroid) {
      final androidInfo = await deviceInfo.androidInfo;
      return {
        'platform': 'android',
        'model': androidInfo.model,
        'version': androidInfo.version.release,
        'app_version': packageInfo.version,
        'blessing': 'Divine-Android-Device',
      };
    } else if (Platform.isIOS) {
      final iosInfo = await deviceInfo.iosInfo;
      return {
        'platform': 'ios',
        'model': iosInfo.model,
        'version': iosInfo.systemVersion,
        'app_version': packageInfo.version,
        'blessing': 'Sacred-iOS-Device',
      };
    }
    
    return {
      'platform': 'unknown',
      'app_version': packageInfo.version,
      'blessing': 'Blessed-Unknown-Device',
    };
  }
  
  Future<void> _validateStoredToken(String token) async {
    try {
      final response = await http.get(
        Uri.parse('${SpiritualMobileConfig.baseApiUrl}/auth/validate'),
        headers: {'Authorization': 'Bearer $token'},
      ).timeout(SpiritualMobileConfig.requestTimeout);
      
      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        _currentUser = SpiritualUser.fromJson(data['user']);
        _isAuthenticated = true;
      } else {
        await _secureStorage.delete(key: 'auth_token');
      }
    } catch (e) {
      await _secureStorage.delete(key: 'auth_token');
    }
  }
  
  Future<void> _checkBiometricAvailability() async {
    try {
      final isAvailable = await _localAuth.canCheckBiometrics;
      final availableBiometrics = await _localAuth.getAvailableBiometrics();
      
      if (kDebugMode) {
        print('🔐 Biometric available: $isAvailable');
        print('🔐 Available biometrics: $availableBiometrics');
      }
    } catch (e) {
      if (kDebugMode) {
        print('❌ Biometric check error: $e');
      }
    }
  }
  
  Future<void> _storeUserData(SpiritualUser user) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('user_data', jsonEncode(user.toJson()));
  }
  
  Future<void> _loadStoredUserData() async {
    final prefs = await SharedPreferences.getInstance();
    final userData = prefs.getString('user_data');
    
    if (userData != null) {
      _currentUser = SpiritualUser.fromJson(jsonDecode(userData));
      _isAuthenticated = true;
      notifyListeners();
    }
  }
  
  Future<void> _clearUserData() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove('user_data');
  }
}

// 🌐 Spiritual API Service
class SpiritualApiService {
  static final SpiritualApiService _instance = SpiritualApiService._internal();
  factory SpiritualApiService() => _instance;
  SpiritualApiService._internal();
  
  final http.Client _client = http.Client();
  final FlutterSecureStorage _secureStorage = const FlutterSecureStorage();
  
  // Get authenticated headers with divine blessing
  Future<Map<String, String>> _getAuthHeaders() async {
    final token = await _secureStorage.read(key: 'auth_token');
    return {
      'Content-Type': 'application/json',
      'Authorization': 'Bearer ${token ?? ''}',
      'X-Spiritual-Blessing': 'Divine-API-Request',
    };
  }
  
  // Spiritual GET request
  Future<Map<String, dynamic>> get(String endpoint) async {
    try {
      final response = await _client.get(
        Uri.parse('${SpiritualMobileConfig.baseApiUrl}$endpoint'),
        headers: await _getAuthHeaders(),
      ).timeout(SpiritualMobileConfig.requestTimeout);
      
      return _handleResponse(response);
    } catch (e) {
      throw Exception('GET request failed: $e');
    }
  }
  
  // Spiritual POST request
  Future<Map<String, dynamic>> post(String endpoint, Map<String, dynamic> data) async {
    try {
      final response = await _client.post(
        Uri.parse('${SpiritualMobileConfig.baseApiUrl}$endpoint'),
        headers: await _getAuthHeaders(),
        body: jsonEncode({...data, 'blessing': 'Divine-POST-Request'}),
      ).timeout(SpiritualMobileConfig.requestTimeout);
      
      return _handleResponse(response);
    } catch (e) {
      throw Exception('POST request failed: $e');
    }
  }
  
  // Spiritual PUT request
  Future<Map<String, dynamic>> put(String endpoint, Map<String, dynamic> data) async {
    try {
      final response = await _client.put(
        Uri.parse('${SpiritualMobileConfig.baseApiUrl}$endpoint'),
        headers: await _getAuthHeaders(),
        body: jsonEncode({...data, 'blessing': 'Divine-PUT-Request'}),
      ).timeout(SpiritualMobileConfig.requestTimeout);
      
      return _handleResponse(response);
    } catch (e) {
      throw Exception('PUT request failed: $e');
    }
  }
  
  // Spiritual DELETE request
  Future<Map<String, dynamic>> delete(String endpoint) async {
    try {
      final response = await _client.delete(
        Uri.parse('${SpiritualMobileConfig.baseApiUrl}$endpoint'),
        headers: await _getAuthHeaders(),
      ).timeout(SpiritualMobileConfig.requestTimeout);
      
      return _handleResponse(response);
    } catch (e) {
      throw Exception('DELETE request failed: $e');
    }
  }
  
  Map<String, dynamic> _handleResponse(http.Response response) {
    if (response.statusCode >= 200 && response.statusCode < 300) {
      return jsonDecode(response.body);
    } else {
      final error = jsonDecode(response.body);
      throw Exception(error['message'] ?? 'API request failed');
    }
  }
  
  void dispose() {
    _client.close();
  }
}

// 🔔 Spiritual Notification Service
class SpiritualNotificationService {
  static final SpiritualNotificationService _instance = SpiritualNotificationService._internal();
  factory SpiritualNotificationService() => _instance;
  SpiritualNotificationService._internal();
  
  final FlutterLocalNotificationsPlugin _notifications = FlutterLocalNotificationsPlugin();
  bool _isInitialized = false;
  
  // Initialize notification service with divine blessing
  Future<void> initialize() async {
    if (_isInitialized) return;
    
    const androidSettings = AndroidInitializationSettings('@mipmap/ic_launcher');
    const iosSettings = DarwinInitializationSettings(
      requestAlertPermission: true,
      requestBadgePermission: true,
      requestSoundPermission: true,
    );
    
    const initSettings = InitializationSettings(
      android: androidSettings,
      iOS: iosSettings,
    );
    
    await _notifications.initialize(
      initSettings,
      onDidReceiveNotificationResponse: _onNotificationTapped,
    );
    
    // Request permissions
    await _requestPermissions();
    
    _isInitialized = true;
    
    if (kDebugMode) {
      print('🔔 Spiritual notification service initialized with divine blessing');
    }
  }
  
  // Show spiritual notification
  Future<void> showNotification({
    required String title,
    required String body,
    String? payload,
    int id = 0,
  }) async {
    if (!_isInitialized) await initialize();
    
    const androidDetails = AndroidNotificationDetails(
      'spiritual_channel',
      'Spiritual Notifications',
      channelDescription: 'Divine notifications with spiritual blessing',
      importance: Importance.high,
      priority: Priority.high,
      icon: '@mipmap/ic_launcher',
    );
    
    const iosDetails = DarwinNotificationDetails(
      presentAlert: true,
      presentBadge: true,
      presentSound: true,
    );
    
    const notificationDetails = NotificationDetails(
      android: androidDetails,
      iOS: iosDetails,
    );
    
    await _notifications.show(
      id,
      '🙏 $title',
      '✨ $body',
      notificationDetails,
      payload: payload,
    );
  }
  
  // Schedule spiritual notification
  Future<void> scheduleNotification({
    required String title,
    required String body,
    required DateTime scheduledDate,
    String? payload,
    int id = 0,
  }) async {
    if (!_isInitialized) await initialize();
    
    const androidDetails = AndroidNotificationDetails(
      'spiritual_scheduled_channel',
      'Scheduled Spiritual Notifications',
      channelDescription: 'Scheduled divine notifications with spiritual blessing',
      importance: Importance.high,
      priority: Priority.high,
    );
    
    const iosDetails = DarwinNotificationDetails();
    
    const notificationDetails = NotificationDetails(
      android: androidDetails,
      iOS: iosDetails,
    );
    
    await _notifications.zonedSchedule(
      id,
      '🙏 $title',
      '✨ $body',
      scheduledDate,
      notificationDetails,
      payload: payload,
      uiLocalNotificationDateInterpretation: UILocalNotificationDateInterpretation.absoluteTime,
    );
  }
  
  Future<void> _requestPermissions() async {
    if (Platform.isAndroid) {
      await Permission.notification.request();
    } else if (Platform.isIOS) {
      await _notifications
          .resolvePlatformSpecificImplementation<IOSFlutterLocalNotificationsPlugin>()
          ?.requestPermissions(
            alert: true,
            badge: true,
            sound: true,
          );
    }
  }
  
  void _onNotificationTapped(NotificationResponse response) {
    if (kDebugMode) {
      print('🔔 Notification tapped: ${response.payload}');
    }
    // Handle notification tap
  }
}

// 🎨 Spiritual Theme Provider
class SpiritualThemeProvider extends ChangeNotifier {
  bool _isDarkMode = false;
  String _currentTheme = 'light';
  
  bool get isDarkMode => _isDarkMode;
  String get currentTheme => _currentTheme;
  
  ThemeData get lightTheme => ThemeData(
    useMaterial3: true,
    colorScheme: ColorScheme.fromSeed(
      seedColor: SpiritualMobileConfig.lightTheme['primary'],
      brightness: Brightness.light,
    ),
    appBarTheme: AppBarTheme(
      backgroundColor: SpiritualMobileConfig.lightTheme['primary'],
      foregroundColor: Colors.white,
      elevation: 0,
    ),
    elevatedButtonTheme: ElevatedButtonThemeData(
      style: ElevatedButton.styleFrom(
        backgroundColor: SpiritualMobileConfig.lightTheme['primary'],
        foregroundColor: Colors.white,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(12),
        ),
      ),
    ),
    cardTheme: CardTheme(
      elevation: 4,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(16),
      ),
    ),
    inputDecorationTheme: InputDecorationTheme(
      border: OutlineInputBorder(
        borderRadius: BorderRadius.circular(12),
      ),
      filled: true,
      fillColor: Colors.grey[50],
    ),
  );
  
  ThemeData get darkTheme => ThemeData(
    useMaterial3: true,
    colorScheme: ColorScheme.fromSeed(
      seedColor: SpiritualMobileConfig.darkTheme['primary'],
      brightness: Brightness.dark,
    ),
    appBarTheme: AppBarTheme(
      backgroundColor: SpiritualMobileConfig.darkTheme['surface'],
      foregroundColor: Colors.white,
      elevation: 0,
    ),
    elevatedButtonTheme: ElevatedButtonThemeData(
      style: ElevatedButton.styleFrom(
        backgroundColor: SpiritualMobileConfig.darkTheme['primary'],
        foregroundColor: Colors.white,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(12),
        ),
      ),
    ),
    cardTheme: CardTheme(
      elevation: 4,
      color: SpiritualMobileConfig.darkTheme['surface'],
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(16),
      ),
    ),
    inputDecorationTheme: InputDecorationTheme(
      border: OutlineInputBorder(
        borderRadius: BorderRadius.circular(12),
      ),
      filled: true,
      fillColor: Colors.grey[800],
    ),
  );
  
  Future<void> initialize() async {
    final prefs = await SharedPreferences.getInstance();
    _isDarkMode = prefs.getBool('dark_mode') ?? false;
    _currentTheme = _isDarkMode ? 'dark' : 'light';
    notifyListeners();
  }
  
  Future<void> toggleTheme() async {
    _isDarkMode = !_isDarkMode;
    _currentTheme = _isDarkMode ? 'dark' : 'light';
    
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool('dark_mode', _isDarkMode);
    
    notifyListeners();
  }
}

// 🏠 Spiritual Home Screen
class SpiritualHomeScreen extends StatefulWidget {
  const SpiritualHomeScreen({super.key});
  
  @override
  State<SpiritualHomeScreen> createState() => _SpiritualHomeScreenState();
}

class _SpiritualHomeScreenState extends State<SpiritualHomeScreen>
    with TickerProviderStateMixin {
  late AnimationController _animationController;
  late Animation<double> _fadeAnimation;
  late Animation<Offset> _slideAnimation;
  
  @override
  void initState() {
    super.initState();
    
    _animationController = AnimationController(
      duration: const Duration(milliseconds: 1500),
      vsync: this,
    );
    
    _fadeAnimation = Tween<double>(
      begin: 0.0,
      end: 1.0,
    ).animate(CurvedAnimation(
      parent: _animationController,
      curve: Curves.easeInOut,
    ));
    
    _slideAnimation = Tween<Offset>(
      begin: const Offset(0, 0.3),
      end: Offset.zero,
    ).animate(CurvedAnimation(
      parent: _animationController,
      curve: Curves.easeOutCubic,
    ));
    
    _animationController.forward();
  }
  
  @override
  void dispose() {
    _animationController.dispose();
    super.dispose();
  }
  
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Row(
          children: [
            Icon(
              Icons.auto_awesome,
              color: SpiritualMobileConfig.spiritualColors['divine_gold'],
            ),
            const SizedBox(width: 8),
            const Text('ZeroLight Orbit'),
          ],
        ),
        actions: [
          Consumer<SpiritualThemeProvider>(
            builder: (context, themeProvider, child) {
              return IconButton(
                icon: Icon(
                  themeProvider.isDarkMode ? Icons.light_mode : Icons.dark_mode,
                ),
                onPressed: () => themeProvider.toggleTheme(),
              );
            },
          ),
          Consumer<SpiritualAuthService>(
            builder: (context, authService, child) {
              return PopupMenuButton(
                icon: CircleAvatar(
                  backgroundImage: authService.currentUser?.profileImageUrl != null
                      ? NetworkImage(authService.currentUser!.profileImageUrl!)
                      : null,
                  child: authService.currentUser?.profileImageUrl == null
                      ? const Icon(Icons.person)
                      : null,
                ),
                itemBuilder: (context) => [
                  PopupMenuItem(
                    child: const ListTile(
                      leading: Icon(Icons.person),
                      title: Text('Profile'),
                    ),
                    onTap: () => _navigateToProfile(),
                  ),
                  PopupMenuItem(
                    child: const ListTile(
                      leading: Icon(Icons.settings),
                      title: Text('Settings'),
                    ),
                    onTap: () => _navigateToSettings(),
                  ),
                  PopupMenuItem(
                    child: const ListTile(
                      leading: Icon(Icons.logout),
                      title: Text('Logout'),
                    ),
                    onTap: () => _logout(),
                  ),
                ],
              );
            },
          ),
        ],
      ),
      body: AnimatedBuilder(
        animation: _animationController,
        builder: (context, child) {
          return FadeTransition(
            opacity: _fadeAnimation,
            child: SlideTransition(
              position: _slideAnimation,
              child: _buildHomeContent(),
            ),
          );
        },
      ),
      floatingActionButton: FloatingActionButton.extended(
        onPressed: _showSpiritualFeatures,
        icon: const Icon(Icons.auto_awesome),
        label: const Text('Spiritual Features'),
        backgroundColor: SpiritualMobileConfig.spiritualColors['divine_gold'],
      ),
    );
  }
  
  Widget _buildHomeContent() {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Welcome Card
          Card(
            child: Padding(
              padding: const EdgeInsets.all(20),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Icon(
                        Icons.waving_hand,
                        color: SpiritualMobileConfig.spiritualColors['divine_gold'],
                        size: 32,
                      ),
                      const SizedBox(width: 12),
                      Expanded(
                        child: Consumer<SpiritualAuthService>(
                          builder: (context, authService, child) {
                            return Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Text(
                                  'Welcome back,',
                                  style: Theme.of(context).textTheme.titleMedium,
                                ),
                                Text(
                                  authService.currentUser?.displayName ?? 'Blessed User',
                                  style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                                    fontWeight: FontWeight.bold,
                                  ),
                                ),
                              ],
                            );
                          },
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 16),
                  Container(
                    padding: const EdgeInsets.all(12),
                    decoration: BoxDecoration(
                      color: SpiritualMobileConfig.spiritualColors['blessed_green']?.withOpacity(0.1),
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: Row(
                      children: [
                        Icon(
                          Icons.verified,
                          color: SpiritualMobileConfig.spiritualColors['blessed_green'],
                        ),
                        const SizedBox(width: 8),
                        const Text('🙏 Blessed with divine protection'),
                      ],
                    ),
                  ),
                ],
              ),
            ),
          ),
          
          const SizedBox(height: 24),
          
          // Quick Actions
          Text(
            'Quick Actions',
            style: Theme.of(context).textTheme.headlineSmall?.copyWith(
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 16),
          
          GridView.count(
            shrinkWrap: true,
            physics: const NeverScrollableScrollPhysics(),
            crossAxisCount: 2,
            crossAxisSpacing: 16,
            mainAxisSpacing: 16,
            children: [
              _buildQuickActionCard(
                icon: Icons.analytics,
                title: 'Analytics',
                subtitle: 'View insights',
                color: SpiritualMobileConfig.spiritualColors['sacred_blue']!,
                onTap: () => _navigateToAnalytics(),
              ),
              _buildQuickActionCard(
                icon: Icons.security,
                title: 'Security',
                subtitle: 'Manage security',
                color: SpiritualMobileConfig.spiritualColors['blessed_green']!,
                onTap: () => _navigateToSecurity(),
              ),
              _buildQuickActionCard(
                icon: Icons.notifications,
                title: 'Notifications',
                subtitle: 'View messages',
                color: SpiritualMobileConfig.spiritualColors['spiritual_purple']!,
                onTap: () => _navigateToNotifications(),
              ),
              _buildQuickActionCard(
                icon: Icons.help,
                title: 'Support',
                subtitle: 'Get help',
                color: SpiritualMobileConfig.spiritualColors['angelic_pink']!,
                onTap: () => _navigateToSupport(),
              ),
            ],
          ),
          
          const SizedBox(height: 24),
          
          // Recent Activity
          Text(
            'Recent Activity',
            style: Theme.of(context).textTheme.headlineSmall?.copyWith(
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 16),
          
          Card(
            child: ListView.separated(
              shrinkWrap: true,
              physics: const NeverScrollableScrollPhysics(),
              itemCount: 5,
              separatorBuilder: (context, index) => const Divider(),
              itemBuilder: (context, index) {
                return ListTile(
                  leading: CircleAvatar(
                    backgroundColor: SpiritualMobileConfig.spiritualColors['divine_gold']?.withOpacity(0.2),
                    child: Icon(
                      Icons.auto_awesome,
                      color: SpiritualMobileConfig.spiritualColors['divine_gold'],
                    ),
                  ),
                  title: Text('Spiritual Activity ${index + 1}'),
                  subtitle: Text('Divine action performed with blessing'),
                  trailing: Text(
                    '${index + 1}h ago',
                    style: Theme.of(context).textTheme.bodySmall,
                  ),
                );
              },
            ),
          ),
        ],
      ),
    );
  }
  
  Widget _buildQuickActionCard({
    required IconData icon,
    required String title,
    required String subtitle,
    required Color color,
    required VoidCallback onTap,
  }) {
    return Card(
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(16),
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: color.withOpacity(0.1),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Icon(
                  icon,
                  color: color,
                  size: 32,
                ),
              ),
              const SizedBox(height: 12),
              Text(
                title,
                style: Theme.of(context).textTheme.titleMedium?.copyWith(
                  fontWeight: FontWeight.bold,
                ),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 4),
              Text(
                subtitle,
                style: Theme.of(context).textTheme.bodySmall,
                textAlign: TextAlign.center,
              ),
            ],
          ),
        ),
      ),
    );
  }
  
  void _navigateToProfile() {
    // Navigate to profile screen
  }
  
  void _navigateToSettings() {
    // Navigate to settings screen
  }
  
  void _navigateToAnalytics() {
    // Navigate to analytics screen
  }
  
  void _navigateToSecurity() {
    // Navigate to security screen
  }
  
  void _navigateToNotifications() {
    // Navigate to notifications screen
  }
  
  void _navigateToSupport() {
    // Navigate to support screen
  }
  
  void _logout() {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Logout'),
        content: const Text('Are you sure you want to logout?'),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancel'),
          ),
          ElevatedButton(
            onPressed: () {
              Navigator.pop(context);
              context.read<SpiritualAuthService>().logout();
            },
            child: const Text('Logout'),
          ),
        ],
      ),
    );
  }
  
  void _showSpiritualFeatures() {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder: (context) => DraggableScrollableSheet(
        initialChildSize: 0.7,
        maxChildSize: 0.9,
        minChildSize: 0.5,
        builder: (context, scrollController) {
          return Container(
            padding: const EdgeInsets.all(20),
            child: Column(
              children: [
                Container(
                  width: 40,
                  height: 4,
                  decoration: BoxDecoration(
                    color: Colors.grey[300],
                    borderRadius: BorderRadius.circular(2),
                  ),
                ),
                const SizedBox(height: 20),
                Text(
                  '✨ Spiritual Features',
                  style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                    fontWeight: FontWeight.bold,
                  ),
                ),
                const SizedBox(height: 20),
                Expanded(
                  child: ListView.builder(
                    controller: scrollController,
                    itemCount: SpiritualMobileConfig.spiritualFeatures.length,
                    itemBuilder: (context, index) {
                      final feature = SpiritualMobileConfig.spiritualFeatures[index];
                      return Card(
                        margin: const EdgeInsets.only(bottom: 12),
                        child: ListTile(
                          leading: CircleAvatar(
                            backgroundColor: SpiritualMobileConfig.spiritualColors['divine_gold']?.withOpacity(0.2),
                            child: Icon(
                              Icons.auto_awesome,
                              color: SpiritualMobileConfig.spiritualColors['divine_gold'],
                            ),
                          ),
                          title: Text(feature),
                          subtitle: const Text('Blessed with divine power'),
                          trailing: const Icon(Icons.arrow_forward_ios),
                          onTap: () {
                            // Handle feature tap
                            Navigator.pop(context);
                            ScaffoldMessenger.of(context).showSnackBar(
                              SnackBar(
                                content: Text('🙏 $feature activated with divine blessing'),
                                backgroundColor: SpiritualMobileConfig.spiritualColors['blessed_green'],
                              ),
                            );
                          },
                        ),
                      );
                    },
                  ),
                ),
              ],
            ),
          );
        },
      ),
    );
  }
}

// 🔐 Spiritual Login Screen
class SpiritualLoginScreen extends StatefulWidget {
  const SpiritualLoginScreen({super.key});
  
  @override
  State<SpiritualLoginScreen> createState() => _SpiritualLoginScreenState();
}

class _SpiritualLoginScreenState extends State<SpiritualLoginScreen>
    with TickerProviderStateMixin {
  final _formKey = GlobalKey<FormState>();
  final _usernameController = TextEditingController();
  final _passwordController = TextEditingController();
  
  late AnimationController _animationController;
  late Animation<double> _fadeAnimation;
  late Animation<Offset> _slideAnimation;
  
  bool _obscurePassword = true;
  
  @override
  void initState() {
    super.initState();
    
    _animationController = AnimationController(
      duration: const Duration(milliseconds: 2000),
      vsync: this,
    );
    
    _fadeAnimation = Tween<double>(
      begin: 0.0,
      end: 1.0,
    ).animate(CurvedAnimation(
      parent: _animationController,
      curve: Curves.easeInOut,
    ));
    
    _slideAnimation = Tween<Offset>(
      begin: const Offset(0, 0.5),
      end: Offset.zero,
    ).animate(CurvedAnimation(
      parent: _animationController,
      curve: Curves.easeOutCubic,
    ));
    
    _animationController.forward();
  }
  
  @override
  void dispose() {
    _animationController.dispose();
    _usernameController.dispose();
    _passwordController.dispose();
    super.dispose();
  }
  
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: AnimatedBuilder(
        animation: _animationController,
        builder: (context, child) {
          return FadeTransition(
            opacity: _fadeAnimation,
            child: SlideTransition(
              position: _slideAnimation,
              child: _buildLoginContent(),
            ),
          );
        },
      ),
    );
  }
  
  Widget _buildLoginContent() {
    return SafeArea(
      child: SingleChildScrollView(
        padding: const EdgeInsets.all(24),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.center,
          children: [
            const SizedBox(height: 60),
            
            // Logo and Title
            Container(
              padding: const EdgeInsets.all(20),
              decoration: BoxDecoration(
                color: SpiritualMobileConfig.spiritualColors['divine_gold']?.withOpacity(0.1),
                borderRadius: BorderRadius.circular(20),
              ),
              child: Icon(
                Icons.auto_awesome,
                size: 80,
                color: SpiritualMobileConfig.spiritualColors['divine_gold'],
              ),
            ),
            
            const SizedBox(height: 24),
            
            Text(
              'ZeroLight Orbit',
              style: Theme.of(context).textTheme.headlineMedium?.copyWith(
                fontWeight: FontWeight.bold,
                color: SpiritualMobileConfig.spiritualColors['sacred_blue'],
              ),
            ),
            
            const SizedBox(height: 8),
            
            Text(
              '🙏 In The Name of GOD',
              style: Theme.of(context).textTheme.titleMedium?.copyWith(
                color: SpiritualMobileConfig.spiritualColors['divine_gold'],
              ),
            ),
            
            const SizedBox(height: 48),
            
            // Login Form
            Form(
              key: _formKey,
              child: Column(
                children: [
                  TextFormField(
                    controller: _usernameController,
                    decoration: const InputDecoration(
                      labelText: 'Username or Email',
                      prefixIcon: Icon(Icons.person),
                    ),
                    validator: (value) {
                      if (value == null || value.isEmpty) {
                        return 'Please enter your username or email';
                      }
                      return null;
                    },
                  ),
                  
                  const SizedBox(height: 16),
                  
                  TextFormField(
                    controller: _passwordController,
                    obscureText: _obscurePassword,
                    decoration: InputDecoration(
                      labelText: 'Password',
                      prefixIcon: const Icon(Icons.lock),
                      suffixIcon: IconButton(
                        icon: Icon(
                          _obscurePassword ? Icons.visibility : Icons.visibility_off,
                        ),
                        onPressed: () {
                          setState(() {
                            _obscurePassword = !_obscurePassword;
                          });
                        },
                      ),
                    ),
                    validator: (value) {
                      if (value == null || value.isEmpty) {
                        return 'Please enter your password';
                      }
                      return null;
                    },
                  ),
                  
                  const SizedBox(height: 24),
                  
                  // Login Button
                  Consumer<SpiritualAuthService>(
                    builder: (context, authService, child) {
                      return SizedBox(
                        width: double.infinity,
                        height: 50,
                        child: ElevatedButton(
                          onPressed: authService.isLoading ? null : _login,
                          child: authService.isLoading
                              ? const CircularProgressIndicator(color: Colors.white)
                              : const Text(
                                  '🙏 Login with Divine Blessing',
                                  style: TextStyle(fontSize: 16),
                                ),
                        ),
                      );
                    },
                  ),
                  
                  const SizedBox(height: 16),
                  
                  // Biometric Login Button
                  SizedBox(
                    width: double.infinity,
                    height: 50,
                    child: OutlinedButton.icon(
                      onPressed: _loginWithBiometrics,
                      icon: const Icon(Icons.fingerprint),
                      label: const Text('Login with Biometrics'),
                    ),
                  ),
                  
                  const SizedBox(height: 24),
                  
                  // Error Message
                  Consumer<SpiritualAuthService>(
                    builder: (context, authService, child) {
                      if (authService.errorMessage != null) {
                        return Container(
                          padding: const EdgeInsets.all(12),
                          decoration: BoxDecoration(
                            color: Colors.red.withOpacity(0.1),
                            borderRadius: BorderRadius.circular(8),
                            border: Border.all(color: Colors.red.withOpacity(0.3)),
                          ),
                          child: Row(
                            children: [
                              const Icon(Icons.error, color: Colors.red),
                              const SizedBox(width: 8),
                              Expanded(
                                child: Text(
                                  authService.errorMessage!,
                                  style: const TextStyle(color: Colors.red),
                                ),
                              ),
                            ],
                          ),
                        );
                      }
                      return const SizedBox.shrink();
                    },
                  ),
                  
                  const SizedBox(height: 24),
                  
                  // Forgot Password
                  TextButton(
                    onPressed: _forgotPassword,
                    child: const Text('Forgot Password?'),
                  ),
                  
                  const SizedBox(height: 16),
                  
                  // Sign Up
                  Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      const Text("Don't have an account? "),
                      TextButton(
                        onPressed: _signUp,
                        child: const Text('Sign Up'),
                      ),
                    ],
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
  
  void _login() async {
    if (_formKey.currentState!.validate()) {
      final authService = context.read<SpiritualAuthService>();
      final success = await authService.login(
        _usernameController.text.trim(),
        _passwordController.text,
      );
      
      if (success && mounted) {
        Navigator.pushReplacement(
          context,
          MaterialPageRoute(builder: (context) => const SpiritualHomeScreen()),
        );
      }
    }
  }
  
  void _loginWithBiometrics() async {
    final authService = context.read<SpiritualAuthService>();
    final success = await authService.authenticateWithBiometrics();
    
    if (success && mounted) {
      Navigator.pushReplacement(
        context,
        MaterialPageRoute(builder: (context) => const SpiritualHomeScreen()),
      );
    }
  }
  
  void _forgotPassword() {
    // Navigate to forgot password screen
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(
        content: Text('🙏 Password reset feature coming soon with divine blessing'),
      ),
    );
  }
  
  void _signUp() {
    // Navigate to sign up screen
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(
        content: Text('🙏 Sign up feature coming soon with divine blessing'),
      ),
    );
  }
}

// 🚀 Main Spiritual Mobile Application
class SpiritualMobileApp extends StatelessWidget {
  const SpiritualMobileApp({super.key});
  
  @override
  Widget build(BuildContext context) {
    return MultiProvider(
      providers: [
        ChangeNotifierProvider(create: (_) => SpiritualAuthService()),
        ChangeNotifierProvider(create: (_) => SpiritualThemeProvider()),
      ],
      child: Consumer<SpiritualThemeProvider>(
        builder: (context, themeProvider, child) {
          return MaterialApp(
            title: SpiritualMobileConfig.appName,
            debugShowCheckedModeBanner: false,
            theme: themeProvider.lightTheme,
            darkTheme: themeProvider.darkTheme,
            themeMode: themeProvider.isDarkMode ? ThemeMode.dark : ThemeMode.light,
            home: Consumer<SpiritualAuthService>(
              builder: (context, authService, child) {
                if (authService.isAuthenticated) {
                  return const SpiritualHomeScreen();
                } else {
                  return const SpiritualLoginScreen();
                }
              },
            ),
          );
        },
      ),
    );
  }
}

// 🎯 Main Application Entry Point
void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  
  // Display spiritual blessing
  displaySpiritualMobileBlessing();
  
  // Initialize services
  await _initializeServices();
  
  // Run the blessed application
  runApp(const SpiritualMobileApp());
}

// Initialize all spiritual services
Future<void> _initializeServices() async {
  try {
    // Initialize notification service
    await SpiritualNotificationService().initialize();
    
    // Initialize theme provider
    final themeProvider = SpiritualThemeProvider();
    await themeProvider.initialize();
    
    // Initialize auth service
    final authService = SpiritualAuthService();
    await authService.initialize();
    
    if (kDebugMode) {
      print('✨ All spiritual services initialized with divine blessing');
    }
    
  } catch (e) {
    if (kDebugMode) {
      print('❌ Error initializing services: $e');
    }
  }
}

// 🙏 Blessed Spiritual Mobile Application
// May this mobile app serve humanity with divine wisdom and blessing
// In The Name of GOD - بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيم
// Alhamdulillahi rabbil alameen - All praise to Allah, Lord of the worlds