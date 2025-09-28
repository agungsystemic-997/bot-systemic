/**
 * 🙏 In The Name of GOD - ZeroLight Orbit Lazy Loading System
 * Blessed Progressive Loading for Spiritual Performance
 * بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيم
 */

// 🌟 Spiritual Lazy Loading Configuration
const SPIRITUAL_LAZY_CONFIG = {
    rootMargin: '50px 0px',
    threshold: [0, 0.1, 0.25, 0.5, 0.75, 1.0],
    loadingClass: 'spiritual-loading',
    loadedClass: 'spiritual-loaded',
    errorClass: 'spiritual-error',
    fadeInDuration: 300,
    retryAttempts: 3,
    retryDelay: 1000,
    blessing: 'Divine-Progressive-Loading',
    guidance: 'Alhamdulillahi-rabbil-alameen'
};

// 🙏 Spiritual Blessing Display
function displaySpiritualBlessing() {
    console.log('🌟 ═══════════════════════════════════════════════════════════════');
    console.log('🙏 بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيم');
    console.log('✨ ZeroLight Orbit Lazy Loading - In The Name of GOD');
    console.log('🚀 Blessed Progressive Loading System');
    console.log('═══════════════════════════════════════════════════════════════ 🌟');
}

// 📊 Performance Metrics Tracker
class SpiritualPerformanceTracker {
    constructor() {
        this.metrics = {
            totalElements: 0,
            loadedElements: 0,
            failedElements: 0,
            averageLoadTime: 0,
            totalLoadTime: 0,
            bandwidthSaved: 0,
            blessing: 'Divine-Performance-Metrics'
        };
        this.loadTimes = [];
        this.startTime = performance.now();
    }

    trackElementLoad(element, loadTime, size = 0) {
        this.metrics.loadedElements++;
        this.loadTimes.push(loadTime);
        this.metrics.totalLoadTime += loadTime;
        this.metrics.averageLoadTime = this.metrics.totalLoadTime / this.metrics.loadedElements;
        this.metrics.bandwidthSaved += size;
        
        this.logMetrics(element, loadTime);
    }

    trackElementError(element, error) {
        this.metrics.failedElements++;
        console.warn(`⚠️ Spiritual loading failed for ${element.tagName}:`, error);
    }

    logMetrics(element, loadTime) {
        const elementInfo = element.dataset.src || element.src || element.tagName;
        console.log(`✨ Blessed loading: ${elementInfo} in ${loadTime.toFixed(2)}ms`);
    }

    getPerformanceReport() {
        const totalTime = performance.now() - this.startTime;
        return {
            ...this.metrics,
            totalTime: totalTime,
            successRate: (this.metrics.loadedElements / this.metrics.totalElements) * 100,
            blessing: 'Divine-Performance-Report-Complete'
        };
    }
}

// 🖼️ Spiritual Image Lazy Loader
class SpiritualImageLoader {
    constructor(tracker) {
        this.tracker = tracker;
        this.loadingImages = new Set();
        this.retryCount = new Map();
    }

    async loadImage(img) {
        if (this.loadingImages.has(img)) return;
        
        this.loadingImages.add(img);
        img.classList.add(SPIRITUAL_LAZY_CONFIG.loadingClass);
        
        const startTime = performance.now();
        const src = img.dataset.src || img.dataset.lazySrc;
        
        if (!src) {
            console.warn('⚠️ No source found for image:', img);
            return;
        }

        try {
            // Create placeholder with spiritual loading animation
            this.createSpiritualPlaceholder(img);
            
            // Preload image
            const preloadImg = new Image();
            
            // Handle different image formats (WebP, AVIF fallback)
            const sources = this.generateImageSources(src);
            
            for (const source of sources) {
                try {
                    await this.loadImageSource(preloadImg, source);
                    break; // Success, exit loop
                } catch (error) {
                    console.log(`🔄 Trying next format for ${img.alt || 'image'}`);
                    continue;
                }
            }
            
            // Apply loaded image with spiritual transition
            img.src = preloadImg.src;
            img.srcset = preloadImg.srcset || '';
            
            // Remove data attributes
            delete img.dataset.src;
            delete img.dataset.lazySrc;
            
            // Apply spiritual loading complete animation
            await this.applySpiritualTransition(img);
            
            const loadTime = performance.now() - startTime;
            this.tracker.trackElementLoad(img, loadTime, this.estimateImageSize(img));
            
        } catch (error) {
            await this.handleImageError(img, error);
        } finally {
            this.loadingImages.delete(img);
        }
    }

    generateImageSources(src) {
        const baseName = src.replace(/\.[^/.]+$/, '');
        const extension = src.split('.').pop().toLowerCase();
        
        // Return sources in order of preference (modern formats first)
        const sources = [];
        
        // AVIF (best compression)
        if (this.supportsFormat('avif')) {
            sources.push(`${baseName}.avif`);
        }
        
        // WebP (good compression, wide support)
        if (this.supportsFormat('webp')) {
            sources.push(`${baseName}.webp`);
        }
        
        // Original format (fallback)
        sources.push(src);
        
        return sources;
    }

    supportsFormat(format) {
        const canvas = document.createElement('canvas');
        canvas.width = 1;
        canvas.height = 1;
        
        switch (format) {
            case 'webp':
                return canvas.toDataURL('image/webp').indexOf('data:image/webp') === 0;
            case 'avif':
                return canvas.toDataURL('image/avif').indexOf('data:image/avif') === 0;
            default:
                return false;
        }
    }

    loadImageSource(img, src) {
        return new Promise((resolve, reject) => {
            img.onload = () => resolve(img);
            img.onerror = () => reject(new Error(`Failed to load ${src}`));
            img.src = src;
        });
    }

    createSpiritualPlaceholder(img) {
        // Create spiritual loading animation
        const placeholder = document.createElement('div');
        placeholder.className = 'spiritual-placeholder';
        placeholder.innerHTML = `
            <div class="spiritual-loading-animation">
                <div class="spiritual-spinner"></div>
                <div class="spiritual-blessing">🙏 Loading with divine grace...</div>
            </div>
        `;
        
        // Insert placeholder
        img.parentNode.insertBefore(placeholder, img);
        img.style.opacity = '0';
        
        // Store reference for cleanup
        img._spiritualPlaceholder = placeholder;
    }

    async applySpiritualTransition(img) {
        return new Promise((resolve) => {
            // Remove placeholder
            if (img._spiritualPlaceholder) {
                img._spiritualPlaceholder.remove();
                delete img._spiritualPlaceholder;
            }
            
            // Apply spiritual fade-in
            img.style.transition = `opacity ${SPIRITUAL_LAZY_CONFIG.fadeInDuration}ms ease-in-out`;
            img.style.opacity = '1';
            img.classList.remove(SPIRITUAL_LAZY_CONFIG.loadingClass);
            img.classList.add(SPIRITUAL_LAZY_CONFIG.loadedClass);
            
            setTimeout(resolve, SPIRITUAL_LAZY_CONFIG.fadeInDuration);
        });
    }

    async handleImageError(img, error) {
        const retryCount = this.retryCount.get(img) || 0;
        
        if (retryCount < SPIRITUAL_LAZY_CONFIG.retryAttempts) {
            this.retryCount.set(img, retryCount + 1);
            
            console.log(`🔄 Retrying image load (${retryCount + 1}/${SPIRITUAL_LAZY_CONFIG.retryAttempts})`);
            
            // Wait before retry with exponential backoff
            await new Promise(resolve => 
                setTimeout(resolve, SPIRITUAL_LAZY_CONFIG.retryDelay * Math.pow(2, retryCount))
            );
            
            return this.loadImage(img);
        }
        
        // Final failure - show error state
        img.classList.remove(SPIRITUAL_LAZY_CONFIG.loadingClass);
        img.classList.add(SPIRITUAL_LAZY_CONFIG.errorClass);
        
        // Create error placeholder
        const errorPlaceholder = document.createElement('div');
        errorPlaceholder.className = 'spiritual-error-placeholder';
        errorPlaceholder.innerHTML = `
            <div class="spiritual-error-content">
                <span class="spiritual-error-icon">⚠️</span>
                <span class="spiritual-error-message">Image could not be loaded</span>
                <button class="spiritual-retry-btn" onclick="window.spiritualLazyLoader.retryImage(this)">
                    🔄 Retry with Divine Grace
                </button>
            </div>
        `;
        
        img.parentNode.insertBefore(errorPlaceholder, img);
        img.style.display = 'none';
        
        this.tracker.trackElementError(img, error);
    }

    estimateImageSize(img) {
        // Estimate size based on dimensions (rough calculation)
        const width = img.naturalWidth || img.width || 800;
        const height = img.naturalHeight || img.height || 600;
        return Math.round((width * height * 3) / 1024); // Rough KB estimate
    }

    retryImage(button) {
        const errorPlaceholder = button.closest('.spiritual-error-placeholder');
        const img = errorPlaceholder.nextElementSibling;
        
        if (img && img.tagName === 'IMG') {
            errorPlaceholder.remove();
            img.style.display = '';
            img.classList.remove(SPIRITUAL_LAZY_CONFIG.errorClass);
            this.retryCount.delete(img);
            this.loadImage(img);
        }
    }
}

// 🎬 Spiritual Video Lazy Loader
class SpiritualVideoLoader {
    constructor(tracker) {
        this.tracker = tracker;
        this.loadingVideos = new Set();
    }

    async loadVideo(video) {
        if (this.loadingVideos.has(video)) return;
        
        this.loadingVideos.add(video);
        video.classList.add(SPIRITUAL_LAZY_CONFIG.loadingClass);
        
        const startTime = performance.now();
        
        try {
            // Load video sources
            const sources = video.querySelectorAll('source[data-src]');
            
            for (const source of sources) {
                source.src = source.dataset.src;
                delete source.dataset.src;
            }
            
            // Load main video source
            if (video.dataset.src) {
                video.src = video.dataset.src;
                delete video.dataset.src;
            }
            
            // Load poster image if present
            if (video.dataset.poster) {
                video.poster = video.dataset.poster;
                delete video.dataset.poster;
            }
            
            // Wait for video to be ready
            await this.waitForVideoReady(video);
            
            video.classList.remove(SPIRITUAL_LAZY_CONFIG.loadingClass);
            video.classList.add(SPIRITUAL_LAZY_CONFIG.loadedClass);
            
            const loadTime = performance.now() - startTime;
            this.tracker.trackElementLoad(video, loadTime);
            
        } catch (error) {
            video.classList.remove(SPIRITUAL_LAZY_CONFIG.loadingClass);
            video.classList.add(SPIRITUAL_LAZY_CONFIG.errorClass);
            this.tracker.trackElementError(video, error);
        } finally {
            this.loadingVideos.delete(video);
        }
    }

    waitForVideoReady(video) {
        return new Promise((resolve, reject) => {
            const timeout = setTimeout(() => {
                reject(new Error('Video load timeout'));
            }, 10000);
            
            video.addEventListener('loadeddata', () => {
                clearTimeout(timeout);
                resolve();
            }, { once: true });
            
            video.addEventListener('error', () => {
                clearTimeout(timeout);
                reject(new Error('Video load error'));
            }, { once: true });
            
            video.load();
        });
    }
}

// 📄 Spiritual Content Lazy Loader
class SpiritualContentLoader {
    constructor(tracker) {
        this.tracker = tracker;
        this.loadingContent = new Set();
    }

    async loadContent(element) {
        if (this.loadingContent.has(element)) return;
        
        this.loadingContent.add(element);
        element.classList.add(SPIRITUAL_LAZY_CONFIG.loadingClass);
        
        const startTime = performance.now();
        const contentUrl = element.dataset.src || element.dataset.content;
        
        if (!contentUrl) {
            console.warn('⚠️ No content URL found for element:', element);
            return;
        }

        try {
            // Fetch content with spiritual blessing
            const response = await fetch(contentUrl, {
                headers: {
                    'X-Spiritual-Blessing': SPIRITUAL_LAZY_CONFIG.blessing,
                    'X-Divine-Guidance': SPIRITUAL_LAZY_CONFIG.guidance
                }
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const content = await response.text();
            
            // Apply content with spiritual transition
            await this.applySpiritualContentTransition(element, content);
            
            const loadTime = performance.now() - startTime;
            this.tracker.trackElementLoad(element, loadTime, content.length);
            
        } catch (error) {
            element.classList.remove(SPIRITUAL_LAZY_CONFIG.loadingClass);
            element.classList.add(SPIRITUAL_LAZY_CONFIG.errorClass);
            element.innerHTML = `
                <div class="spiritual-content-error">
                    <span class="spiritual-error-icon">⚠️</span>
                    <span class="spiritual-error-message">Content could not be loaded</span>
                </div>
            `;
            this.tracker.trackElementError(element, error);
        } finally {
            this.loadingContent.delete(element);
        }
    }

    async applySpiritualContentTransition(element, content) {
        return new Promise((resolve) => {
            // Fade out current content
            element.style.transition = `opacity ${SPIRITUAL_LAZY_CONFIG.fadeInDuration / 2}ms ease-out`;
            element.style.opacity = '0';
            
            setTimeout(() => {
                // Replace content
                element.innerHTML = content;
                
                // Fade in new content
                element.style.transition = `opacity ${SPIRITUAL_LAZY_CONFIG.fadeInDuration}ms ease-in`;
                element.style.opacity = '1';
                element.classList.remove(SPIRITUAL_LAZY_CONFIG.loadingClass);
                element.classList.add(SPIRITUAL_LAZY_CONFIG.loadedClass);
                
                // Clean up data attributes
                delete element.dataset.src;
                delete element.dataset.content;
                
                setTimeout(resolve, SPIRITUAL_LAZY_CONFIG.fadeInDuration);
            }, SPIRITUAL_LAZY_CONFIG.fadeInDuration / 2);
        });
    }
}

// 🌟 Main Spiritual Lazy Loader
class SpiritualLazyLoader {
    constructor(options = {}) {
        this.config = { ...SPIRITUAL_LAZY_CONFIG, ...options };
        this.tracker = new SpiritualPerformanceTracker();
        this.imageLoader = new SpiritualImageLoader(this.tracker);
        this.videoLoader = new SpiritualVideoLoader(this.tracker);
        this.contentLoader = new SpiritualContentLoader(this.tracker);
        
        this.observer = null;
        this.elements = new Set();
        
        this.init();
    }

    init() {
        displaySpiritualBlessing();
        
        // Check for Intersection Observer support
        if (!('IntersectionObserver' in window)) {
            console.warn('⚠️ IntersectionObserver not supported, loading all content immediately');
            this.loadAllContent();
            return;
        }
        
        // Create intersection observer with spiritual configuration
        this.observer = new IntersectionObserver(
            this.handleIntersection.bind(this),
            {
                rootMargin: this.config.rootMargin,
                threshold: this.config.threshold
            }
        );
        
        // Find and observe all lazy elements
        this.findAndObserveElements();
        
        // Set up performance monitoring
        this.setupPerformanceMonitoring();
        
        console.log('✨ Spiritual Lazy Loader initialized with divine blessing');
    }

    findAndObserveElements() {
        // Find images with lazy loading
        const lazyImages = document.querySelectorAll('img[data-src], img[data-lazy-src]');
        lazyImages.forEach(img => {
            this.elements.add(img);
            this.observer.observe(img);
        });
        
        // Find videos with lazy loading
        const lazyVideos = document.querySelectorAll('video[data-src], video[data-poster]');
        lazyVideos.forEach(video => {
            this.elements.add(video);
            this.observer.observe(video);
        });
        
        // Find content elements with lazy loading
        const lazyContent = document.querySelectorAll('[data-content], [data-src]:not(img):not(video)');
        lazyContent.forEach(element => {
            this.elements.add(element);
            this.observer.observe(element);
        });
        
        this.tracker.metrics.totalElements = this.elements.size;
        console.log(`🔍 Found ${this.elements.size} elements for spiritual lazy loading`);
    }

    handleIntersection(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const element = entry.target;
                
                // Load based on element type
                if (element.tagName === 'IMG') {
                    this.imageLoader.loadImage(element);
                } else if (element.tagName === 'VIDEO') {
                    this.videoLoader.loadVideo(element);
                } else {
                    this.contentLoader.loadContent(element);
                }
                
                // Stop observing this element
                this.observer.unobserve(element);
                this.elements.delete(element);
            }
        });
    }

    setupPerformanceMonitoring() {
        // Monitor page load performance
        window.addEventListener('load', () => {
            setTimeout(() => {
                const report = this.tracker.getPerformanceReport();
                console.log('📊 Spiritual Performance Report:', report);
                
                // Send to analytics if configured
                if (window.gtag) {
                    window.gtag('event', 'spiritual_lazy_loading', {
                        custom_parameter_1: report.successRate,
                        custom_parameter_2: report.averageLoadTime,
                        custom_parameter_3: report.bandwidthSaved
                    });
                }
            }, 2000);
        });
        
        // Monitor visibility changes
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                console.log('🌙 Page hidden - pausing spiritual loading');
            } else {
                console.log('☀️ Page visible - resuming spiritual loading');
            }
        });
    }

    loadAllContent() {
        // Fallback for browsers without IntersectionObserver
        const allLazyElements = document.querySelectorAll('[data-src], [data-lazy-src], [data-content]');
        
        allLazyElements.forEach(element => {
            if (element.tagName === 'IMG') {
                this.imageLoader.loadImage(element);
            } else if (element.tagName === 'VIDEO') {
                this.videoLoader.loadVideo(element);
            } else {
                this.contentLoader.loadContent(element);
            }
        });
    }

    // Public API methods
    refresh() {
        this.findAndObserveElements();
        console.log('🔄 Spiritual lazy loader refreshed');
    }

    destroy() {
        if (this.observer) {
            this.observer.disconnect();
        }
        this.elements.clear();
        console.log('🙏 Spiritual lazy loader destroyed with gratitude');
    }

    getPerformanceReport() {
        return this.tracker.getPerformanceReport();
    }

    // Retry failed image (called from error placeholder)
    retryImage(button) {
        this.imageLoader.retryImage(button);
    }
}

// 🎨 Spiritual CSS Styles
const spiritualStyles = `
.spiritual-loading {
    position: relative;
    overflow: hidden;
}

.spiritual-placeholder {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: linear-gradient(45deg, #f0f0f0, #e0e0e0);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1;
}

.spiritual-loading-animation {
    text-align: center;
    color: #666;
}

.spiritual-spinner {
    width: 40px;
    height: 40px;
    border: 3px solid #f3f3f3;
    border-top: 3px solid #4CAF50;
    border-radius: 50%;
    animation: spiritual-spin 1s linear infinite;
    margin: 0 auto 10px;
}

.spiritual-blessing {
    font-size: 12px;
    font-style: italic;
    opacity: 0.8;
}

@keyframes spiritual-spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

.spiritual-loaded {
    animation: spiritual-fade-in 0.3s ease-in-out;
}

@keyframes spiritual-fade-in {
    from { opacity: 0; }
    to { opacity: 1; }
}

.spiritual-error-placeholder {
    padding: 20px;
    text-align: center;
    background: #f8f8f8;
    border: 1px dashed #ddd;
    border-radius: 4px;
}

.spiritual-error-content {
    color: #666;
}

.spiritual-error-icon {
    font-size: 24px;
    display: block;
    margin-bottom: 10px;
}

.spiritual-retry-btn {
    margin-top: 10px;
    padding: 8px 16px;
    background: #4CAF50;
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-size: 12px;
}

.spiritual-retry-btn:hover {
    background: #45a049;
}

.spiritual-content-error {
    padding: 20px;
    text-align: center;
    color: #666;
    background: #f8f8f8;
    border-radius: 4px;
}
`;

// 🚀 Auto-initialization
function initSpiritualLazyLoader() {
    // Inject spiritual styles
    const styleSheet = document.createElement('style');
    styleSheet.textContent = spiritualStyles;
    document.head.appendChild(styleSheet);
    
    // Initialize lazy loader when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => {
            window.spiritualLazyLoader = new SpiritualLazyLoader();
        });
    } else {
        window.spiritualLazyLoader = new SpiritualLazyLoader();
    }
}

// 🌟 Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        SpiritualLazyLoader,
        SpiritualImageLoader,
        SpiritualVideoLoader,
        SpiritualContentLoader,
        SpiritualPerformanceTracker,
        initSpiritualLazyLoader
    };
} else {
    // Browser environment - auto-initialize
    initSpiritualLazyLoader();
}

// 🙏 Blessed Lazy Loading System
// May this progressive loading serve users with divine performance and grace
// Alhamdulillahi rabbil alameen - All praise to Allah, Lord of the worlds