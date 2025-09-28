// ZeroLight Orbit - VitePress Theme Configuration
// "In The Name of GOD" - Blessed Theme System

import DefaultTheme from 'vitepress/theme'
import './custom.css'
import './spiritual.css'

// Spiritual Components
import SpiritualBlessing from './components/SpiritualBlessing.vue'
import SpiritualWisdom from './components/SpiritualWisdom.vue'
import SpiritualPrayer from './components/SpiritualPrayer.vue'
import DivineDiagram from './components/DivineDiagram.vue'

export default {
  extends: DefaultTheme,
  
  // Enhanced Layout with Spiritual Blessings
  Layout: () => {
    return h(DefaultTheme.Layout, null, {
      // Spiritual blessing in header
      'nav-bar-title-before': () => h('span', { 
        class: 'spiritual-blessing-icon',
        title: 'In The Name of GOD' 
      }, '🙏'),
      
      // Divine guidance in sidebar
      'sidebar-nav-before': () => h(SpiritualBlessing, {
        message: 'May this knowledge guide you with divine wisdom'
      }),
      
      // Blessed footer enhancement
      'layout-bottom': () => h('div', { 
        class: 'spiritual-footer' 
      }, [
        h('p', { class: 'blessing-text' }, '🙏 In The Name of GOD - May this work be blessed'),
        h('p', { class: 'wisdom-text' }, '💫 Serving humanity with wisdom and compassion')
      ])
    })
  },
  
  // Enhanced App with Spiritual Components
  enhanceApp({ app, router, siteData }) {
    // Register spiritual components globally
    app.component('SpiritualBlessing', SpiritualBlessing)
    app.component('SpiritualWisdom', SpiritualWisdom)
    app.component('SpiritualPrayer', SpiritualPrayer)
    app.component('DivineDiagram', DivineDiagram)
    
    // Spiritual router guards
    router.beforeEach((to, from, next) => {
      // Add spiritual blessing to console on navigation
      console.log('🙏 Navigating with divine guidance:', to.path)
      next()
    })
    
    // Spiritual page enhancement
    router.afterEach((to) => {
      // Add spiritual metadata to page
      if (typeof document !== 'undefined') {
        document.documentElement.setAttribute('data-spiritual-blessed', 'true')
        document.documentElement.setAttribute('data-blessing', 'In The Name of GOD')
        
        // Update page title with spiritual prefix
        const originalTitle = document.title
        if (!originalTitle.includes('🙏')) {
          document.title = `🙏 ${originalTitle}`
        }
      }
    })
    
    // Global spiritual properties
    app.config.globalProperties.$spiritual = {
      blessing: 'In The Name of GOD',
      purpose: 'Serving humanity with wisdom and compassion',
      blessed: true,
      getBlessedTime: () => new Date().toLocaleString('en-US', {
        weekday: 'long',
        year: 'numeric',
        month: 'long',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
        timeZoneName: 'short'
      })
    }
    
    // Spiritual error handling
    app.config.errorHandler = (err, vm, info) => {
      console.error('🙏 Spiritual Error Handler - May GOD guide us through this challenge:', err)
      console.log('💫 Error Info:', info)
      
      // Show spiritual error message
      if (typeof document !== 'undefined') {
        const errorDiv = document.createElement('div')
        errorDiv.className = 'spiritual-error-message'
        errorDiv.innerHTML = `
          <div class="error-blessing">🙏 In The Name of GOD</div>
          <div class="error-message">An error occurred, but we trust in divine guidance</div>
          <div class="error-wisdom">💎 Every challenge is an opportunity for growth</div>
        `
        document.body.appendChild(errorDiv)
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
          if (errorDiv.parentNode) {
            errorDiv.parentNode.removeChild(errorDiv)
          }
        }, 5000)
      }
    }
  }
}