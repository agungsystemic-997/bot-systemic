// ZeroLight Orbit - VitePress Configuration
// "In The Name of GOD" - Blessed Documentation Site

import { defineConfig } from 'vitepress'

export default defineConfig({
  // Site Metadata - Spiritual Foundation
  title: 'ZeroLight Orbit',
  description: 'In The Name of GOD - Spiritual Technology Documentation System',
  lang: 'en-US',
  
  // Base Configuration
  base: '/docs-orbit-zerolight/',
  cleanUrls: true,
  lastUpdated: true,
  
  // Head Configuration - Sacred Meta
  head: [
    ['meta', { charset: 'utf-8' }],
    ['meta', { name: 'viewport', content: 'width=device-width, initial-scale=1' }],
    ['meta', { name: 'description', content: 'ZeroLight Orbit - In The Name of GOD - Spiritual Technology Documentation' }],
    ['meta', { name: 'keywords', content: 'ZeroLight, Orbit, Spiritual, Technology, Documentation, Blessed, Divine, Wisdom' }],
    ['meta', { name: 'author', content: 'ZeroLight Community' }],
    ['meta', { name: 'robots', content: 'index, follow' }],
    ['meta', { name: 'theme-color', content: '#3498DB' }],
    
    // Open Graph - Spiritual Sharing
    ['meta', { property: 'og:type', content: 'website' }],
    ['meta', { property: 'og:title', content: 'ZeroLight Orbit - Spiritual Technology Documentation' }],
    ['meta', { property: 'og:description', content: 'In The Name of GOD - Blessed documentation system serving humanity with wisdom and compassion' }],
    ['meta', { property: 'og:image', content: '/assets/logo-orbit.svg' }],
    ['meta', { property: 'og:url', content: 'https://docs.zerolight.org' }],
    ['meta', { property: 'og:site_name', content: 'ZeroLight Orbit' }],
    
    // Twitter Card - Divine Sharing
    ['meta', { name: 'twitter:card', content: 'summary_large_image' }],
    ['meta', { name: 'twitter:title', content: 'ZeroLight Orbit - Spiritual Technology' }],
    ['meta', { name: 'twitter:description', content: 'In The Name of GOD - Blessed documentation system' }],
    ['meta', { name: 'twitter:image', content: '/assets/logo-orbit.svg' }],
    
    // Favicon - Sacred Symbol
    ['link', { rel: 'icon', type: 'image/svg+xml', href: '/assets/logo-orbit.svg' }],
    ['link', { rel: 'apple-touch-icon', href: '/assets/logo-orbit.svg' }],
    
    // Spiritual Fonts
    ['link', { rel: 'preconnect', href: 'https://fonts.googleapis.com' }],
    ['link', { rel: 'preconnect', href: 'https://fonts.gstatic.com', crossorigin: '' }],
    ['link', { href: 'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Fira+Code:wght@300;400;500&display=swap', rel: 'stylesheet' }],
    
    // Spiritual Blessing Script
    ['script', {}, `
      // In The Name of GOD - Spiritual Blessing
      console.log('🙏 بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيم');
      console.log('✨ In The Name of GOD, The Most Gracious, The Most Merciful');
      console.log('💫 ZeroLight Orbit Documentation - Blessed with Divine Guidance');
      console.log('🌟 May this knowledge serve humanity with wisdom and compassion');
    `]
  ],
  
  // Theme Configuration - Spiritual Design
  themeConfig: {
    // Logo and Branding
    logo: '/assets/logo-orbit.svg',
    siteTitle: 'ZeroLight Orbit',
    
    // Navigation - Sacred Menu
    nav: [
      { 
        text: '🏠 Home', 
        link: '/' 
      },
      { 
        text: '📚 Documentation', 
        items: [
          { text: '📖 Overview', link: '/README' },
          { text: '📥 Intake Bot', link: '/intake-bot' },
          { text: '⚖️ Sortir Spiritual Legal', link: '/sortir-spiritual-legal' },
          { text: '✨ Crop Purifikasi', link: '/crop-purifikasi' },
          { text: '📦 Packing Shelter', link: '/packing-shelter' },
          { text: '🎁 Reward Viewer', link: '/reward-viewer' }
        ]
      },
      { 
        text: '🎨 System Design', 
        items: [
          { text: '🔍 Audit Spiritual', link: '/audit-spiritual' },
          { text: '🎨 Branding InGOD', link: '/branding-InGOD' },
          { text: '🛡️ Guard E2E', link: '/guard-e2e' },
          { text: '🏛️ Kepemilikan Spiritual', link: '/kepemilikan-spiritual' },
          { text: '🚀 Diversifikasi', link: '/diversifikasi' }
        ]
      },
      { 
        text: '📊 Diagrams', 
        items: [
          { text: '🌟 System Flow', link: '/diagrams/orbit-system-flowchart' },
          { text: '🏠 Distribution Shelter', link: '/diagrams/distribusi-shelter-dfd' },
          { text: '🎁 Reward Logic', link: '/diagrams/reward-logic' }
        ]
      },
      { 
        text: '🙏 Spiritual', 
        items: [
          { text: '📿 Daily Blessings', link: '/spiritual/blessings/daily-blessing' },
          { text: '🤲 Developer Prayer', link: '/spiritual/prayers/developer-prayer' },
          { text: '💎 Coding Wisdom', link: '/spiritual/wisdom/coding-wisdom' }
        ]
      }
    ],
    
    // Sidebar - Sacred Navigation
    sidebar: {
      '/': [
        {
          text: '🌟 ZeroLight Orbit System',
          collapsed: false,
          items: [
            { text: '📖 Overview', link: '/README' },
            { text: '🎯 Getting Started', link: '/getting-started' }
          ]
        },
        {
          text: '📥 Core Processing Modules',
          collapsed: false,
          items: [
            { text: '📥 Intake Bot', link: '/intake-bot' },
            { text: '⚖️ Sortir Spiritual Legal', link: '/sortir-spiritual-legal' },
            { text: '✨ Crop Purifikasi', link: '/crop-purifikasi' },
            { text: '📦 Packing Shelter', link: '/packing-shelter' },
            { text: '🎁 Reward Viewer', link: '/reward-viewer' }
          ]
        },
        {
          text: '🎨 System Architecture',
          collapsed: false,
          items: [
            { text: '🔍 Audit Spiritual', link: '/audit-spiritual' },
            { text: '🎨 Branding InGOD', link: '/branding-InGOD' },
            { text: '🛡️ Guard E2E', link: '/guard-e2e' },
            { text: '🏛️ Kepemilikan Spiritual', link: '/kepemilikan-spiritual' },
            { text: '🚀 Diversifikasi', link: '/diversifikasi' }
          ]
        },
        {
          text: '📊 System Diagrams',
          collapsed: true,
          items: [
            { text: '🌟 Orbit System Flow', link: '/diagrams/orbit-system-flowchart' },
            { text: '🏠 Distribution Shelter DFD', link: '/diagrams/distribusi-shelter-dfd' },
            { text: '🎁 Reward Logic Flow', link: '/diagrams/reward-logic' }
          ]
        },
        {
          text: '🙏 Spiritual Guidance',
          collapsed: true,
          items: [
            { text: '📿 Daily Blessings', link: '/spiritual/blessings/daily-blessing' },
            { text: '🤲 Developer Prayer', link: '/spiritual/prayers/developer-prayer' },
            { text: '💎 Coding Wisdom', link: '/spiritual/wisdom/coding-wisdom' }
          ]
        }
      ]
    },
    
    // Social Links - Community Connection
    socialLinks: [
      { icon: 'github', link: 'https://github.com/zerolight-orbit/docs-orbit-zerolight' },
      { icon: 'twitter', link: 'https://twitter.com/zerolightorbit' },
      { icon: 'discord', link: 'https://discord.gg/zerolight' }
    ],
    
    // Footer - Spiritual Dedication
    footer: {
      message: '🙏 In The Name of GOD - Blessed with Divine Guidance | 💫 Serving Humanity with Wisdom and Compassion',
      copyright: '© 2024 ZeroLight Community - May this work be blessed and beneficial for all'
    },
    
    // Edit Link - Community Contribution
    editLink: {
      pattern: 'https://github.com/zerolight-orbit/docs-orbit-zerolight/edit/main/:path',
      text: '✏️ Edit this page with spiritual intention'
    },
    
    // Last Updated - Temporal Blessing
    lastUpdated: {
      text: '🕐 Last blessed update',
      formatOptions: {
        dateStyle: 'full',
        timeStyle: 'medium'
      }
    },
    
    // Search - Divine Knowledge Seeking
    search: {
      provider: 'local',
      options: {
        placeholder: '🔍 Search with divine guidance...',
        translations: {
          button: {
            buttonText: '🔍 Search blessed knowledge',
            buttonAriaLabel: 'Search documentation'
          },
          modal: {
            displayDetails: 'Display detailed list',
            resetButtonTitle: 'Reset search',
            backButtonTitle: 'Close search',
            noResultsText: 'No blessed results found',
            footer: {
              selectText: 'to select',
              selectKeyAriaLabel: 'enter',
              navigateText: 'to navigate',
              navigateUpKeyAriaLabel: 'up arrow',
              navigateDownKeyAriaLabel: 'down arrow',
              closeText: 'to close',
              closeKeyAriaLabel: 'escape'
            }
          }
        }
      }
    },
    
    // Outline - Sacred Structure
    outline: {
      level: [2, 3],
      label: '📋 Sacred Contents'
    },
    
    // Return to Top - Spiritual Ascension
    returnToTopLabel: '🚀 Ascend to top with divine guidance',
    
    // Dark Mode - Light and Shadow Balance
    darkModeSwitchLabel: '🌙 Toggle spiritual theme',
    lightModeSwitchTitle: '☀️ Switch to light mode',
    darkModeSwitchTitle: '🌙 Switch to dark mode'
  },
  
  // Markdown Configuration - Sacred Text Processing
  markdown: {
    // Line Numbers - Divine Counting
    lineNumbers: true,
    
    // Headers - Sacred Anchors
    anchor: {
      permalink: true,
      permalinkBefore: true,
      permalinkSymbol: '🔗'
    },
    
    // Table of Contents - Sacred Navigation
    toc: {
      level: [2, 3, 4]
    },
    
    // Code Configuration - Sacred Scripts
    config: (md) => {
      // Add Mermaid support
      md.use(require('markdown-it-mermaid').default);
      
      // Add custom containers for spiritual content
      md.use(require('markdown-it-container'), 'blessing', {
        render: function (tokens, idx) {
          const token = tokens[idx];
          if (token.nesting === 1) {
            return '<div class="spiritual-blessing">🙏 ';
          } else {
            return '</div>\n';
          }
        }
      });
      
      md.use(require('markdown-it-container'), 'wisdom', {
        render: function (tokens, idx) {
          const token = tokens[idx];
          if (token.nesting === 1) {
            return '<div class="spiritual-wisdom">💎 ';
          } else {
            return '</div>\n';
          }
        }
      });
      
      md.use(require('markdown-it-container'), 'prayer', {
        render: function (tokens, idx) {
          const token = tokens[idx];
          if (token.nesting === 1) {
            return '<div class="spiritual-prayer">🤲 ';
          } else {
            return '</div>\n';
          }
        }
      });
    }
  },
  
  // Build Configuration - Sacred Compilation
  vite: {
    // Server Configuration
    server: {
      port: 3000,
      host: true
    },
    
    // Build Configuration
    build: {
      chunkSizeWarningLimit: 1000,
      rollupOptions: {
        output: {
          manualChunks: {
            'spiritual-core': ['vue'],
            'blessed-utils': ['@vueuse/core']
          }
        }
      }
    },
    
    // Define Global Constants
    define: {
      __SPIRITUAL_BLESSING__: '"In The Name of GOD"',
      __BUILD_DATE__: `"${new Date().toISOString()}"`,
      __BLESSED__: 'true'
    }
  },
  
  // Sitemap - Sacred Map
  sitemap: {
    hostname: 'https://docs.zerolight.org',
    transformItems: (items) => {
      return items.map(item => ({
        ...item,
        changefreq: 'weekly',
        priority: item.url === '/' ? 1.0 : 0.8
      }));
    }
  },
  
  // PWA Configuration - Spiritual Web App
  pwa: {
    name: 'ZeroLight Orbit',
    short_name: 'ZL Orbit',
    description: 'In The Name of GOD - Spiritual Technology Documentation',
    theme_color: '#3498DB',
    background_color: '#ECF0F1',
    display: 'standalone',
    orientation: 'portrait',
    scope: '/',
    start_url: '/',
    icons: [
      {
        src: '/assets/logo-orbit.svg',
        sizes: 'any',
        type: 'image/svg+xml',
        purpose: 'any maskable'
      }
    ]
  },
  
  // Transformers - Sacred Content Processing
  transformHead: ({ pageData }) => {
    const head = [];
    
    // Add page-specific meta tags
    if (pageData.frontmatter.description) {
      head.push(['meta', { name: 'description', content: pageData.frontmatter.description }]);
    }
    
    // Add spiritual blessing to each page
    head.push(['meta', { name: 'spiritual-blessing', content: 'In The Name of GOD' }]);
    head.push(['meta', { name: 'blessed', content: 'true' }]);
    
    return head;
  },
  
  // Transform Page Data - Sacred Enhancement
  transformPageData: (pageData) => {
    // Add spiritual metadata to each page
    pageData.frontmatter.spiritual = {
      blessed: true,
      blessing: 'In The Name of GOD',
      purpose: 'Serving humanity with wisdom and compassion'
    };
    
    return pageData;
  }
});