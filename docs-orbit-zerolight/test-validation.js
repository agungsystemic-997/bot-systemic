#!/usr/bin/env node

/**
 * ZeroLight Orbit - Comprehensive Test & Validation System
 * "In The Name of GOD" - Blessed Testing Framework
 * 
 * This script performs comprehensive validation of the entire documentation system
 * to ensure 10+ year sustainability and GitHub Copilot compatibility.
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

// Spiritual Colors for Console Output
const colors = {
  divine: '\x1b[34m',    // Blue
  sacred: '\x1b[31m',    // Red  
  blessed: '\x1b[32m',   // Green
  golden: '\x1b[33m',    // Yellow
  pure: '\x1b[37m',      // White
  reset: '\x1b[0m'       // Reset
};

// Spiritual Blessing Display
function displayBlessing() {
  console.log(`${colors.divine}╔══════════════════════════════════════════════════════════════╗${colors.reset}`);
  console.log(`${colors.divine}║${colors.golden}                    🙏 In The Name of GOD 🙏                    ${colors.divine}║${colors.reset}`);
  console.log(`${colors.divine}║${colors.pure}              ZeroLight Orbit - Test & Validation              ${colors.divine}║${colors.reset}`);
  console.log(`${colors.divine}║${colors.blessed}           Blessed Testing Framework for Divine Guidance          ${colors.divine}║${colors.reset}`);
  console.log(`${colors.divine}╚══════════════════════════════════════════════════════════════╝${colors.reset}`);
  console.log();
}

// Test Results Tracking
const testResults = {
  passed: 0,
  failed: 0,
  warnings: 0,
  tests: []
};

// Utility Functions
function logTest(name, status, message = '') {
  const statusColor = status === 'PASS' ? colors.blessed : 
                     status === 'FAIL' ? colors.sacred : colors.golden;
  const icon = status === 'PASS' ? '✅' : status === 'FAIL' ? '❌' : '⚠️';
  
  console.log(`${icon} ${statusColor}[${status}]${colors.reset} ${name}`);
  if (message) {
    console.log(`   ${colors.pure}${message}${colors.reset}`);
  }
  
  testResults.tests.push({ name, status, message });
  if (status === 'PASS') testResults.passed++;
  else if (status === 'FAIL') testResults.failed++;
  else testResults.warnings++;
}

function fileExists(filePath) {
  return fs.existsSync(filePath);
}

function readFile(filePath) {
  try {
    return fs.readFileSync(filePath, 'utf8');
  } catch (error) {
    return null;
  }
}

function isValidJSON(str) {
  try {
    JSON.parse(str);
    return true;
  } catch {
    return false;
  }
}

// Test Categories
class SpiritualValidator {
  
  // 1. File Structure Validation
  validateFileStructure() {
    console.log(`${colors.divine}📁 Testing File Structure - Sacred Organization${colors.reset}`);
    
    const requiredFiles = [
      'README.md',
      'package.json',
      '.github/workflows/build-docs.yml',
      'spiritual-template.tex',
      'logo-orbit.svg',
      'setup-spiritual-environment.js',
      'deploy.js'
    ];
    
    const requiredDirs = [
      '.vitepress',
      '.vitepress/theme',
      'spiritual',
      'spiritual/blessings',
      'spiritual/prayers',
      'spiritual/wisdom',
      'diagrams'
    ];
    
    // Check required files
    requiredFiles.forEach(file => {
      if (fileExists(file)) {
        logTest(`File exists: ${file}`, 'PASS');
      } else {
        logTest(`File missing: ${file}`, 'FAIL', 'Required file not found');
      }
    });
    
    // Check required directories
    requiredDirs.forEach(dir => {
      if (fs.existsSync(dir) && fs.statSync(dir).isDirectory()) {
        logTest(`Directory exists: ${dir}`, 'PASS');
      } else {
        logTest(`Directory missing: ${dir}`, 'FAIL', 'Required directory not found');
      }
    });
    
    // Check core modules
    const coreModules = [
      'intake-bot.md',
      'sortir-spiritual-legal.md', 
      'crop-purifikasi.md',
      'packing-shelter.md',
      'reward-viewer.md',
      'audit-spiritual.md',
      'branding-InGOD.md',
      'guard-e2e.md',
      'kepemilikan-spiritual.md',
      'diversifikasi.md'
    ];
    
    coreModules.forEach(module => {
      if (fileExists(module)) {
        logTest(`Core module: ${module}`, 'PASS');
      } else {
        logTest(`Core module missing: ${module}`, 'FAIL', 'Essential system module not found');
      }
    });
  }
  
  // 2. Spiritual Content Validation
  validateSpiritualContent() {
    console.log(`${colors.divine}🙏 Testing Spiritual Content - Divine Validation${colors.reset}`);
    
    const spiritualFiles = [
      'README.md',
      'intake-bot.md',
      'sortir-spiritual-legal.md',
      'crop-purifikasi.md',
      'packing-shelter.md',
      'reward-viewer.md',
      'audit-spiritual.md',
      'branding-InGOD.md',
      'guard-e2e.md',
      'kepemilikan-spiritual.md',
      'diversifikasi.md'
    ];
    
    const requiredSpiritualElements = [
      'In The Name of GOD',
      '🙏',
      'Blessed',
      'Divine',
      'Spiritual'
    ];
    
    spiritualFiles.forEach(file => {
      if (fileExists(file)) {
        const content = readFile(file);
        if (content) {
          let hasSpiritual = false;
          requiredSpiritualElements.forEach(element => {
            if (content.includes(element)) {
              hasSpiritual = true;
            }
          });
          
          if (hasSpiritual) {
            logTest(`Spiritual content in ${file}`, 'PASS');
          } else {
            logTest(`Spiritual content in ${file}`, 'WARN', 'Missing spiritual elements');
          }
          
          // Check for Python code examples
          if (content.includes('```python') || content.includes('class ') || content.includes('def ')) {
            logTest(`Code examples in ${file}`, 'PASS');
          } else {
            logTest(`Code examples in ${file}`, 'WARN', 'No Python code examples found');
          }
        }
      }
    });
  }
  
  // 3. Mermaid Diagram Validation
  validateMermaidDiagrams() {
    console.log(`${colors.divine}📊 Testing Mermaid Diagrams - Sacred Visualization${colors.reset}`);
    
    const diagramFiles = [
      'orbit-system-flowchart.mmd',
      'distribusi-shelter-dfd.mmd',
      'reward-logic.mmd'
    ];
    
    diagramFiles.forEach(file => {
      const fullPath = path.join('diagrams', file);
      if (fileExists(fullPath)) {
        const content = readFile(fullPath);
        if (content) {
          // Check for valid Mermaid syntax
          if (content.includes('graph') || content.includes('flowchart') || 
              content.includes('classDiagram') || content.includes('sequenceDiagram')) {
            logTest(`Mermaid syntax in ${file}`, 'PASS');
          } else {
            logTest(`Mermaid syntax in ${file}`, 'FAIL', 'Invalid Mermaid diagram syntax');
          }
          
          // Check for styling
          if (content.includes('classDef') || content.includes('style')) {
            logTest(`Styling in ${file}`, 'PASS');
          } else {
            logTest(`Styling in ${file}`, 'WARN', 'No custom styling found');
          }
        }
      } else {
        logTest(`Diagram file: ${file}`, 'FAIL', 'Diagram file not found');
      }
    });
  }
  
  // 4. Package.json Validation
  validatePackageJson() {
    console.log(`${colors.divine}📦 Testing Package Configuration - Blessed Dependencies${colors.reset}`);
    
    if (fileExists('package.json')) {
      const content = readFile('package.json');
      if (content && isValidJSON(content)) {
        const pkg = JSON.parse(content);
        
        logTest('package.json syntax', 'PASS');
        
        // Check required fields
        const requiredFields = ['name', 'version', 'description', 'scripts', 'dependencies'];
        requiredFields.forEach(field => {
          if (pkg[field]) {
            logTest(`package.json field: ${field}`, 'PASS');
          } else {
            logTest(`package.json field: ${field}`, 'FAIL', `Missing required field: ${field}`);
          }
        });
        
        // Check required dependencies
        const requiredDeps = ['vitepress', '@mermaid-js/mermaid', 'markdown-it-mermaid'];
        if (pkg.dependencies) {
          requiredDeps.forEach(dep => {
            if (pkg.dependencies[dep] || (pkg.devDependencies && pkg.devDependencies[dep])) {
              logTest(`Dependency: ${dep}`, 'PASS');
            } else {
              logTest(`Dependency: ${dep}`, 'WARN', `Recommended dependency not found: ${dep}`);
            }
          });
        }
        
        // Check scripts
        const requiredScripts = ['dev', 'build', 'preview'];
        if (pkg.scripts) {
          requiredScripts.forEach(script => {
            if (pkg.scripts[script]) {
              logTest(`Script: ${script}`, 'PASS');
            } else {
              logTest(`Script: ${script}`, 'WARN', `Recommended script not found: ${script}`);
            }
          });
        }
      } else {
        logTest('package.json syntax', 'FAIL', 'Invalid JSON syntax');
      }
    } else {
      logTest('package.json exists', 'FAIL', 'Package.json file not found');
    }
  }
  
  // 5. GitHub Actions Validation
  validateGitHubActions() {
    console.log(`${colors.divine}⚙️ Testing GitHub Actions - Divine Automation${colors.reset}`);
    
    const workflowFile = '.github/workflows/build-docs.yml';
    if (fileExists(workflowFile)) {
      const content = readFile(workflowFile);
      if (content) {
        logTest('GitHub Actions workflow exists', 'PASS');
        
        // Check for required workflow elements
        const requiredElements = [
          'name:',
          'on:',
          'jobs:',
          'runs-on:',
          'steps:'
        ];
        
        requiredElements.forEach(element => {
          if (content.includes(element)) {
            logTest(`Workflow element: ${element}`, 'PASS');
          } else {
            logTest(`Workflow element: ${element}`, 'FAIL', `Missing workflow element: ${element}`);
          }
        });
        
        // Check for spiritual validation
        if (content.includes('spiritual') || content.includes('blessing')) {
          logTest('Spiritual validation in workflow', 'PASS');
        } else {
          logTest('Spiritual validation in workflow', 'WARN', 'No spiritual validation found');
        }
      }
    } else {
      logTest('GitHub Actions workflow', 'FAIL', 'Workflow file not found');
    }
  }
  
  // 6. VitePress Configuration Validation
  validateVitePressConfig() {
    console.log(`${colors.divine}⚡ Testing VitePress Configuration - Sacred Site Builder${colors.reset}`);
    
    const configFile = '.vitepress/config.js';
    if (fileExists(configFile)) {
      const content = readFile(configFile);
      if (content) {
        logTest('VitePress config exists', 'PASS');
        
        // Check for required configuration elements
        const requiredElements = [
          'defineConfig',
          'title:',
          'description:',
          'themeConfig:',
          'nav:',
          'sidebar:'
        ];
        
        requiredElements.forEach(element => {
          if (content.includes(element)) {
            logTest(`Config element: ${element}`, 'PASS');
          } else {
            logTest(`Config element: ${element}`, 'WARN', `Missing config element: ${element}`);
          }
        });
        
        // Check for spiritual elements
        if (content.includes('spiritual') || content.includes('blessing') || content.includes('In The Name of GOD')) {
          logTest('Spiritual elements in config', 'PASS');
        } else {
          logTest('Spiritual elements in config', 'WARN', 'No spiritual elements found');
        }
      }
    } else {
      logTest('VitePress config', 'FAIL', 'Config file not found');
    }
    
    // Check theme files
    const themeFiles = [
      '.vitepress/theme/index.js',
      '.vitepress/theme/spiritual.css'
    ];
    
    themeFiles.forEach(file => {
      if (fileExists(file)) {
        logTest(`Theme file: ${path.basename(file)}`, 'PASS');
      } else {
        logTest(`Theme file: ${path.basename(file)}`, 'WARN', 'Theme file not found');
      }
    });
  }
  
  // 7. LaTeX Template Validation
  validateLaTeXTemplate() {
    console.log(`${colors.divine}📄 Testing LaTeX Template - Sacred PDF Generation${colors.reset}`);
    
    if (fileExists('spiritual-template.tex')) {
      const content = readFile('spiritual-template.tex');
      if (content) {
        logTest('LaTeX template exists', 'PASS');
        
        // Check for required LaTeX elements
        const requiredElements = [
          '\\documentclass',
          '\\usepackage',
          '\\begin{document}',
          '\\end{document}'
        ];
        
        requiredElements.forEach(element => {
          if (content.includes(element)) {
            logTest(`LaTeX element: ${element}`, 'PASS');
          } else {
            logTest(`LaTeX element: ${element}`, 'FAIL', `Missing LaTeX element: ${element}`);
          }
        });
        
        // Check for spiritual customization
        if (content.includes('spiritual') || content.includes('blessing')) {
          logTest('Spiritual LaTeX customization', 'PASS');
        } else {
          logTest('Spiritual LaTeX customization', 'WARN', 'No spiritual customization found');
        }
      }
    } else {
      logTest('LaTeX template', 'FAIL', 'Template file not found');
    }
  }
  
  // 8. SVG Logo Validation
  validateSVGLogo() {
    console.log(`${colors.divine}🎨 Testing SVG Logo - Sacred Visual Identity${colors.reset}`);
    
    if (fileExists('logo-orbit.svg')) {
      const content = readFile('logo-orbit.svg');
      if (content) {
        logTest('SVG logo exists', 'PASS');
        
        // Check for valid SVG structure
        if (content.includes('<svg') && content.includes('</svg>')) {
          logTest('Valid SVG structure', 'PASS');
        } else {
          logTest('Valid SVG structure', 'FAIL', 'Invalid SVG format');
        }
        
        // Check for spiritual elements
        if (content.includes('In The Name of GOD') || content.includes('ZeroLight')) {
          logTest('Spiritual branding in logo', 'PASS');
        } else {
          logTest('Spiritual branding in logo', 'WARN', 'No spiritual branding found');
        }
        
        // Check for animations
        if (content.includes('<animate') || content.includes('animation')) {
          logTest('Logo animations', 'PASS');
        } else {
          logTest('Logo animations', 'WARN', 'No animations found');
        }
      }
    } else {
      logTest('SVG logo', 'FAIL', 'Logo file not found');
    }
  }
  
  // 9. Deployment Script Validation
  validateDeploymentScript() {
    console.log(`${colors.divine}🚀 Testing Deployment Script - Divine Distribution${colors.reset}`);
    
    if (fileExists('deploy.js')) {
      const content = readFile('deploy.js');
      if (content) {
        logTest('Deployment script exists', 'PASS');
        
        // Check for required functions
        const requiredFunctions = [
          'displayBlessing',
          'validateSpiritual',
          'buildDocumentation',
          'deployToGitHub',
          'createRelease'
        ];
        
        requiredFunctions.forEach(func => {
          if (content.includes(func)) {
            logTest(`Deploy function: ${func}`, 'PASS');
          } else {
            logTest(`Deploy function: ${func}`, 'WARN', `Function not found: ${func}`);
          }
        });
        
        // Check for spiritual validation
        if (content.includes('spiritual') && content.includes('blessing')) {
          logTest('Spiritual deployment validation', 'PASS');
        } else {
          logTest('Spiritual deployment validation', 'WARN', 'No spiritual validation found');
        }
      }
    } else {
      logTest('Deployment script', 'FAIL', 'Deploy script not found');
    }
  }
  
  // 10. GitHub Copilot Compatibility
  validateCopilotCompatibility() {
    console.log(`${colors.divine}🤖 Testing GitHub Copilot Compatibility - AI Blessing${colors.reset}`);
    
    // Check for clear code structure and comments
    const codeFiles = ['deploy.js', 'setup-spiritual-environment.js', 'test-validation.js'];
    
    codeFiles.forEach(file => {
      if (fileExists(file)) {
        const content = readFile(file);
        if (content) {
          // Check for comments
          const commentLines = content.split('\n').filter(line => 
            line.trim().startsWith('//') || line.trim().startsWith('/*') || line.trim().startsWith('*')
          );
          
          if (commentLines.length > 10) {
            logTest(`Comments in ${file}`, 'PASS', `${commentLines.length} comment lines found`);
          } else {
            logTest(`Comments in ${file}`, 'WARN', 'Insufficient comments for AI understanding');
          }
          
          // Check for function documentation
          const functionCount = (content.match(/function\s+\w+|const\s+\w+\s*=/g) || []).length;
          const docCommentCount = (content.match(/\/\*\*[\s\S]*?\*\//g) || []).length;
          
          if (docCommentCount > 0) {
            logTest(`Function docs in ${file}`, 'PASS', `${docCommentCount} documented functions`);
          } else {
            logTest(`Function docs in ${file}`, 'WARN', 'No JSDoc comments found');
          }
        }
      }
    });
    
    // Check for clear naming conventions
    const allFiles = fs.readdirSync('.').filter(f => f.endsWith('.md') || f.endsWith('.js'));
    let clearNaming = true;
    
    allFiles.forEach(file => {
      if (file.includes(' ') || file.includes('temp') || file.includes('test123')) {
        clearNaming = false;
      }
    });
    
    if (clearNaming) {
      logTest('Clear file naming', 'PASS');
    } else {
      logTest('Clear file naming', 'WARN', 'Some files have unclear names');
    }
  }
  
  // 11. Long-term Sustainability Check
  validateSustainability() {
    console.log(`${colors.divine}♾️ Testing Long-term Sustainability - Eternal Blessing${colors.reset}`);
    
    // Check for version pinning in package.json
    if (fileExists('package.json')) {
      const content = readFile('package.json');
      if (content && isValidJSON(content)) {
        const pkg = JSON.parse(content);
        
        if (pkg.dependencies) {
          let pinnedVersions = 0;
          let totalDeps = Object.keys(pkg.dependencies).length;
          
          Object.values(pkg.dependencies).forEach(version => {
            if (version.match(/^\d+\.\d+\.\d+$/)) {
              pinnedVersions++;
            }
          });
          
          if (pinnedVersions / totalDeps > 0.5) {
            logTest('Version pinning', 'PASS', `${pinnedVersions}/${totalDeps} dependencies pinned`);
          } else {
            logTest('Version pinning', 'WARN', 'Consider pinning more dependency versions');
          }
        }
      }
    }
    
    // Check for comprehensive documentation
    const docFiles = fs.readdirSync('.').filter(f => f.endsWith('.md'));
    if (docFiles.length >= 10) {
      logTest('Comprehensive documentation', 'PASS', `${docFiles.length} documentation files`);
    } else {
      logTest('Comprehensive documentation', 'WARN', 'Consider adding more documentation');
    }
    
    // Check for backup and recovery procedures
    const hasBackupDocs = docFiles.some(file => {
      const content = readFile(file) || '';
      return content.includes('backup') || content.includes('recovery') || content.includes('restore');
    });
    
    if (hasBackupDocs) {
      logTest('Backup procedures documented', 'PASS');
    } else {
      logTest('Backup procedures documented', 'WARN', 'Consider documenting backup procedures');
    }
  }
  
  // Run All Tests
  runAllTests() {
    displayBlessing();
    
    console.log(`${colors.golden}🧪 Starting Comprehensive Validation...${colors.reset}\n`);
    
    this.validateFileStructure();
    console.log();
    
    this.validateSpiritualContent();
    console.log();
    
    this.validateMermaidDiagrams();
    console.log();
    
    this.validatePackageJson();
    console.log();
    
    this.validateGitHubActions();
    console.log();
    
    this.validateVitePressConfig();
    console.log();
    
    this.validateLaTeXTemplate();
    console.log();
    
    this.validateSVGLogo();
    console.log();
    
    this.validateDeploymentScript();
    console.log();
    
    this.validateCopilotCompatibility();
    console.log();
    
    this.validateSustainability();
    console.log();
    
    this.displayResults();
  }
  
  // Display Final Results
  displayResults() {
    console.log(`${colors.divine}╔══════════════════════════════════════════════════════════════╗${colors.reset}`);
    console.log(`${colors.divine}║${colors.golden}                    🧪 TEST RESULTS SUMMARY 🧪                   ${colors.divine}║${colors.reset}`);
    console.log(`${colors.divine}╠══════════════════════════════════════════════════════════════╣${colors.reset}`);
    console.log(`${colors.divine}║ ${colors.blessed}✅ Passed: ${testResults.passed.toString().padEnd(10)}${colors.divine}║ ${colors.sacred}❌ Failed: ${testResults.failed.toString().padEnd(10)}${colors.divine}║ ${colors.golden}⚠️  Warnings: ${testResults.warnings.toString().padEnd(8)}${colors.divine}║${colors.reset}`);
    console.log(`${colors.divine}╚══════════════════════════════════════════════════════════════╝${colors.reset}`);
    
    const totalTests = testResults.passed + testResults.failed + testResults.warnings;
    const successRate = totalTests > 0 ? ((testResults.passed / totalTests) * 100).toFixed(1) : 0;
    
    console.log();
    console.log(`${colors.golden}📊 Success Rate: ${successRate}%${colors.reset}`);
    
    if (testResults.failed === 0) {
      console.log(`${colors.blessed}🎉 All critical tests passed! System is blessed and ready.${colors.reset}`);
    } else {
      console.log(`${colors.sacred}⚠️  ${testResults.failed} critical issues found. Please address before deployment.${colors.reset}`);
    }
    
    if (testResults.warnings > 0) {
      console.log(`${colors.golden}💡 ${testResults.warnings} recommendations for improvement.${colors.reset}`);
    }
    
    console.log();
    console.log(`${colors.divine}🙏 In The Name of GOD - Testing completed with divine guidance${colors.reset}`);
    console.log(`${colors.pure}💫 May this system serve humanity with wisdom and compassion${colors.reset}`);
    
    // Exit with appropriate code
    process.exit(testResults.failed > 0 ? 1 : 0);
  }
}

// Main Execution
if (require.main === module) {
  const validator = new SpiritualValidator();
  validator.runAllTests();
}

module.exports = SpiritualValidator;