# DocFoundry Security Configuration Guide

This guide covers comprehensive security configuration for DocFoundry, including authentication, authorization, data protection, and security best practices.

## Table of Contents

1. [Security Overview](#security-overview)
2. [Authentication & Authorization](#authentication--authorization)
3. [API Security](#api-security)
4. [Data Protection](#data-protection)
5. [Network Security](#network-security)
6. [Input Validation & Sanitization](#input-validation--sanitization)
7. [Rate Limiting & DDoS Protection](#rate-limiting--ddos-protection)
8. [Audit Logging & Monitoring](#audit-logging--monitoring)
9. [Security Headers](#security-headers)
10. [Encryption](#encryption)
11. [Security Best Practices](#security-best-practices)
12. [Compliance & Standards](#compliance--standards)
13. [Security Troubleshooting](#security-troubleshooting)

## Security Overview

DocFoundry implements multiple layers of security to protect your data and ensure secure operations:

- **Authentication**: Multi-factor authentication with JWT tokens
- **Authorization**: Role-based access control (RBAC)
- **Data Protection**: Encryption at rest and in transit
- **Network Security**: HTTPS, CORS, and security headers
- **Input Validation**: Comprehensive input sanitization
- **Rate Limiting**: Protection against abuse and DDoS
- **Audit Logging**: Complete security event tracking
- **Compliance**: SOC 2, GDPR, and industry standards

## Authentication & Authorization

### 1. JWT Token Configuration

```bash
# JWT Settings
JWT_SECRET_KEY=your-super-secret-jwt-key-min-32-chars
JWT_ALGORITHM=HS256
JWT_EXPIRATION=3600  # 1 hour
JWT_REFRESH_EXPIRATION=604800  # 7 days
JWT_ISSUER=docfoundry
JWT_AUDIENCE=docfoundry-api

# Token Security
JWT_REQUIRE_CLAIMS=true
JWT_VERIFY_SIGNATURE=true
JWT_VERIFY_EXPIRATION=true
JWT_LEEWAY=10  # seconds
```

### 2. Multi-Factor Authentication (MFA)

```bash
# MFA Configuration
MFA_ENABLED=true
MFA_ISSUER=DocFoundry
MFA_BACKUP_CODES=10
MFA_WINDOW=1  # TOTP window
MFA_RATE_LIMIT=5/minute

# Supported MFA Methods
MFA_METHODS=totp,sms,email
MFA_DEFAULT_METHOD=totp
MFA_ENFORCE_FOR_ADMIN=true
```

### 3. Role-Based Access Control (RBAC)

```bash
# RBAC Configuration
RBAC_ENABLED=true
RBAC_DEFAULT_ROLE=viewer
RBAC_ADMIN_ROLE=admin
RBAC_STRICT_MODE=true

# Permission Inheritance
RBAC_INHERIT_PERMISSIONS=true
RBAC_CACHE_TTL=300  # 5 minutes
```

#### Default Roles and Permissions

| Role | Permissions | Description |
|------|-------------|-------------|
| **admin** | All permissions | Full system access |
| **editor** | read, write, update | Content management |
| **analyst** | read, analyze, export | Data analysis |
| **viewer** | read | Read-only access |
| **guest** | limited_read | Restricted access |

#### Custom Role Configuration

```python
# Custom roles configuration
CUSTOM_ROLES = {
    "data_scientist": {
        "permissions": ["read", "analyze", "export", "lineage_view"],
        "description": "Data analysis and lineage access"
    },
    "compliance_officer": {
        "permissions": ["read", "audit_view", "report_generate"],
        "description": "Compliance and audit access"
    }
}
```

### 4. Session Management

```bash
# Session Configuration
SESSION_TIMEOUT=1800  # 30 minutes
SESSION_ABSOLUTE_TIMEOUT=28800  # 8 hours
SESSION_COOKIE_SECURE=true
SESSION_COOKIE_HTTPONLY=true
SESSION_COOKIE_SAMESITE=Strict

# Session Storage
SESSION_STORAGE=redis
SESSION_REDIS_URL=redis://localhost:6379/2
SESSION_KEY_PREFIX=docfoundry:session:
```

## API Security

### 1. API Key Management

```bash
# API Key Configuration
API_KEY_ENABLED=true
API_KEY_HEADER=X-API-Key
API_KEY_QUERY_PARAM=api_key
API_KEY_LENGTH=32
API_KEY_EXPIRATION=2592000  # 30 days

# API Key Security
API_KEY_HASH_ALGORITHM=sha256
API_KEY_RATE_LIMIT=1000/hour
API_KEY_IP_WHITELIST=enabled
```

#### API Key Generation

```bash
# Generate new API key
python scripts/generate_api_key.py --user-id 123 --name "Production API" --expires-in 30d

# List API keys
python scripts/list_api_keys.py --user-id 123

# Revoke API key
python scripts/revoke_api_key.py --key-id abc123
```

### 2. OAuth 2.0 Integration

```bash
# OAuth Configuration
OAUTH_ENABLED=true
OAUTH_PROVIDERS=google,github,microsoft

# Google OAuth
GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-client-secret
GOOGLE_REDIRECT_URI=https://yourdomain.com/auth/google/callback

# GitHub OAuth
GITHUB_CLIENT_ID=your-github-client-id
GITHUB_CLIENT_SECRET=your-github-client-secret
GITHUB_REDIRECT_URI=https://yourdomain.com/auth/github/callback
```

### 3. API Versioning & Deprecation

```bash
# API Versioning
API_VERSION_HEADER=X-API-Version
API_DEFAULT_VERSION=v1
API_SUPPORTED_VERSIONS=v1,v2
API_DEPRECATION_WARNINGS=true

# Version-specific rate limits
API_V1_RATE_LIMIT=500/hour
API_V2_RATE_LIMIT=1000/hour
```

## Data Protection

### 1. Encryption at Rest

```bash
# Database Encryption
DB_ENCRYPTION_ENABLED=true
DB_ENCRYPTION_KEY=your-32-char-encryption-key
DB_ENCRYPTION_ALGORITHM=AES-256-GCM

# File Encryption
FILE_ENCRYPTION_ENABLED=true
FILE_ENCRYPTION_KEY=your-file-encryption-key
FILE_ENCRYPTION_ALGORITHM=AES-256-CBC
```

### 2. Encryption in Transit

```bash
# TLS Configuration
TLS_VERSION=1.2,1.3
TLS_CIPHERS=ECDHE-RSA-AES256-GCM-SHA384,ECDHE-RSA-AES128-GCM-SHA256
TLS_CERT_PATH=/etc/ssl/certs/docfoundry.crt
TLS_KEY_PATH=/etc/ssl/private/docfoundry.key

# Certificate Management
CERT_AUTO_RENEWAL=true
CERT_RENEWAL_DAYS=30
CERT_PROVIDER=letsencrypt
```

### 3. Data Masking & Anonymization

```bash
# Data Masking
DATA_MASKING_ENABLED=true
DATA_MASKING_FIELDS=email,phone,ssn,credit_card
DATA_MASKING_ALGORITHM=sha256

# PII Detection
PII_DETECTION_ENABLED=true
PII_DETECTION_CONFIDENCE=0.8
PII_AUTO_MASK=true
```

## Network Security

### 1. CORS Configuration

```bash
# CORS Settings
CORS_ENABLED=true
CORS_ORIGINS=https://yourdomain.com,https://app.yourdomain.com
CORS_ALLOW_CREDENTIALS=true
CORS_ALLOW_METHODS=GET,POST,PUT,DELETE,OPTIONS
CORS_ALLOW_HEADERS=Content-Type,Authorization,X-API-Key,X-Requested-With
CORS_EXPOSE_HEADERS=X-Total-Count,X-Rate-Limit-Remaining
CORS_MAX_AGE=86400  # 24 hours

# CORS Security
CORS_STRICT_MODE=true
CORS_VALIDATE_ORIGIN=true
```

### 2. IP Whitelisting & Blacklisting

```bash
# IP Access Control
IP_WHITELIST_ENABLED=false
IP_WHITELIST=192.168.1.0/24,10.0.0.0/8

IP_BLACKLIST_ENABLED=true
IP_BLACKLIST_AUTO_UPDATE=true
IP_BLACKLIST_SOURCES=spamhaus,malwaredomains

# Geolocation Blocking
GEO_BLOCKING_ENABLED=false
GEO_BLOCKED_COUNTRIES=CN,RU,KP
GEO_ALLOWED_COUNTRIES=US,CA,GB,DE,FR
```

### 3. Firewall Configuration

```bash
# Application Firewall
WAF_ENABLED=true
WAF_MODE=block  # monitor, block
WAF_RULES=owasp_crs,custom

# DDoS Protection
DDOS_PROTECTION_ENABLED=true
DDOS_THRESHOLD=1000/minute
DDOS_BAN_DURATION=3600  # 1 hour
```

## Input Validation & Sanitization

### 1. Input Validation Rules

```bash
# Validation Configuration
VALIDATION_STRICT=true
VALIDATION_MAX_REQUEST_SIZE=10MB
VALIDATION_MAX_FIELD_LENGTH=1000
VALIDATION_ALLOW_HTML=false

# File Upload Validation
FILE_UPLOAD_MAX_SIZE=100MB
FILE_UPLOAD_ALLOWED_TYPES=pdf,doc,docx,txt,md,csv,json
FILE_UPLOAD_SCAN_MALWARE=true
FILE_UPLOAD_QUARANTINE=true
```

### 2. SQL Injection Prevention

```bash
# SQL Security
SQL_INJECTION_PROTECTION=true
SQL_PARAMETERIZED_QUERIES=true
SQL_QUERY_LOGGING=true
SQL_SUSPICIOUS_QUERY_ALERT=true

# Database Security
DB_READ_ONLY_USER=docfoundry_ro
DB_QUERY_TIMEOUT=30
DB_MAX_CONNECTIONS=100
```

### 3. XSS Protection

```bash
# XSS Prevention
XSS_PROTECTION_ENABLED=true
XSS_FILTER_MODE=block
XSS_SANITIZE_HTML=true
XSS_ALLOWED_TAGS=p,br,strong,em,ul,ol,li

# Content Security Policy
CSP_ENABLED=true
CSP_REPORT_ONLY=false
CSP_REPORT_URI=/security/csp-report
```

## Rate Limiting & DDoS Protection

### 1. Rate Limiting Configuration

```bash
# Global Rate Limits
RATE_LIMIT_ENABLED=true
RATE_LIMIT_STORAGE=redis
RATE_LIMIT_KEY_FUNC=ip_and_user

# Per-Endpoint Limits
RATE_LIMIT_LOGIN=5/minute
RATE_LIMIT_API_READ=1000/hour
RATE_LIMIT_API_WRITE=100/hour
RATE_LIMIT_UPLOAD=10/hour
RATE_LIMIT_SEARCH=50/minute

# Burst Protection
RATE_LIMIT_BURST_MULTIPLIER=2
RATE_LIMIT_BURST_WINDOW=60  # seconds
```

### 2. Advanced Rate Limiting

```bash
# Adaptive Rate Limiting
ADAPTIVE_RATE_LIMIT=true
ADAPTIVE_THRESHOLD=0.8
ADAPTIVE_COOLDOWN=300  # 5 minutes

# User-based Limits
USER_RATE_LIMIT_PREMIUM=5000/hour
USER_RATE_LIMIT_STANDARD=1000/hour
USER_RATE_LIMIT_FREE=100/hour
```

### 3. DDoS Mitigation

```bash
# DDoS Detection
DDOS_DETECTION_ENABLED=true
DDOS_THRESHOLD_REQUESTS=10000/minute
DDOS_THRESHOLD_BANDWIDTH=100MB/minute
DDOS_DETECTION_WINDOW=60  # seconds

# DDoS Response
DDOS_AUTO_BAN=true
DDOS_BAN_DURATION=3600  # 1 hour
DDOS_ALERT_WEBHOOK=https://alerts.yourdomain.com/ddos
```

## Audit Logging & Monitoring

### 1. Security Event Logging

```bash
# Audit Logging
AUDIT_LOGGING_ENABLED=true
AUDIT_LOG_LEVEL=INFO
AUDIT_LOG_FORMAT=json
AUDIT_LOG_FILE=/var/log/docfoundry/audit.log

# Event Categories
AUDIT_LOG_AUTH=true
AUDIT_LOG_ACCESS=true
AUDIT_LOG_DATA_CHANGES=true
AUDIT_LOG_ADMIN_ACTIONS=true
AUDIT_LOG_SECURITY_EVENTS=true
```

### 2. Security Monitoring

```bash
# Security Monitoring
SECURITY_MONITORING_ENABLED=true
SECURITY_ALERT_WEBHOOK=https://alerts.yourdomain.com/security
SECURITY_ALERT_EMAIL=security@yourdomain.com

# Anomaly Detection
ANOMALY_DETECTION_ENABLED=true
ANOMALY_THRESHOLD=3  # standard deviations
ANOMALY_LEARNING_PERIOD=7  # days
```

### 3. Compliance Logging

```bash
# Compliance Requirements
COMPLIANCE_LOGGING=true
COMPLIANCE_RETENTION=2555  # 7 years in days
COMPLIANCE_ENCRYPTION=true
COMPLIANCE_IMMUTABLE=true

# GDPR Compliance
GDPR_ENABLED=true
GDPR_DATA_RETENTION=1095  # 3 years
GDPR_RIGHT_TO_ERASURE=true
GDPR_DATA_PORTABILITY=true
```

## Security Headers

### 1. HTTP Security Headers

```bash
# Security Headers
SECURITY_HEADERS_ENABLED=true

# HSTS (HTTP Strict Transport Security)
HSTS_ENABLED=true
HSTS_MAX_AGE=31536000  # 1 year
HSTS_INCLUDE_SUBDOMAINS=true
HSTS_PRELOAD=true

# Content Security Policy
CSP_DEFAULT_SRC="'self'"
CSP_SCRIPT_SRC="'self' 'unsafe-inline'"
CSP_STYLE_SRC="'self' 'unsafe-inline'"
CSP_IMG_SRC="'self' data: https:"
CSP_FONT_SRC="'self'"
CSP_CONNECT_SRC="'self'"
CSP_FRAME_ANCESTORS="'none'"

# Other Security Headers
X_FRAME_OPTIONS=DENY
X_CONTENT_TYPE_OPTIONS=nosniff
X_XSS_PROTECTION="1; mode=block"
REFERRER_POLICY=strict-origin-when-cross-origin
PERMISSIONS_POLICY="geolocation=(), microphone=(), camera=()"
```

### 2. Custom Security Headers

```bash
# Custom Headers
CUSTOM_SECURITY_HEADERS=true
SERVER_HEADER_HIDE=true
X_POWERED_BY_HIDE=true

# API Security Headers
API_SECURITY_HEADERS=true
X_API_VERSION_HEADER=true
X_RATE_LIMIT_HEADERS=true
```

## Encryption

### 1. Encryption Standards

```bash
# Encryption Configuration
ENCRYPTION_ALGORITHM=AES-256-GCM
ENCRYPTION_KEY_SIZE=256
ENCRYPTION_IV_SIZE=12

# Key Management
KEY_ROTATION_ENABLED=true
KEY_ROTATION_INTERVAL=90  # days
KEY_BACKUP_ENABLED=true
KEY_ESCROW_ENABLED=false
```

### 2. Password Security

```bash
# Password Policy
PASSWORD_MIN_LENGTH=12
PASSWORD_REQUIRE_UPPERCASE=true
PASSWORD_REQUIRE_LOWERCASE=true
PASSWORD_REQUIRE_NUMBERS=true
PASSWORD_REQUIRE_SYMBOLS=true
PASSWORD_HISTORY=5
PASSWORD_EXPIRY=90  # days

# Password Hashing
PASSWORD_HASH_ALGORITHM=argon2id
PASSWORD_HASH_ITERATIONS=3
PASSWORD_HASH_MEMORY=65536  # KB
PASSWORD_HASH_PARALLELISM=4
```

### 3. Cryptographic Keys

```bash
# Key Generation
KEY_GENERATION_ALGORITHM=RSA-4096
KEY_GENERATION_ENTROPY_SOURCE=/dev/urandom

# Key Storage
KEY_STORAGE_TYPE=hsm  # hsm, file, env
KEY_STORAGE_PATH=/etc/docfoundry/keys
KEY_STORAGE_ENCRYPTION=true
```

## Security Best Practices

### 1. Development Security

```bash
# Secure Development
SECURE_CODING_ENABLED=true
CODE_SCANNING_ENABLED=true
DEPENDENCY_SCANNING=true
SECRET_SCANNING=true

# Security Testing
SECURITY_TESTING_ENABLED=true
PENETRATION_TESTING=quarterly
VULNERABILITY_SCANNING=weekly
```

### 2. Deployment Security

```bash
# Container Security
CONTAINER_SECURITY_SCANNING=true
CONTAINER_ROOTLESS=true
CONTAINER_READ_ONLY=true
CONTAINER_NO_NEW_PRIVILEGES=true

# Infrastructure Security
INFRASTRUCTURE_AS_CODE=true
SECURITY_BASELINE_SCANNING=true
COMPLIANCE_MONITORING=true
```

### 3. Operational Security

```bash
# Security Operations
SECURITY_INCIDENT_RESPONSE=true
SECURITY_AWARENESS_TRAINING=true
SECURITY_POLICY_ENFORCEMENT=true

# Backup Security
BACKUP_ENCRYPTION=true
BACKUP_INTEGRITY_CHECKS=true
BACKUP_ACCESS_CONTROL=true
```

## Compliance & Standards

### 1. SOC 2 Compliance

```bash
# SOC 2 Configuration
SOC2_COMPLIANCE=true
SOC2_AUDIT_LOGGING=true
SOC2_ACCESS_CONTROLS=true
SOC2_CHANGE_MANAGEMENT=true
SOC2_MONITORING=true
```

### 2. GDPR Compliance

```bash
# GDPR Configuration
GDPR_COMPLIANCE=true
GDPR_CONSENT_MANAGEMENT=true
GDPR_DATA_MINIMIZATION=true
GDPR_BREACH_NOTIFICATION=true
GDPR_DPO_CONTACT=dpo@yourdomain.com
```

### 3. Industry Standards

```bash
# Security Standards
ISO27001_COMPLIANCE=true
NIST_FRAMEWORK=true
OWASP_TOP10_PROTECTION=true
CIS_CONTROLS=true
```

## Security Troubleshooting

### 1. Authentication Issues

```bash
# Debug Authentication
# Check JWT token validity
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/auth/verify

# Validate API key
curl -H "X-API-Key: $API_KEY" http://localhost:8000/auth/validate

# Check user permissions
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/auth/permissions
```

### 2. Rate Limiting Issues

```bash
# Check rate limit status
curl -I http://localhost:8000/api/status
# Look for X-RateLimit-* headers

# Reset rate limits (admin only)
curl -X POST -H "Authorization: Bearer $ADMIN_TOKEN" \
  http://localhost:8000/admin/rate-limits/reset

# Check Redis rate limit data
redis-cli KEYS "rate_limit:*"
```

### 3. Security Event Investigation

```bash
# Search security logs
grep "SECURITY_EVENT" /var/log/docfoundry/audit.log

# Check failed authentication attempts
grep "AUTH_FAILED" /var/log/docfoundry/audit.log | tail -20

# Monitor suspicious activity
tail -f /var/log/docfoundry/audit.log | grep -E "(SUSPICIOUS|ANOMALY|THREAT)"
```

### 4. SSL/TLS Issues

```bash
# Test SSL configuration
openssl s_client -connect yourdomain.com:443 -servername yourdomain.com

# Check certificate validity
openssl x509 -in /etc/ssl/certs/docfoundry.crt -text -noout

# Verify cipher suites
nmap --script ssl-enum-ciphers -p 443 yourdomain.com
```

### 5. Security Monitoring

```bash
# Check security metrics
curl http://localhost:9090/metrics | grep security_

# Monitor failed requests
curl http://localhost:9090/metrics | grep http_requests_total | grep "status=\"4"

# Check rate limit violations
curl http://localhost:9090/metrics | grep rate_limit_exceeded_total
```

## Security Checklist

### Pre-Production Security Checklist

- [ ] **Authentication & Authorization**
  - [ ] JWT tokens properly configured
  - [ ] MFA enabled for admin accounts
  - [ ] RBAC permissions reviewed
  - [ ] Session management configured

- [ ] **API Security**
  - [ ] API keys generated and distributed
  - [ ] Rate limiting configured
  - [ ] Input validation enabled
  - [ ] CORS properly configured

- [ ] **Data Protection**
  - [ ] Encryption at rest enabled
  - [ ] TLS/SSL certificates installed
  - [ ] Database encryption configured
  - [ ] PII detection enabled

- [ ] **Network Security**
  - [ ] Firewall rules configured
  - [ ] Security headers enabled
  - [ ] IP whitelisting/blacklisting set
  - [ ] DDoS protection enabled

- [ ] **Monitoring & Logging**
  - [ ] Audit logging enabled
  - [ ] Security monitoring configured
  - [ ] Alert webhooks set up
  - [ ] Log retention policies set

- [ ] **Compliance**
  - [ ] GDPR compliance configured
  - [ ] SOC 2 controls implemented
  - [ ] Security policies documented
  - [ ] Incident response plan ready

### Regular Security Maintenance

- [ ] **Weekly**
  - [ ] Review security logs
  - [ ] Check for failed authentication attempts
  - [ ] Monitor rate limit violations
  - [ ] Update security blacklists

- [ ] **Monthly**
  - [ ] Rotate API keys
  - [ ] Review user permissions
  - [ ] Update security configurations
  - [ ] Test backup and recovery

- [ ] **Quarterly**
  - [ ] Security assessment
  - [ ] Penetration testing
  - [ ] Policy review and updates
  - [ ] Security training

## Emergency Response

### Security Incident Response

1. **Immediate Actions**
   ```bash
   # Block suspicious IP
   iptables -A INPUT -s SUSPICIOUS_IP -j DROP
   
   # Disable compromised user
   python scripts/disable_user.py --user-id COMPROMISED_USER_ID
   
   # Revoke API keys
   python scripts/revoke_api_key.py --key-id COMPROMISED_KEY_ID
   ```

2. **Investigation**
   ```bash
   # Collect security logs
   grep "$(date +'%Y-%m-%d')" /var/log/docfoundry/audit.log > incident_logs.txt
   
   # Check access patterns
   grep "SUSPICIOUS_IP" /var/log/docfoundry/access.log
   
   # Review authentication events
   grep "AUTH_" /var/log/docfoundry/audit.log | grep "$(date +'%Y-%m-%d')"
   ```

3. **Recovery**
   ```bash
   # Reset passwords
   python scripts/force_password_reset.py --user-id AFFECTED_USER_ID
   
   # Regenerate secrets
   python scripts/rotate_secrets.py --all
   
   # Update security configurations
   python scripts/update_security_config.py --incident-response
   ```

## Additional Resources

- [OWASP Security Guidelines](https://owasp.org/)
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)
- [CIS Controls](https://www.cisecurity.org/controls/)
- [DocFoundry Security Updates](https://github.com/your-org/docfoundry/security)

For security questions or to report vulnerabilities:
- Email: security@yourdomain.com
- Security Portal: https://security.yourdomain.com
- Bug Bounty: https://bugbounty.yourdomain.com