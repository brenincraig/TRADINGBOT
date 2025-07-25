#!/bin/bash

# Crypto Trading Bot Deployment Script
# Supports AWS EC2, DigitalOcean, Linode, and other VPS providers

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="crypto-trading-bot"
DOCKER_IMAGE="crypto-trading-bot:latest"
BACKUP_DIR="/opt/backups/crypto-trading-bot"
LOG_FILE="/var/log/crypto-trading-bot-deploy.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
    exit 1
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

# Check if running as root
check_root() {
    if [[ $EUID -eq 0 ]]; then
        error "This script should not be run as root for security reasons"
    fi
}

# Check system requirements
check_requirements() {
    log "Checking system requirements..."
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install Docker first."
    fi
    
    # Check if Docker Compose is installed
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed. Please install Docker Compose first."
    fi
    
    # Check available disk space (minimum 5GB)
    available_space=$(df / | awk 'NR==2 {print $4}')
    if [[ $available_space -lt 5242880 ]]; then  # 5GB in KB
        warning "Less than 5GB disk space available. Consider freeing up space."
    fi
    
    # Check available memory (minimum 2GB)
    available_memory=$(free -m | awk 'NR==2{print $7}')
    if [[ $available_memory -lt 2048 ]]; then
        warning "Less than 2GB memory available. Performance may be affected."
    fi
    
    log "System requirements check completed"
}

# Install system dependencies
install_dependencies() {
    log "Installing system dependencies..."
    
    # Update package list
    sudo apt-get update
    
    # Install required packages
    sudo apt-get install -y \
        curl \
        wget \
        git \
        htop \
        nano \
        unzip \
        software-properties-common \
        apt-transport-https \
        ca-certificates \
        gnupg \
        lsb-release
    
    # Install Docker if not present
    if ! command -v docker &> /dev/null; then
        log "Installing Docker..."
        curl -fsSL https://get.docker.com -o get-docker.sh
        sudo sh get-docker.sh
        sudo usermod -aG docker $USER
        rm get-docker.sh
    fi
    
    # Install Docker Compose if not present
    if ! command -v docker-compose &> /dev/null; then
        log "Installing Docker Compose..."
        sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
        sudo chmod +x /usr/local/bin/docker-compose
    fi
    
    log "Dependencies installation completed"
}

# Setup firewall
setup_firewall() {
    log "Setting up firewall..."
    
    # Install UFW if not present
    if ! command -v ufw &> /dev/null; then
        sudo apt-get install -y ufw
    fi
    
    # Configure UFW
    sudo ufw --force reset
    sudo ufw default deny incoming
    sudo ufw default allow outgoing
    
    # Allow SSH
    sudo ufw allow ssh
    
    # Allow HTTP and HTTPS
    sudo ufw allow 80/tcp
    sudo ufw allow 443/tcp
    
    # Allow application ports
    sudo ufw allow 5000/tcp  # Flask app
    sudo ufw allow 3000/tcp  # Grafana (if enabled)
    sudo ufw allow 9090/tcp  # Prometheus (if enabled)
    
    # Enable firewall
    sudo ufw --force enable
    
    log "Firewall setup completed"
}

# Create necessary directories
create_directories() {
    log "Creating necessary directories..."
    
    # Create application directories
    sudo mkdir -p /opt/crypto-trading-bot
    sudo mkdir -p /opt/crypto-trading-bot/data
    sudo mkdir -p /opt/crypto-trading-bot/logs
    sudo mkdir -p /opt/crypto-trading-bot/backups
    sudo mkdir -p /opt/crypto-trading-bot/config
    
    # Create backup directory
    sudo mkdir -p "$BACKUP_DIR"
    
    # Set permissions
    sudo chown -R $USER:$USER /opt/crypto-trading-bot
    sudo chown -R $USER:$USER "$BACKUP_DIR"
    
    log "Directories created successfully"
}

# Copy application files
copy_files() {
    log "Copying application files..."
    
    # Copy all files to deployment directory
    cp -r "$SCRIPT_DIR"/* /opt/crypto-trading-bot/
    
    # Copy environment file
    if [[ -f "$SCRIPT_DIR/.env.production" ]]; then
        cp "$SCRIPT_DIR/.env.production" /opt/crypto-trading-bot/.env
        log "Production environment file copied"
    else
        warning "Production environment file not found. Please create .env file manually."
    fi
    
    # Set proper permissions
    chmod +x /opt/crypto-trading-bot/deploy.sh
    chmod 600 /opt/crypto-trading-bot/.env
    
    log "Files copied successfully"
}

# Setup SSL certificates (Let's Encrypt)
setup_ssl() {
    local domain=$1
    
    if [[ -z "$domain" ]]; then
        info "No domain provided, skipping SSL setup"
        return
    fi
    
    log "Setting up SSL certificates for domain: $domain"
    
    # Install Certbot
    sudo apt-get install -y certbot
    
    # Generate certificates
    sudo certbot certonly --standalone -d "$domain" --non-interactive --agree-tos --email admin@"$domain"
    
    # Copy certificates to config directory
    sudo cp /etc/letsencrypt/live/"$domain"/fullchain.pem /opt/crypto-trading-bot/config/ssl/cert.pem
    sudo cp /etc/letsencrypt/live/"$domain"/privkey.pem /opt/crypto-trading-bot/config/ssl/key.pem
    
    # Set permissions
    sudo chown $USER:$USER /opt/crypto-trading-bot/config/ssl/*
    chmod 600 /opt/crypto-trading-bot/config/ssl/*
    
    # Setup auto-renewal
    echo "0 12 * * * /usr/bin/certbot renew --quiet" | sudo crontab -
    
    log "SSL certificates setup completed"
}

# Build and start services
deploy_services() {
    log "Building and starting services..."
    
    cd /opt/crypto-trading-bot
    
    # Build Docker images
    docker-compose build --no-cache
    
    # Start services
    docker-compose up -d
    
    # Wait for services to be ready
    log "Waiting for services to start..."
    sleep 30
    
    # Check service health
    if docker-compose ps | grep -q "Up"; then
        log "Services started successfully"
    else
        error "Failed to start services"
    fi
    
    log "Deployment completed successfully"
}

# Setup monitoring
setup_monitoring() {
    log "Setting up monitoring..."
    
    cd /opt/crypto-trading-bot
    
    # Start monitoring services
    docker-compose --profile monitoring up -d
    
    log "Monitoring setup completed"
}

# Setup logging
setup_logging() {
    log "Setting up logging..."
    
    cd /opt/crypto-trading-bot
    
    # Start logging services
    docker-compose --profile logging up -d
    
    # Setup log rotation
    sudo tee /etc/logrotate.d/crypto-trading-bot > /dev/null <<EOF
/opt/crypto-trading-bot/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 $USER $USER
    postrotate
        docker-compose -f /opt/crypto-trading-bot/docker-compose.yml restart trading-bot
    endscript
}
EOF
    
    log "Logging setup completed"
}

# Setup backup
setup_backup() {
    log "Setting up backup system..."
    
    # Create backup script
    sudo tee /usr/local/bin/backup-trading-bot.sh > /dev/null <<'EOF'
#!/bin/bash

BACKUP_DIR="/opt/backups/crypto-trading-bot"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/backup_$TIMESTAMP.tar.gz"

# Create backup
cd /opt/crypto-trading-bot
tar -czf "$BACKUP_FILE" \
    --exclude='logs/*' \
    --exclude='*.pyc' \
    --exclude='__pycache__' \
    .

# Keep only last 30 backups
find "$BACKUP_DIR" -name "backup_*.tar.gz" -type f -mtime +30 -delete

echo "Backup created: $BACKUP_FILE"
EOF
    
    sudo chmod +x /usr/local/bin/backup-trading-bot.sh
    
    # Setup cron job for daily backups
    (crontab -l 2>/dev/null; echo "0 2 * * * /usr/local/bin/backup-trading-bot.sh") | crontab -
    
    log "Backup system setup completed"
}

# Setup system service
setup_systemd() {
    log "Setting up systemd service..."
    
    sudo tee /etc/systemd/system/crypto-trading-bot.service > /dev/null <<EOF
[Unit]
Description=Crypto Trading Bot
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/opt/crypto-trading-bot
ExecStart=/usr/local/bin/docker-compose up -d
ExecStop=/usr/local/bin/docker-compose down
TimeoutStartSec=0
User=$USER
Group=$USER

[Install]
WantedBy=multi-user.target
EOF
    
    # Reload systemd and enable service
    sudo systemctl daemon-reload
    sudo systemctl enable crypto-trading-bot.service
    
    log "Systemd service setup completed"
}

# Health check
health_check() {
    log "Performing health check..."
    
    # Check if services are running
    cd /opt/crypto-trading-bot
    
    if ! docker-compose ps | grep -q "Up"; then
        error "Services are not running properly"
    fi
    
    # Check API endpoint
    if curl -f http://localhost:5000/api/status > /dev/null 2>&1; then
        log "API health check passed"
    else
        warning "API health check failed"
    fi
    
    # Check database connection
    if docker-compose exec -T postgres pg_isready -U trading_user > /dev/null 2>&1; then
        log "Database health check passed"
    else
        warning "Database health check failed"
    fi
    
    log "Health check completed"
}

# Display deployment information
show_info() {
    log "Deployment Information:"
    echo "=========================="
    echo "Application URL: http://$(curl -s ifconfig.me):5000"
    echo "Dashboard: http://$(curl -s ifconfig.me):5000"
    echo "Grafana (if enabled): http://$(curl -s ifconfig.me):3000"
    echo "Prometheus (if enabled): http://$(curl -s ifconfig.me):9090"
    echo ""
    echo "Logs location: /opt/crypto-trading-bot/logs"
    echo "Data location: /opt/crypto-trading-bot/data"
    echo "Backup location: $BACKUP_DIR"
    echo ""
    echo "Useful commands:"
    echo "  View logs: docker-compose -f /opt/crypto-trading-bot/docker-compose.yml logs -f"
    echo "  Restart: docker-compose -f /opt/crypto-trading-bot/docker-compose.yml restart"
    echo "  Stop: docker-compose -f /opt/crypto-trading-bot/docker-compose.yml down"
    echo "  Start: docker-compose -f /opt/crypto-trading-bot/docker-compose.yml up -d"
    echo "=========================="
}

# Main deployment function
main() {
    local domain=""
    local enable_monitoring=false
    local enable_logging=false
    local skip_ssl=false
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --domain)
                domain="$2"
                shift 2
                ;;
            --enable-monitoring)
                enable_monitoring=true
                shift
                ;;
            --enable-logging)
                enable_logging=true
                shift
                ;;
            --skip-ssl)
                skip_ssl=true
                shift
                ;;
            --help)
                echo "Usage: $0 [OPTIONS]"
                echo "Options:"
                echo "  --domain DOMAIN         Set domain for SSL certificates"
                echo "  --enable-monitoring     Enable Prometheus and Grafana"
                echo "  --enable-logging        Enable ELK stack for logging"
                echo "  --skip-ssl             Skip SSL certificate setup"
                echo "  --help                 Show this help message"
                exit 0
                ;;
            *)
                error "Unknown option: $1"
                ;;
        esac
    done
    
    log "Starting Crypto Trading Bot deployment..."
    
    # Pre-deployment checks
    check_root
    check_requirements
    
    # Installation steps
    install_dependencies
    setup_firewall
    create_directories
    copy_files
    
    # SSL setup
    if [[ ! "$skip_ssl" == true ]] && [[ -n "$domain" ]]; then
        setup_ssl "$domain"
    fi
    
    # Deploy services
    deploy_services
    
    # Optional components
    if [[ "$enable_monitoring" == true ]]; then
        setup_monitoring
    fi
    
    if [[ "$enable_logging" == true ]]; then
        setup_logging
    fi
    
    # System setup
    setup_backup
    setup_systemd
    
    # Final checks
    health_check
    show_info
    
    log "Deployment completed successfully!"
    log "Please review the configuration files and update API keys before starting trading."
}

# Run main function with all arguments
main "$@"

