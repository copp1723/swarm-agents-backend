# 🚀 PRODUCTION OPTIMIZATION ENHANCEMENTS

## Advanced Monitoring and Performance System

import os
import time
import psutil
import logging
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, g
from functools import wraps
import redis
from sqlalchemy import event
from sqlalchemy.engine import Engine
import sqlite3

# Configure advanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('swarm_agents.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Advanced performance monitoring and analytics"""
    
    def __init__(self, app=None):
        self.app = app
        self.metrics = {
            'requests': 0,
            'response_times': [],
            'errors': 0,
            'agent_calls': 0,
            'db_queries': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        self.start_time = datetime.now()
        
        if app:
            self.init_app(app)
    
    def init_app(self, app):
        """Initialize monitoring with Flask app"""
        app.before_request(self.before_request)
        app.after_request(self.after_request)
        app.teardown_appcontext(self.teardown)
        
        # Add monitoring endpoints
        app.add_url_rule('/api/monitoring/health', 'health_check', self.health_check)
        app.add_url_rule('/api/monitoring/metrics', 'metrics', self.get_metrics)
        app.add_url_rule('/api/monitoring/performance', 'performance', self.get_performance)
    
    def before_request(self):
        """Track request start time"""
        g.start_time = time.time()
        self.metrics['requests'] += 1
        
        # Log request details
        logger.info(f"Request: {request.method} {request.path} from {request.remote_addr}")
    
    def after_request(self, response):
        """Track response time and status"""
        if hasattr(g, 'start_time'):
            response_time = time.time() - g.start_time
            self.metrics['response_times'].append(response_time)
            
            # Keep only last 1000 response times
            if len(self.metrics['response_times']) > 1000:
                self.metrics['response_times'] = self.metrics['response_times'][-1000:]
            
            # Track errors
            if response.status_code >= 400:
                self.metrics['errors'] += 1
                logger.warning(f"Error response: {response.status_code} for {request.path}")
            
            # Add performance headers
            response.headers['X-Response-Time'] = f"{response_time:.3f}s"
            response.headers['X-Request-ID'] = str(self.metrics['requests'])
        
        return response
    
    def teardown(self, exception):
        """Clean up after request"""
        if exception:
            logger.error(f"Request exception: {exception}")
            self.metrics['errors'] += 1
    
    def health_check(self):
        """Comprehensive health check endpoint"""
        try:
            # Check system resources
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Check database connectivity
            db_healthy = self.check_database_health()
            
            # Check external services
            services_healthy = self.check_external_services()
            
            health_status = {
                'status': 'healthy' if db_healthy and services_healthy else 'degraded',
                'timestamp': datetime.now().isoformat(),
                'uptime': str(datetime.now() - self.start_time),
                'system': {
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'memory_available': memory.available,
                    'disk_percent': disk.percent,
                    'disk_free': disk.free
                },
                'database': {
                    'healthy': db_healthy,
                    'queries_total': self.metrics['db_queries']
                },
                'services': {
                    'healthy': services_healthy,
                    'agent_calls': self.metrics['agent_calls']
                },
                'requests': {
                    'total': self.metrics['requests'],
                    'errors': self.metrics['errors'],
                    'error_rate': self.metrics['errors'] / max(self.metrics['requests'], 1) * 100
                }
            }
            
            return jsonify(health_status)
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return jsonify({
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500
    
    def get_metrics(self):
        """Get detailed performance metrics"""
        response_times = self.metrics['response_times']
        
        metrics = {
            'requests': {
                'total': self.metrics['requests'],
                'errors': self.metrics['errors'],
                'error_rate': self.metrics['errors'] / max(self.metrics['requests'], 1) * 100
            },
            'performance': {
                'avg_response_time': sum(response_times) / len(response_times) if response_times else 0,
                'min_response_time': min(response_times) if response_times else 0,
                'max_response_time': max(response_times) if response_times else 0,
                'p95_response_time': self.percentile(response_times, 95) if response_times else 0,
                'p99_response_time': self.percentile(response_times, 99) if response_times else 0
            },
            'database': {
                'queries': self.metrics['db_queries']
            },
            'cache': {
                'hits': self.metrics['cache_hits'],
                'misses': self.metrics['cache_misses'],
                'hit_rate': self.metrics['cache_hits'] / max(self.metrics['cache_hits'] + self.metrics['cache_misses'], 1) * 100
            },
            'agents': {
                'calls': self.metrics['agent_calls']
            },
            'uptime': str(datetime.now() - self.start_time)
        }
        
        return jsonify(metrics)
    
    def get_performance(self):
        """Get real-time performance data"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            performance = {
                'timestamp': datetime.now().isoformat(),
                'cpu': {
                    'percent': cpu_percent,
                    'count': psutil.cpu_count()
                },
                'memory': {
                    'percent': memory.percent,
                    'total': memory.total,
                    'available': memory.available,
                    'used': memory.used
                },
                'requests': {
                    'active': getattr(g, 'active_requests', 0),
                    'total': self.metrics['requests']
                }
            }
            
            return jsonify(performance)
            
        except Exception as e:
            logger.error(f"Performance check failed: {e}")
            return jsonify({'error': str(e)}), 500
    
    def check_database_health(self):
        """Check database connectivity and performance"""
        try:
            # Simple database query to check connectivity
            from src.models.swarm_models import db
            db.engine.execute('SELECT 1')
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    
    def check_external_services(self):
        """Check external service connectivity"""
        try:
            # Check OpenRouter API
            import requests
            openrouter_key = os.getenv('OPENROUTER_API_KEY')
            if openrouter_key and openrouter_key != 'your-openrouter-api-key':
                response = requests.get(
                    'https://openrouter.ai/api/v1/models',
                    headers={'Authorization': f'Bearer {openrouter_key}'},
                    timeout=5
                )
                if response.status_code != 200:
                    return False
            
            return True
        except Exception as e:
            logger.error(f"External services health check failed: {e}")
            return False
    
    def percentile(self, data, percentile):
        """Calculate percentile of data"""
        if not data:
            return 0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def track_agent_call(self):
        """Track agent API calls"""
        self.metrics['agent_calls'] += 1
    
    def track_db_query(self):
        """Track database queries"""
        self.metrics['db_queries'] += 1
    
    def track_cache_hit(self):
        """Track cache hits"""
        self.metrics['cache_hits'] += 1
    
    def track_cache_miss(self):
        """Track cache misses"""
        self.metrics['cache_misses'] += 1


class CacheManager:
    """Advanced caching system for improved performance"""
    
    def __init__(self, app=None):
        self.app = app
        self.cache = {}
        self.cache_stats = {'hits': 0, 'misses': 0}
        
        # Try to use Redis if available, fallback to in-memory
        try:
            import redis
            self.redis_client = redis.Redis(
                host=os.getenv('REDIS_HOST', 'localhost'),
                port=int(os.getenv('REDIS_PORT', 6379)),
                db=0,
                decode_responses=True
            )
            self.use_redis = True
            logger.info("Using Redis for caching")
        except:
            self.redis_client = None
            self.use_redis = False
            logger.info("Using in-memory caching")
        
        if app:
            self.init_app(app)
    
    def init_app(self, app):
        """Initialize caching with Flask app"""
        app.cache = self
    
    def get(self, key):
        """Get value from cache"""
        try:
            if self.use_redis:
                value = self.redis_client.get(key)
                if value:
                    self.cache_stats['hits'] += 1
                    return value
            else:
                if key in self.cache:
                    # Check expiration
                    item = self.cache[key]
                    if item['expires'] > datetime.now():
                        self.cache_stats['hits'] += 1
                        return item['value']
                    else:
                        del self.cache[key]
            
            self.cache_stats['misses'] += 1
            return None
            
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            self.cache_stats['misses'] += 1
            return None
    
    def set(self, key, value, timeout=300):
        """Set value in cache with timeout"""
        try:
            if self.use_redis:
                self.redis_client.setex(key, timeout, value)
            else:
                self.cache[key] = {
                    'value': value,
                    'expires': datetime.now() + timedelta(seconds=timeout)
                }
                
                # Clean up expired items periodically
                if len(self.cache) > 1000:
                    self.cleanup_expired()
                    
        except Exception as e:
            logger.error(f"Cache set error: {e}")
    
    def delete(self, key):
        """Delete value from cache"""
        try:
            if self.use_redis:
                self.redis_client.delete(key)
            else:
                self.cache.pop(key, None)
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
    
    def cleanup_expired(self):
        """Clean up expired cache items"""
        if not self.use_redis:
            now = datetime.now()
            expired_keys = [
                key for key, item in self.cache.items()
                if item['expires'] <= now
            ]
            for key in expired_keys:
                del self.cache[key]
    
    def get_stats(self):
        """Get cache statistics"""
        total = self.cache_stats['hits'] + self.cache_stats['misses']
        hit_rate = (self.cache_stats['hits'] / total * 100) if total > 0 else 0
        
        return {
            'hits': self.cache_stats['hits'],
            'misses': self.cache_stats['misses'],
            'hit_rate': hit_rate,
            'total_requests': total,
            'cache_size': len(self.cache) if not self.use_redis else 'redis'
        }


def cache_response(timeout=300, key_func=None):
    """Decorator for caching API responses"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{f.__name__}:{request.path}:{request.query_string.decode()}"
            
            # Try to get from cache
            cached_result = current_app.cache.get(cache_key)
            if cached_result:
                return cached_result
            
            # Execute function and cache result
            result = f(*args, **kwargs)
            current_app.cache.set(cache_key, result, timeout)
            
            return result
        return decorated_function
    return decorator


# Database query monitoring
@event.listens_for(Engine, "before_cursor_execute")
def receive_before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    """Track database query execution"""
    context._query_start_time = time.time()

@event.listens_for(Engine, "after_cursor_execute")
def receive_after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    """Log slow database queries"""
    total = time.time() - context._query_start_time
    if total > 0.1:  # Log queries slower than 100ms
        logger.warning(f"Slow query ({total:.3f}s): {statement[:100]}...")


class DatabaseOptimizer:
    """Database optimization utilities"""
    
    @staticmethod
    def optimize_sqlite():
        """Optimize SQLite for production use"""
        optimizations = [
            "PRAGMA journal_mode=WAL;",
            "PRAGMA synchronous=NORMAL;",
            "PRAGMA cache_size=10000;",
            "PRAGMA temp_store=MEMORY;",
            "PRAGMA mmap_size=268435456;"  # 256MB
        ]
        
        try:
            from src.models.swarm_models import db
            for pragma in optimizations:
                db.engine.execute(pragma)
            logger.info("SQLite optimizations applied")
        except Exception as e:
            logger.error(f"SQLite optimization failed: {e}")
    
    @staticmethod
    def create_indexes():
        """Create database indexes for better performance"""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id);",
            "CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp);",
            "CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON conversations(user_id);",
            "CREATE INDEX IF NOT EXISTS idx_tasks_agent_id ON tasks(agent_id);",
            "CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);"
        ]
        
        try:
            from src.models.swarm_models import db
            for index_sql in indexes:
                db.engine.execute(index_sql)
            logger.info("Database indexes created")
        except Exception as e:
            logger.error(f"Index creation failed: {e}")


# Initialize monitoring and caching
monitor = PerformanceMonitor()
cache_manager = CacheManager()

def init_production_optimizations(app):
    """Initialize all production optimizations"""
    monitor.init_app(app)
    cache_manager.init_app(app)
    
    # Apply database optimizations
    with app.app_context():
        DatabaseOptimizer.optimize_sqlite()
        DatabaseOptimizer.create_indexes()
    
    logger.info("Production optimizations initialized")
    
    return app

