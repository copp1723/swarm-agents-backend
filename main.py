# 🚀 SWARM MULTI-AGENT SYSTEM - PRODUCTION MAIN
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv('.env.production' if os.getenv('FLASK_ENV') == 'production' else '.env')

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.routes.swarm_routes import create_complete_app, socketio

# Create the Flask application
app = create_complete_app()

if __name__ == '__main__':
    # Development server
    port = int(os.getenv('PORT', 5001))
    host = os.getenv('HOST', '0.0.0.0')
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    print(f'🚀 Starting Swarm Multi-Agent System on {host}:{port}')
    print(f'🔧 Debug mode: {debug}')
    print(f'🌍 Environment: {os.getenv("FLASK_ENV", "development")}')
    
    socketio.run(
        app, 
        host=host, 
        port=port, 
        debug=debug,
        allow_unsafe_werkzeug=True
    )
else:
    # Production server (gunicorn)
    print('🚀 Swarm Multi-Agent System - Production Mode')
    print(f'🔧 Environment: {os.getenv("FLASK_ENV", "production")}')
    
    # This is the WSGI application that gunicorn will use
    application = app

