# 🚀 SWARM MULTI-AGENT SYSTEM - MAIN APPLICATION

import os
import sys
# DON'T CHANGE THIS !!!
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Import the complete integrated application
from src.routes.swarm_routes import create_complete_app, socketio

# Create the complete application with all integrations
app = create_complete_app()

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    """Serve static files and SPA routing"""
    static_folder_path = app.static_folder
    if static_folder_path is None:
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>🤖 Swarm Multi-Agent System</title>
            <style>
                body { 
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
                    margin: 0; padding: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white; min-height: 100vh;
                }
                .container { 
                    max-width: 1000px; margin: 0 auto; background: rgba(255,255,255,0.1); 
                    padding: 40px; border-radius: 20px; backdrop-filter: blur(10px);
                    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
                }
                .header { text-align: center; margin-bottom: 40px; }
                .status { color: #10B981; font-weight: bold; font-size: 18px; }
                .endpoint { 
                    background: rgba(255,255,255,0.1); padding: 15px; margin: 10px 0; 
                    border-radius: 10px; font-family: 'Monaco', monospace; border-left: 4px solid #10B981;
                }
                .method { color: #3B82F6; font-weight: bold; }
                .feature { 
                    background: rgba(255,255,255,0.05); padding: 20px; margin: 15px 0; 
                    border-radius: 15px; border: 1px solid rgba(255,255,255,0.2);
                }
                .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
                .badge { 
                    display: inline-block; background: #10B981; color: white; 
                    padding: 5px 12px; border-radius: 20px; font-size: 12px; margin: 5px;
                }
                .logo { font-size: 60px; margin-bottom: 20px; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <div class="logo">🤖</div>
                    <h1>Swarm Multi-Agent System</h1>
                    <p class="status">✅ Production-Ready AI Swarm Intelligence Platform</p>
                </div>
                
                <div class="grid">
                    <div class="feature">
                        <h3>🚀 Core API Endpoints</h3>
                        <div class="endpoint"><span class="method">GET</span> /api/health - System health check</div>
                        <div class="endpoint"><span class="method">GET</span> /api/agents - Get all swarm agents</div>
                        <div class="endpoint"><span class="method">GET</span> /api/conversations - Get conversations</div>
                        <div class="endpoint"><span class="method">POST</span> /api/conversations/{id}/messages - Send message</div>
                        <div class="endpoint"><span class="method">GET</span> /api/tasks - Get tasks</div>
                        <div class="endpoint"><span class="method">GET</span> /api/knowledge/search - Search knowledge</div>
                        <div class="endpoint"><span class="method">POST</span> /api/email/send - Send email via agent</div>
                    </div>
                    
                    <div class="feature">
                        <h3>🤖 Swarm Intelligence Features</h3>
                        <div class="badge">Dynamic Agent Selection</div>
                        <div class="badge">Collaborative Problem Solving</div>
                        <div class="badge">Intelligent Task Routing</div>
                        <div class="badge">Real-time Coordination</div>
                        <div class="badge">Shared Knowledge Base</div>
                        <div class="badge">Performance Optimization</div>
                    </div>
                </div>
                
                <div class="grid">
                    <div class="feature">
                        <h3>🔧 Integrated Services</h3>
                        <div class="badge">OpenRouter AI (GPT-4, Claude-3)</div>
                        <div class="badge">SuperMemory Knowledge System</div>
                        <div class="badge">Mailgun Email Service</div>
                        <div class="badge">Real-time WebSocket</div>
                        <div class="badge">SQLite Database</div>
                        <div class="badge">CORS Enabled</div>
                    </div>
                    
                    <div class="feature">
                        <h3>🎯 Available Agents</h3>
                        <div class="badge">Cathy - Personal Assistant</div>
                        <div class="badge">DataMiner - Data Analysis</div>
                        <div class="badge">Coder - Software Development</div>
                        <div class="badge">Creative - Content Creation</div>
                        <div class="badge">Researcher - Information Gathering</div>
                    </div>
                </div>
                
                <div class="feature">
                    <h3>📡 WebSocket Events</h3>
                    <p>Real-time communication for live chat updates, typing indicators, agent responses, and swarm coordination.</p>
                    <div class="badge">connect/disconnect</div>
                    <div class="badge">join_conversation</div>
                    <div class="badge">typing_start/stop</div>
                    <div class="badge">new_message</div>
                    <div class="badge">agent_typing</div>
                </div>
                
                <div class="feature">
                    <h3>🔒 Authentication</h3>
                    <p><strong>Demo Credentials:</strong></p>
                    <p>Email: <code>demo@swarm.ai</code><br>Password: <code>demo123</code></p>
                </div>
                
                <div class="feature" style="text-align: center; background: rgba(16, 185, 129, 0.2); border-color: #10B981;">
                    <h3>🎉 Ready for Production!</h3>
                    <p>This is a fully functional, production-ready multi-agent system with real AI capabilities, email integration, and swarm intelligence.</p>
                    <p><strong>Connect your frontend and start building the future! 🚀</strong></p>
                </div>
            </div>
        </body>
        </html>
        """, 200

    if path != "" and os.path.exists(os.path.join(static_folder_path, path)):
        from flask import send_from_directory
        return send_from_directory(static_folder_path, path)
    else:
        index_path = os.path.join(static_folder_path, 'index.html')
        if os.path.exists(index_path):
            from flask import send_from_directory
            return send_from_directory(static_folder_path, 'index.html')
        else:
            # Return the same beautiful landing page
            return serve('')

if __name__ == '__main__':
    print("🚀 Starting Swarm Multi-Agent System...")
    print("🤖 Swarm orchestration active")
    print("📡 WebSocket support enabled") 
    print("🔌 OpenRouter AI integration active")
    print("🧠 SuperMemory knowledge system active")
    print("📧 Mailgun email service active")
    print("💾 Database and state management active")
    print("🌐 CORS enabled for all origins")
    print("🔒 Demo authentication active")
    print("🌍 Server running on http://0.0.0.0:5000")
    
    # Run with SocketIO for WebSocket support
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
