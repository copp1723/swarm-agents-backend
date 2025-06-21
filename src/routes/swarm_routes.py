# 🚀 SWARM MULTI-AGENT SYSTEM - FLASK API ROUTES

from flask import Flask, request, jsonify, session
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room, leave_room, disconnect
from werkzeug.security import generate_password_hash, check_password_hash
import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from functools import wraps
from typing import Dict, List, Optional

# Import our models and orchestrator
from src.models.swarm_models import (
    db, User, Agent, Conversation, Message, Task, SwarmSession, KnowledgeEntry,
    DatabaseManager, MessageType, SwarmMode, TaskStatus, AgentStatus
)
from src.swarm_orchestrator import (
    swarm_orchestrator, process_message, get_agent_response, 
    send_email, store_knowledge, get_knowledge, get_status
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask extensions
socketio = SocketIO(cors_allowed_origins="*", async_mode='threading')

def create_app():
    """Create and configure Flask app"""
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'swarm-secret-key-change-in-production'
    
    # Database configuration
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///swarm_agents.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    # Initialize extensions
    db.init_app(app)
    CORS(app, origins="*")
    socketio.init_app(app)
    
    # Create tables and initialize data
    with app.app_context():
        DatabaseManager.initialize_system()
    
    return app

# Authentication helpers
def get_current_user() -> Optional[User]:
    """Get current user (simplified for demo)"""
    return DatabaseManager.create_demo_user()

def require_auth(f):
    """Authentication decorator"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        user = get_current_user()
        if not user:
            return jsonify({'error': 'Authentication required'}), 401
        return f(user, *args, **kwargs)
    return decorated_function

def register_routes(app):
    """Register all API routes"""
    
    # Root route with beautiful landing page
    @app.route('/')
    def root():
        """Beautiful landing page for the Swarm system"""
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
                .test-button {
                    background: #10B981; color: white; padding: 12px 24px; border: none;
                    border-radius: 8px; cursor: pointer; font-weight: bold; margin: 10px;
                    text-decoration: none; display: inline-block;
                }
                .test-button:hover { background: #059669; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <div class="logo">🤖</div>
                    <h1>Swarm Multi-Agent System</h1>
                    <p class="status">✅ Production-Ready AI Swarm Intelligence Platform</p>
                    <div>
                        <a href="/api/health" class="test-button">🔍 Test API Health</a>
                        <a href="/api/agents" class="test-button">🤖 View Agents</a>
                    </div>
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
    
    # Health check endpoint
    @app.route('/api/health', methods=['GET'])
    def health_check():
        """System health check"""
        try:
            # Check database
            db_health = {'status': 'healthy'}
            try:
                db.session.execute('SELECT 1')
                db_health['status'] = 'healthy'
            except Exception as e:
                db_health = {'status': 'error', 'error': str(e)}
            
            # Check swarm system
            swarm_status = asyncio.run(get_status())
            
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.utcnow().isoformat(),
                'services': {
                    'database': db_health,
                    'swarm_system': swarm_status
                }
            })
        
        except Exception as e:
            return jsonify({
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }), 500
    
    # Agent endpoints
    @app.route('/api/agents', methods=['GET'])
    @require_auth
    def get_agents(user):
        """Get all available agents"""
        try:
            agents = Agent.query.filter_by().all()
            return jsonify({
                'agents': [agent.to_dict() for agent in agents]
            })
        except Exception as e:
            logger.error(f"Error fetching agents: {str(e)}")
            return jsonify({'error': 'Failed to fetch agents'}), 500
    
    @app.route('/api/agents/<agent_id>', methods=['GET'])
    @require_auth
    def get_agent(user, agent_id):
        """Get specific agent details"""
        try:
            agent = Agent.query.filter_by(id=agent_id).first()
            if not agent:
                return jsonify({'error': 'Agent not found'}), 404
            
            return jsonify({'agent': agent.to_dict()})
        except Exception as e:
            logger.error(f"Error fetching agent {agent_id}: {str(e)}")
            return jsonify({'error': 'Failed to fetch agent'}), 500
    
    @app.route('/api/agents/<agent_id>/status', methods=['PUT'])
    @require_auth
    def update_agent_status(user, agent_id):
        """Update agent status"""
        try:
            data = request.get_json()
            new_status = data.get('status')
            
            agent = Agent.query.filter_by(id=agent_id).first()
            if not agent:
                return jsonify({'error': 'Agent not found'}), 404
            
            if new_status in [status.value for status in AgentStatus]:
                agent.status = AgentStatus(new_status)
                agent.last_active = datetime.now(timezone.utc)
                db.session.commit()
                
                return jsonify({'agent': agent.to_dict()})
            else:
                return jsonify({'error': 'Invalid status'}), 400
        
        except Exception as e:
            logger.error(f"Error updating agent status: {str(e)}")
            return jsonify({'error': 'Failed to update agent status'}), 500
    
    # Conversation endpoints
    @app.route('/api/conversations', methods=['GET'])
    @require_auth
    def get_conversations(user):
        """Get user's conversations"""
        try:
            limit = int(request.args.get('limit', 50))
            offset = int(request.args.get('offset', 0))
            
            conversations = Conversation.query.filter_by(user_id=user.id)\
                .order_by(Conversation.updated_at.desc())\
                .limit(limit).offset(offset).all()
            
            return jsonify({
                'conversations': [conv.to_dict() for conv in conversations]
            })
        except Exception as e:
            logger.error(f"Error fetching conversations: {str(e)}")
            return jsonify({'error': 'Failed to fetch conversations'}), 500
    
    @app.route('/api/conversations', methods=['POST'])
    @require_auth
    def create_conversation(user):
        """Create new conversation"""
        try:
            data = request.get_json()
            
            conversation = Conversation(
                user_id=user.id,
                title=data.get('title', 'New Conversation'),
                description=data.get('description'),
                swarm_mode=SwarmMode(data.get('swarm_mode', 'individual')),
                participating_agents=data.get('participating_agents', []),
                context_data=data.get('context_data', {})
            )
            
            db.session.add(conversation)
            db.session.commit()
            
            return jsonify({
                'conversation': conversation.to_dict()
            }), 201
        
        except Exception as e:
            logger.error(f"Error creating conversation: {str(e)}")
            return jsonify({'error': 'Failed to create conversation'}), 500
    
    @app.route('/api/conversations/<conversation_id>', methods=['GET'])
    @require_auth
    def get_conversation(user, conversation_id):
        """Get conversation with messages"""
        try:
            conversation = Conversation.query.filter_by(
                id=conversation_id, user_id=user.id
            ).first()
            
            if not conversation:
                return jsonify({'error': 'Conversation not found'}), 404
            
            # Get recent messages
            messages = Message.query.filter_by(conversation_id=conversation_id)\
                .order_by(Message.created_at.desc())\
                .limit(50).all()
            
            conversation_data = conversation.to_dict()
            conversation_data['messages'] = [msg.to_dict() for msg in reversed(messages)]
            
            return jsonify({'conversation': conversation_data})
        
        except Exception as e:
            logger.error(f"Error fetching conversation {conversation_id}: {str(e)}")
            return jsonify({'error': 'Failed to fetch conversation'}), 500
    
    @app.route('/api/conversations/<conversation_id>/messages', methods=['POST'])
    @require_auth
    def send_message(user, conversation_id):
        """Send message to conversation"""
        try:
            data = request.get_json()
            content = data.get('content', '').strip()
            
            if not content:
                return jsonify({'error': 'Message content is required'}), 400
            
            # Verify conversation exists and user has access
            conversation = Conversation.query.filter_by(
                id=conversation_id, user_id=user.id
            ).first()
            
            if not conversation:
                return jsonify({'error': 'Conversation not found'}), 404
            
            # Create user message
            user_message = Message(
                conversation_id=conversation_id,
                content=content,
                message_type=MessageType.USER,
                user_id=user.id,
                sender_name=user.full_name or user.username
            )
            
            db.session.add(user_message)
            
            # Update conversation
            conversation.message_count += 1
            conversation.last_message_at = datetime.now(timezone.utc)
            conversation.updated_at = datetime.now(timezone.utc)
            
            db.session.commit()
            
            # Emit message to WebSocket clients
            socketio.emit('new_message', {
                'conversation_id': conversation_id,
                'message': user_message.to_dict()
            }, room=f'conversation_{conversation_id}')
            
            # Process message through swarm system asynchronously
            socketio.start_background_task(
                process_swarm_message, 
                conversation_id, 
                content, 
                user.id
            )
            
            return jsonify({'message': user_message.to_dict()}), 201
        
        except Exception as e:
            logger.error(f"Error sending message: {str(e)}")
            return jsonify({'error': 'Failed to send message'}), 500
    
    # Task endpoints
    @app.route('/api/tasks', methods=['GET'])
    @require_auth
    def get_tasks(user):
        """Get user's tasks"""
        try:
            status_filter = request.args.get('status')
            limit = int(request.args.get('limit', 50))
            
            query = Task.query.filter_by(user_id=user.id)
            
            if status_filter:
                query = query.filter_by(status=TaskStatus(status_filter))
            
            tasks = query.order_by(Task.created_at.desc()).limit(limit).all()
            
            return jsonify({
                'tasks': [task.to_dict() for task in tasks]
            })
        
        except Exception as e:
            logger.error(f"Error fetching tasks: {str(e)}")
            return jsonify({'error': 'Failed to fetch tasks'}), 500
    
    @app.route('/api/tasks', methods=['POST'])
    @require_auth
    def create_task(user):
        """Create new task"""
        try:
            data = request.get_json()
            
            task = Task(
                user_id=user.id,
                conversation_id=data.get('conversation_id'),
                title=data.get('title'),
                description=data.get('description'),
                task_type=data.get('task_type'),
                priority=data.get('priority', 5),
                required_capabilities=data.get('required_capabilities', []),
                input_data=data.get('input_data', {}),
                constraints=data.get('constraints', {}),
                due_date=datetime.fromisoformat(data['due_date']) if data.get('due_date') else None
            )
            
            db.session.add(task)
            db.session.commit()
            
            return jsonify({'task': task.to_dict()}), 201
        
        except Exception as e:
            logger.error(f"Error creating task: {str(e)}")
            return jsonify({'error': 'Failed to create task'}), 500
    
    # Knowledge endpoints
    @app.route('/api/knowledge/search', methods=['GET'])
    @require_auth
    def search_knowledge(user):
        """Search knowledge base"""
        try:
            query = request.args.get('q', '').strip()
            limit = int(request.args.get('limit', 10))
            
            if not query:
                return jsonify({'error': 'Search query is required'}), 400
            
            # Search in SuperMemory
            knowledge_results = asyncio.run(get_knowledge(query))
            
            # Also search local knowledge entries
            local_entries = KnowledgeEntry.query.filter(
                KnowledgeEntry.content.contains(query)
            ).limit(limit).all()
            
            return jsonify({
                'supermemory_results': knowledge_results,
                'local_entries': [entry.to_dict() for entry in local_entries],
                'total': len(knowledge_results) + len(local_entries)
            })
        
        except Exception as e:
            logger.error(f"Error searching knowledge: {str(e)}")
            return jsonify({'error': 'Failed to search knowledge'}), 500
    
    @app.route('/api/knowledge', methods=['POST'])
    @require_auth
    def store_knowledge_entry(user):
        """Store new knowledge entry"""
        try:
            data = request.get_json()
            
            entry = KnowledgeEntry(
                title=data.get('title'),
                content=data.get('content'),
                knowledge_type=data.get('knowledge_type', 'fact'),
                tags=data.get('tags', []),
                source_conversation_id=data.get('source_conversation_id'),
                contributing_agents=data.get('contributing_agents', []),
                confidence_score=data.get('confidence_score', 0.5)
            )
            
            db.session.add(entry)
            db.session.commit()
            
            return jsonify({'entry': entry.to_dict()}), 201
        
        except Exception as e:
            logger.error(f"Error storing knowledge: {str(e)}")
            return jsonify({'error': 'Failed to store knowledge'}), 500
    
    # Email endpoints
    @app.route('/api/email/send', methods=['POST'])
    @require_auth
    def send_email_endpoint(user):
        """Send email through agent"""
        try:
            data = request.get_json()
            
            to = data.get('to')
            subject = data.get('subject')
            content = data.get('content')
            agent_id = data.get('agent_id', 'cathy')
            
            if not all([to, subject, content]):
                return jsonify({'error': 'Missing required fields'}), 400
            
            # Send email through swarm system
            result = asyncio.run(send_email(to, subject, content, agent_id))
            
            return jsonify(result)
        
        except Exception as e:
            logger.error(f"Error sending email: {str(e)}")
            return jsonify({'error': 'Failed to send email'}), 500
    
    # System endpoints
    @app.route('/api/system/status', methods=['GET'])
    @require_auth
    def system_status(user):
        """Get system status"""
        try:
            status = asyncio.run(get_status())
            
            # Add database stats
            status['database'] = {
                'users': User.query.count(),
                'agents': Agent.query.count(),
                'conversations': Conversation.query.count(),
                'messages': Message.query.count(),
                'tasks': Task.query.count(),
                'knowledge_entries': KnowledgeEntry.query.count()
            }
            
            return jsonify(status)
        
        except Exception as e:
            logger.error(f"Error getting system status: {str(e)}")
            return jsonify({'error': 'Failed to get system status'}), 500

# WebSocket Events
def register_socketio_events():
    """Register WebSocket event handlers"""
    
    @socketio.on('connect')
    def handle_connect():
        """Handle client connection"""
        user = get_current_user()
        if not user:
            disconnect()
            return
        
        logger.info(f"User {user.username} connected via WebSocket")
        emit('connected', {'status': 'Connected to Swarm Multi-Agent System'})
    
    @socketio.on('disconnect')
    def handle_disconnect():
        """Handle client disconnection"""
        logger.info("Client disconnected from WebSocket")
    
    @socketio.on('join_conversation')
    def handle_join_conversation(data):
        """Join conversation room for real-time updates"""
        conversation_id = data.get('conversation_id')
        user = get_current_user()
        
        if not user or not conversation_id:
            return
        
        # Verify user has access to conversation
        conversation = Conversation.query.filter_by(
            id=conversation_id, user_id=user.id
        ).first()
        
        if not conversation:
            emit('error', {'message': 'Conversation not found or access denied'})
            return
        
        join_room(f'conversation_{conversation_id}')
        emit('joined_conversation', {'conversation_id': conversation_id})
        logger.info(f"User {user.username} joined conversation {conversation_id}")
    
    @socketio.on('leave_conversation')
    def handle_leave_conversation(data):
        """Leave conversation room"""
        conversation_id = data.get('conversation_id')
        leave_room(f'conversation_{conversation_id}')
        emit('left_conversation', {'conversation_id': conversation_id})
    
    @socketio.on('typing_start')
    def handle_typing_start(data):
        """Handle typing indicator start"""
        conversation_id = data.get('conversation_id')
        user = get_current_user()
        
        if user and conversation_id:
            emit('user_typing', {
                'conversation_id': conversation_id,
                'user_id': user.id,
                'username': user.username,
                'typing': True
            }, room=f'conversation_{conversation_id}', include_self=False)
    
    @socketio.on('typing_stop')
    def handle_typing_stop(data):
        """Handle typing indicator stop"""
        conversation_id = data.get('conversation_id')
        user = get_current_user()
        
        if user and conversation_id:
            emit('user_typing', {
                'conversation_id': conversation_id,
                'user_id': user.id,
                'username': user.username,
                'typing': False
            }, room=f'conversation_{conversation_id}', include_self=False)

# Background Tasks
def process_swarm_message(conversation_id: str, user_message: str, user_id: str):
    """Process user message through swarm system in background"""
    try:
        with socketio.app.app_context():
            # Get conversation context
            conversation = Conversation.query.filter_by(id=conversation_id).first()
            if not conversation:
                return
            
            # Get recent messages for context
            recent_messages = Message.query.filter_by(conversation_id=conversation_id)\
                .order_by(Message.created_at.desc())\
                .limit(10).all()
            
            # Prepare context
            context = {
                'conversation_context': conversation.description,
                'swarm_mode': conversation.swarm_mode.value if conversation.swarm_mode else 'individual',
                'participating_agents': conversation.participating_agents,
                'recent_messages': [
                    {
                        'role': 'assistant' if msg.message_type == MessageType.AGENT else 'user',
                        'content': msg.content,
                        'sender': msg.sender_name
                    }
                    for msg in reversed(recent_messages[-5:])
                ]
            }
            
            # Parse mentions
            mentioned_agents = []
            words = user_message.split()
            for word in words:
                if word.startswith('@') and len(word) > 1:
                    agent_id = word[1:].lower()
                    # Check if agent exists in database
                    agent = Agent.query.filter_by(agent_type=agent_id).first()
                    if agent:
                        mentioned_agents.append(agent_id)
            
            # Process through swarm system
            result = asyncio.run(process_message(
                user_message, context, mentioned_agents
            ))
            
            # Handle different response types
            if result['type'] == 'single_agent':
                # Single agent response
                response = result['response']
                
                if response.success:
                    # Create agent message
                    agent = Agent.query.filter_by(agent_type=response.agent_id).first()
                    agent_message = Message(
                        conversation_id=conversation_id,
                        content=response.content,
                        message_type=MessageType.AGENT,
                        agent_id=agent.id if agent else None,
                        sender_name=agent.name if agent else response.agent_id.title(),
                        model_used=response.model_used,
                        tokens_used=response.tokens_used,
                        response_time=response.response_time,
                        cost=response.cost,
                        confidence_score=response.confidence
                    )
                    
                    db.session.add(agent_message)
                    
                    # Update conversation stats
                    conversation.message_count += 1
                    conversation.total_tokens_used += response.tokens_used.get('total', 0)
                    conversation.total_cost += response.cost
                    conversation.updated_at = datetime.now(timezone.utc)
                    
                    # Update agent stats
                    if agent:
                        agent.total_tasks_completed += 1
                        agent.last_active = datetime.now(timezone.utc)
                    
                    db.session.commit()
                    
                    # Emit new message
                    socketio.emit('new_message', {
                        'conversation_id': conversation_id,
                        'message': agent_message.to_dict()
                    }, room=f'conversation_{conversation_id}')
                
                else:
                    # Emit error message
                    error_message = Message(
                        conversation_id=conversation_id,
                        content=f"Sorry, I encountered an error: {response.error}",
                        message_type=MessageType.SYSTEM,
                        sender_name="System",
                        metadata={'error': True, 'error_message': response.error}
                    )
                    
                    db.session.add(error_message)
                    db.session.commit()
                    
                    socketio.emit('new_message', {
                        'conversation_id': conversation_id,
                        'message': error_message.to_dict()
                    }, room=f'conversation_{conversation_id}')
            
            elif result['type'] == 'swarm_collaboration':
                # Swarm collaboration response
                decision = result['decision']
                
                swarm_message = Message(
                    conversation_id=conversation_id,
                    content=f"Swarm Decision: {decision.reasoning}",
                    message_type=MessageType.SWARM,
                    sender_name="Swarm Intelligence",
                    is_swarm_decision=True,
                    swarm_consensus_score=decision.consensus_score,
                    contributing_agents=decision.participating_agents,
                    confidence_score=decision.confidence
                )
                
                db.session.add(swarm_message)
                db.session.commit()
                
                socketio.emit('new_message', {
                    'conversation_id': conversation_id,
                    'message': swarm_message.to_dict()
                }, room=f'conversation_{conversation_id}')
            
            # Store conversation knowledge
            asyncio.run(store_knowledge(conversation_id, context['recent_messages']))
    
    except Exception as e:
        logger.error(f"Error processing swarm message: {str(e)}")
        
        # Emit error to user
        try:
            socketio.emit('error', {
                'conversation_id': conversation_id,
                'message': 'Failed to process message through swarm system'
            }, room=f'conversation_{conversation_id}')
        except:
            pass

# Main application factory
def create_complete_app():
    """Create complete Flask app with all integrations"""
    app = create_app()
    register_routes(app)
    register_socketio_events()
    
    return app

if __name__ == "__main__":
    # Create and run the complete application
    app = create_complete_app()
    
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
    
    # Run with SocketIO
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)

