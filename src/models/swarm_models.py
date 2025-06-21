# 🚀 SWARM MULTI-AGENT SYSTEM - DATABASE MODELS

from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timezone
from enum import Enum
import uuid
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

db = SQLAlchemy()

# Enums for better type safety
class AgentStatus(Enum):
    IDLE = "idle"
    ACTIVE = "active"
    BUSY = "busy"
    ERROR = "error"
    OFFLINE = "offline"

class TaskStatus(Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class MessageType(Enum):
    USER = "user"
    AGENT = "agent"
    SYSTEM = "system"
    SWARM = "swarm"

class SwarmMode(Enum):
    INDIVIDUAL = "individual"
    COLLABORATIVE = "collaborative"
    COMPETITIVE = "competitive"
    HIERARCHICAL = "hierarchical"

# Core Models

class User(db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    username = db.Column(db.String(100), unique=True, nullable=False, index=True)
    full_name = db.Column(db.String(255))
    password_hash = db.Column(db.String(255), nullable=False)
    
    # User preferences
    preferred_models = db.Column(db.JSON, default=list)
    swarm_preferences = db.Column(db.JSON, default=dict)
    ui_preferences = db.Column(db.JSON, default=dict)
    
    # Status and permissions
    is_active = db.Column(db.Boolean, default=True)
    is_verified = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    last_login = db.Column(db.DateTime)
    
    # Relationships
    conversations = db.relationship('Conversation', backref='user', lazy='dynamic')
    tasks = db.relationship('Task', backref='user', lazy='dynamic')
    
    def to_dict(self):
        return {
            'id': self.id,
            'email': self.email,
            'username': self.username,
            'full_name': self.full_name,
            'preferred_models': self.preferred_models,
            'swarm_preferences': self.swarm_preferences,
            'ui_preferences': self.ui_preferences,
            'is_active': self.is_active,
            'is_verified': self.is_verified,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_login': self.last_login.isoformat() if self.last_login else None
        }

class Agent(db.Model):
    __tablename__ = 'agents'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = db.Column(db.String(100), nullable=False)
    agent_type = db.Column(db.String(50), nullable=False)  # 'cathy', 'dataminer', 'coder', etc.
    description = db.Column(db.Text)
    
    # Agent configuration
    capabilities = db.Column(db.JSON, default=list)  # List of capabilities
    preferred_models = db.Column(db.JSON, default=list)  # Preferred AI models
    system_prompt = db.Column(db.Text)
    personality_traits = db.Column(db.JSON, default=dict)
    
    # Swarm behavior
    collaboration_style = db.Column(db.String(50), default='cooperative')
    leadership_score = db.Column(db.Float, default=0.5)  # 0-1 scale
    specialization_areas = db.Column(db.JSON, default=list)
    
    # Status and performance
    status = db.Column(db.Enum(AgentStatus), default=AgentStatus.IDLE)
    current_load = db.Column(db.Integer, default=0)  # Number of active tasks
    max_concurrent_tasks = db.Column(db.Integer, default=3)
    
    # Performance metrics
    total_tasks_completed = db.Column(db.Integer, default=0)
    average_response_time = db.Column(db.Float, default=0.0)
    success_rate = db.Column(db.Float, default=1.0)
    user_satisfaction_score = db.Column(db.Float, default=0.0)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    last_active = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    
    # Relationships
    messages = db.relationship('Message', backref='agent', lazy='dynamic')
    tasks = db.relationship('Task', backref='assigned_agent', lazy='dynamic')
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'agent_type': self.agent_type,
            'description': self.description,
            'capabilities': self.capabilities,
            'preferred_models': self.preferred_models,
            'system_prompt': self.system_prompt,
            'personality_traits': self.personality_traits,
            'collaboration_style': self.collaboration_style,
            'leadership_score': self.leadership_score,
            'specialization_areas': self.specialization_areas,
            'status': self.status.value if self.status else None,
            'current_load': self.current_load,
            'max_concurrent_tasks': self.max_concurrent_tasks,
            'total_tasks_completed': self.total_tasks_completed,
            'average_response_time': self.average_response_time,
            'success_rate': self.success_rate,
            'user_satisfaction_score': self.user_satisfaction_score,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_active': self.last_active.isoformat() if self.last_active else None
        }

class Conversation(db.Model):
    __tablename__ = 'conversations'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(db.String(36), db.ForeignKey('users.id'), nullable=False, index=True)
    title = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text)
    
    # Swarm configuration
    swarm_mode = db.Column(db.Enum(SwarmMode), default=SwarmMode.INDIVIDUAL)
    participating_agents = db.Column(db.JSON, default=list)  # List of agent IDs
    swarm_leader = db.Column(db.String(36), db.ForeignKey('agents.id'))
    
    # Conversation context
    context_data = db.Column(db.JSON, default=dict)
    shared_memory_id = db.Column(db.String(255))  # SuperMemory reference
    
    # Status and metadata
    is_active = db.Column(db.Boolean, default=True)
    message_count = db.Column(db.Integer, default=0)
    total_tokens_used = db.Column(db.Integer, default=0)
    total_cost = db.Column(db.Float, default=0.0)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    last_message_at = db.Column(db.DateTime)
    
    # Relationships
    messages = db.relationship('Message', backref='conversation', lazy='dynamic', cascade='all, delete-orphan')
    tasks = db.relationship('Task', backref='conversation', lazy='dynamic')
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'title': self.title,
            'description': self.description,
            'swarm_mode': self.swarm_mode.value if self.swarm_mode else None,
            'participating_agents': self.participating_agents,
            'swarm_leader': self.swarm_leader,
            'context_data': self.context_data,
            'shared_memory_id': self.shared_memory_id,
            'is_active': self.is_active,
            'message_count': self.message_count,
            'total_tokens_used': self.total_tokens_used,
            'total_cost': self.total_cost,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'last_message_at': self.last_message_at.isoformat() if self.last_message_at else None
        }

class Message(db.Model):
    __tablename__ = 'messages'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_id = db.Column(db.String(36), db.ForeignKey('conversations.id'), nullable=False, index=True)
    
    # Message content
    content = db.Column(db.Text, nullable=False)
    message_type = db.Column(db.Enum(MessageType), nullable=False)
    
    # Sender information
    user_id = db.Column(db.String(36), db.ForeignKey('users.id'), nullable=True, index=True)
    agent_id = db.Column(db.String(36), db.ForeignKey('agents.id'), nullable=True, index=True)
    sender_name = db.Column(db.String(255))
    
    # Message metadata
    mentions = db.Column(db.JSON, default=list)  # List of mentioned agent IDs
    attachments = db.Column(db.JSON, default=list)  # File attachments
    message_metadata = db.Column(db.JSON, default=dict)  # Additional metadata (renamed to avoid SQLAlchemy conflict)
    
    # AI response metadata (for agent messages)
    model_used = db.Column(db.String(255))
    tokens_used = db.Column(db.JSON, default=dict)  # {'input': X, 'output': Y}
    response_time = db.Column(db.Float)  # Response time in seconds
    cost = db.Column(db.Float, default=0.0)
    confidence_score = db.Column(db.Float)  # Agent confidence in response
    
    # Swarm coordination
    is_swarm_decision = db.Column(db.Boolean, default=False)
    swarm_consensus_score = db.Column(db.Float)  # If this was a swarm decision
    contributing_agents = db.Column(db.JSON, default=list)  # Agents that contributed
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    
    def to_dict(self):
        return {
            'id': self.id,
            'conversation_id': self.conversation_id,
            'content': self.content,
            'message_type': self.message_type.value if self.message_type else None,
            'user_id': self.user_id,
            'agent_id': self.agent_id,
            'sender_name': self.sender_name,
            'mentions': self.mentions,
            'attachments': self.attachments,
            'message_metadata': self.message_metadata,
            'model_used': self.model_used,
            'tokens_used': self.tokens_used,
            'response_time': self.response_time,
            'cost': self.cost,
            'confidence_score': self.confidence_score,
            'is_swarm_decision': self.is_swarm_decision,
            'swarm_consensus_score': self.swarm_consensus_score,
            'contributing_agents': self.contributing_agents,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class Task(db.Model):
    __tablename__ = 'tasks'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(db.String(36), db.ForeignKey('users.id'), nullable=False, index=True)
    conversation_id = db.Column(db.String(36), db.ForeignKey('conversations.id'), nullable=True, index=True)
    
    # Task definition
    title = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text)
    task_type = db.Column(db.String(100))  # 'email', 'analysis', 'coding', etc.
    priority = db.Column(db.Integer, default=5)  # 1-10 scale
    
    # Task assignment
    assigned_agent_id = db.Column(db.String(36), db.ForeignKey('agents.id'), nullable=True, index=True)
    swarm_assignment = db.Column(db.JSON, default=list)  # Multiple agents for swarm tasks
    assignment_strategy = db.Column(db.String(50), default='auto')  # 'auto', 'manual', 'swarm'
    
    # Task execution
    status = db.Column(db.Enum(TaskStatus), default=TaskStatus.PENDING)
    progress_percentage = db.Column(db.Float, default=0.0)
    execution_steps = db.Column(db.JSON, default=list)
    current_step = db.Column(db.Integer, default=0)
    
    # Task context and requirements
    required_capabilities = db.Column(db.JSON, default=list)
    input_data = db.Column(db.JSON, default=dict)
    output_data = db.Column(db.JSON, default=dict)
    constraints = db.Column(db.JSON, default=dict)
    
    # Performance tracking
    estimated_duration = db.Column(db.Integer)  # Estimated duration in minutes
    actual_duration = db.Column(db.Integer)  # Actual duration in minutes
    quality_score = db.Column(db.Float)  # User-rated quality (1-5)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    assigned_at = db.Column(db.DateTime)
    started_at = db.Column(db.DateTime)
    completed_at = db.Column(db.DateTime)
    due_date = db.Column(db.DateTime)
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'conversation_id': self.conversation_id,
            'title': self.title,
            'description': self.description,
            'task_type': self.task_type,
            'priority': self.priority,
            'assigned_agent_id': self.assigned_agent_id,
            'swarm_assignment': self.swarm_assignment,
            'assignment_strategy': self.assignment_strategy,
            'status': self.status.value if self.status else None,
            'progress_percentage': self.progress_percentage,
            'execution_steps': self.execution_steps,
            'current_step': self.current_step,
            'required_capabilities': self.required_capabilities,
            'input_data': self.input_data,
            'output_data': self.output_data,
            'constraints': self.constraints,
            'estimated_duration': self.estimated_duration,
            'actual_duration': self.actual_duration,
            'quality_score': self.quality_score,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'assigned_at': self.assigned_at.isoformat() if self.assigned_at else None,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'due_date': self.due_date.isoformat() if self.due_date else None
        }

class SwarmSession(db.Model):
    __tablename__ = 'swarm_sessions'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_id = db.Column(db.String(36), db.ForeignKey('conversations.id'), nullable=False, index=True)
    
    # Swarm configuration
    session_type = db.Column(db.String(50))  # 'brainstorm', 'problem_solve', 'review', etc.
    participating_agents = db.Column(db.JSON, nullable=False)  # List of agent IDs
    leader_agent_id = db.Column(db.String(36), db.ForeignKey('agents.id'))
    
    # Session state
    current_phase = db.Column(db.String(50), default='initialization')
    phase_data = db.Column(db.JSON, default=dict)
    session_context = db.Column(db.JSON, default=dict)
    
    # Performance metrics
    consensus_score = db.Column(db.Float, default=0.0)  # How well agents agree
    efficiency_score = db.Column(db.Float, default=0.0)  # How efficiently they work
    quality_score = db.Column(db.Float, default=0.0)  # Quality of output
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    started_at = db.Column(db.DateTime)
    ended_at = db.Column(db.DateTime)
    
    def to_dict(self):
        return {
            'id': self.id,
            'conversation_id': self.conversation_id,
            'session_type': self.session_type,
            'participating_agents': self.participating_agents,
            'leader_agent_id': self.leader_agent_id,
            'current_phase': self.current_phase,
            'phase_data': self.phase_data,
            'session_context': self.session_context,
            'consensus_score': self.consensus_score,
            'efficiency_score': self.efficiency_score,
            'quality_score': self.quality_score,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'ended_at': self.ended_at.isoformat() if self.ended_at else None
        }

class KnowledgeEntry(db.Model):
    __tablename__ = 'knowledge_entries'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Knowledge content
    title = db.Column(db.String(255), nullable=False)
    content = db.Column(db.Text, nullable=False)
    knowledge_type = db.Column(db.String(50))  # 'fact', 'procedure', 'experience', etc.
    tags = db.Column(db.JSON, default=list)
    
    # Source information
    source_conversation_id = db.Column(db.String(36), db.ForeignKey('conversations.id'))
    source_message_id = db.Column(db.String(36), db.ForeignKey('messages.id'))
    contributing_agents = db.Column(db.JSON, default=list)
    
    # Knowledge metadata
    confidence_score = db.Column(db.Float, default=0.5)
    usage_count = db.Column(db.Integer, default=0)
    last_used = db.Column(db.DateTime)
    supermemory_id = db.Column(db.String(255))  # Reference to SuperMemory
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    def to_dict(self):
        return {
            'id': self.id,
            'title': self.title,
            'content': self.content,
            'knowledge_type': self.knowledge_type,
            'tags': self.tags,
            'source_conversation_id': self.source_conversation_id,
            'source_message_id': self.source_message_id,
            'contributing_agents': self.contributing_agents,
            'confidence_score': self.confidence_score,
            'usage_count': self.usage_count,
            'last_used': self.last_used.isoformat() if self.last_used else None,
            'supermemory_id': self.supermemory_id,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

# Data Access Layer Classes

class DatabaseManager:
    """Centralized database operations manager"""
    
    @staticmethod
    def create_demo_user() -> User:
        """Create demo user for testing"""
        demo_user = User.query.filter_by(email='demo@swarm.ai').first()
        
        if not demo_user:
            from werkzeug.security import generate_password_hash
            demo_user = User(
                email='demo@swarm.ai',
                username='demo',
                password_hash=generate_password_hash('demo123'),
                full_name='Demo User',
                is_active=True,
                is_verified=True,
                preferred_models=['gpt-4', 'claude-3-sonnet'],
                swarm_preferences={
                    'default_mode': 'collaborative',
                    'max_agents': 5,
                    'auto_assign': True
                }
            )
            db.session.add(demo_user)
            db.session.commit()
        
        return demo_user
    
    @staticmethod
    def create_default_agents():
        """Create default agent types"""
        default_agents = [
            {
                'name': 'Cathy',
                'agent_type': 'cathy',
                'description': 'Personal assistant and task coordinator',
                'capabilities': ['task_management', 'scheduling', 'email', 'general_assistance'],
                'system_prompt': 'You are Cathy, a helpful personal assistant focused on productivity and organization.',
                'personality_traits': {'helpful': 0.9, 'organized': 0.95, 'friendly': 0.8},
                'collaboration_style': 'coordinator',
                'leadership_score': 0.8,
                'specialization_areas': ['productivity', 'organization', 'communication']
            },
            {
                'name': 'DataMiner',
                'agent_type': 'dataminer',
                'description': 'Data analysis and insights specialist',
                'capabilities': ['data_analysis', 'visualization', 'statistics', 'research'],
                'system_prompt': 'You are DataMiner, an expert in data analysis and extracting insights from complex datasets.',
                'personality_traits': {'analytical': 0.95, 'precise': 0.9, 'curious': 0.85},
                'collaboration_style': 'specialist',
                'leadership_score': 0.6,
                'specialization_areas': ['data_science', 'analytics', 'research']
            },
            {
                'name': 'Coder',
                'agent_type': 'coder',
                'description': 'Software development and programming expert',
                'capabilities': ['coding', 'debugging', 'architecture', 'code_review'],
                'system_prompt': 'You are Coder, a skilled software developer who writes clean, efficient code.',
                'personality_traits': {'logical': 0.95, 'detail_oriented': 0.9, 'innovative': 0.8},
                'collaboration_style': 'implementer',
                'leadership_score': 0.7,
                'specialization_areas': ['programming', 'software_engineering', 'technical_architecture']
            },
            {
                'name': 'Creative',
                'agent_type': 'creative',
                'description': 'Creative content and design specialist',
                'capabilities': ['content_creation', 'design', 'brainstorming', 'storytelling'],
                'system_prompt': 'You are Creative, an imaginative agent focused on generating original ideas and content.',
                'personality_traits': {'creative': 0.95, 'expressive': 0.9, 'intuitive': 0.85},
                'collaboration_style': 'innovator',
                'leadership_score': 0.5,
                'specialization_areas': ['content_creation', 'design', 'marketing']
            },
            {
                'name': 'Researcher',
                'agent_type': 'researcher',
                'description': 'Research and information gathering expert',
                'capabilities': ['research', 'fact_checking', 'analysis', 'synthesis'],
                'system_prompt': 'You are Researcher, dedicated to finding accurate information and conducting thorough research.',
                'personality_traits': {'thorough': 0.95, 'skeptical': 0.8, 'methodical': 0.9},
                'collaboration_style': 'investigator',
                'leadership_score': 0.6,
                'specialization_areas': ['research', 'fact_checking', 'information_synthesis']
            }
        ]
        
        for agent_data in default_agents:
            existing_agent = Agent.query.filter_by(agent_type=agent_data['agent_type']).first()
            if not existing_agent:
                agent = Agent(**agent_data)
                db.session.add(agent)
        
        db.session.commit()
    
    @staticmethod
    def initialize_system():
        """Initialize the system with default data"""
        try:
            # Create tables
            db.create_all()
            
            # Create demo user
            DatabaseManager.create_demo_user()
            
            # Create default agents
            DatabaseManager.create_default_agents()
            
            print("✅ Database initialized successfully")
            
        except Exception as e:
            print(f"❌ Error initializing database: {str(e)}")
            db.session.rollback()
            raise

