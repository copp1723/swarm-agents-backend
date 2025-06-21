# 🤖 SWARM AGENT ORCHESTRATION SYSTEM

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import aiohttp
import requests
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration from environment variables
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY', 'your-openrouter-api-key')
SUPERMEMORY_API_KEY = os.getenv('SUPERMEMORY_API_KEY', 'your-supermemory-api-key')
MAILGUN_API_KEY = os.getenv('MAILGUN_API_KEY', 'your-mailgun-api-key')
MAILGUN_DOMAIN = os.getenv('MAILGUN_DOMAIN', 'your-domain.com')

@dataclass
class AgentResponse:
    """Response from an agent"""
    agent_id: str
    content: str
    confidence: float
    model_used: str
    tokens_used: Dict[str, int]
    response_time: float
    cost: float
    success: bool
    error: Optional[str] = None
    metadata: Dict[str, Any] = None

@dataclass
class SwarmDecision:
    """Decision made by the swarm"""
    decision_type: str
    selected_agent: Optional[str]
    consensus_score: float
    participating_agents: List[str]
    reasoning: str
    confidence: float
    alternative_options: List[Dict] = None

class SwarmStrategy(Enum):
    """Different swarm coordination strategies"""
    COMPETITIVE = "competitive"  # Agents compete, best response wins
    COLLABORATIVE = "collaborative"  # Agents work together
    HIERARCHICAL = "hierarchical"  # Leader delegates to specialists
    CONSENSUS = "consensus"  # All agents must agree
    SPECIALIST = "specialist"  # Route to most qualified agent

class OpenRouterClient:
    """OpenRouter API client for AI model interactions"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1"
        self.session = None
        
        # Available models with their capabilities
        self.models = {
            "gpt-4": {
                "name": "GPT-4",
                "cost_per_1k_tokens": {"input": 0.03, "output": 0.06},
                "max_tokens": 8192,
                "capabilities": ["reasoning", "coding", "analysis", "creative"]
            },
            "claude-3-sonnet": {
                "name": "Claude 3 Sonnet",
                "cost_per_1k_tokens": {"input": 0.003, "output": 0.015},
                "max_tokens": 4096,
                "capabilities": ["reasoning", "analysis", "writing", "research"]
            },
            "gpt-3.5-turbo": {
                "name": "GPT-3.5 Turbo",
                "cost_per_1k_tokens": {"input": 0.001, "output": 0.002},
                "max_tokens": 4096,
                "capabilities": ["general", "coding", "analysis"]
            }
        }
    
    async def get_session(self):
        """Get or create aiohttp session"""
        if not self.session:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def close_session(self):
        """Close aiohttp session"""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def generate_response(self, messages: List[Dict], model: str = "gpt-3.5-turbo", 
                              max_tokens: int = 1000, temperature: float = 0.7) -> AgentResponse:
        """Generate response using OpenRouter API"""
        start_time = time.time()
        
        try:
            session = await self.get_session()
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://swarm-agents.ai",
                "X-Title": "Swarm Multi-Agent System"
            }
            
            payload = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False
            }
            
            async with session.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload
            ) as response:
                
                if response.status == 200:
                    data = await response.json()
                    
                    # Extract response data
                    choice = data['choices'][0]
                    usage = data.get('usage', {})
                    
                    # Calculate cost
                    model_info = self.models.get(model, self.models["gpt-3.5-turbo"])
                    input_cost = (usage.get('prompt_tokens', 0) / 1000) * model_info['cost_per_1k_tokens']['input']
                    output_cost = (usage.get('completion_tokens', 0) / 1000) * model_info['cost_per_1k_tokens']['output']
                    total_cost = input_cost + output_cost
                    
                    response_time = time.time() - start_time
                    
                    return AgentResponse(
                        agent_id="openrouter",
                        content=choice['message']['content'],
                        confidence=0.8,  # Default confidence
                        model_used=model,
                        tokens_used={
                            'input': usage.get('prompt_tokens', 0),
                            'output': usage.get('completion_tokens', 0),
                            'total': usage.get('total_tokens', 0)
                        },
                        response_time=response_time,
                        cost=total_cost,
                        success=True,
                        metadata={'finish_reason': choice.get('finish_reason')}
                    )
                
                else:
                    error_text = await response.text()
                    logger.error(f"OpenRouter API error: {response.status} - {error_text}")
                    
                    return AgentResponse(
                        agent_id="openrouter",
                        content="",
                        confidence=0.0,
                        model_used=model,
                        tokens_used={},
                        response_time=time.time() - start_time,
                        cost=0.0,
                        success=False,
                        error=f"API Error: {response.status} - {error_text}"
                    )
        
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return AgentResponse(
                agent_id="openrouter",
                content="",
                confidence=0.0,
                model_used=model,
                tokens_used={},
                response_time=time.time() - start_time,
                cost=0.0,
                success=False,
                error=str(e)
            )

class SuperMemoryClient:
    """SuperMemory client for shared knowledge management"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.supermemory.ai/v1"  # Adjust based on actual API
    
    async def store_knowledge(self, content: str, tags: List[str] = None, 
                            metadata: Dict = None) -> Dict:
        """Store knowledge in SuperMemory"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "content": content,
                "tags": tags or [],
                "metadata": metadata or {}
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/memories",
                    headers=headers,
                    json=payload
                ) as response:
                    
                    if response.status == 200:
                        return await response.json()
                    else:
                        logger.error(f"SuperMemory store error: {response.status}")
                        return {"success": False, "error": f"HTTP {response.status}"}
        
        except Exception as e:
            logger.error(f"Error storing knowledge: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def retrieve_knowledge(self, query: str, limit: int = 10) -> List[Dict]:
        """Retrieve relevant knowledge from SuperMemory"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            params = {
                "query": query,
                "limit": limit
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/memories/search",
                    headers=headers,
                    params=params
                ) as response:
                    
                    if response.status == 200:
                        data = await response.json()
                        return data.get("memories", [])
                    else:
                        logger.error(f"SuperMemory retrieve error: {response.status}")
                        return []
        
        except Exception as e:
            logger.error(f"Error retrieving knowledge: {str(e)}")
            return []

class MailgunClient:
    """Mailgun client for email functionality"""
    
    def __init__(self, api_key: str, domain: str):
        self.api_key = api_key
        self.domain = domain
        self.base_url = f"https://api.mailgun.net/v3/{domain}"
    
    async def send_email(self, to: str, subject: str, text: str, 
                        html: str = None, from_email: str = None) -> Dict:
        """Send email via Mailgun"""
        try:
            if not from_email:
                from_email = f"swarm@{self.domain}"
            
            data = {
                "from": from_email,
                "to": to,
                "subject": subject,
                "text": text
            }
            
            if html:
                data["html"] = html
            
            auth = ("api", self.api_key)
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/messages",
                    auth=aiohttp.BasicAuth("api", self.api_key),
                    data=data
                ) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        return {"success": True, "message_id": result.get("id")}
                    else:
                        error_text = await response.text()
                        logger.error(f"Mailgun error: {response.status} - {error_text}")
                        return {"success": False, "error": f"HTTP {response.status}"}
        
        except Exception as e:
            logger.error(f"Error sending email: {str(e)}")
            return {"success": False, "error": str(e)}

class SwarmAgent:
    """Individual agent in the swarm"""
    
    def __init__(self, agent_id: str, agent_type: str, capabilities: List[str],
                 system_prompt: str, preferred_models: List[str] = None):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.capabilities = capabilities
        self.system_prompt = system_prompt
        self.preferred_models = preferred_models or ["gpt-3.5-turbo"]
        self.performance_history = []
        self.current_load = 0
        self.max_concurrent_tasks = 3
    
    def can_handle_task(self, required_capabilities: List[str]) -> float:
        """Calculate how well this agent can handle a task (0-1 score)"""
        if not required_capabilities:
            return 0.5  # Neutral score for general tasks
        
        matching_capabilities = set(self.capabilities) & set(required_capabilities)
        score = len(matching_capabilities) / len(required_capabilities)
        
        # Adjust for current load
        load_factor = 1.0 - (self.current_load / self.max_concurrent_tasks)
        
        return score * load_factor
    
    def get_preferred_model(self, task_type: str = None) -> str:
        """Get preferred model for this agent/task"""
        # Simple model selection logic
        if task_type == "coding" and "gpt-4" in self.preferred_models:
            return "gpt-4"
        elif task_type == "analysis" and "claude-3-sonnet" in self.preferred_models:
            return "claude-3-sonnet"
        else:
            return self.preferred_models[0]

class SwarmOrchestrator:
    """Main orchestrator for the swarm system"""
    
    def __init__(self):
        self.openrouter = OpenRouterClient(OPENROUTER_API_KEY)
        self.supermemory = SuperMemoryClient(SUPERMEMORY_API_KEY)
        self.mailgun = MailgunClient(MAILGUN_API_KEY, MAILGUN_DOMAIN)
        
        self.agents = {}
        self.active_sessions = {}
        self.performance_metrics = {}
        
        # Initialize default agents
        self._initialize_default_agents()
    
    def _initialize_default_agents(self):
        """Initialize default swarm agents"""
        default_agents = [
            SwarmAgent(
                agent_id="cathy",
                agent_type="assistant",
                capabilities=["task_management", "scheduling", "email", "general_assistance"],
                system_prompt="You are Cathy, a helpful personal assistant focused on productivity and organization. You excel at coordinating tasks and managing schedules.",
                preferred_models=["gpt-3.5-turbo", "gpt-4"]
            ),
            SwarmAgent(
                agent_id="dataminer",
                agent_type="analyst",
                capabilities=["data_analysis", "visualization", "statistics", "research"],
                system_prompt="You are DataMiner, an expert in data analysis and extracting insights from complex datasets. You provide thorough, accurate analysis.",
                preferred_models=["claude-3-sonnet", "gpt-4"]
            ),
            SwarmAgent(
                agent_id="coder",
                agent_type="developer",
                capabilities=["coding", "debugging", "architecture", "code_review"],
                system_prompt="You are Coder, a skilled software developer who writes clean, efficient code and provides excellent technical guidance.",
                preferred_models=["gpt-4", "claude-3-sonnet"]
            ),
            SwarmAgent(
                agent_id="creative",
                agent_type="creator",
                capabilities=["content_creation", "design", "brainstorming", "storytelling"],
                system_prompt="You are Creative, an imaginative agent focused on generating original ideas and compelling content.",
                preferred_models=["gpt-4", "gpt-3.5-turbo"]
            ),
            SwarmAgent(
                agent_id="researcher",
                agent_type="investigator",
                capabilities=["research", "fact_checking", "analysis", "synthesis"],
                system_prompt="You are Researcher, dedicated to finding accurate information and conducting thorough research with proper citations.",
                preferred_models=["claude-3-sonnet", "gpt-4"]
            )
        ]
        
        for agent in default_agents:
            self.agents[agent.agent_id] = agent
        
        logger.info(f"Initialized {len(self.agents)} default agents")
    
    def select_agent_for_task(self, required_capabilities: List[str], 
                            strategy: SwarmStrategy = SwarmStrategy.SPECIALIST) -> Optional[str]:
        """Select the best agent for a task based on strategy"""
        
        if strategy == SwarmStrategy.SPECIALIST:
            # Find the most qualified agent
            best_agent = None
            best_score = 0.0
            
            for agent_id, agent in self.agents.items():
                score = agent.can_handle_task(required_capabilities)
                if score > best_score:
                    best_score = score
                    best_agent = agent_id
            
            return best_agent if best_score > 0.3 else None
        
        elif strategy == SwarmStrategy.COLLABORATIVE:
            # Return multiple agents for collaboration
            qualified_agents = []
            for agent_id, agent in self.agents.items():
                score = agent.can_handle_task(required_capabilities)
                if score > 0.2:
                    qualified_agents.append((agent_id, score))
            
            # Return top 3 agents
            qualified_agents.sort(key=lambda x: x[1], reverse=True)
            return [agent_id for agent_id, _ in qualified_agents[:3]]
        
        return None
    
    async def generate_agent_response(self, agent_id: str, messages: List[Dict],
                                    context: Dict = None) -> AgentResponse:
        """Generate response from a specific agent"""
        
        if agent_id not in self.agents:
            return AgentResponse(
                agent_id=agent_id,
                content="",
                confidence=0.0,
                model_used="",
                tokens_used={},
                response_time=0.0,
                cost=0.0,
                success=False,
                error=f"Agent {agent_id} not found"
            )
        
        agent = self.agents[agent_id]
        
        # Prepare messages with system prompt
        enhanced_messages = [
            {"role": "system", "content": agent.system_prompt}
        ] + messages
        
        # Add context if available
        if context:
            context_message = f"Context: {json.dumps(context, indent=2)}"
            enhanced_messages.insert(1, {"role": "system", "content": context_message})
        
        # Get preferred model for this agent
        model = agent.get_preferred_model(context.get('task_type') if context else None)
        
        # Generate response
        response = await self.openrouter.generate_response(
            messages=enhanced_messages,
            model=model,
            max_tokens=1000,
            temperature=0.7
        )
        
        # Update agent performance
        agent.performance_history.append({
            'timestamp': datetime.now(timezone.utc),
            'response_time': response.response_time,
            'success': response.success,
            'cost': response.cost
        })
        
        # Update response with agent info
        response.agent_id = agent_id
        
        return response
    
    async def swarm_collaboration(self, messages: List[Dict], participating_agents: List[str],
                                strategy: SwarmStrategy = SwarmStrategy.COLLABORATIVE,
                                context: Dict = None) -> SwarmDecision:
        """Coordinate multiple agents for collaborative problem solving"""
        
        start_time = time.time()
        agent_responses = []
        
        # Generate responses from all participating agents
        tasks = []
        for agent_id in participating_agents:
            if agent_id in self.agents:
                task = self.generate_agent_response(agent_id, messages, context)
                tasks.append(task)
        
        # Wait for all responses
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process responses
        valid_responses = []
        for response in responses:
            if isinstance(response, AgentResponse) and response.success:
                valid_responses.append(response)
        
        if not valid_responses:
            return SwarmDecision(
                decision_type="error",
                selected_agent=None,
                consensus_score=0.0,
                participating_agents=participating_agents,
                reasoning="No valid responses from agents",
                confidence=0.0
            )
        
        # Apply strategy to select best response
        if strategy == SwarmStrategy.COMPETITIVE:
            # Select response with highest confidence
            best_response = max(valid_responses, key=lambda r: r.confidence)
            
            return SwarmDecision(
                decision_type="competitive_selection",
                selected_agent=best_response.agent_id,
                consensus_score=best_response.confidence,
                participating_agents=participating_agents,
                reasoning=f"Selected {best_response.agent_id} with highest confidence ({best_response.confidence:.2f})",
                confidence=best_response.confidence,
                alternative_options=[
                    {"agent": r.agent_id, "confidence": r.confidence, "content": r.content[:100]}
                    for r in valid_responses if r.agent_id != best_response.agent_id
                ]
            )
        
        elif strategy == SwarmStrategy.CONSENSUS:
            # Combine responses and calculate consensus
            combined_content = "\n\n".join([
                f"**{r.agent_id}**: {r.content}" for r in valid_responses
            ])
            
            avg_confidence = sum(r.confidence for r in valid_responses) / len(valid_responses)
            
            return SwarmDecision(
                decision_type="consensus",
                selected_agent="swarm_consensus",
                consensus_score=avg_confidence,
                participating_agents=participating_agents,
                reasoning="Combined insights from all participating agents",
                confidence=avg_confidence
            )
        
        # Default: return best response
        best_response = max(valid_responses, key=lambda r: r.confidence)
        return SwarmDecision(
            decision_type="default_selection",
            selected_agent=best_response.agent_id,
            consensus_score=best_response.confidence,
            participating_agents=participating_agents,
            reasoning=f"Default selection of {best_response.agent_id}",
            confidence=best_response.confidence
        )
    
    async def process_user_message(self, user_message: str, conversation_context: Dict = None,
                                 mentioned_agents: List[str] = None) -> Dict:
        """Process a user message and coordinate appropriate agent responses"""
        
        # Parse mentions from message if not provided
        if mentioned_agents is None:
            mentioned_agents = []
            words = user_message.split()
            for word in words:
                if word.startswith('@') and len(word) > 1:
                    agent_id = word[1:].lower()
                    if agent_id in self.agents:
                        mentioned_agents.append(agent_id)
        
        # Prepare message format
        messages = [{"role": "user", "content": user_message}]
        
        # Add conversation context if available
        if conversation_context and conversation_context.get('recent_messages'):
            for msg in conversation_context['recent_messages'][-5:]:  # Last 5 messages
                messages.insert(-1, {
                    "role": msg.get('role', 'user'),
                    "content": msg.get('content', '')
                })
        
        # Determine strategy based on number of mentioned agents
        if len(mentioned_agents) == 0:
            # No specific agents mentioned, use intelligent routing
            required_capabilities = self._extract_capabilities_from_message(user_message)
            selected_agent = self.select_agent_for_task(required_capabilities)
            
            if selected_agent:
                response = await self.generate_agent_response(
                    selected_agent, messages, conversation_context
                )
                
                return {
                    "type": "single_agent",
                    "agent_id": selected_agent,
                    "response": response,
                    "strategy": "auto_routing"
                }
        
        elif len(mentioned_agents) == 1:
            # Single agent mentioned
            agent_id = mentioned_agents[0]
            response = await self.generate_agent_response(
                agent_id, messages, conversation_context
            )
            
            return {
                "type": "single_agent",
                "agent_id": agent_id,
                "response": response,
                "strategy": "direct_mention"
            }
        
        else:
            # Multiple agents mentioned, use swarm collaboration
            decision = await self.swarm_collaboration(
                messages, mentioned_agents, SwarmStrategy.COLLABORATIVE, conversation_context
            )
            
            return {
                "type": "swarm_collaboration",
                "decision": decision,
                "participating_agents": mentioned_agents,
                "strategy": "collaborative"
            }
        
        # Fallback: use default assistant
        response = await self.generate_agent_response(
            "cathy", messages, conversation_context
        )
        
        return {
            "type": "fallback",
            "agent_id": "cathy",
            "response": response,
            "strategy": "fallback"
        }
    
    def _extract_capabilities_from_message(self, message: str) -> List[str]:
        """Extract required capabilities from user message"""
        message_lower = message.lower()
        capabilities = []
        
        # Simple keyword matching (can be enhanced with NLP)
        capability_keywords = {
            "data_analysis": ["analyze", "data", "statistics", "chart", "graph", "trends"],
            "coding": ["code", "program", "debug", "script", "function", "algorithm"],
            "email": ["email", "send", "message", "contact", "mail"],
            "research": ["research", "find", "search", "investigate", "study"],
            "content_creation": ["write", "create", "content", "article", "blog", "story"],
            "task_management": ["schedule", "plan", "organize", "task", "todo", "reminder"]
        }
        
        for capability, keywords in capability_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                capabilities.append(capability)
        
        return capabilities if capabilities else ["general_assistance"]
    
    async def store_conversation_knowledge(self, conversation_id: str, 
                                         messages: List[Dict]) -> bool:
        """Store conversation knowledge in SuperMemory"""
        try:
            # Extract key insights from conversation
            content = "\n".join([
                f"{msg.get('sender', 'User')}: {msg.get('content', '')}"
                for msg in messages[-10:]  # Last 10 messages
            ])
            
            tags = ["conversation", conversation_id]
            metadata = {
                "conversation_id": conversation_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "message_count": len(messages)
            }
            
            result = await self.supermemory.store_knowledge(content, tags, metadata)
            return result.get("success", False)
        
        except Exception as e:
            logger.error(f"Error storing conversation knowledge: {str(e)}")
            return False
    
    async def get_relevant_knowledge(self, query: str) -> List[Dict]:
        """Retrieve relevant knowledge for a query"""
        try:
            return await self.supermemory.retrieve_knowledge(query, limit=5)
        except Exception as e:
            logger.error(f"Error retrieving knowledge: {str(e)}")
            return []
    
    async def send_email_via_agent(self, to: str, subject: str, content: str,
                                 agent_id: str = "cathy") -> Dict:
        """Send email through an agent"""
        try:
            # Generate email content using agent
            messages = [
                {"role": "user", "content": f"Help me send an email to {to} with subject '{subject}' and this content: {content}"}
            ]
            
            response = await self.generate_agent_response(agent_id, messages)
            
            if response.success:
                # Send actual email
                email_result = await self.mailgun.send_email(
                    to=to,
                    subject=subject,
                    text=content,
                    from_email=f"{agent_id}@{MAILGUN_DOMAIN}"
                )
                
                return {
                    "success": email_result.get("success", False),
                    "agent_response": response.content,
                    "email_result": email_result
                }
            else:
                return {
                    "success": False,
                    "error": response.error,
                    "agent_response": None
                }
        
        except Exception as e:
            logger.error(f"Error sending email via agent: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def get_system_status(self) -> Dict:
        """Get overall system status"""
        return {
            "agents": {
                agent_id: {
                    "status": "active",
                    "current_load": agent.current_load,
                    "max_load": agent.max_concurrent_tasks,
                    "capabilities": agent.capabilities,
                    "performance_history_count": len(agent.performance_history)
                }
                for agent_id, agent in self.agents.items()
            },
            "active_sessions": len(self.active_sessions),
            "openrouter_status": "connected" if self.openrouter.api_key else "disconnected",
            "supermemory_status": "connected" if self.supermemory.api_key else "disconnected",
            "mailgun_status": "connected" if self.mailgun.api_key else "disconnected"
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        await self.openrouter.close_session()

# Global orchestrator instance
swarm_orchestrator = SwarmOrchestrator()

# Convenience functions for external use
async def process_message(user_message: str, conversation_context: Dict = None,
                        mentioned_agents: List[str] = None) -> Dict:
    """Process a user message through the swarm system"""
    return await swarm_orchestrator.process_user_message(
        user_message, conversation_context, mentioned_agents
    )

async def get_agent_response(agent_id: str, messages: List[Dict], 
                           context: Dict = None) -> AgentResponse:
    """Get response from a specific agent"""
    return await swarm_orchestrator.generate_agent_response(agent_id, messages, context)

async def send_email(to: str, subject: str, content: str, agent_id: str = "cathy") -> Dict:
    """Send email through an agent"""
    return await swarm_orchestrator.send_email_via_agent(to, subject, content, agent_id)

async def store_knowledge(conversation_id: str, messages: List[Dict]) -> bool:
    """Store conversation knowledge"""
    return await swarm_orchestrator.store_conversation_knowledge(conversation_id, messages)

async def get_knowledge(query: str) -> List[Dict]:
    """Get relevant knowledge"""
    return await swarm_orchestrator.get_relevant_knowledge(query)

async def get_status() -> Dict:
    """Get system status"""
    return await swarm_orchestrator.get_system_status()

# Initialize logging
logger.info("🤖 Swarm Agent Orchestration System initialized")
logger.info(f"✅ OpenRouter API configured")
logger.info(f"✅ SuperMemory API configured") 
logger.info(f"✅ Mailgun API configured")
logger.info(f"🚀 {len(swarm_orchestrator.agents)} agents ready for deployment")

