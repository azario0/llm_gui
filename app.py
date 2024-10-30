import customtkinter as ctk
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor
import uuid
from dataclasses import dataclass
from typing import Dict, Optional
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from collections import deque
import textwrap
import os

class ChatMemory:
    def __init__(self, max_memories=4):
        self.max_memories = max_memories
        self.memories = deque(maxlen=max_memories)
        self.current_conversation = []
    
    def add_exchange(self, user_input: str, assistant_response: str):
        """Add a new exchange to the current conversation"""
        self.current_conversation.extend([
            f"User: {user_input}",
            f"Assistant: {assistant_response}"
        ])
    
    def add_summary(self, summary: str):
        """Add a new summary to the memories"""
        self.memories.append(summary)
        self.current_conversation = []  # Reset current conversation
    
    def get_context(self) -> str:
        """Get all memories as context"""
        if not self.memories:
            return "No previous context."
        return "\n".join([f"Memory {i+1}: {memory}" for i, memory in enumerate(self.memories)])
    
    def get_current_conversation(self) -> str:
        """Get current conversation as string"""
        if not self.current_conversation:
            return ""
        return "\n".join(self.current_conversation)
    
    def is_full(self) -> bool:
        """Check if memory limit is reached"""
        return len(self.memories) >= self.max_memories
    
    def save_to_file(self, chat_id, filename="chat_memories.json"):
        """Save memories to file"""
        data = {
            "memories": list(self.memories),
            "current_conversation": self.current_conversation
        }
        try:
            with open(filename, 'r') as f:
                all_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            all_data = {}
            
        all_data[chat_id] = data
        
        with open(filename, 'w') as f:
            json.dump(all_data, f)
    
    def load_from_file(self, chat_id, filename="chat_memories.json"):
        """Load memories from file"""
        try:
            with open(filename, 'r') as f:
                all_data = json.load(f)
                if chat_id in all_data:
                    data = all_data[chat_id]
                    self.memories = deque(data["memories"], maxlen=self.max_memories)
                    self.current_conversation = data["current_conversation"]
        except FileNotFoundError:
            pass

@dataclass
class ChatTab:
    display: ctk.CTkTextbox
    memory: ChatMemory
    name: str

class ChatMemoryManager:
    def __init__(self):
        self.tabs: Dict[str, ChatTab] = {}
        self.current_tab_id: Optional[str] = None
        
    def create_new_chat(self, display: ctk.CTkTextbox, tab_name: str) -> str:
        """Create a new chat tab with its own memory"""
        chat_id = str(uuid.uuid4())
        self.tabs[chat_id] = ChatTab(
            display=display,
            memory=ChatMemory(),
            name=tab_name
        )
        self.current_tab_id = chat_id
        self.load_chat_memory(chat_id)
        return chat_id
    
    def switch_chat(self, chat_id: str):
        """Switch to a different chat tab"""
        if chat_id in self.tabs:
            self.current_tab_id = chat_id
            
    def get_current_chat(self) -> Optional[ChatTab]:
        """Get the current active chat tab"""
        if self.current_tab_id:
            return self.tabs.get(self.current_tab_id)
        return None
        
    def save_chat_memory(self, chat_id: str):
        """Save the chat memory for a specific tab"""
        if chat_id in self.tabs:
            self.tabs[chat_id].memory.save_to_file(chat_id)
            
    def load_chat_memory(self, chat_id: str):
        """Load the chat memory for a specific tab"""
        if chat_id in self.tabs:
            self.tabs[chat_id].memory.load_from_file(chat_id)
            
    def get_memory_status(self, chat_id: Optional[str] = None) -> tuple[int, int]:
        """Get current memory usage status for a specific tab"""
        if chat_id is None:
            chat_id = self.current_tab_id
        if chat_id in self.tabs:
            memories = self.tabs[chat_id].memory.memories
            return len(memories), self.tabs[chat_id].memory.max_memories
        return 0, 0
    
class ChatbotApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Chatbot with Memory")
        self.root.geometry("800x600")
        
        # Configure the grid layout
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)
        
        # Configure color theme
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # Create main container
        self.main_container = ctk.CTkFrame(self.root)
        self.main_container.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        self.main_container.grid_columnconfigure(0, weight=1)
        self.main_container.grid_rowconfigure(1, weight=1)
        
        # Create header with memory status and new chat button
        self.header_frame = ctk.CTkFrame(self.main_container)
        self.header_frame.grid(row=0, column=0, sticky="ew", pady=(0, 20))
        self.header_frame.grid_columnconfigure(1, weight=1)
        
        self.header = ctk.CTkLabel(
            self.header_frame,
            text="AI Assistant",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        self.header.grid(row=0, column=0, padx=10)
        
        self.memory_status = ctk.CTkLabel(
            self.header_frame,
            text="Memories: 0/4",
            font=ctk.CTkFont(size=12)
        )
        self.memory_status.grid(row=0, column=1, padx=10)
        
        self.new_chat_button = ctk.CTkButton(
            self.header_frame,
            text="New Chat",
            width=100,
            command=self.new_chat
        )
        self.new_chat_button.grid(row=0, column=2, padx=10)
        
        # Initialize tab system
        self.tabs = {}
        self.current_tab_id = None
        
        
        # Create input container
        self.input_container = ctk.CTkFrame(self.main_container)
        self.input_container.grid(row=2, column=0, sticky="ew", padx=10)
        self.input_container.grid_columnconfigure(0, weight=1)
        
        # Create input field
        self.input_field = ctk.CTkEntry(
            self.input_container,
            placeholder_text="Type your message here...",
            font=ctk.CTkFont(size=14),
            height=40
        )
        self.input_field.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        
        # Create send button
        self.send_button = ctk.CTkButton(
            self.input_container,
            text="Send",
            width=100,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold"),
            command=self.send_message
        )
        self.send_button.grid(row=0, column=1)
        
        # Bind Enter key to send message
        self.input_field.bind("<Return>", lambda event: self.send_message())
        
        # Initialize Gemini LLM
        self.initialize_llm()
        
        # Initialize prompt templates
        self.chat_template = PromptTemplate(
            input_variables=["context", "input"],
            template=textwrap.dedent("""
                Previous conversation context:
                {context}
                
                You are a helpful and friendly AI assistant.
                Respond to the following input in a clear and engaging way:
                {input}
            """).strip()
        )
        
        self.summary_template = PromptTemplate(
            input_variables=["conversation"],
            template=textwrap.dedent("""
                Please provide a brief, one-sentence summary of the following conversation exchange:
                {conversation}
                
                Summary:
            """).strip()
        )
        
        # Create chains using RunnableSequence
        self.chat_chain = self.chat_template | self.llm
        self.summary_chain = self.summary_template | self.llm
        self.tab_to_id = {}  # Add this to store mapping between tab names and chat IDs
        self.memory_manager = ChatMemoryManager()
        self.init_tabs()
        
    def initialize_llm(self):
        """Initialize the Gemini LLM with LangChain"""
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            self.add_message("System", "Error: GOOGLE_API_KEY not found in environment variables.")
            return
            
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=api_key,
            temperature=0.7,
            top_p=0.9,
        )
    
    def init_tabs(self):
        """Initialize the tab view"""
        self.tab_view = ctk.CTkTabview(self.main_container)
        self.tab_view.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 20))
        
        # Add tab change event
        self.tab_view.configure(command=self.on_tab_change)
        
        self.new_chat()

    def on_tab_change(self):
        """Handle tab change events"""
        current_tab = self.tab_view.get()
        if current_tab in self.tab_to_id:
            chat_id = self.tab_to_id[current_tab]
            self.memory_manager.switch_chat(chat_id)
            self.update_memory_status()
        
    def new_chat(self):
        """Create a new chat tab"""
        tab_name = f"Chat {len(self.memory_manager.tabs) + 1}"
        
        # Create new tab
        self.tab_view.add(tab_name)
        tab = self.tab_view.tab(tab_name)
        
        # Create chat display for new tab
        chat_display = ctk.CTkTextbox(
            tab,
            wrap="word",
            font=ctk.CTkFont(size=14),
            height=400
        )
        chat_display.pack(fill="both", expand=True)
        
        # Create new chat in memory manager
        chat_id = self.memory_manager.create_new_chat(chat_display, tab_name)
        self.tab_to_id[tab_name] = chat_id
        
        # Switch to new tab
        self.tab_view.set(tab_name)
        
        # Add welcome message
        self.add_message("Assistant", "Hello! How can I help you today?")
        self.update_memory_status()
        
    def show_memory_full_dialog(self):
        """Show dialog when memory is full"""
        dialog = ctk.CTkInputDialog(
            text="Memory limit reached! Please start a new chat.",
            title="Memory Full"
        )
        self.new_chat()
    
    def get_current_chat_display(self):
        """Get the chat display of the current tab"""
        if current_chat := self.memory_manager.get_current_chat():
            return current_chat.display
        return None
    
    def get_current_memory(self):
        """Get the memory of the current tab"""
        if current_chat := self.memory_manager.get_current_chat():
            return current_chat.memory
        return None 
    
    def update_memory_status(self):
        """Update the memory status display"""
        if current_chat := self.memory_manager.get_current_chat():
            used, max_memories = self.memory_manager.get_memory_status()
            self.memory_status.configure(
                text=f"Memories: {used}/{max_memories}"
            )
    
    def add_message(self, sender, message):
        """Add a message to the chat display"""
        display = self.get_current_chat_display()
        display.configure(state="normal")
        
        # Add sender with bold font
        sender_text = f"\n{sender}: " if display.get("1.0", "end-1c") else f"{sender}: "
        display.insert("end", sender_text)
        
        # Insert the actual message
        display.insert("end", f"{message}\n")
        
        display.configure(state="disabled")
        display.see("end")
    
    async def summarize_conversation(self):
        """Summarize the current conversation"""
        memory = self.get_current_memory()
        conversation = memory.get_current_conversation()
        
        if conversation:
            try:
                # Run synchronous chain.invoke in a thread pool
                with ThreadPoolExecutor() as executor:
                    summary_response = await asyncio.get_event_loop().run_in_executor(
                        executor,
                        lambda: self.summary_chain.invoke({"conversation": conversation})
                    )
                
                memory.add_summary(summary_response.content)
                self.update_memory_status()
                memory.save_to_file(self.current_tab_id)
                
                if memory.is_full():
                    self.root.after(0, self.show_memory_full_dialog)
                    
            except Exception as e:
                print(f"Error summarizing conversation: {e}")
    
    def send_message(self):
        """Send a message and get response from the AI"""
        memory = self.get_current_memory()
        if not memory:
            return
            
        if memory.is_full():
            self.show_memory_full_dialog()
            return
            
        user_input = self.input_field.get().strip()
        if not user_input:
            return
        
        # Clear input field
        self.input_field.delete(0, "end")
        
        # Add user message to chat
        self.add_message("You", user_input)
        
        # Disable input while processing
        self.input_field.configure(state="disabled")
        self.send_button.configure(state="disabled")
        
        # Create thread for AI response
        thread = threading.Thread(target=self.get_ai_response, args=(user_input,))
        thread.start()
    
    def get_ai_response(self, user_input):
        """Get response from AI in a separate thread"""
        try:
            # Get context from current chat memory
            memory = self.get_current_memory()
            if not memory:
                raise Exception("No active chat found")
                
            context = memory.get_context()
            
            # Get response using context
            response = self.chat_chain.invoke({
                "context": context,
                "input": user_input
            })
            
            # Add exchange to memory
            memory.add_exchange(user_input, response.content)
            
            # Save chat memory after update
            self.memory_manager.save_chat_memory(self.memory_manager.current_tab_id)
            
            # Schedule adding the response to the main thread
            self.root.after(0, self.add_message, "Assistant", response.content)
            
            # Run summarization
            asyncio.run(self.summarize_conversation())
            
        except Exception as e:
            self.root.after(0, self.add_message, "System", f"Error: {str(e)}")
        finally:
            self.root.after(0, lambda: self.input_field.configure(state="normal"))
            self.root.after(0, lambda: self.send_button.configure(state="normal"))
            self.root.after(0, lambda: self.input_field.focus())
def main():
    root = ctk.CTk()
    app = ChatbotApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()