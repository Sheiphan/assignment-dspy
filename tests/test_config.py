#!/usr/bin/env python3
"""
Test script to demonstrate .env file configuration loading
"""

from assignment.config import config
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

def test_config():
    """Test and display the configuration"""
    
    console.print(Panel("[bold green]Configuration Test[/bold green]"))
    
    # Create table showing current configuration
    table = Table(title="Current Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Source", style="yellow")
    
    # Test various config values
    table.add_row("OpenAI API Key", "***SET***" if config.openai_api_key else "NOT SET", ".env or ENV")
    table.add_row("LLM Model", config.openai_llm_model, ".env or ENV or default")
    table.add_row("Embedding Model", config.openai_embedding_model, ".env or ENV or default") 
    table.add_row("Confidence Threshold", str(config.confidence_threshold), ".env or ENV or default")
    table.add_row("Max Chunk Size", str(config.max_chunk_size), ".env or ENV or default")
    table.add_row("Chunk Overlap", str(config.chunk_overlap), ".env or ENV or default")
    table.add_row("Vector Search Top K", str(config.vector_search_top_k), ".env or ENV or default")
    table.add_row("Log Level", config.log_level, ".env or ENV or default")
    table.add_row("Save Results", str(config.save_results), ".env or ENV or default")
    table.add_row("Output Directory", config.output_directory, ".env or ENV or default")
    
    console.print(table)
    
    # Show instructions
    if not config.openai_api_key:
        console.print(Panel(
            "[yellow]To set up your API key:[/yellow]\n\n"
            "1. cp env_template.txt .env\n"
            "2. Edit .env and replace 'your_openai_api_key_here' with your actual key\n"
            "3. Run this test again to see the change"
        ))
    else:
        console.print(Panel(
            "[green]✓ Configuration loaded successfully![/green]\n\n"
            "The system is ready to run with your settings."
        ))

if __name__ == "__main__":
    test_config() 