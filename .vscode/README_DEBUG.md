# VSCode Debugging Guide for Data Transformation System

## Quick Start

1. **Open VSCode** in this project directory
2. **Install Python Extension** (if not already installed)
3. **Set breakpoints** by clicking in the gutter next to line numbers
4. **Press F5** or go to Run & Debug panel and select a configuration

## Available Debug Configurations

### 1. Debug main.py
- **Purpose**: Debug the main script with mock responses (no API key required)
- **Use when**: Testing basic functionality, debugging logic without API calls
- **Environment**: Uses mock data for OpenAI API calls

### 2. Debug main.py (with API)  
- **Purpose**: Debug with real OpenAI API calls
- **Use when**: Testing full functionality with actual API responses
- **Requirements**: Set `OPENAI_API_KEY` environment variable
- **Environment**: Uses real OpenAI API

### 3. Debug demo.py
- **Purpose**: Debug the demonstration script
- **Use when**: Testing all three demo scenarios

## Debugging Tips

### Setting Breakpoints
- Click in the gutter (left of line numbers) to set breakpoints
- Red dots indicate active breakpoints
- Breakpoints will pause execution at that line

### Key Places to Set Breakpoints

#### In main.py:
```python
# Line ~280: Schema processing starts
schema_paths = self.schema_processor.flatten_schema(schema)

# Line ~300: Skeleton extraction begins  
skeleton = await self._extract_skeleton(schema_paths)

# Line ~320: Field population starts
populated_data, confidence_scores, flagged_fields = await self._populate_fields(skeleton, schema_paths)

# Line ~400: Vector search
relevant_chunks = await self.vector_index.search(search_query, top_k=3)

# Line ~500: Mock value creation (for debugging without API)
mock_value = self._create_mock_skeleton_value(path.path, chunks_text)
```

### Debugging Async Code
- The debugger handles async/await automatically
- Step through async calls with F10 (Step Over) or F11 (Step Into)
- Watch variables in the Variables panel on the left

### Inspecting Variables
- **Variables Panel**: Shows local and global variables
- **Watch Panel**: Add expressions to monitor (e.g., `len(chunks)`, `result.confidence_scores`)
- **Call Stack**: Shows the function call hierarchy
- **Debug Console**: Execute Python expressions in the current context

### Common Debug Scenarios

#### 1. Schema Processing Issues
```python
# Set breakpoint at line ~286
self._extract_paths(schema, "", paths, True)
# Inspect: schema, paths variables
```

#### 2. Chunking Problems  
```python
# Set breakpoint in SemanticChunker.chunk_document()
# Inspect: text, paragraphs, chunks variables
```

#### 3. Vector Search Issues
```python
# Set breakpoint in VectorIndex.search()
# Inspect: query_embedding, scores, indices
```

#### 4. Extraction Pipeline Debugging
```python
# Set breakpoint in ExtractionPipeline.extract()
# Step through each phase and inspect intermediate results
```

## Debug Console Commands

While debugging, you can use the Debug Console to:

```python
# Check variable values
print(f"Number of chunks: {len(chunks)}")

# Examine schema paths
[p.path for p in schema_paths[:5]]

# Check confidence scores
{k: v for k, v in confidence_scores.items() if v < 0.5}

# Inspect document chunks
chunks[0].content[:100]
```

## Troubleshooting

### Python Interpreter Not Found
1. Open Command Palette (Ctrl+Shift+P / Cmd+Shift+P)
2. Type "Python: Select Interpreter"
3. Choose `./.venv/bin/python`

### Breakpoints Not Hitting
1. Ensure you're using the correct debug configuration
2. Check that the file you're editing matches the one being executed
3. Verify the virtual environment is activated

### Import Errors
1. Check that all dependencies are installed: `pip install -r requirements.txt`
2. Verify `PYTHONPATH` is set correctly in the launch configuration
3. Ensure you're in the correct working directory

### API Key Issues
1. For real API testing, set environment variable:
   ```bash
   export OPENAI_API_KEY="your-key-here"
   ```
2. Or create a `.env` file with:
   ```
   OPENAI_API_KEY=your-key-here
   ```

## Performance Debugging

### Memory Usage
- Use the Debug Console to check object sizes:
  ```python
  import sys
  sys.getsizeof(chunks)
  ```

### Execution Time
- Add timing breakpoints:
  ```python
  import time
  start = time.time()
  # ... code to debug ...
  print(f"Execution time: {time.time() - start:.2f}s")
  ```

### Vector Index Performance
- Check index statistics:
  ```python
  print(f"Index size: {self.vector_index.index.ntotal}")
  print(f"Embedding dimension: {self.vector_index.embedding_dim}")
  ```

## Advanced Debugging

### Conditional Breakpoints
- Right-click on a breakpoint
- Add condition (e.g., `confidence < 0.5`)
- Breakpoint only triggers when condition is true

### Logpoints
- Right-click in gutter
- Choose "Add Logpoint"
- Enter expression to log (e.g., `"Processing chunk {chunk.chunk_id}"`)

### Exception Breakpoints
- In Run & Debug panel, check "Uncaught Exceptions"
- Debugger will pause on any unhandled exception 