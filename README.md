# Vizzy Chat

Vizzy Chat provides a single conversational interface where users can create, transform, iterate, and deploy visual, narrative, and experiential content. It is powered by OpenAI's GPT-4o and DALL-E 3 models, orchestrated through a LangChain + LangGraph state machine pipeline, and presented via a polished Streamlit chat UI.

---

## Table of Contents

1. [Features](#features)
2. [Tech Stack](#tech-stack)
3. [Architecture](#architecture)
4. [Project Structure](#project-structure)
5. [How It Works](#how-it-works)
6. [Setup and Installation](#setup-and-installation)
7. [Configuration](#configuration)
8. [Usage Guide](#usage-guide)
9. [Design Decisions](#design-decisions)
10. [Prompt Engineering Strategy](#prompt-engineering-strategy)
11. [Future Roadmap](#future-roadmap)


---

## Features

### Core Capabilities

| Capability                       | Description                                                                                                            |
| -------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| **Intent Detection**             | LLM-powered classification of user messages into 8 distinct creative intents using GPT-4o-mini in JSON mode            |
| **Dual Mode System**             | Home mode (personal creativity, emotional resonance) and Business mode (marketing, brand strategy, premium aesthetics) |
| **Image Generation**             | DALL-E 3 image generation with LLM-refined prompts for high-quality, non-generic outputs                               |
| **Multi-Variation Output**       | Generate up to 3 visually distinct variations per request, each using a different aesthetic angle                      |
| **Natural Language Iteration**   | Refine previous outputs with plain English feedback like "make it warmer", "more minimal", or "try a different style"  |
| **Taste Memory**                 | JSON-backed persistent preference engine that learns from interactions and subtly influences future outputs            |
| **Multi-Step Creative Pipeline** | Story generation followed by automatic scene extraction followed by per-scene image generation                         |
| **Marketing Intelligence**       | Brand-aware copy generation with premium positioning, strategic messaging, and anti-cliche enforcement                 |
| **One-Click Download**           | Download any generated image directly from the chat interface                                                          |
| **Real-Time Status Feedback**    | Human-centred progress messages during generation (no technical jargon or loading spinners)                            |

### Supported Creative Intents

The intent engine classifies every user message into one of these 8 categories, which determines the execution pipeline:

| Intent                     | Example Prompt                                            | Pipeline                                                |
| -------------------------- | --------------------------------------------------------- | ------------------------------------------------------- |
| `visual_creation`          | "Paint something that feels like how my last year felt"   | Image generation                                        |
| `image_transformation`     | "Make this more dramatic and cinematic"                   | Image generation with transformation context            |
| `story_generation`         | "Write a bedtime story about a brave little fox"          | Text generation (narrative)                             |
| `marketing_asset`          | "Design a quote poster for my living room"                | Image generation with brand-aware prompting             |
| `emotional_interpretation` | "Create something that captures the feeling of nostalgia" | Image generation with emotional depth                   |
| `multi_step_creative`      | "Generate a story for my kids, then visualise each scene" | Text then scene extraction then per-scene images        |
| `iteration_refinement`     | "Make it warmer and more inviting"                        | Delta extraction then prompt revision then regeneration |
| `general_conversation`     | "What can you help me create?"                            | Conversational reply with creative guidance             |

---

## Tech Stack

| Component                | Technology            | Purpose                                                                |
| ------------------------ | --------------------- | ---------------------------------------------------------------------- |
| **Frontend**             | Streamlit 1.31+       | Chat UI, sidebar controls, image gallery, download buttons             |
| **LLM Orchestration**    | LangChain 0.1.9+      | Structured LLM calls, message formatting, model abstraction            |
| **Pipeline Engine**      | LangGraph 0.0.26+     | State machine for routing intents to execution pipelines               |
| **Chat Model**           | OpenAI GPT-4o         | Primary generation model for narratives, marketing copy, conversations |
| **Classification Model** | OpenAI GPT-4o-mini    | Fast, cheap intent classification with JSON mode                       |
| **Image Model**          | OpenAI DALL-E 3       | Image generation from LLM-refined prompts                              |
| **Data Validation**      | Pydantic 2.6+         | Structured output models (IntentResult, config validation)             |
| **Environment**          | python-dotenv         | Secure API key and configuration management                            |
| **Image Processing**     | Pillow 10.2+          | Image format conversion, download utilities                            |
| **Logging**              | Python stdlib logging | Rotating file logs + console output, zero external dependencies        |

---

## Architecture

### High-Level Data Flow

```
User Message (text input or suggestion button)
        |
        v
+-------------------+
|   Intent Engine    |  <-- GPT-4o-mini, JSON mode, LangChain ChatOpenAI
|   (classify)       |      Outputs: intent label + confidence + reasoning
+--------+----------+
         |
         v
+-------------------+
| Pathway Selector   |  <-- LangGraph StateGraph with conditional edges
| (route)            |      Maps intent -> pipeline pathway
+--------+----------+
         |
    +----+-----+----------+-----------+----------+
    |          |          |           |          |
    v          v          v           v          v
  Image      Text    Multi-Step   Iteration  Conversation
  Pipeline   Pipeline  Pipeline    Engine     Pipeline
    |          |          |           |          |
    +----+-----+----------+-----------+----------+
         |
         v
+-------------------+
|   Memory Engine    |  <-- Saves interaction history
|   (persist)        |      Learns preferences over time
+--------+----------+
         |
         v
  Streamlit Chat UI
  (text + images + download buttons + action buttons)
```

### Pipeline Details

**Image Pipeline** (`run_image_pipeline`):

1. Receives user request, mode, and preferences
2. Calls `build_image_prompt()` to create a structured creative brief
3. Sends the brief to GPT-4o-mini via `_refine_dalle_prompt()` to produce a concise, high-quality DALL-E prompt (under 950 characters)
4. Sends the refined prompt to DALL-E 3 for image generation
5. Returns image URLs and the prompt used (for iteration support)

**Text Pipeline** (`run_text_pipeline`):

1. Builds a system prompt with mode context and memory injection
2. Generates narrative content or marketing copy via GPT-4o
3. Returns formatted text output

**Multi-Step Pipeline** (`run_multi_step_pipeline`):

1. Generates a full narrative via GPT-4o
2. Extracts numbered scenes from the narrative using JSON mode
3. Generates one image per scene (up to 3 scenes) via the image pipeline
4. Returns the story text plus scene-labelled images

**Iteration Engine** (`handle_iteration`):

1. Extracts the refinement delta from the user's feedback using GPT-4o
2. Locates the previous DALL-E prompt from conversation history
3. Revises the prompt incorporating the delta
4. Regenerates the image with the revised prompt

---

## Project Structure

```
vizzy_chat/
|
|-- app.py                      # Main Streamlit application (chat UI, sidebar, processing loop)
|-- config.py                   # Central configuration (env vars, constants, status messages)
|-- requirements.txt            # Python dependencies
|-- .env                        # API keys and settings (not committed to git)
|
|-- core/                       # Brain: orchestration and intelligence
|   |-- __init__.py
|   |-- intent_engine.py        # LangChain-based LLM intent classification (8 intents)
|   |-- pathway_selector.py     # LangGraph StateGraph - routes intent to pipeline
|   |-- generation_engine.py    # Executes image, text, multi-step, conversation pipelines
|   |-- iteration_engine.py     # Handles refinement: delta extraction + prompt revision
|   |-- memory_engine.py        # Persistent taste memory (JSON file backend)
|
|-- services/                   # API layer: thin wrappers over external services
|   |-- __init__.py
|   |-- openai_service.py       # OpenAI SDK wrapper (chat completion + image generation)
|   |-- image_service.py        # Image variation generation, prompt refinement, download
|   |-- text_service.py         # Narrative, marketing copy, conversational reply, scene extraction
|
|-- storage/                    # Persistent data
|   |-- __init__.py
|   |-- memory.json             # User preferences and interaction history
|
|-- utils/                      # Shared utilities
|   |-- __init__.py
|   |-- prompt_builder.py       # All prompt templates (system, image, story, iteration, classification)
|   |-- logger.py               # Stdlib logging with RotatingFileHandler
|
|-- logs/                       # Application log files (auto-created)
|   |-- vizzy_chat.log          # Rotating log (500KB max, 3 backups)
```

---

## How It Works

### 1. User Sends a Message

The user types a message in the chat input or clicks a suggestion button. The message is stored in Streamlit session state and displayed in the chat.

### 2. Intent Classification

The message is sent to `classify_intent()` in `intent_engine.py`. This function:

- Creates a LangChain `ChatOpenAI` instance configured for GPT-4o-mini with JSON response format
- Builds a classification prompt that includes the user's message, current mode (Home/Business), and recent conversation history (last 4 messages for context)
- Parses the JSON response into an `IntentResult` object containing: intent label, confidence score (0-1), and reasoning

### 3. Pipeline Routing

The `PipelineState` (a TypedDict) is constructed with all context: user message, intent, mode, preferences, and conversation history. This state is passed to the compiled LangGraph `StateGraph`.

The graph has the following structure:

- **Entry node**: `route_intent` maps the intent label to a pathway string (e.g., "image", "text", "multi_step", "iteration", "conversation")
- **Conditional edge**: `should_generate_images` reads the pathway and routes to the appropriate generation node
- **Terminal nodes**: Each generation node (generate_images, generate_text, multi_step, iterate, converse) executes its pipeline and returns the updated state
- **End**: All terminal nodes connect to END

### 4. Content Generation

The selected pipeline node calls the appropriate functions in `generation_engine.py`, which delegates to the services layer (`image_service.py`, `text_service.py`, `openai_service.py`).

### 5. Response Display

The Streamlit app renders the pipeline output:

- Text content is displayed as markdown
- Images are displayed in a responsive column grid with variation labels
- Download buttons are generated for each image
- Quick-action buttons (Regenerate, Make warmer, More minimal, More premium, 3 Variations) allow one-click follow-up

### 6. Memory Persistence

After each interaction, the user's message, intent, and mode are saved to `storage/memory.json`. Over time, the memory engine builds a taste profile that subtly influences prompt generation.

---

## Setup and Installation

### Prerequisites

- **Python 3.10 or higher** (tested with Python 3.14)
- An **OpenAI API key** with access to GPT-4o, GPT-4o-mini, and DALL-E 3
- **pip** package manager

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd "chat window"
```

### Step 2: Create and Activate Virtual Environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r vizzy_chat/requirements.txt
```

This installs: Streamlit, OpenAI SDK, LangChain, LangGraph, Pydantic, Pillow, python-dotenv, and requests.

### Step 4: Configure Your API Key

Create or edit `vizzy_chat/.env`:

```env
OPENAI_API_KEY=sk-your-actual-api-key-here
```

### Step 5: Run the Application

```bash
cd vizzy_chat
streamlit run app.py
```

The app will open automatically at **http://localhost:8501**.

If Streamlit does not open automatically, you can also run:

```bash
python -m streamlit run app.py --server.port 8501
```

---

## Configuration

All configuration is managed through environment variables in `vizzy_chat/.env`. The values are loaded by `config.py` at startup.

| Variable                   | Default                  | Description                                    |
| -------------------------- | ------------------------ | ---------------------------------------------- |
| `OPENAI_API_KEY`           | _(required)_             | Your OpenAI API key                            |
| `VIZZY_CHAT_MODEL`         | `gpt-4o`                 | Primary LLM model for text generation          |
| `VIZZY_IMAGE_MODEL`        | `dall-e-3`               | Image generation model                         |
| `VIZZY_EMBEDDING_MODEL`    | `text-embedding-3-small` | Embedding model (reserved for future use)      |
| `VIZZY_DEFAULT_VARIATIONS` | `1`                      | Number of image variations per request (1-3)   |
| `VIZZY_LOG_LEVEL`          | `INFO`                   | Logging level (DEBUG, INFO, WARNING, ERROR)    |
| `VIZZY_MEMORY_BACKEND`     | `json`                   | Memory storage backend (currently "json" only) |

---

## Usage Guide

### Home Mode

Home mode is designed for personal creative projects. The system emphasises:

- Emotional resonance and personal meaning
- Warm, evocative aesthetics
- Storytelling and narrative depth

**Example prompts:**

- "Paint something that feels like how my last year felt"
- "Create a dreamlike version of a childhood memory"
- "Generate a story for my kids, then visualise each scene"
- "Design a quote poster for my living room"

### Business Mode

Business mode is designed for professional and brand-oriented work. The system emphasises:

- Premium, brand-aligned aesthetics
- Strategic messaging and positioning
- Marketing-ready output quality

**Example prompts:**

- "Create premium-looking visuals for our new product"
- "Design a seasonal ambiance visual for evenings"
- "Create a sale poster that doesn't feel cheap"
- "Generate brand-themed artwork using warm, earthy tones"

### Iterating on Results

After receiving a response, you can refine it naturally:

- Click **Regenerate** to get a fresh version
- Click **Make warmer** to shift the colour palette toward warm tones
- Click **More minimal** to simplify the composition
- Click **More premium** to elevate perceived quality
- Click **3 Variations** to get three distinct visual directions
- Or type any custom feedback: "less dramatic", "add more blue", "make the typography bolder"

---

## Design Decisions

### Why LangChain + LangGraph?

**LangChain** provides a clean abstraction over OpenAI's API with structured output support, retry logic, and model-agnostic interfaces. This means the underlying model can be swapped (e.g., from GPT-4o to Claude or Llama) without changing any pipeline code.

**LangGraph** enables the creative pipeline as a **state machine** with typed state, conditional edges, and explicit routing. This makes the orchestration:

- **Visible**: The entire flow is defined declaratively in `pathway_selector.py`
- **Extensible**: Adding a new pipeline path requires only a new node, a new intent label, and a routing entry
- **Testable**: Each node function can be unit-tested independently with a mock state dict

### Why GPT-4o-mini for Classification?

Intent classification does not require the full reasoning power of GPT-4o. Using GPT-4o-mini for this step reduces latency by approximately 50% and cost by approximately 90%, while maintaining high classification accuracy. The heavier GPT-4o model is reserved for content generation where quality matters most.

### Why a Two-Step Image Prompt Process?

Rather than sending the user's raw text directly to DALL-E 3, the system first builds a structured creative brief (with mode context, variation guidance, and preferences) and then uses GPT-4o-mini to refine it into a concise, vivid DALL-E prompt under 950 characters. This two-step approach:

- Produces consistently higher-quality images
- Avoids DALL-E's tendency to be overly literal with user text
- Allows injection of mode and preference context without polluting the visual prompt

### Why JSON for Memory?

The MVP uses JSON file storage (`storage/memory.json`) for zero-dependency persistence. The `memory_engine.py` module exposes a clean interface (`get_preferences()`, `update_preferences()`, `add_history_entry()`) so the backend can be swapped to SQLite, Redis, or PostgreSQL without changing any consumer code.

### Why Stdlib Logging Instead of Loguru?

The initial implementation used Loguru, but it caused import issues across different Python environments. The stdlib `logging` module with `RotatingFileHandler` provides the same functionality (rotating file logs, console output, configurable levels) with zero external dependencies and guaranteed compatibility.

---

## Prompt Engineering Strategy

The prompt system in `utils/prompt_builder.py` follows four key principles:

### 1. Separation of Concerns

System prompts (role, mode context, personality) are built separately from user prompts (request, variation guidance, memory injection). This allows each component to be tuned independently.

### 2. Anti-Generic Enforcement

All prompts explicitly instruct against:

- Cliche AI-generated textures (smooth gradients, over-saturation)
- Stock-photo compositions
- Generic motivational poster aesthetics
- Exclamation marks and salesy language (in marketing mode)

### 3. Mode-Aware Context

- **Home mode** prompts emphasise emotion, personal meaning, warmth, and intimacy
- **Business mode** prompts emphasise strategy, restraint, premium perception, and brand intelligence (think Apple keynote, not clearance flyer)

### 4. Taste Memory Injection

If the user has accumulated preferences (e.g., preference for warm tones, minimalist compositions), these are subtly woven into prompts as style guidance rather than explicit instructions. The user never sees "based on your preference for warm tones" - the output simply feels more aligned over time.

---

## Future Roadmap

| Phase | Feature                        | Description                                                                         | Effort |
| ----- | ------------------------------ | ----------------------------------------------------------------------------------- | ------ |
| v1.1  | Image Upload + Transformation  | Upload images and apply creative transformations via the Vision API                 | Medium |
| v1.2  | Database Memory Backend        | Replace JSON with SQLite or PostgreSQL for production-grade persistence             | Low    |
| v1.3  | User Authentication            | Multi-user support with individual taste profiles and session management            | Medium |
| v1.4  | Gallery View                   | Browse, search, and re-use past creations across sessions                           | Low    |
| v2.0  | Brand Profile System           | Upload logos, colour palettes, and brand guidelines for consistent outputs          | High   |
| v2.1  | Batch Generation               | Generate multiple assets in one request (e.g., "10 seasonal visuals for Instagram") | Medium |
| v2.2  | Video and Animation            | Integration with Runway or Pika for motion content generation                       | High   |
| v2.3  | Team Collaboration             | Shared workspaces with roles, comments, and approval workflows                      | High   |
| v3.0  | Embedding-Based Style Matching | Use vector embeddings to find and suggest similar visual styles                     | Medium |
| v3.1  | Plugin System                  | Allow custom creative pipelines to be registered and executed                       | High   |
| v3.2  | Headless API                   | REST/GraphQL API layer for programmatic access without the Streamlit UI             | Medium |

---
