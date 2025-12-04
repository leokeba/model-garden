<script lang="ts">
  import { api } from "$lib/api/client";
  import Badge from "$lib/components/Badge.svelte";
  import Button from "$lib/components/Button.svelte";
  import Card from "$lib/components/Card.svelte";
  import GpuMemoryProfile from "$lib/components/GpuMemoryProfile.svelte";
  import ModelLoader from "$lib/components/ModelLoader.svelte";
  import { onMount } from "svelte";

  // Inference status
  let inferenceStatus = $state<any>(null);
  let loadingStatus = $state(true);
  let statusError = $state("");

  // Chat mode
  type Message = {
    role: "user" | "assistant" | "system";
    content: string;
    timestamp: Date;
    image?: string; // Base64 or URL
    imagePreview?: string; // For display
  };

  let messages: Message[] = $state([]);
  let currentInput = $state("");
  let isGenerating = $state(false);
  let streamingContent = $state("");

  // Image input
  let currentImage = $state<string | null>(null);
  let currentImagePreview = $state<string | null>(null);
  let imageInputRef: HTMLInputElement | null = $state(null);
  let isDragging = $state(false);

  // Settings
  let mode = $state<"chat" | "completion">("chat");
  let settings = $state({
    temperature: 0.7,
    max_tokens: 500,
    top_p: 0.9,
    top_k: 50,
    stream: true,
  });

  // System prompt for chat mode
  let systemPrompt = $state("You are a helpful AI assistant.");
  let showSettings = $state(false);

  async function loadInferenceStatus() {
    try {
      loadingStatus = true;
      statusError = "";
      inferenceStatus = await api.getInferenceStatus();
    } catch (error) {
      console.error("Failed to load inference status:", error);
      statusError =
        error instanceof Error ? error.message : "Failed to load status";
    } finally {
      loadingStatus = false;
    }
  }

  async function sendMessage() {
    if (
      (!currentInput.trim() && !currentImage) ||
      !inferenceStatus?.loaded ||
      isGenerating
    )
      return;

    const userMessage: Message = {
      role: "user",
      content: currentInput.trim(),
      timestamp: new Date(),
      image: currentImage || undefined,
      imagePreview: currentImagePreview || undefined,
    };

    messages = [...messages, userMessage];
    const inputToSend = currentInput;
    const imageToSend = currentImage;
    currentInput = "";
    currentImage = null;
    currentImagePreview = null;
    isGenerating = true;
    streamingContent = "";

    try {
      if (mode === "chat") {
        await streamChatCompletion(inputToSend, imageToSend);
      } else {
        await streamCompletion(inputToSend);
      }
    } catch (error) {
      console.error("Generation error:", error);
      messages = [
        ...messages,
        {
          role: "assistant",
          content: "Error: Failed to generate response. Please try again.",
          timestamp: new Date(),
        },
      ];
    } finally {
      isGenerating = false;
    }
  }

  async function streamChatCompletion(
    userInput: string,
    image?: string | null,
  ) {
    const assistantMessage: Message = {
      role: "assistant",
      content: "",
      timestamp: new Date(),
    };
    messages = [...messages, assistantMessage];

    try {
      // Prepare chat messages with system prompt
      const chatMessages: any[] = [
        { role: "system", content: systemPrompt },
        ...messages.slice(0, -1).map((m) => {
          // Include image in message if present (OpenAI multimodal format)
          if (m.image) {
            return {
              role: m.role,
              content: [
                {
                  type: "image_url",
                  image_url: {
                    url: m.image.startsWith("data:")
                      ? m.image
                      : `data:image/jpeg;base64,${m.image}`,
                  },
                },
                { type: "text", text: m.content || "What is in this image?" },
              ],
            };
          }
          return { role: m.role, content: m.content };
        }),
      ];

      // Add current user message with optional image
      if (image) {
        chatMessages.push({
          role: "user",
          content: [
            {
              type: "image_url",
              image_url: {
                url: image.startsWith("data:")
                  ? image
                  : `data:image/jpeg;base64,${image}`,
              },
            },
            { type: "text", text: userInput || "What is in this image?" },
          ],
        });
      } else {
        chatMessages.push({ role: "user", content: userInput });
      }

      const response = await fetch(`/api/v1/chat/completions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages: chatMessages,
          temperature: settings.temperature,
          max_tokens: settings.max_tokens,
          top_p: settings.top_p,
          stream: settings.stream,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      if (settings.stream && response.body) {
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() || "";

          for (const line of lines) {
            if (line.startsWith("data: ")) {
              const data = line.slice(6);
              if (data === "[DONE]") continue;

              try {
                const parsed = JSON.parse(data);
                const content = parsed.choices[0]?.delta?.content || "";
                if (content) {
                  streamingContent += content;
                  messages[messages.length - 1].content = streamingContent;
                }
              } catch (e) {
                console.error("Parse error:", e);
              }
            }
          }
        }
      } else {
        const data = await response.json();
        messages[messages.length - 1].content = data.choices[0].message.content;
      }
    } catch (error) {
      messages[messages.length - 1].content = "Error generating response";
      throw error;
    } finally {
      streamingContent = "";
    }
  }

  async function streamCompletion(prompt: string) {
    const assistantMessage: Message = {
      role: "assistant",
      content: "",
      timestamp: new Date(),
    };
    messages = [...messages, assistantMessage];

    try {
      const response = await fetch(`/api/v1/inference/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt: prompt,
          temperature: settings.temperature,
          max_tokens: settings.max_tokens,
          top_p: settings.top_p,
          top_k: settings.top_k,
          stream: settings.stream,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      if (settings.stream && response.body) {
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() || "";

          for (const line of lines) {
            if (line.startsWith("data: ")) {
              const data = line.slice(6);
              if (data === "[DONE]") continue;

              try {
                const parsed = JSON.parse(data);
                const content = parsed.text || "";
                if (content) {
                  streamingContent += content;
                  messages[messages.length - 1].content = streamingContent;
                }
              } catch (e) {
                console.error("Parse error:", e);
              }
            }
          }
        }
      } else {
        const data = await response.json();
        messages[messages.length - 1].content = data.text || data.content || "";
      }
    } catch (error) {
      messages[messages.length - 1].content = "Error generating response";
      throw error;
    } finally {
      streamingContent = "";
    }
  }

  function clearConversation() {
    if (confirm("Clear all messages?")) {
      messages = [];
      streamingContent = "";
    }
  }

  function formatTime(date: Date) {
    return date.toLocaleTimeString("en-US", {
      hour: "2-digit",
      minute: "2-digit",
    });
  }

  function handleKeyDown(event: KeyboardEvent) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    }
  }

  // Image handling functions
  function handleImageSelect(event: Event) {
    const input = event.target as HTMLInputElement;
    if (input.files && input.files[0]) {
      processImageFile(input.files[0]);
    }
  }

  function processImageFile(file: File) {
    if (!file.type.startsWith("image/")) {
      alert("Please select an image file");
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      const result = e.target?.result as string;
      currentImage = result; // Full data URL
      currentImagePreview = result;
    };
    reader.readAsDataURL(file);
  }

  function handlePaste(event: ClipboardEvent) {
    const items = event.clipboardData?.items;
    if (!items) return;

    for (const item of items) {
      if (item.type.startsWith("image/")) {
        event.preventDefault();
        const file = item.getAsFile();
        if (file) processImageFile(file);
        break;
      }
    }
  }

  function handleDrop(event: DragEvent) {
    event.preventDefault();
    isDragging = false;

    const files = event.dataTransfer?.files;
    if (files && files[0]) {
      processImageFile(files[0]);
    }
  }

  function handleDragOver(event: DragEvent) {
    event.preventDefault();
    isDragging = true;
  }

  function handleDragLeave() {
    isDragging = false;
  }

  function clearImage() {
    currentImage = null;
    currentImagePreview = null;
    if (imageInputRef) imageInputRef.value = "";
  }

  onMount(() => {
    loadInferenceStatus();
    // Refresh status every 10 seconds
    const interval = setInterval(loadInferenceStatus, 10000);
    return () => clearInterval(interval);
  });
</script>

<svelte:head>
  <title>Inference - Model Garden</title>
</svelte:head>

<div class="min-h-screen bg-gray-50 pt-6">
  <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
    <!-- Header -->
    <div class="flex justify-between items-center mb-8">
      <div>
        <div class="flex items-center gap-4">
          <h1 class="text-3xl font-bold text-gray-900">Inference</h1>
          {#if inferenceStatus?.loaded}
            <Badge variant="success">Model Loaded</Badge>
          {:else if !loadingStatus}
            <Badge variant="warning">No Model</Badge>
          {/if}
        </div>
        <p class="mt-2 text-sm text-gray-600">
          Chat with your models or run completions
        </p>
      </div>
      <div class="flex gap-3">
        <Button
          onclick={() => (showSettings = !showSettings)}
          variant="secondary"
        >
          ⚙️ Settings
        </Button>
        <Button onclick={clearConversation} variant="secondary">
          🗑️ Clear
        </Button>
      </div>
    </div>
    <div class="grid grid-cols-1 lg:grid-cols-4 gap-6">
      <!-- Main Chat Area -->
      <div class="lg:col-span-3">
        <Card class="h-[calc(100vh-16rem)] flex flex-col">
          <!-- Messages -->
          <div class="flex-1 overflow-y-auto p-6 space-y-4">
            {#if messages.length === 0}
              <div class="text-center py-12">
                <div class="text-6xl mb-4">💬</div>
                <h3 class="text-xl font-semibold text-gray-700 mb-2">
                  Start a Conversation
                </h3>
                <p class="text-gray-500">
                  {inferenceStatus?.model_info?.is_vision_adapter ||
                  inferenceStatus?.model_info?.model_path
                    ?.toLowerCase()
                    .includes("vl")
                    ? "Vision model loaded! You can send images with your messages."
                    : "Select a model and type a message below"}
                </p>
              </div>
            {:else}
              {#each messages as message, index}
                <div
                  class="flex {message.role === 'user'
                    ? 'justify-end'
                    : 'justify-start'}"
                >
                  <div
                    class="max-w-[80%] rounded-lg px-4 py-3 {message.role ===
                    'user'
                      ? 'bg-primary-600 text-white'
                      : 'bg-gray-100 text-gray-900'}"
                  >
                    <div class="flex items-start gap-2">
                      <div class="flex-1">
                        <div class="text-xs opacity-70 mb-1">
                          {message.role === "user" ? "You" : "Assistant"} · {formatTime(
                            message.timestamp,
                          )}
                        </div>
                        <!-- Show image if present -->
                        {#if message.imagePreview}
                          <div class="mb-2">
                            <img
                              src={message.imagePreview}
                              alt="Attached"
                              class="max-w-[200px] max-h-[150px] rounded-lg object-cover"
                            />
                          </div>
                        {/if}
                        <div class="whitespace-pre-wrap break-words">
                          {message.content}{#if isGenerating && index === messages.length - 1 && message.role === "assistant"}<span
                              class="inline-block w-1 h-4 bg-gray-600 animate-pulse ml-1"
                            ></span>{/if}
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              {/each}
            {/if}
          </div>

          <!-- Input Area -->
          <div
            class="border-t border-gray-200 p-4 {isDragging
              ? 'bg-primary-50 border-primary-300'
              : ''}"
            ondrop={handleDrop}
            ondragover={handleDragOver}
            ondragleave={handleDragLeave}
            role="region"
          >
            <!-- Image Preview -->
            {#if currentImagePreview}
              <div class="mb-3 flex items-start gap-2">
                <div class="relative inline-block">
                  <img
                    src={currentImagePreview}
                    alt="To send"
                    class="max-w-[120px] max-h-[80px] rounded-lg object-cover border border-gray-300"
                  />
                  <button
                    type="button"
                    onclick={clearImage}
                    class="absolute -top-2 -right-2 w-5 h-5 bg-red-500 text-white rounded-full flex items-center justify-center text-xs hover:bg-red-600"
                    aria-label="Remove image"
                  >
                    ✕
                  </button>
                </div>
                <span class="text-xs text-gray-500 mt-1">Image attached</span>
              </div>
            {/if}

            <div class="flex gap-3">
              <!-- Image Upload Button -->
              <input
                type="file"
                accept="image/*"
                onchange={handleImageSelect}
                bind:this={imageInputRef}
                class="hidden"
                id="image-input"
              />
              <button
                type="button"
                onclick={() => imageInputRef?.click()}
                disabled={isGenerating ||
                  !inferenceStatus?.loaded ||
                  loadingStatus}
                class="px-3 py-2 border border-gray-300 rounded-lg hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-1 text-gray-600 self-end"
                title="Attach image (or paste/drag)"
              >
                🖼️
              </button>

              <textarea
                bind:value={currentInput}
                onkeydown={handleKeyDown}
                onpaste={handlePaste}
                placeholder={isGenerating
                  ? "Generating..."
                  : inferenceStatus?.loaded
                    ? "Type your message... (Enter to send, Ctrl+V to paste image)"
                    : "Load a model first..."}
                rows="2"
                disabled={isGenerating ||
                  !inferenceStatus?.loaded ||
                  loadingStatus}
                class="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500 resize-none disabled:bg-gray-100 disabled:cursor-not-allowed"
              ></textarea>
              <Button
                onclick={sendMessage}
                variant="primary"
                disabled={(!currentInput.trim() && !currentImage) ||
                  isGenerating ||
                  !inferenceStatus?.loaded ||
                  loadingStatus}
                loading={isGenerating}
                class="self-end"
              >
                {isGenerating ? "..." : "Send"}
              </Button>
            </div>
            {#if isDragging}
              <div class="mt-2 text-sm text-primary-600 text-center">
                Drop image here
              </div>
            {/if}
          </div>
        </Card>
      </div>

      <!-- Sidebar -->
      <div class="space-y-6">
        <!-- Model Loader -->
        <ModelLoader
          compact
          onModelLoaded={loadInferenceStatus}
          onModelUnloaded={loadInferenceStatus}
        />

        <!-- GPU Memory Profile -->
        {#if inferenceStatus?.loaded}
          <GpuMemoryProfile compact refreshInterval={10000} />
        {/if}

        <!-- Mode Selection -->
        <Card>
          <div class="p-4">
            <h3 class="text-lg font-semibold text-gray-900 mb-3">Mode</h3>
            <div class="space-y-2">
              <label class="flex items-center">
                <input
                  type="radio"
                  bind:group={mode}
                  value="chat"
                  class="mr-2"
                />
                <span class="text-sm">Chat (with context)</span>
              </label>
              <label class="flex items-center">
                <input
                  type="radio"
                  bind:group={mode}
                  value="completion"
                  class="mr-2"
                />
                <span class="text-sm">Completion (stateless)</span>
              </label>
            </div>
          </div>
        </Card>

        <!-- Settings Panel -->
        {#if showSettings}
          <Card>
            <div class="p-4">
              <h3 class="text-lg font-semibold text-gray-900 mb-3">
                Generation Settings
              </h3>

              <div class="space-y-4">
                <div>
                  <label
                    for="temperature"
                    class="block text-sm font-medium text-gray-700 mb-1"
                  >
                    Temperature: {settings.temperature}
                  </label>
                  <input
                    type="range"
                    id="temperature"
                    bind:value={settings.temperature}
                    min="0.0"
                    max="2.0"
                    step="0.1"
                    class="w-full"
                  />
                  <p class="text-xs text-gray-500 mt-1">
                    Higher = more creative
                  </p>
                </div>

                <div>
                  <label
                    for="max-tokens"
                    class="block text-sm font-medium text-gray-700 mb-1"
                  >
                    Max Tokens
                  </label>
                  <input
                    type="number"
                    id="max-tokens"
                    bind:value={settings.max_tokens}
                    min="1"
                    max="2000"
                    class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 text-sm"
                  />
                </div>

                <div>
                  <label
                    for="top-p"
                    class="block text-sm font-medium text-gray-700 mb-1"
                  >
                    Top P: {settings.top_p}
                  </label>
                  <input
                    type="range"
                    id="top-p"
                    bind:value={settings.top_p}
                    min="0.0"
                    max="1.0"
                    step="0.05"
                    class="w-full"
                  />
                </div>

                <div>
                  <label
                    for="top-k"
                    class="block text-sm font-medium text-gray-700 mb-1"
                  >
                    Top K
                  </label>
                  <input
                    type="number"
                    id="top-k"
                    bind:value={settings.top_k}
                    min="1"
                    max="100"
                    class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 text-sm"
                  />
                </div>

                <div>
                  <label class="flex items-center">
                    <input
                      type="checkbox"
                      bind:checked={settings.stream}
                      class="mr-2"
                    />
                    <span class="text-sm font-medium text-gray-700"
                      >Enable Streaming</span
                    >
                  </label>
                </div>
              </div>
            </div>
          </Card>

          {#if mode === "chat"}
            <Card>
              <div class="p-4">
                <h3 class="text-lg font-semibold text-gray-900 mb-3">
                  System Prompt
                </h3>
                <textarea
                  bind:value={systemPrompt}
                  rows="4"
                  class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 text-sm resize-none"
                  placeholder="Set the assistant's behavior..."
                ></textarea>
              </div>
            </Card>
          {/if}
        {/if}

        <!-- Stats -->
        <Card>
          <div class="p-4">
            <h3 class="text-lg font-semibold text-gray-900 mb-3">
              Conversation
            </h3>
            <div class="space-y-2 text-sm">
              <div class="flex justify-between">
                <span class="text-gray-600">Messages:</span>
                <span class="font-medium">{messages.length}</span>
              </div>
              <div class="flex justify-between">
                <span class="text-gray-600">Status:</span>
                <span
                  class="font-medium {isGenerating
                    ? 'text-green-600'
                    : 'text-gray-900'}"
                >
                  {isGenerating ? "🟢 Generating" : "⚪ Idle"}
                </span>
              </div>
            </div>
          </div>
        </Card>
      </div>
    </div>
  </div>
</div>
