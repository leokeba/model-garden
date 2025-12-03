<script lang="ts">
  import { goto } from "$app/navigation";
  import { page } from "$app/stores";
  import ModelLoader from "$lib/components/ModelLoader.svelte";
  import { onMount } from "svelte";

  let selectedModelPath = $state("");
  let useHuggingFaceHub = $state(false);
  let huggingFaceModelId = $state("");

  onMount(() => {
    // Pre-select model from URL parameter
    const modelParam = $page.url.searchParams.get("model");
    const hfModelParam = $page.url.searchParams.get("hf_model");

    if (hfModelParam) {
      // HuggingFace model from browse page
      useHuggingFaceHub = true;
      huggingFaceModelId = decodeURIComponent(hfModelParam);
    } else if (modelParam) {
      // Local model parameter
      useHuggingFaceHub = false;
      selectedModelPath = decodeURIComponent(modelParam);
    }
  });

  function handleModelLoaded() {
    // Redirect to inference page after model is loaded
    goto("/inference");
  }
</script>

<svelte:head>
  <title>Load Model - Model Garden</title>
</svelte:head>

<div class="min-h-screen bg-gray-50 pt-6">
  <div class="max-w-2xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
    <!-- Header -->
    <div class="mb-8">
      <h1 class="text-3xl font-bold text-gray-900">Load Model for Inference</h1>
      <p class="mt-2 text-sm text-gray-600">
        Load a model from your local storage or HuggingFace Hub
      </p>
    </div>

    <!-- Model Loader Component -->
    <ModelLoader {selectedModelPath} onModelLoaded={handleModelLoaded} />

    <!-- Help Text -->
    <div class="mt-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
      <h3 class="font-medium text-blue-900 mb-2">💡 Tips</h3>
      <ul class="text-sm text-blue-800 space-y-1">
        <li>• Only one model can be loaded at a time for inference</li>
        <li>• Loading a new model will automatically unload the current one</li>
        <li>
          • 🤗 <strong>HuggingFace Hub:</strong> Load models directly using model
          IDs
        </li>
        <li>
          • <strong>Local Models:</strong> Use models you've trained or downloaded
        </li>
        <li>
          • <strong>GPU Memory = Auto (0%):</strong> Calculates optimal memory based
          on model size
        </li>
        <li>
          • <strong>Disable CUDA Graphs:</strong> Saves ~2GB memory but slower inference
        </li>
        <li>
          • After loading, you'll be redirected to the <a
            href="/inference"
            class="underline font-medium">Inference page</a
          >
        </li>
      </ul>
    </div>
  </div>
</div>
